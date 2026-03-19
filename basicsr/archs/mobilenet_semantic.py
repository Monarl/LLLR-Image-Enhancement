"""
MobileNetV3-Small Semantic Feature Extractor

Architecture:
    Input Image [B, 3, H, W]
         │
         ▼
    ┌───────────────────────────────────────┐
    │ MobileNetV3-Small (Frozen)            │
    │   ├── Layer 0-3:  features_low  (24ch)│
    │   ├── Layer 4-8:  features_mid  (48ch)│
    │   └── Layer 9-11: features_high (96ch)│
    └───────────────────────────────────────┘
         │
         ▼
    ┌───────────────────────────────────────┐
    │ Per-scale Projection                  │
    │   Project each tap → embed_dim        │
    └───────────────────────────────────────┘
         │
         ▼
    ┌───────────────────────────────────────┐
    │ Per-stage Multi-scale Fusion          │
    │   Stage k learns its own mixture of   │
    │   low/mid/high semantic taps          │
    └───────────────────────────────────────┘
         │
    S_fk [B, embed_dim, H, W] (unique per stage)

Usage similar to Mini-ASPP but extracts features from the original image:
    # In MambaIRv2LLIESR
    s_features = self.semantic_extractor(input_image)  # Extract once
    for k, layer in enumerate(self.layers):
        x = layer(x, ...)
        s_fk = s_features[k]  # Stage-specific MobileNet semantic feature
        m_fk = self.isdm_stages[k](x, i_fk, s_fk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class MobileNetV3SemanticExtractor(nn.Module):
    """
    MobileNetV3-Small based semantic feature extractor.

    Extracts multi-scale features from a frozen MobileNetV3-Small backbone
    and projects them to the target embedding dimension for each ASSG stage.

    Args:
        embed_dim (int): Output channel dimension (to match ASSG embed_dim). Default: 48.
        num_stages (int): Number of ASSG stages to provide features for. Default: 4.
        freeze_backbone (bool): Whether to freeze MobileNetV3 weights. Default: True.
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: True.

    Input:
        x: RGB image tensor [B, 3, H, W] in [0, 1] range.

    Output:
        List of num_stages semantic feature tensors, each [B, embed_dim, H, W].
    """

    def __init__(
        self,
        embed_dim: int = 48,
        num_stages: int = 4,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for MobileNetV3SemanticExtractor. "
                "Install with: pip install torchvision"
            )

        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.freeze_backbone = freeze_backbone

        # Load MobileNetV3-Small with ImageNet pretrained weights
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # MobileNetV3-Small feature layer structure:
        # Layer  0: Conv2d(3, 16, 3, 2) + BN + HSwish  - 16ch, H/2
        # Layer  1: InvertedResidual (16→16)          - 16ch, H/2
        # Layer  2: InvertedResidual (16→24, stride=2) - 24ch, H/4
        # Layer  3: InvertedResidual (24→24)          - 24ch, H/4
        # Layer  4: InvertedResidual (24→40, stride=2) - 40ch, H/8
        # Layer  5: InvertedResidual (40→40)          - 40ch, H/8
        # Layer  6: InvertedResidual (40→40)          - 40ch, H/8
        # Layer  7: InvertedResidual (40→48)          - 48ch, H/8
        # Layer  8: InvertedResidual (48→48)          - 48ch, H/8
        # Layer  9: InvertedResidual (48→96, stride=2) - 96ch, H/16
        # Layer 10: InvertedResidual (96→96)          - 96ch, H/16
        # Layer 11: InvertedResidual (96→96)          - 96ch, H/16
        # Layer 12: Conv2d+BN+HSwish (96→576)         - 576ch, H/16

        # We'll tap into three scales:
        # low:  after layer 3 (24ch, H/4)
        # mid:  after layer 8 (48ch, H/8)
        # high: after layer 11 (96ch, H/16)

        self.tap_indices = [3, 8, 11]  # Feature tap points
        self.tap_channels = [24, 48, 96]  # Output channels at each tap

        self.num_scales = len(self.tap_channels)

        # Project each MobileNet tap to the ASSG embedding dimension first.
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for channels in self.tap_channels
        ])

        # Each ASSG stage gets its own multi-scale fusion head
        self.stage_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * self.num_scales, embed_dim * 2,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_stages)
        ])

        # Stage-dependent scale weights encourage shallow ASSG stages to use
        # lower-level taps first and deeper stages to use higher-level taps.
        self.stage_scale_logits = nn.Parameter(self._init_stage_scale_logits())

        # Lightweight post-fusion refinement per stage.
        self.stage_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1,
                          groups=embed_dim, bias=False),  # Depthwise
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),  # Pointwise
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            for _ in range(num_stages)
        ])

        # ImageNet normalization constants
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _init_stage_scale_logits(self) -> torch.Tensor:
        """Bias early stages toward low-level taps and later stages toward high-level taps."""
        stage_positions = torch.linspace(0.0, 1.0, steps=self.num_stages).unsqueeze(1)
        scale_positions = torch.linspace(0.0, 1.0, steps=self.num_scales).unsqueeze(0)
        return -2.0 * torch.abs(stage_positions - scale_positions)

    def _extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from MobileNetV3-Small backbone.

        Args:
            x: Input tensor [B, 3, H, W] normalized for ImageNet.

        Returns:
            List of feature tensors at different scales.
        """
        features = []

        # Get the feature extractor layers
        backbone_features = self.backbone.features

        for idx, layer in enumerate(backbone_features):
            x = layer(x)
            if idx in self.tap_indices:
                features.append(x)

        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract semantic features for all ASSG stages.

        Args:
            x: Input RGB image [B, 3, H, W] in [0, 1] range.

        Returns:
            List of num_stages semantic feature tensors, each [B, embed_dim, H, W].
        """
        B, C, H, W = x.shape

        # Normalize for ImageNet (backbone expects ImageNet normalization)
        x_norm = (x - self.mean) / self.std

        # Set backbone to eval mode during inference (important for BatchNorm)
        if self.freeze_backbone:
            self.backbone.eval()

        # Extract multi-scale features
        with torch.set_grad_enabled(not self.freeze_backbone):
            multi_scale_features = self._extract_backbone_features(x_norm)

        projected_features = []
        for projector, feat in zip(self.scale_projectors, multi_scale_features):
            feat = projector(feat)
            if feat.shape[2:] != (H, W):
                feat = F.interpolate(
                    feat, size=(H, W),
                    mode='bilinear', align_corners=False
                )
            projected_features.append(feat)

        stage_features = []
        for k in range(self.num_stages):
            scale_weights = torch.softmax(self.stage_scale_logits[k], dim=0)
            weighted_features = [
                feat * scale_weights[scale_idx]
                for scale_idx, feat in enumerate(projected_features)
            ]
            fused = self.stage_fusion[k](torch.cat(weighted_features, dim=1))
            s_fk = self.stage_refine[k](fused) + fused
            stage_features.append(s_fk)

        return stage_features

    def count_params(self, trainable_only: bool = True) -> int:
        """
        Count parameters.

        Args:
            trainable_only: If True, count only trainable params. Default: True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def count_backbone_params(self) -> int:
        """Count backbone (MobileNetV3) parameters."""
        return sum(p.numel() for p in self.backbone.parameters())


def create_mobilenet_semantic_extractor(
    embed_dim: int = 48,
    num_stages: int = 4,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> MobileNetV3SemanticExtractor:
    """
    Factory function to create MobileNetV3-based semantic feature extractor.

    Args:
        embed_dim: Output channel dimension (should match ASSG embed_dim).
        num_stages: Number of ASSG stages.
        freeze_backbone: Whether to freeze MobileNetV3 weights.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        MobileNetV3SemanticExtractor instance.
    """
    return MobileNetV3SemanticExtractor(
        embed_dim=embed_dim,
        num_stages=num_stages,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    # ===== Unit Tests =====
    import torch

    print("=" * 60)
    print("MobileNetV3 Semantic Extractor Unit Tests")
    print("=" * 60)

    # Test 1: Basic instantiation and forward pass
    print("\n--- Test 1: Basic Forward Pass ---")
    extractor = MobileNetV3SemanticExtractor(embed_dim=48, num_stages=4)
    x = torch.randn(2, 3, 64, 64)
    features = extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"Number of output features: {len(features)}")
    for k, f in enumerate(features):
        print(f"  Stage {k}: {f.shape}")
        assert f.shape == (2, 48, 64, 64), f"Shape mismatch at stage {k}"
    print("PASSED")

    # Test 2: Parameter counts
    print("\n--- Test 2: Parameter Counts ---")
    trainable_params = extractor.count_params(trainable_only=True)
    total_params = extractor.count_params(trainable_only=False)
    backbone_params = extractor.count_backbone_params()
    print(f"Backbone (MobileNetV3-Small) params: {backbone_params:,}")
    print(f"Total params (frozen + trainable): {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable fraction: {trainable_params / total_params * 100:.1f}%")
    # Backbone should be ~2.5M params
    assert 2_000_000 < backbone_params < 3_000_000, \
        f"Unexpected backbone param count: {backbone_params}"
    print("PASSED")

    # Test 3: Gradient flow (only trainable parts)
    print("\n--- Test 3: Gradient Flow ---")
    extractor_grad = MobileNetV3SemanticExtractor(
        embed_dim=48, num_stages=4, freeze_backbone=True
    )
    x_grad = torch.randn(1, 3, 32, 32, requires_grad=True)
    feats = extractor_grad(x_grad)
    loss = sum(f.sum() for f in feats)
    loss.backward()

    # Check that backbone has no gradient
    backbone_grads = [p.grad for p in extractor_grad.backbone.parameters() if p.grad is not None]
    assert len(backbone_grads) == 0, "Backbone should have no gradients when frozen"
    print("Backbone correctly frozen (no gradients)")

    # Check that stage-wise semantic heads have gradients
    projector_grads = [
        p.grad for p in extractor_grad.scale_projectors.parameters()
        if p.grad is not None
    ]
    assert len(projector_grads) > 0, "Scale projectors should have gradients"
    print("Scale projectors have gradients")

    fusion_grads = [
        p.grad for p in extractor_grad.stage_fusion.parameters()
        if p.grad is not None
    ]
    assert len(fusion_grads) > 0, "Stage fusion should have gradients"
    print("Stage fusion layers have gradients")

    refine_grads = [p.grad for p in extractor_grad.stage_refine.parameters() if p.grad is not None]
    assert len(refine_grads) > 0, "Stage refinement should have gradients"
    print("Stage refinement layers have gradients")
    print("PASSED")

    # Test 4: Different input sizes
    print("\n--- Test 4: Various Input Sizes ---")
    extractor_size = MobileNetV3SemanticExtractor(embed_dim=48, num_stages=4)
    for h, w in [(32, 32), (64, 64), (128, 128), (48, 64), (96, 72)]:
        x_var = torch.randn(1, 3, h, w)
        feats_var = extractor_size(x_var)
        for k, f in enumerate(feats_var):
            assert f.shape == (1, 48, h, w), f"Failed for {h}x{w} at stage {k}: {f.shape}"
        print(f"  {h}x{w}: OK")
    print("PASSED")

    # Test 5: Comparison with Mini-ASPP params
    print("\n--- Test 5: Parameter Comparison ---")
    from mini_aspp import create_mini_aspp_stages
    mini_aspp_stages = create_mini_aspp_stages(num_stages=4, in_channels=48)
    mini_aspp_params = sum(p.numel() for p in mini_aspp_stages.parameters())

    print(f"Mini-ASPP (4 stages):     {mini_aspp_params:>10,} (all trainable)")
    print(f"MobileNetV3 (trainable):  {trainable_params:>10,}")
    print(f"MobileNetV3 (backbone):   {backbone_params:>10,} (frozen)")
    print(f"MobileNetV3 (total):      {total_params:>10,}")
    print("PASSED")

    # Test 6: Factory function
    print("\n--- Test 6: Factory Function ---")
    extractor_factory = create_mobilenet_semantic_extractor(
        embed_dim=48, num_stages=4, freeze_backbone=True, pretrained=True
    )
    x_factory = torch.randn(1, 3, 64, 64)
    feats_factory = extractor_factory(x_factory)
    assert len(feats_factory) == 4
    print("Factory function works correctly")
    print("PASSED")

    print("\n" + "=" * 60)
    print("All tests passed! MobileNetV3 Semantic Extractor is ready.")
    print("=" * 60)
