"""
MobileNetV3-Small Semantic Feature Extractor

Replaces Mini-ASPP with a frozen pretrained MobileNetV3-Small backbone for
semantic feature extraction. Provides true semantic understanding (trained on
ImageNet) at a fraction of HRNet's ~65M cost (~2.5M params, frozen).

Unlike Mini-ASPP which operates on ASSG intermediate features, this module
processes the input image directly and provides multi-scale semantic features
at each ASSG stage through feature projection and interpolation.

Architecture:
    Input Image [B, 3, H, W]
         │
         ▼
    ┌───────────────────────────────────────┐
    │ MobileNetV3-Small (Frozen)            │
    │   ├── Layer 0-3:  features_low  (24ch)│
    │   ├── Layer 4-8:  features_mid  (40ch)│
    │   └── Layer 9-12: features_high (96ch)│
    └───────────────────────────────────────┘
         │
         ▼
    ┌───────────────────────────────────────┐
    │ Feature Fusion & Projection           │
    │   Concat multi-scale → Conv → embed_dim│
    └───────────────────────────────────────┘
         │
    S_fk [B, embed_dim, H, W] (per stage)

Usage similar to Mini-ASPP but extracts features from the original image:
    # In MambaIRv2LLIESR
    s_features = self.semantic_extractor(input_image)  # Extract once
    for k, layer in enumerate(self.layers):
        x = layer(x, ...)
        s_fk = s_features[k]  # Use cached feature for this stage
        m_fk = self.isdm_stages[k](x, i_fk, s_fk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

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

        # Total channels after concatenation (upsampled to same resolution)
        total_channels = sum(self.tap_channels)  # 24 + 48 + 96 = 168

        # Fusion: combine multi-scale features → project to embed_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Per-stage refinement (lightweight) - allows stage-specific adaptation
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

        # Upsample all features to the highest resolution (H/4 from tap[0])
        target_size = multi_scale_features[0].shape[2:]  # H/4 x W/4

        upsampled_features = []
        for feat in multi_scale_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            upsampled_features.append(feat)

        # Concatenate multi-scale features
        fused = torch.cat(upsampled_features, dim=1)  # [B, 168, H/4, W/4]

        # Apply fusion to get embed_dim channels
        fused = self.fusion(fused)  # [B, embed_dim, H/4, W/4]

        # Upsample to input resolution for compatibility with ASSG output
        fused = F.interpolate(
            fused, size=(H, W),
            mode='bilinear', align_corners=False
        )  # [B, embed_dim, H, W]

        # Generate per-stage features with stage-specific refinement
        stage_features = []
        for k in range(self.num_stages):
            s_fk = self.stage_refine[k](fused)
            if k > 0:
                # Add residual connection from fused for deeper stages
                s_fk = s_fk + fused
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

    # Check that fusion and refinement have gradients
    fusion_grads = [p.grad for p in extractor_grad.fusion.parameters() if p.grad is not None]
    assert len(fusion_grads) > 0, "Fusion should have gradients"
    print("Fusion layers have gradients")

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
