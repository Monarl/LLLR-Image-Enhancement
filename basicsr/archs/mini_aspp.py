"""
Mini-ASPP (Atrous Spatial Pyramid Pooling) Semantic Feature Extraction Module

Replaces the heavy external HRNet backbone (~40M+ params) used in UltraIS/UltraBM
with a lightweight internal semantic extraction module (<1M params total).

The Mini-ASPP extracts multi-scale semantic context from ASSG block output features
(R_fk) to produce semantic features (S_fk) for ISDM-lite dual modulation.

Architecture (per stage):
    Input R_fk [B, C, H, W]
         │
         ├──► Branch 1: Conv 1x1 ──────────────────────► Local details
         │
         ├──► Branch 2: DW-Conv 3x3 (d=2) + PW 1x1 ──► Short-range context
         │
         ├──► Branch 3: DW-Conv 3x3 (d=4) + PW 1x1 ──► Mid-range context
         │
         ├──► Branch 4: DW-Conv 3x3 (d=6) + PW 1x1 ──► Long-range context
         │
         └──► Fusion: Concat [B1,B2,B3,B4] → Conv 1x1 → (+x) → S_fk [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MiniASPP(nn.Module):
    """
    Mini Atrous Spatial Pyramid Pooling module for semantic feature extraction.

    Applies parallel dilated convolution branches to capture context at multiple
    receptive field scales, then fuses them into a single semantic feature map.
    A residual skip connection is added when in_channels == out_channels.

    Args:
        in_channels (int): Number of input channels (embed_dim from MambaIRv2).
        out_channels (int, optional): Number of output channels. Defaults to in_channels.
        dilations (tuple): Dilation rates for branches 2, 3, and 4. Default: (2, 4, 6).
        bias (bool): Whether to use bias in convolution layers. Default: False.

    Input:
        x: Feature tensor [B, in_channels, H, W] from ASSG block output (R_fk).

    Output:
        Semantic feature tensor [B, out_channels, H, W] (S_fk).

    Example:
        >>> mini_aspp = MiniASPP(in_channels=48)
        >>> x = torch.randn(2, 48, 64, 64)
        >>> s_fk = mini_aspp(x)
        >>> print(s_fk.shape)  # [2, 48, 64, 64]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dilations: Tuple[int, int, int] = (2, 4, 6),
        bias: bool = False,
    ):
        super(MiniASPP, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations

        # Branch 1: Conv 1x1 — captures local channel-wise details
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Branch 2: Depthwise Conv 3x3 (dilation=2) + Pointwise Conv 1x1
        # Effective receptive field: 5x5 — short-range context
        d1 = dilations[0]
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=d1, dilation=d1,
                groups=in_channels, bias=bias,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Branch 3: Depthwise Conv 3x3 (dilation=4) + Pointwise Conv 1x1
        # Effective receptive field: 9x9 — mid-range context
        d2 = dilations[1]
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=d2, dilation=d2,
                groups=in_channels, bias=bias,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Branch 4: Depthwise Conv 3x3 (dilation=6) + Pointwise Conv 1x1
        # Effective receptive field: 13x13 — long-range context
        d3 = dilations[2]
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=d3, dilation=d3,
                groups=in_channels, bias=bias,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Fusion: Concatenate all 4 branches → reduce to output channels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Residual skip only when channels are preserved
        self.use_residual = (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mini-ASPP.

        Args:
            x: Input feature tensor [B, in_channels, H, W] from ASSG block (R_fk).

        Returns:
            Semantic feature tensor [B, out_channels, H, W] (S_fk).
        """
        b1 = self.branch1(x)   # [B, C, H, W] — local
        b2 = self.branch2(x)   # [B, C, H, W] — short-range  (d=2)
        b3 = self.branch3(x)   # [B, C, H, W] — mid-range    (d=4)
        b4 = self.branch4(x)   # [B, C, H, W] — long-range   (d=6)

        fused = torch.cat([b1, b2, b3, b4], dim=1)  # [B, 4*C, H, W]
        out = self.fusion(fused)                     # [B, out_channels, H, W]

        if self.use_residual:
            out = out + x

        return out

    def count_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, input_resolution: Tuple[int, int]) -> float:
        """
        Estimate FLOPs for given input resolution.

        Args:
            input_resolution: (H, W) spatial dimensions.

        Returns:
            Approximate FLOPs count.
        """
        h, w = input_resolution
        hw = h * w
        c_in = self.in_channels
        c_out = self.out_channels

        # Branch 1: Conv 1x1 → c_in * c_in * HW
        flops_b1 = c_in * c_in * hw

        # Branches 2/3/4: DW 3x3 → c_in * 9 * HW, PW 1x1 → c_in * c_in * HW
        flops_dilated = (c_in * 9 * hw + c_in * c_in * hw) * 3

        # Fusion: Conv 1x1 → 4*c_in * c_out * HW
        flops_fusion = 4 * c_in * c_out * hw

        return flops_b1 + flops_dilated + flops_fusion


# ---------------------------------------------------------------------------
#  Factory: create nn.ModuleList of MiniASPP for N ASSG stages
# ---------------------------------------------------------------------------

def create_mini_aspp_stages(
    num_stages: int,
    in_channels: int,
    out_channels: Optional[int] = None,
    dilations: Tuple[int, int, int] = (2, 4, 6),
    bias: bool = False,
) -> nn.ModuleList:
    """
    Create an nn.ModuleList of independent MiniASPP instances, one per ASSG stage.

    Usage in MambaIRv2 forward_features():
        self.mini_aspp_list = create_mini_aspp_stages(num_stages=4, in_channels=48)
        ...
        for k, layer in enumerate(self.layers):
            x = layer(x, x_size, params)
            x_2d = x.transpose(1, 2).view(B, C, H, W)
            s_fk = self.mini_aspp_list[k](x_2d)  # [B, 48, H, W]

    Args:
        num_stages: Number of ASSG stages (len(depths) in MambaIRv2).
        in_channels: Input channels per stage (embed_dim).
        out_channels: Output channels per stage. Defaults to in_channels.
        dilations: Dilation rates for branches 2, 3, and 4.
        bias: Whether to use bias in convolutions.

    Returns:
        nn.ModuleList of MiniASPP instances.
    """
    return nn.ModuleList([
        MiniASPP(in_channels, out_channels, dilations, bias)
        for _ in range(num_stages)
    ])


if __name__ == "__main__":
    # ===== Unit Tests =====
    import torch

    print("=" * 60)
    print("Mini-ASPP Module Unit Tests")
    print("=" * 60)

    # Test 1: Basic MiniASPP with residual
    print("\n--- Test 1: Basic MiniASPP (embed_dim=48, residual active) ---")
    mini_aspp = MiniASPP(in_channels=48)
    x = torch.randn(2, 48, 64, 64)
    s_fk = mini_aspp(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {s_fk.shape}")
    print(f"Parameters:   {mini_aspp.count_params():,}")
    print(f"FLOPs (64x64): {mini_aspp.count_flops((64, 64)):,.0f}")
    print(f"Residual:     {mini_aspp.use_residual}")
    assert s_fk.shape == (2, 48, 64, 64), f"Shape mismatch: {s_fk.shape}"
    assert mini_aspp.use_residual
    print("PASSED")

    # Test 2: Different out_channels — no residual
    print("\n--- Test 2: MiniASPP with different out_channels (no residual) ---")
    mini_aspp2 = MiniASPP(in_channels=48, out_channels=64)
    s_fk2 = mini_aspp2(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {s_fk2.shape}")
    print(f"Parameters:   {mini_aspp2.count_params():,}")
    print(f"Residual:     {mini_aspp2.use_residual}")
    assert s_fk2.shape == (2, 64, 64, 64), f"Shape mismatch: {s_fk2.shape}"
    assert not mini_aspp2.use_residual
    print("PASSED")

    # Test 3: create_mini_aspp_stages factory
    print("\n--- Test 3: create_mini_aspp_stages (4 stages) ---")
    aspp_stages = create_mini_aspp_stages(num_stages=4, in_channels=48)
    print(f"Number of stages: {len(aspp_stages)}")
    total_params = sum(p.numel() for p in aspp_stages.parameters() if p.requires_grad)
    for k in range(4):
        r_fk = torch.randn(2, 48, 64, 64)
        s_fk = aspp_stages[k](r_fk)
        print(f"  Stage {k}: {s_fk.shape}")
        assert s_fk.shape == (2, 48, 64, 64), f"Stage {k} shape mismatch: {s_fk.shape}"
    print(f"Total parameters: {total_params:,}")
    print("PASSED")

    # Test 4: Independent weights verification
    print("\n--- Test 4: Independent weights (stages don't share) ---")
    aspp_stages2 = create_mini_aspp_stages(num_stages=4, in_channels=48)
    for k in range(1, 4):
        assert aspp_stages2[k] is not aspp_stages2[0], \
            f"Stage {k} should not be same object as stage 0"
    print("All stages are independent instances")
    print("PASSED")

    # Test 5: Gradient flow
    print("\n--- Test 5: Gradient Flow ---")
    mini_aspp_grad = MiniASPP(in_channels=48)
    x_grad = torch.randn(1, 48, 32, 32, requires_grad=True)
    out = mini_aspp_grad(x_grad)
    loss = out.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient on input"
    assert x_grad.grad.shape == x_grad.shape, "Gradient shape mismatch"
    for name, param in mini_aspp_grad.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("Gradient flows through all parameters")
    print("PASSED")

    # Test 6: Different input sizes
    print("\n--- Test 6: Various Input Sizes ---")
    mini_aspp_size = MiniASPP(in_channels=48)
    for h, w in [(32, 32), (64, 64), (128, 128), (48, 64), (96, 72)]:
        x_var = torch.randn(1, 48, h, w)
        out_var = mini_aspp_size(x_var)
        assert out_var.shape == (1, 48, h, w), f"Failed for {h}x{w}: {out_var.shape}"
        print(f"  {h}x{w}: OK")
    print("PASSED")

    # Test 7: Parameter comparison with HRNet
    print("\n--- Test 7: Parameter Count Comparison ---")
    hrnet_params = 40_000_000  # ~40M for HRNet-W48
    ms_params = total_params
    ratio = hrnet_params / ms_params
    print(f"HRNet-W48 params:         ~{hrnet_params:>12,}")
    print(f"MiniASPP (4 stages):       {ms_params:>12,}")
    print(f"Reduction ratio:           {ratio:>12.1f}x lighter")
    assert ms_params < 1_000_000, \
        f"Mini-ASPP should be < 1M params, got {ms_params:,}"
    print("PASSED (< 1M params)")

    print("\n" + "=" * 60)
    print("All tests passed! Mini-ASPP module is ready for integration.")
    print("=" * 60)
