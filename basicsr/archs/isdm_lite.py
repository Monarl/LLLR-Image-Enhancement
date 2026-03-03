"""
ISDM-lite (Illumination-Semantic Dual Modulation — Lightweight)

Adapted from UltraIS/UltraBM TransformerBlock (cross-attention modulation) for
integration with MambaIRv2 ASSG blocks in a low-light super-resolution pipeline.

Architecture per stage:

    R_fk ──► IMU(I_fk) ──► R̃_fk ──► SMU(S_fk) ──► M_fk
                ↑                        ↑
          Illumination              Semantic
          features (IGM)         features (Mini-ASPP)

Each modulation unit (IMU / SMU) performs:
    1. LayerNorm both inputs
    2. Channel cross-attention: Q from guidance, KV from main feature
    3. Residual connection
    4. LayerNorm → Gated FFN → Residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from typing import List, Optional, Tuple

from einops import rearrange


# ---------------------------------------------------------------------------
#  Helper: LayerNorm for 4D tensors [B, C, H, W]
# ---------------------------------------------------------------------------

def _to_3d(x: torch.Tensor) -> torch.Tensor:
    """Reshape [B, C, H, W] → [B, H*W, C] for LayerNorm."""
    return rearrange(x, 'b c h w -> b (h w) c')


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Reshape [B, H*W, C] → [B, C, H, W] after LayerNorm."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    """
    LayerNorm without bias — normalizes by variance only.

    Args:
        normalized_shape (int): Channel dimension to normalize over.
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H*W, C]
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm4d(nn.Module):
    """
    LayerNorm wrapper for 4D tensors [B, C, H, W].

    Converts to 3D ([B, H*W, C]), applies BiasFree LayerNorm on C, converts back.

    Args:
        dim (int): Channel dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.body = BiasFreeLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


# ---------------------------------------------------------------------------
#  GELU activation (for Gated FFN)
# ---------------------------------------------------------------------------

class GELU(nn.Module):
    """GELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


# ---------------------------------------------------------------------------
#  Gated Feed-Forward Network
# ---------------------------------------------------------------------------

class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network with depthwise convolution.

    Architecture:
        x → Conv1x1 (expand to 2*hidden) → DWConv3x3 → chunk → GELU(a)⊙b → Conv1x1 (project)

    Args:
        dim (int): Input/output channel dimension.
        ffn_expansion_factor (float): Hidden dimension = dim * factor. Default: 2.66.
        bias (bool): Whether convolutions use bias. Default: False.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.gelu = GELU()

        # Expand: dim → 2 * hidden (two branches for gating)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # Depthwise conv for local spatial mixing
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )

        # Contract: hidden → dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim, H, W]
        Returns:
            [B, dim, H, W]
        """
        x = self.project_in(x)                      # [B, 2*hidden, H, W]
        x1, x2 = self.dwconv(x).chunk(2, dim=1)     # Each: [B, hidden, H, W]
        x = self.gelu(x1) * x2                      # Gated activation
        x = self.project_out(x)                      # [B, dim, H, W]
        return x


# ---------------------------------------------------------------------------
#  Channel Cross-Attention
# ---------------------------------------------------------------------------

class ChannelCrossAttention(nn.Module):
    """
    Channel-wise (transposed) cross-attention mechanism.

    Q is generated from the **guidance** feature (illumination or semantic),
    K and V from the **main** feature (reflectance / ASSG output).

    Args:
        dim (int): Feature channel dimension.
        num_heads (int): Number of attention heads. Default: 2.
        bias (bool): Whether convolutions use bias. Default: False.
    """

    def __init__(self, dim: int, num_heads: int = 2, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads

        # Learnable temperature scaling per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # K, V projections from main feature
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2, dim * 2,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 2, bias=bias,
        )

        # Q projection from guidance feature
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=3, stride=1, padding=1,
            groups=dim, bias=bias,
        )

        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Main feature (R_fk or R̃_fk) — [B, dim, H, W]
            y: Guidance feature (I_fk or S_fk, already spatially aligned) — [B, dim, H, W]

        Returns:
            Modulated feature — [B, dim, H, W]
        """
        b, c, h, w = x.shape

        # Generate K, V from main feature x
        kv = self.kv_dwconv(self.kv(x))             # [B, 2*dim, H, W]
        k, v = kv.chunk(2, dim=1)                    # Each: [B, dim, H, W]

        # Generate Q from guidance feature y
        q = self.q_dwconv(self.q(y))                 # [B, dim, H, W]

        # Reshape for multi-head attention
        # [B, dim, H, W] → [B, num_heads, C_per_head, H*W]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 normalize Q and K along spatial dimension
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Channel-wise attention (transposed attention)
        # [B, heads, C_head, HW] @ [B, heads, HW, C_head] → [B, heads, C_head, C_head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        # [B, heads, C_head, C_head] @ [B, heads, C_head, HW] → [B, heads, C_head, HW]
        out = attn @ v

        # Reshape back to [B, dim, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)                  # [B, dim, H, W]
        return out


# ---------------------------------------------------------------------------
#  Modulation Unit (IMU or SMU)
# ---------------------------------------------------------------------------

class ModulationUnit(nn.Module):
    """
    A single modulation unit — used as either IMU or SMU.

    Applies cross-attention between a main feature and a guidance feature,
    followed by a gated feed-forward network, with residual connections.

    Flow:
        main, guidance → Norm(main), Norm(guidance) → CrossAttn(main, guidance)
        → + main (residual) → Norm → GatedFFN → + residual
        → output

    Args:
        dim (int): Feature channel dimension.
        num_heads (int): Number of attention heads. Default: 2.
        ffn_expansion_factor (float): FFN hidden expansion. Default: 2.66.
        bias (bool): Whether layers use bias. Default: False.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()

        # Shared LayerNorm for both main and guidance (matches UltraIS)
        self.norm1 = LayerNorm4d(dim)

        # Channel cross-attention: Q from guidance, KV from main
        self.attn = ChannelCrossAttention(dim, num_heads, bias)

        # LayerNorm before FFN
        self.norm2 = LayerNorm4d(dim)

        # Gated FFN for post-attention refinement
        self.ffn = GatedFFN(dim, ffn_expansion_factor, bias)

    def forward(self, main: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main:     Main feature — [B, dim, H, W]
            guidance: Guidance feature (already spatially aligned) — [B, dim, H, W]

        Returns:
            Modulated feature — [B, dim, H, W]
        """
        # Normalize both features through shared norm
        main_normed = self.norm1(main)
        guidance_normed = self.norm1(guidance)

        # Cross-attention + residual
        main = main + self.attn(main_normed, guidance_normed)

        # FFN + residual
        main = main + self.ffn(self.norm2(main))

        return main


# ---------------------------------------------------------------------------
#  ISDM-lite: Dual Modulation for One Stage
# ---------------------------------------------------------------------------

class ISDMLite(nn.Module):
    """
    ISDM-lite (Illumination-Semantic Dual Modulation — Lightweight) for one stage.

    Applies two sequential modulation units:
        1. IMU: R_fk modulated by illumination features I_fk → R̃_fk
        2. SMU: R̃_fk modulated by semantic features S_fk → M_fk

    Handles spatial interpolation when I_fk resolution differs from R_fk
    (I_fk comes from IGM's multi-scale U-Net decoder, S_fk from Mini-ASPP
    which operates at the same resolution as R_fk).

    Args:
        dim (int): Feature channel dimension (embed_dim from MambaIRv2).
        num_heads (int): Number of attention heads. Default: 2.
        ffn_expansion_factor (float): FFN hidden expansion. Default: 2.66.
        bias (bool): Whether layers use bias. Default: False.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()

        # IMU: Illumination Modulated Unit
        self.imu = ModulationUnit(dim, num_heads, ffn_expansion_factor, bias)

        # SMU: Semantic Modulated Unit
        self.smu = ModulationUnit(dim, num_heads, ffn_expansion_factor, bias)

    def forward(
        self,
        r_fk: torch.Tensor,
        i_fk: torch.Tensor,
        s_fk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dual modulation: R_fk → IMU(I_fk) → R̃_fk → SMU(S_fk) → M_fk

        Args:
            r_fk: ASSG block output (reflectance features) — [B, dim, H, W]
            i_fk: Illumination features from IGM — [B, dim, H', W'] (may differ spatially)
            s_fk: Semantic features from Mini-ASPP — [B, dim, H, W] (same resolution as r_fk)

        Returns:
            m_fk: Dual-modulated features — [B, dim, H, W]
        """
        # Spatially align I_fk to R_fk resolution if needed
        if i_fk.shape[2:] != r_fk.shape[2:]:
            i_fk = F.interpolate(
                i_fk,
                size=r_fk.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        # Stage 1: IMU — modulate with illumination
        r_tilde = self.imu(r_fk, i_fk)  # [B, dim, H, W]

        # Stage 2: SMU — modulate with semantics
        m_fk = self.smu(r_tilde, s_fk)  # [B, dim, H, W]

        return m_fk


# ---------------------------------------------------------------------------
#  Factory: create nn.ModuleList of ISDMLite for N ASSG stages
# ---------------------------------------------------------------------------

def create_isdm_lite_stages(
    num_stages: int,
    dim: int,
    num_heads: int = 2,
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
) -> nn.ModuleList:
    """
    Create an nn.ModuleList of independent ISDMLite instances, one per ASSG stage.

    Usage in MambaIRv2 forward_features():
        self.isdm_stages = create_isdm_lite_stages(num_stages=4, dim=48)
        ...
        for k, layer in enumerate(self.layers):
            x = layer(x, x_size, params)
            x_2d = x.transpose(1, 2).view(B, C, H, W)
            s_fk = self.mini_aspp_list[k](x_2d)
            m_fk = self.isdm_stages[k](x_2d, i_fk_list[ill_idx], s_fk)
            x = m_fk.flatten(2).transpose(1, 2)

    Args:
        num_stages: Number of ASSG stages (len(depths) in MambaIRv2).
        dim: Feature channel dimension (embed_dim).
        num_heads: Attention heads per modulation unit.
        ffn_expansion_factor: FFN hidden expansion.
        bias: Whether layers use bias.

    Returns:
        nn.ModuleList of ISDMLite instances.
    """
    return nn.ModuleList([
        ISDMLite(dim, num_heads, ffn_expansion_factor, bias)
        for _ in range(num_stages)
    ])


# ---------------------------------------------------------------------------
#  Parameter / FLOPs estimation utilities
# ---------------------------------------------------------------------------

def estimate_isdm_lite_params(dim: int = 48, num_heads: int = 2,
                              ffn_expansion_factor: float = 2.66,
                              num_stages: int = 4) -> dict:
    """
    Estimate parameter count for ISDM-lite configuration.

    Returns:
        Dictionary with parameter breakdown.
    """
    hidden = int(dim * ffn_expansion_factor)

    # ChannelCrossAttention params
    attn_params = (
        dim * dim * 2  +               # kv: 1x1 conv (dim → 2*dim)
        dim * 2 * 9    +               # kv_dwconv: 3x3 DW (2*dim)
        dim * dim      +               # q: 1x1 conv (dim → dim)
        dim * 9        +               # q_dwconv: 3x3 DW (dim)
        dim * dim      +               # project_out: 1x1 conv
        num_heads                       # temperature
    )

    # GatedFFN params
    ffn_params = (
        dim * hidden * 2  +            # project_in: 1x1 (dim → 2*hidden)
        hidden * 2 * 9    +            # dwconv: 3x3 DW (2*hidden)
        hidden * dim                    # project_out: 1x1 (hidden → dim)
    )

    # LayerNorm params (BiasFree: only weight, no bias)
    ln_params = dim  # per LayerNorm instance

    # ModulationUnit = Attention + FFN + 2 LayerNorms
    mod_unit_params = attn_params + ffn_params + ln_params * 2

    # ISDMLite = IMU + SMU = 2 ModulationUnits
    isdm_per_stage = mod_unit_params * 2

    # Total
    total = isdm_per_stage * num_stages

    return {
        'attention_params': attn_params,
        'ffn_params': ffn_params,
        'layernorm_params': ln_params,
        'modulation_unit_params': mod_unit_params,
        'isdm_per_stage_params': isdm_per_stage,
        'total_params': total,
        'num_stages': num_stages,
    }
