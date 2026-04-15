"""
MambaIRv2 LLIE+SR Architecture
(Low-Light Image Enhancement + Super-Resolution)

Combines MambaIRv2's ASSG backbone with:
  - IGM (Illumination Guidance Module) — multi-scale illumination features
  - Mini-ASPP — lightweight internal semantic extraction (replaces HRNet)
  - ISDM-lite — dual modulation (illumination + semantic) per ASSG stage

Data Flow:
    Input X (LLLR) ──┬──► conv_first ──► [ASSG → Mini-ASPP → ISDM-lite] × N ──► conv_after_body ──► Upsample ──► Ŷ
                     │                                    ▲
                     └──► (gray guidance) ──► IGM ──► I_fk list ──┘

Each ASSG stage k:
    x = ASSG_k(x)                               # MambaIRv2 state-space + window attention
    x_2d = reshape(x)                            # [B, H*W, C] → [B, C, H, W]
    S_fk = MiniASPP_k(x_2d)                     # semantic features
    I_fk = igm_features[idx]                     # illumination features (bilinear interp if needed)
    M_fk = ISDMLite_k(x_2d, I_fk, S_fk)         # dual modulation: IMU(I_fk) → SMU(S_fk)
    x = reshape(M_fk)                            # [B, C, H, W] → [B, H*W, C]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

# Reuse ASSG blocks and utilities from MambaIRv2Light
from basicsr.archs.mambairv2light_arch import (
    ASSB,
    PatchEmbed,
    PatchUnEmbed,
    Upsample,
    UpsampleOneStep,
    window_partition,
)

# Our custom modules
from basicsr.archs.igm_module import IGMModule
from basicsr.archs.mini_aspp import create_mini_aspp_stages
from basicsr.archs.isdm_lite import create_isdm_lite_stages


@torch.no_grad()
def compute_illumination_guidance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 2-channel illumination guidance IG(L_g) from an RGB image.

    Matches the numpy implementation in paired_image_dataset.py but uses
    pure PyTorch so it works on GPU tensors.

    Args:
        x: RGB image tensor [B, 3, H, W] in [0, 1] range.

    Returns:
        gray: 2-channel guidance [B, 2, H, W]
            Channel 0: IG(L_g) = A(L_g) + B(L_g)
            Channel 1: L_g (inverted grayscale)
    """
    # Inverted luminance (same formula as dataset)
    r, g, b = x[:, 0:1] + 1, x[:, 1:2] + 1, x[:, 2:3] + 1
    gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0  # [B, 1, H, W]

    # --- Structure Expansion A(L_g): neighbor max-pooling ---
    # Horizontal max
    mx = torch.max(gray[:, :, :-1, :], gray[:, :, 1:, :])
    mx = torch.cat([mx, gray[:, :, -1:, :]], dim=2)
    # Vertical max
    my = torch.max(mx[:, :, :, :-1], mx[:, :, :, 1:])
    max_out = torch.cat([my, mx[:, :, :, -1:]], dim=3)

    # --- Edge Preservation B(L_g): local gradient differences ---
    x1 = gray[:, :, :-1, :] - gray[:, :, 1:, :]
    x1 = torch.cat([x1, gray[:, :, -1:, :]], dim=2)

    x2 = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    x2 = torch.cat([gray[:, :, :1, :], x2], dim=2)

    y1 = gray[:, :, :, :-1] - gray[:, :, :, 1:]
    y1 = torch.cat([y1, gray[:, :, :, -1:]], dim=3)

    y2 = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    y2 = torch.cat([gray[:, :, :, :1], y2], dim=3)

    edge_out = (torch.abs(x1) + torch.abs(x2) + torch.abs(y1) + torch.abs(y2)) / 4.0

    # IG(L_g) = A(L_g) + B(L_g)
    guidance = max_out + edge_out  # [B, 1, H, W]

    return torch.cat([guidance, gray], dim=1)  # [B, 2, H, W]


# ---------------------------------------------------------------------------
#  Main Architecture
# ---------------------------------------------------------------------------

@ARCH_REGISTRY.register()
class MambaIRv2LLIESR(nn.Module):
    """
    MambaIRv2 with Illumination-Semantic Dual Modulation for
    joint Low-Light Image Enhancement and Super-Resolution.

    Extends MambaIRv2Light by integrating:
      1. IGM (Illumination Guidance Module) — produces multi-scale I_fk
      2. Mini-ASPP — extracts semantic features S_fk from ASSG output
      3. ISDM-lite — dual modulation of ASSG features using I_fk and S_fk

    Args:
        img_size (int): Input image size. Default: 64.
        patch_size (int): Patch size. Default: 1.
        in_chans (int): Number of input channels. Default: 3.
        embed_dim (int): Feature dimension. Default: 48.
        d_state (int): SSM hidden state dimension. Default: 8.
        depths (tuple[int]): Depths per ASSG stage. Default: (5, 5, 5, 5).
        num_heads (tuple[int]): Attention heads per stage. Default: (4, 4, 4, 4).
        window_size (int): Window-MHSA window size. Default: 16.
        inner_rank (int): Prompt decomposition rank. Default: 32.
        num_tokens (int): Prompt pool size. Default: 64.
        convffn_kernel_size (int): ConvFFN kernel size. Default: 5.
        mlp_ratio (float): MLP expansion ratio. Default: 1.0.
        upscale (int): Upsampling factor. Default: 4.
        img_range (float): Image value range. Default: 1.0.
        upsampler (str): Upsampling method. Default: 'pixelshuffledirect'.
        resi_connection (str): Residual connection type. Default: '1conv'.
        igm_n_feat (int): IGM feature channels (should match embed_dim). Default: 48.
        isdm_num_heads (int): ISDM-lite attention heads. Default: 2.
        isdm_ffn_expansion (float): ISDM-lite FFN expansion. Default: 2.66.
        use_reflectance_input (bool): If True, backbone input is Retinex
            reflectance x / illumination_map (UltraIS-style). Default: True.
    """

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=48,
        d_state=8,
        depths=(5, 5, 5, 5),
        num_heads=(4, 4, 4, 4),
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
        # --- LLIE+SR specific ---
        igm_n_feat=48,
        isdm_num_heads=2,
        isdm_ffn_expansion=2.66,
        use_reflectance_input=True,
        **kwargs,
    ):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        num_stages = len(depths)

        assert num_stages <= 5, (
            f"num_stages ({num_stages}) must be <= 5 "
            f"(IGM produces 5 I_fk levels)"
        )

        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.use_reflectance_input = use_reflectance_input

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # ================================================================
        # 1. Shallow Feature Extraction
        # ================================================================
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ================================================================
        # 2. Deep Feature Extraction — ASSG Blocks
        # ================================================================
        self.num_layers = num_stages
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Relative position index for window self-attention
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA',
                             relative_position_index_SA)

        # ASSG blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ASSB(
                dim=embed_dim,
                d_state=d_state,
                idx=i_layer,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # Residual conv after body
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ================================================================
        # 3. Reconstruction / Upsampling
        # ================================================================
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # For denoising / CAR (no upsampling)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        # ================================================================
        # 4. Illumination Guidance Module (IGM)
        # ================================================================
        self.igm = IGMModule(n_feat=igm_n_feat)

        # ================================================================
        # 5. Mini-ASPP semantic extraction (one per ASSG stage)
        # ================================================================
        self.mini_aspp_stages = create_mini_aspp_stages(
            num_stages=num_stages,
            in_channels=embed_dim,
        )

        # ================================================================
        # 6. ISDM-lite dual modulation (one per ASSG stage)
        # ================================================================
        self.isdm_stages = create_isdm_lite_stages(
            num_stages=num_stages,
            dim=embed_dim,
            num_heads=isdm_num_heads,
            ffn_expansion_factor=isdm_ffn_expansion,
        )

        # Initialize weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    #  Weight Initialization
    # ------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # ------------------------------------------------------------------
    #  Attention Utilities
    # ------------------------------------------------------------------

    def calculate_rpi_sa(self):
        """Calculate relative position index for window self-attention."""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        """Calculate shifted-window attention mask."""
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    # ------------------------------------------------------------------
    #  Forward: Deep Features with ISDM-lite Modulation
    # ------------------------------------------------------------------

    def forward_features(self, x, i_fk_list, params):
        """
        Deep feature extraction with per-stage ISDM-lite modulation.

        Args:
            x: Shallow features [B, C, H, W] from conv_first.
            i_fk_list: Multi-scale illumination features from IGM
                       (list of 5 tensors at different spatial scales).
            params: Attention mask parameters dict.

        Returns:
            Deep features [B, C, H, W].
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # [B, H*W, C]

        if self.ape:
            x = x + self.absolute_pos_embed

        num_stages = len(self.layers)
        B = x.shape[0]
        C = self.embed_dim
        H, W = x_size

        for k, layer in enumerate(self.layers):
            # --- ASSG block (Window-MHSA + ASSM) ---
            x = layer(x, x_size, params)  # [B, H*W, C]

            # --- Convert to 2D for Mini-ASPP and ISDM-lite ---
            x_2d = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

            # --- Mini-ASPP: extract semantic features ---
            s_fk = self.mini_aspp_stages[k](x_2d)  # [B, C, H, W]

            # --- Select I_fk for this stage ---
            # IGM produces 5 levels [0..4], we use the finest N levels
            # General formula: i_fk_idx = (5 - num_stages) + k
            # For N=4: stages [0,1,2,3] → I_fk[1,2,3,4] (skip bottleneck)
            # For N=3: stages [0,1,2]   → I_fk[2,3,4]
            i_fk_idx = (5 - num_stages) + k
            i_fk = i_fk_list[i_fk_idx]  # [B, C, H', W']

            # --- ISDM-lite: dual modulation ---
            # IMU: modulate with illumination → SMU: modulate with semantics
            m_fk = self.isdm_stages[k](x_2d, i_fk, s_fk)  # [B, C, H, W]

            # --- Convert back to 1D for next ASSG block ---
            x = m_fk.flatten(2).transpose(1, 2)  # [B, H*W, C]

        x = self.norm(x)  # [B, H*W, C]
        x = self.patch_unembed(x, x_size)  # [B, C, H, W]

        return x

    # ------------------------------------------------------------------
    #  Forward: Full Pipeline
    # ------------------------------------------------------------------

    def forward(self, x, gray=None):
        """
        Full forward pass: LLIE + SR.

        Args:
            x: Low-light low-resolution input [B, 3, H, W] in [0, 1].
            gray: Pre-computed 2-channel illumination guidance [B, 2, H, W].
                  If None, computed automatically from x.

        Returns:
            Super-resolved enhanced output [B, 3, H*scale, W*scale].

        Note:
            When use_reflectance_input=True, the main backbone receives
            reflectance (x / illumination) instead of raw low-light x.
        """
        # --- Padding to multiple of window_size ---
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad

        # Reflect-pad x
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        # --- Illumination guidance ---
        if gray is None:
            # Compute from un-normalized x (still in [0, 1] range)
            gray = compute_illumination_guidance(x)
        else:
            # Pad gray identically
            gray = torch.cat(
                [gray, torch.flip(gray, [2])], 2
            )[:, :, :h, :]
            gray = torch.cat(
                [gray, torch.flip(gray, [3])], 3
            )[:, :, :, :w]

        # --- IGM: illumination features ---
        i_fk_list, illumination_map = self.igm(gray)
        # Store for potential illumination loss in model class
        self._illumination_map = illumination_map

        # --- Retinex-style decomposition: estimate reflectance ---
        # Match UltraIS behavior: divide by predicted illumination then clamp.
        if self.use_reflectance_input:
            reflectance = torch.clamp(
                x / illumination_map.repeat(1, 3, 1, 1), 0.0, 1.0
            )
            self._reflectance = reflectance
            x = reflectance

        # --- Normalize x for main backbone ---
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # --- Attention masks ---
        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': self.relative_position_index_SA,
        }

        # --- Main forward (same branching as MambaIRv2Light) ---
        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, i_fk_list, params)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, i_fk_list, params)
            ) + x
            x = self.upsample(x)

        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, i_fk_list, params)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(
                F.interpolate(x, scale_factor=2, mode='nearest')
            ))
            x = self.lrelu(self.conv_up2(
                F.interpolate(x, scale_factor=2, mode='nearest')
            ))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))

        else:
            # Denoising / CAR (no upsampling)
            x_first = self.conv_first(x)
            res = self.conv_after_body(
                self.forward_features(x_first, i_fk_list, params)
            ) + x_first
            x = x + self.conv_last(res)

        # --- De-normalize ---
        x = x / self.img_range + self.mean

        # --- Remove padding ---
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x
