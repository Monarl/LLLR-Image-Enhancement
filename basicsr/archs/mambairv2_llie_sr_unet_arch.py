"""
MambaIRv2 LLIE-SR U-Net Architecture.

This variant keeps the current LLIE-SR pipeline intact and introduces a
separate multi-resolution backbone inspired by UltraIS and MambaIRUNet:

    reflectance -> shallow conv
                -> encoder level 1 (full resolution)
                -> encoder level 2 (1/2 resolution)
                -> latent        (1/4 resolution)
                -> decoder level 2 with skip fusion
                -> decoder level 1 with skip fusion
                -> optional refinement
                -> SR head

Illumination and semantic modulation are applied at the latent and decoder
levels rather than once per flat ASSG stage.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import trunc_normal_
from basicsr.archs.igm_module import IGMModule, RCBdown, RCBup
from basicsr.archs.isdm_lite import ISDMLite
from basicsr.archs.mambairv2light_arch import (
    ASSB,
    PatchEmbed,
    PatchUnEmbed,
    Upsample,
    UpsampleOneStep,
    window_partition,
)
from basicsr.archs.mini_aspp import MiniASPP
from basicsr.utils.registry import ARCH_REGISTRY

try:
    from basicsr.archs.mobilenet_semantic import create_mobilenet_semantic_extractor
    MOBILENET_AVAILABLE = True
except ImportError:
    MOBILENET_AVAILABLE = False


@torch.no_grad()
def compute_illumination_guidance(x: torch.Tensor) -> torch.Tensor:
    """Compute 2-channel illumination guidance from an RGB image."""
    r, g, b = x[:, 0:1] + 1, x[:, 1:2] + 1, x[:, 2:3] + 1
    gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0

    mx = torch.max(gray[:, :, :-1, :], gray[:, :, 1:, :])
    mx = torch.cat([mx, gray[:, :, -1:, :]], dim=2)
    my = torch.max(mx[:, :, :, :-1], mx[:, :, :, 1:])
    max_out = torch.cat([my, mx[:, :, :, -1:]], dim=3)

    x1 = gray[:, :, :-1, :] - gray[:, :, 1:, :]
    x1 = torch.cat([x1, gray[:, :, -1:, :]], dim=2)
    x2 = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    x2 = torch.cat([gray[:, :, :1, :], x2], dim=2)
    y1 = gray[:, :, :, :-1] - gray[:, :, :, 1:]
    y1 = torch.cat([y1, gray[:, :, :, -1:]], dim=3)
    y2 = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    y2 = torch.cat([gray[:, :, :, :1], y2], dim=3)
    edge_out = (torch.abs(x1) + torch.abs(x2) + torch.abs(y1) + torch.abs(y2)) / 4.0

    guidance = max_out + edge_out
    return torch.cat([guidance, gray], dim=1)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


class SpatialDownsample(nn.Module):
    """Halve spatial size and double channels."""

    def __init__(self, num_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SpatialUpsample(nn.Module):
    """Double spatial size and halve channels."""

    def __init__(self, num_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MambaStage2D(nn.Module):
    """ASSB stage wrapper that accepts and returns 2D feature maps."""

    def __init__(
        self,
        dim,
        d_state,
        idx,
        depth,
        num_heads,
        window_size,
        inner_rank,
        num_tokens,
        convffn_kernel_size,
        mlp_ratio,
        qkv_bias,
        norm_layer,
        use_checkpoint,
        img_size,
        resi_connection,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=None,
        )
        self.block = ASSB(
            dim=dim,
            d_state=d_state,
            idx=idx,
            input_resolution=(img_size, img_size),
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=1,
            resi_connection=resi_connection,
        )
        self.register_buffer(
            'relative_position_index_SA',
            self._calculate_rpi_sa(),
            persistent=False,
        )

    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def _calculate_mask(self, x_size, device):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1), device=device)
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
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        if x_size[0] < self.window_size or x_size[1] < self.window_size:
            raise ValueError(
                f'Input size {x_size} is smaller than window size {self.window_size}.'
            )
        if x_size[0] % self.window_size != 0 or x_size[1] % self.window_size != 0:
            raise ValueError(
                f'Input size {x_size} must be divisible by window size {self.window_size}.'
            )

        tokens = self.patch_embed(x)
        params = {
            'attn_mask': self._calculate_mask(x_size, x.device),
            'rpi_sa': self.relative_position_index_SA,
        }
        tokens = self.block(tokens, x_size, params)
        return self.patch_unembed(tokens, x_size)


def _make_projector(in_channels: int, out_channels: int) -> nn.Module:
    if in_channels == out_channels:
        return nn.Identity()
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


@ARCH_REGISTRY.register()
class MambaIRv2LLIESRUNet(nn.Module):
    """Three-level U-Net variant of the LLIE-SR architecture."""

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=64,
        d_state=8,
        encoder_depths=(4, 4),
        latent_depth=6,
        decoder_depths=(4, 4),
        refinement_depth=2,
        encoder_heads=(4, 4),
        latent_heads=8,
        decoder_heads=(4, 4),
        refinement_heads=4,
        stage_window_sizes=(16, 8, 4, 8, 16, 16),
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=2.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        igm_n_feat=64,
        isdm_num_heads=2,
        isdm_ffn_expansion=2.66,
        semantic_extractor='mobilenet',
        **kwargs,
    ):
        super().__init__()

        if len(encoder_depths) != 2 or len(decoder_depths) != 2:
            raise ValueError('encoder_depths and decoder_depths must each have length 2.')
        if len(encoder_heads) != 2 or len(decoder_heads) != 2:
            raise ValueError('encoder_heads and decoder_heads must each have length 2.')
        if len(stage_window_sizes) != 6:
            raise ValueError('stage_window_sizes must have 6 values.')

        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = stage_window_sizes[0]
        self.semantic_extractor_type = semantic_extractor.lower()
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.stage_window_sizes = tuple(stage_window_sizes)

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        dim_l1 = embed_dim
        dim_l2 = embed_dim * 2
        dim_latent = embed_dim * 4
        num_feat = 64
        num_out_ch = in_chans

        self.pad_multiple = 1
        full_res_factors = (
            stage_window_sizes[0],
            stage_window_sizes[4],
            stage_window_sizes[5],
            2 * stage_window_sizes[1],
            2 * stage_window_sizes[3],
            4 * stage_window_sizes[2],
            4,
        )
        for factor in full_res_factors:
            self.pad_multiple = _lcm(self.pad_multiple, factor)

        self.conv_first = nn.Conv2d(in_chans, dim_l1, 3, 1, 1)

        self.encoder_level1 = MambaStage2D(
            dim=dim_l1,
            d_state=d_state,
            idx=0,
            depth=encoder_depths[0],
            num_heads=encoder_heads[0],
            window_size=stage_window_sizes[0],
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            resi_connection=resi_connection,
        )
        self.encoder_context1 = RCBdown(dim_l1)
        self.down1_2 = SpatialDownsample(dim_l1)

        self.encoder_level2 = MambaStage2D(
            dim=dim_l2,
            d_state=d_state,
            idx=1,
            depth=encoder_depths[1],
            num_heads=encoder_heads[1],
            window_size=stage_window_sizes[1],
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            img_size=max(1, img_size // 2),
            resi_connection=resi_connection,
        )
        self.encoder_context2 = RCBdown(dim_l2)
        self.down2_3 = SpatialDownsample(dim_l2)

        self.latent = MambaStage2D(
            dim=dim_latent,
            d_state=d_state,
            idx=2,
            depth=latent_depth,
            num_heads=latent_heads,
            window_size=stage_window_sizes[2],
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            img_size=max(1, img_size // 4),
            resi_connection=resi_connection,
        )

        self.up3_2 = SpatialUpsample(dim_latent)
        self.fuse_level2 = RCBup(dim_l2)
        self.decoder_level2 = MambaStage2D(
            dim=dim_l2,
            d_state=d_state,
            idx=3,
            depth=decoder_depths[0],
            num_heads=decoder_heads[0],
            window_size=stage_window_sizes[3],
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            img_size=max(1, img_size // 2),
            resi_connection=resi_connection,
        )

        self.up2_1 = SpatialUpsample(dim_l2)
        self.fuse_level1 = RCBup(dim_l1)
        self.decoder_level1 = MambaStage2D(
            dim=dim_l1,
            d_state=d_state,
            idx=4,
            depth=decoder_depths[1],
            num_heads=decoder_heads[1],
            window_size=stage_window_sizes[4],
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            resi_connection=resi_connection,
        )

        self.refinement = None
        if refinement_depth > 0:
            self.refinement = MambaStage2D(
                dim=dim_l1,
                d_state=d_state,
                idx=5,
                depth=refinement_depth,
                num_heads=refinement_heads,
                window_size=stage_window_sizes[5],
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                resi_connection=resi_connection,
            )

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(dim_l1, dim_l1, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(dim_l1, dim_l1 // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim_l1 // 4, dim_l1 // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim_l1 // 4, dim_l1, 3, 1, 1),
            )
        else:
            raise ValueError(f'Unsupported resi_connection: {resi_connection}')

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(dim_l1, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(
                upscale, dim_l1, num_out_ch, (img_size, img_size)
            )
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(dim_l1, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(dim_l1, num_out_ch, 3, 1, 1)

        self.igm = IGMModule(n_feat=igm_n_feat)

        if self.semantic_extractor_type == 'mobilenet':
            if not MOBILENET_AVAILABLE:
                raise ImportError(
                    "MobileNetV3 semantic extractor requires torchvision. "
                    "Install with: pip install torchvision"
                )
            self.mobilenet_semantic = create_mobilenet_semantic_extractor(
                embed_dim=embed_dim,
                num_stages=3,
                freeze_backbone=True,
                pretrained=True,
            )
            self.semantic_stages = None
            self.semantic_projectors = nn.ModuleList([
                _make_projector(embed_dim, dim_latent),
                _make_projector(embed_dim, dim_l2),
                _make_projector(embed_dim, dim_l1),
            ])
        elif self.semantic_extractor_type == 'mini_aspp':
            self.mobilenet_semantic = None
            self.semantic_projectors = None
            self.semantic_stages = nn.ModuleList([
                MiniASPP(dim_latent, dim_latent),
                MiniASPP(dim_l2, dim_l2),
                MiniASPP(dim_l1, dim_l1),
            ])
        else:
            raise ValueError(
                f"Unknown semantic_extractor: {semantic_extractor}. "
                f"Choose from: 'mini_aspp', 'mobilenet'"
            )

        self.illumination_projectors = nn.ModuleList([
            _make_projector(igm_n_feat, dim_latent),
            _make_projector(igm_n_feat, dim_l2),
            _make_projector(igm_n_feat, dim_l1),
        ])
        self.modulation_stages = nn.ModuleList([
            ISDMLite(dim_latent, isdm_num_heads, isdm_ffn_expansion),
            ISDMLite(dim_l2, isdm_num_heads, isdm_ffn_expansion),
            ISDMLite(dim_l1, isdm_num_heads, isdm_ffn_expansion),
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _pad_inputs(self, x: torch.Tensor, gray: torch.Tensor = None):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.pad_multiple
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad

        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        if gray is None:
            gray = compute_illumination_guidance(x)
        else:
            gray = torch.cat([gray, torch.flip(gray, [2])], 2)[:, :, :h, :]
            gray = torch.cat([gray, torch.flip(gray, [3])], 3)[:, :, :, :w]

        return x, gray, h_ori, w_ori

    def _prepare_semantic_cache(self, x: torch.Tensor):
        if self.semantic_extractor_type != 'mobilenet':
            return None
        semantic_features = self.mobilenet_semantic(x)
        return [semantic_features[2], semantic_features[1], semantic_features[0]]

    def _get_semantic_guidance(self, feature: torch.Tensor, semantic_cache, mod_idx: int):
        if self.semantic_extractor_type == 'mini_aspp':
            return self.semantic_stages[mod_idx](feature)

        semantic = semantic_cache[mod_idx]
        if semantic.shape[2:] != feature.shape[2:]:
            semantic = F.interpolate(
                semantic, size=feature.shape[2:], mode='bilinear', align_corners=False
            )
        return self.semantic_projectors[mod_idx](semantic)

    def _get_illumination_guidance(self, feature: torch.Tensor, i_fk: torch.Tensor, mod_idx: int):
        if i_fk.shape[2:] != feature.shape[2:]:
            i_fk = F.interpolate(
                i_fk, size=feature.shape[2:], mode='bilinear', align_corners=False
            )
        return self.illumination_projectors[mod_idx](i_fk)

    def _apply_modulation(self, feature: torch.Tensor, i_fk: torch.Tensor, semantic_cache, mod_idx: int):
        illumination = self._get_illumination_guidance(feature, i_fk, mod_idx)
        semantic = self._get_semantic_guidance(feature, semantic_cache, mod_idx)
        return self.modulation_stages[mod_idx](feature, illumination, semantic)

    def forward(self, x, gray=None):
        x, gray, h_ori, w_ori = self._pad_inputs(x, gray)

        i_fk_list, illumination_map = self.igm(gray)
        self._illumination_map = illumination_map

        # Keep the detached illumination path because it has been more stable
        # for the current LLIE-SR training recipe.
        ill_3ch = illumination_map.detach().repeat(1, 3, 1, 1)
        ill_3ch = ill_3ch.clamp(min=1e-4)
        reflectance = torch.clamp(x / ill_3ch, 0.0, 1.0)
        self._reflectance = reflectance

        semantic_cache = self._prepare_semantic_cache(x)

        self.mean = self.mean.type_as(reflectance)
        backbone_input = (reflectance - self.mean) * self.img_range

        shallow = self.conv_first(backbone_input)

        enc1 = self.encoder_level1(shallow)
        enc1 = self.encoder_context1(enc1)

        enc2 = self.down1_2(enc1)
        enc2 = self.encoder_level2(enc2)
        enc2 = self.encoder_context2(enc2)

        latent = self.down2_3(enc2)
        latent = self.latent(latent)
        latent = self._apply_modulation(latent, i_fk_list[2], semantic_cache, mod_idx=0)

        dec2 = self.up3_2(latent)
        dec2 = self.fuse_level2(torch.cat([dec2, enc2], dim=1))
        dec2 = self.decoder_level2(dec2)
        dec2 = self._apply_modulation(dec2, i_fk_list[3], semantic_cache, mod_idx=1)

        dec1 = self.up2_1(dec2)
        dec1 = self.fuse_level1(torch.cat([dec1, enc1], dim=1))
        dec1 = self.decoder_level1(dec1)
        dec1 = self._apply_modulation(dec1, i_fk_list[4], semantic_cache, mod_idx=2)

        if self.refinement is not None:
            dec1 = self.refinement(dec1)

        body = self.conv_after_body(dec1) + shallow

        if self.upsampler == 'pixelshuffle':
            out = self.conv_before_upsample(body)
            out = self.conv_last(self.upsample(out))
        elif self.upsampler == 'pixelshuffledirect':
            out = self.upsample(body)
        elif self.upsampler == 'nearest+conv':
            out = self.conv_before_upsample(body)
            out = self.lrelu(self.conv_up1(F.interpolate(out, scale_factor=2, mode='nearest')))
            out = self.lrelu(self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.conv_hr(out)))
        else:
            out = backbone_input + self.conv_last(body)

        out = out / self.img_range + self.mean
        out = out[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return out
