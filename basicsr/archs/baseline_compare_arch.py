import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

from analysis.model_zoo.pan import PAN
from analysis.model_zoo.swinIR import SwinIR
from analysis.model_zoo.hat import HAT
from analysis.model_zoo.srformer import SRFormer


@ARCH_REGISTRY.register()
class BaselinePAN(nn.Module):
    """BasicSR-registered PAN wrapper for RELLISUR retraining."""

    def __init__(self, in_chans=3, out_chans=3, nf=40, unf=24, nb=16, upscale=2):
        super().__init__()
        self.net = PAN(
            in_nc=in_chans,
            out_nc=out_chans,
            nf=nf,
            unf=unf,
            nb=nb,
            scale=upscale,
        )

    def forward(self, x):
        return self.net(x)


@ARCH_REGISTRY.register()
class BaselineSwinIR(nn.Module):
    """BasicSR-registered SwinIR wrapper for RELLISUR retraining."""

    def __init__(
        self,
        upscale=2,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        upsampler="pixelshuffledirect",
        img_range=1.0,
    ):
        super().__init__()
        self.net = SwinIR(
            upscale=upscale,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            img_range=img_range,
        )

    def forward(self, x):
        return self.net(x)


@ARCH_REGISTRY.register()
class BaselineHAT(nn.Module):
    """BasicSR-registered HAT wrapper for RELLISUR retraining."""

    def __init__(
        self,
        upscale=2,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=16,
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        img_range=1.0,
    ):
        super().__init__()
        self.net = HAT(
            upscale=upscale,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            img_range=img_range,
        )

    def forward(self, x):
        return self.net(x)


@ARCH_REGISTRY.register()
class BaselineSRFormer(nn.Module):
    """BasicSR-registered SRFormer wrapper for RELLISUR retraining."""

    def __init__(
        self,
        upscale=2,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        upsampler="pixelshuffledirect",
        img_range=1.0,
    ):
        super().__init__()
        self.net = SRFormer(
            upscale=upscale,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            img_range=img_range,
        )

    def forward(self, x):
        return self.net(x)
