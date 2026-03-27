"""
MambaIRv2 LLIE+SR Model Class.

Training/testing logic for the joint Low-Light Image Enhancement and
Super-Resolution model. Extends the base SRModel to handle:
  - Illumination guidance ('gray') from the dataset
  - Passing gray to the network alongside LQ input
  - Multiple losses: L1 + Perceptual + SSIM + Illumination + TV
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MambaIRv2LLIESRModel(SRModel):
    """MambaIRv2 LLIE+SR model with illumination guidance."""

    def setup_optimizers(self):
        """Override to support differential learning rates.

        If ``train.backbone_lr_scale`` is set in the YAML (e.g. 0.1), the
        backbone parameters (conv_first, layers, norm, conv_after_body,
        upsample) are placed in a separate param group with
        ``lr = base_lr * backbone_lr_scale``, while the new LLIE modules
        (igm, mini_aspp_stages, isdm_stages) use the full base lr.

        If ``backbone_lr_scale`` is absent or None the method falls back to
        the standard single-group behaviour (identical to SRModel).
        """
        train_opt = self.opt["train"]
        backbone_lr_scale = train_opt.get("backbone_lr_scale", None)

        if backbone_lr_scale is None:
            super().setup_optimizers()
            return

        logger = get_root_logger()
        logger.info(
            f"Differential LR: backbone_lr_scale={backbone_lr_scale}. "
            f"New modules (igm / mini_aspp / isdm) use full lr; "
            f"backbone uses lr * {backbone_lr_scale}."
        )

        new_module_prefixes = ("igm.", "mini_aspp_stages.", "isdm_stages.")

        backbone_params = []
        new_module_params = []
        for name, param in self.net_g.named_parameters():
            if not param.requires_grad:
                logger.warning(f"Params {name} will not be optimized.")
                continue
            if any(name.startswith(prefix) for prefix in new_module_prefixes):
                new_module_params.append(param)
            else:
                backbone_params.append(param)

        optim_type = train_opt["optim_g"].pop("type")
        base_lr = train_opt["optim_g"]["lr"]
        optim_kwargs = {k: v for k, v in train_opt["optim_g"].items() if k != "lr"}

        param_groups = [
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_scale,
                "name": "backbone",
            },
            {"params": new_module_params, "lr": base_lr, "name": "new_modules"},
        ]

        self.optimizer_g = self.get_optimizer(
            optim_type, param_groups, lr=base_lr, **optim_kwargs
        )
        self.optimizers.append(self.optimizer_g)

        logger.info(
            f"  backbone params : {sum(p.numel() for p in backbone_params):,} "
            f"(lr={base_lr * backbone_lr_scale:.2e})"
        )
        logger.info(
            f"  new module params: {sum(p.numel() for p in new_module_params):,} "
            f"(lr={base_lr:.2e})"
        )

    def init_training_settings(self):
        """Override to add SSIM, illumination, Retinex-illumination, and TV losses."""
        super().init_training_settings()

        train_opt = self.opt["train"]

        if train_opt.get("ssim_opt"):
            self.cri_ssim = build_loss(train_opt["ssim_opt"]).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get("illumination_opt"):
            self.cri_illumination = build_loss(train_opt["illumination_opt"]).to(
                self.device
            )
        else:
            self.cri_illumination = None

        if train_opt.get("retinex_ill_opt"):
            self.cri_retinex_ill = build_loss(train_opt["retinex_ill_opt"]).to(
                self.device
            )
        else:
            self.cri_retinex_ill = None

        if train_opt.get("tv_opt"):
            self.cri_tv = build_loss(train_opt["tv_opt"]).to(self.device)
        else:
            self.cri_tv = None

    def feed_data(self, data):
        """Load LQ, GT, and illumination guidance from data dict."""
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)
        if "gray" in data:
            self.gray = data["gray"].to(self.device)
        else:
            self.gray = None

    def optimize_parameters(self, current_iter):
        """Forward + backward with illumination guidance and multi-loss."""
        logger = get_root_logger()
        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq, self.gray)

        if torch.isnan(self.output).any() or torch.isinf(self.output).any():
            logger.warning(
                f"[Iter {current_iter}] NaN/Inf in network output. Skipping batch."
            )
            return

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            if torch.isnan(l_pix) or torch.isinf(l_pix):
                logger.warning(
                    f"[Iter {current_iter}] NaN/Inf in pixel loss. Skipping batch."
                )
                return
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                if torch.isnan(l_percep) or torch.isinf(l_percep):
                    logger.warning(
                        f"[Iter {current_iter}] NaN/Inf in perceptual loss. "
                        f"Skipping batch."
                    )
                    return
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                if torch.isnan(l_style) or torch.isinf(l_style):
                    logger.warning(
                        f"[Iter {current_iter}] NaN/Inf in style loss. Skipping batch."
                    )
                    return
                l_total += l_style
                loss_dict["l_style"] = l_style

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            if torch.isnan(l_ssim) or torch.isinf(l_ssim):
                logger.warning(
                    f"[Iter {current_iter}] NaN/Inf in SSIM loss. Skipping batch."
                )
                return
            l_total += l_ssim
            loss_dict["l_ssim"] = l_ssim

        if self.cri_illumination:
            l_light = self.cri_illumination(self.output, self.gt)
            if torch.isnan(l_light) or torch.isinf(l_light):
                logger.warning(
                    f"[Iter {current_iter}] NaN/Inf in illumination loss. "
                    f"Skipping batch."
                )
                return
            l_total += l_light
            loss_dict["l_light"] = l_light

        if self.cri_retinex_ill:
            net = self.net_g.module if hasattr(self.net_g, "module") else self.net_g
            pred_ill_map = net._illumination_map
            if self.gray is not None:
                target_gray = self.gray[:, 1:2, :, :]
                if pred_ill_map.shape[2:] != target_gray.shape[2:]:
                    target_gray = F.interpolate(
                        target_gray,
                        size=pred_ill_map.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                l_retinex_ill = self.cri_retinex_ill(pred_ill_map, target_gray)
                if not (torch.isnan(l_retinex_ill) or torch.isinf(l_retinex_ill)):
                    l_total += l_retinex_ill
                    loss_dict["l_retinex_ill"] = l_retinex_ill

        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            if torch.isnan(l_tv) or torch.isinf(l_tv):
                logger.warning(
                    f"[Iter {current_iter}] NaN/Inf in TV loss. Skipping batch."
                )
                return
            l_total += l_tv
            loss_dict["l_tv"] = l_tv

        loss_dict["l_total"] = l_total.detach()
        l_total.backward()

        grad_clip = self.opt["train"].get("grad_clip_norm", 0.05)
        nn_utils.clip_grad_norm_(self.net_g.parameters(), max_norm=grad_clip)

        bad_grad = False
        for name, param in self.net_g.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                logger.warning(
                    f"[Iter {current_iter}] NaN/Inf gradient in {name}. "
                    f"Skipping update."
                )
                bad_grad = True
                break

        if bad_grad:
            self.optimizer_g.zero_grad()
            try:
                for state in list(self.optimizer_g.state.values()):
                    for key, value in list(state.items()):
                        if isinstance(value, torch.Tensor):
                            value.zero_()
            except Exception:
                logger.warning("Failed to fully clear optimizer state tensors.")
            return

        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """Run full-image inference with consistent illumination guidance.

        This follows the same evaluation pattern used by UltraIS: feed the
        full low-light input and its full gray guidance in one forward pass,
        and let the architecture handle internal padding. Avoids visible
        square seams from per-tile Retinex guidance recomputation.
        """
        net = self.net_g_ema if hasattr(self, "net_g_ema") else self.net_g
        gray = getattr(self, "gray", None)

        net.eval()
        with torch.no_grad():
            self.output = net(self.lq, gray)

        if not hasattr(self, "net_g_ema"):
            self.net_g.train()
