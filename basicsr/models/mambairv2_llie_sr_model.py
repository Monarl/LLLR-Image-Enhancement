"""
MambaIRv2 LLIE+SR Model Class

Training/testing logic for the joint Low-Light Image Enhancement and
Super-Resolution model. Extends the base SRModel to handle:
  - Illumination guidance ('gray') from the dataset
  - Passing gray to the network alongside LQ input
  - Multiple losses: L1 + Perceptual + SSIM + Illumination + TV
"""

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from collections import OrderedDict

from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaIRv2LLIESRModel(SRModel):
    """MambaIRv2 LLIE+SR model with illumination guidance.

    Extends SRModel to:
      1. Load illumination guidance ('gray') from dataset in feed_data()
      2. Pass gray to network in optimize_parameters() and test()
      3. Use multi-loss training: L1 + Perceptual + SSIM + Illumination + TV
    """

    def init_training_settings(self):
        """Override to add SSIM, Illumination, and TV losses."""
        # Call parent to set up pixel_opt, perceptual_opt, optimizer, scheduler
        super().init_training_settings()

        train_opt = self.opt['train']

        # SSIM loss
        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        # Illumination loss (Y-channel L1)
        if train_opt.get('illumination_opt'):
            self.cri_illumination = build_loss(
                train_opt['illumination_opt']).to(self.device)
        else:
            self.cri_illumination = None

        # Total Variation loss
        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None

    def feed_data(self, data):
        """Load LQ, GT, and illumination guidance from data dict."""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # Illumination guidance (2-channel: [IG(L_g), L_g])
        # Computed in paired_image_dataset.py when use_illguidance=True
        if 'gray' in data:
            self.gray = data['gray'].to(self.device)
        else:
            self.gray = None  # Architecture will compute internally

    def optimize_parameters(self, current_iter):
        """Forward + backward with illumination guidance and multi-loss."""
        logger = get_root_logger()
        self.optimizer_g.zero_grad()

        # Forward: pass both LQ and illumination guidance
        self.output = self.net_g(self.lq, self.gray)

        # NaN/Inf check on output
        if torch.isnan(self.output).any() or torch.isinf(self.output).any():
            logger.warning(
                f'[Iter {current_iter}] NaN/Inf in network output. '
                f'Skipping batch.'
            )
            return

        l_total = 0
        loss_dict = OrderedDict()

        # --- Loss 1: Pixel loss (L_rec = ||pred - gt||_1) ---
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            if torch.isnan(l_pix) or torch.isinf(l_pix):
                logger.warning(
                    f'[Iter {current_iter}] NaN/Inf in pixel loss. '
                    f'Skipping batch.'
                )
                return
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # --- Loss 2: Perceptual loss (L_perc = VGG feature L1) ---
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                if torch.isnan(l_percep) or torch.isinf(l_percep):
                    logger.warning(
                        f'[Iter {current_iter}] NaN/Inf in perceptual loss. '
                        f'Skipping batch.'
                    )
                    return
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                if torch.isnan(l_style) or torch.isinf(l_style):
                    logger.warning(
                        f'[Iter {current_iter}] NaN/Inf in style loss. '
                        f'Skipping batch.'
                    )
                    return
                l_total += l_style
                loss_dict['l_style'] = l_style

        # --- Loss 3: SSIM loss (L_ssim = 1 - SSIM(pred, gt)) ---
        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            if torch.isnan(l_ssim) or torch.isinf(l_ssim):
                logger.warning(
                    f'[Iter {current_iter}] NaN/Inf in SSIM loss. '
                    f'Skipping batch.'
                )
                return
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        # --- Loss 4: Illumination loss (L_light = ||Y(pred) - Y(gt)||_1) ---
        if self.cri_illumination:
            l_light = self.cri_illumination(self.output, self.gt)
            if torch.isnan(l_light) or torch.isinf(l_light):
                logger.warning(
                    f'[Iter {current_iter}] NaN/Inf in illumination loss. '
                    f'Skipping batch.'
                )
                return
            l_total += l_light
            loss_dict['l_light'] = l_light

        # --- Loss 5: Total Variation loss (L_tv for spatial smoothness) ---
        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            if torch.isnan(l_tv) or torch.isinf(l_tv):
                logger.warning(
                    f'[Iter {current_iter}] NaN/Inf in TV loss. '
                    f'Skipping batch.'
                )
                return
            l_total += l_tv
            loss_dict['l_tv'] = l_tv

        # Backward
        l_total.backward()
        nn_utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)

        # Check gradients for NaN/Inf
        bad_grad = False
        for name, p in self.net_g.named_parameters():
            if p.grad is not None and (
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            ):
                logger.warning(
                    f'[Iter {current_iter}] NaN/Inf gradient in {name}. '
                    f'Skipping update.'
                )
                bad_grad = True
                break

        if bad_grad:
            self.optimizer_g.zero_grad()
            try:
                for state in list(self.optimizer_g.state.values()):
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            v.zero_()
            except Exception:
                logger.warning(
                    'Failed to fully clear optimizer state tensors.'
                )
            return

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """Inference with partitioning (adapted from MambaIRv2LightModel).

        Splits LQ into overlapping patches (~200×200), processes each
        independently (gray=None → computed per-patch by the arch),
        and merges results.  This avoids OOM on full-resolution images
        (e.g. 625×625 RELLISUR val/test).
        """
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1
        split_token_w = w // 200 + 1

        # Padding so dimensions are divisible by split_token counts
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h
        split_w = W // split_token_w

        # Overlap shave
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w

        # Build partition slices (with overlap)
        slices = []
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h,
                                (i + 1) * split_h + shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w,
                                 (j + 1) * split_w + shave_w)
                slices.append((top, left))

        # Extract partitions
        img_chops = [img[..., top, left] for top, left in slices]

        # Choose model
        net = (self.net_g_ema if hasattr(self, 'net_g_ema')
               else self.net_g)
        net.eval()

        with torch.no_grad():
            outputs = []
            for chop in img_chops:
                # gray=None → arch computes illumination guidance per-patch
                out = net(chop, None)
                outputs.append(out)

            _img = torch.zeros(1, C, H * scale, W * scale)
            for i in range(ral):
                for j in range(row):
                    top = slice(i * split_h * scale,
                                (i + 1) * split_h * scale)
                    left = slice(j * split_w * scale,
                                 (j + 1) * split_w * scale)
                    _top = (slice(0, split_h * scale) if i == 0
                            else slice(shave_h * scale,
                                       (shave_h + split_h) * scale))
                    _left = (slice(0, split_w * scale) if j == 0
                             else slice(shave_w * scale,
                                        (shave_w + split_w) * scale))
                    _img[..., top, left] = \
                        outputs[i * row + j][..., _top, _left]
            self.output = _img

        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

        # Remove padding
        _, _, h_out, w_out = self.output.size()
        self.output = self.output[
            :, :,
            :h_out - mod_pad_h * scale,
            :w_out - mod_pad_w * scale
        ]
