"""Inference VRAM and parameter-size analysis for MambaIRv2 LLIE+SR.

Mirrors the real test pipeline used by MambaIRv2LLIESRModel.test():
  - Builds the network from the YAML's ``network_g`` block.
  - Loads pretrained weights from ``path.pretrain_network_g`` (if present).
  - Loads real LLLR test images via PairedImageDataset (with use_illguidance).
  - Calls ``net(lq, gray)`` inside ``torch.no_grad()`` exactly like ``test()``.
  - Records ``torch.cuda.max_memory_allocated`` per image.

Reports trainable param count, on-device weight bytes, and per-image peak VRAM
(allocated + reserved). Also supports synthetic resolutions via ``--resolutions``
for what-if analysis.

Usage:
    # Real dataset (recommended — matches paper-grade numbers):
    python analysis/inference_vram.py \\
        --opt options/test/mambairv2_llie_sr/test_MambaIRv2_LLIESR_x2.yml \\
        --num-images 8 --half

    # Synthetic shapes (no dataset needed):
    python analysis/inference_vram.py \\
        --opt options/test/mambairv2_llie_sr/test_MambaIRv2_LLIESR_x4.yml \\
        --synthetic --resolutions 256x256 384x384 512x512
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _add_repo_root_to_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_opt(opt_path: str) -> dict:
    with open(opt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(opt: dict, device: torch.device):
    from basicsr.archs import build_network
    return build_network(opt["network_g"]).to(device).eval()


def _load_checkpoint(model: torch.nn.Module, opt: dict, strict: bool = True):
    path_opt = opt.get("path", {}) or {}
    ckpt_path = path_opt.get("pretrain_network_g")
    if not ckpt_path:
        print("[checkpoint] no pretrain_network_g in YAML — using randomly initialised weights")
        return None
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_add_repo_root_to_path(), ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"[checkpoint] file not found: {ckpt_path} — using randomly initialised weights")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state, key = ckpt["params_ema"], "params_ema"
        elif "params" in ckpt:
            state, key = ckpt["params"], "params"
        else:
            state, key = ckpt, "root"
    else:
        state, key = ckpt, "root"

    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    print(f"[checkpoint] loaded {ckpt_path} (key={key}); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_test_dataset(opt: dict):
    """Build the first dataset block under ``datasets`` (e.g. ``test_1``)."""
    from basicsr.data import build_dataset

    datasets_block = opt.get("datasets") or {}
    if not datasets_block:
        raise RuntimeError("No 'datasets' block in YAML.")
    ds_name, ds_opt = next(iter(datasets_block.items()))
    ds_opt = dict(ds_opt)
    ds_opt.setdefault("phase", "test")
    ds_opt.setdefault("scale", opt.get("scale", 1))
    return ds_name, build_dataset(ds_opt)


# ---------------------------------------------------------------------------
# Param stats
# ---------------------------------------------------------------------------

def _param_stats(model: torch.nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    bytes_fp32 = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "buffers": buffers,
        "bytes_fp32": bytes_fp32,
        "bytes_fp16": bytes_fp32 // 2,
    }


# ---------------------------------------------------------------------------
# VRAM measurement helpers
# ---------------------------------------------------------------------------

def _format_mb(b):
    return f"{b / 1024**2:>8.1f} MB"


def _measure(model, lq, gray, device):
    """One forward pass, return peak alloc/reserved + latency."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    base_alloc = torch.cuda.memory_allocated(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        starter.record()
        out = model(lq, gray)
        ender.record()
        torch.cuda.synchronize(device)

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    elapsed_ms = starter.elapsed_time(ender)

    return {
        "out_shape": tuple(out.shape),
        "weights_mb": base_alloc / 1024**2,
        "peak_alloc_mb": peak_alloc / 1024**2,
        "peak_reserved_mb": peak_reserved / 1024**2,
        "activations_mb": (peak_alloc - base_alloc) / 1024**2,
        "latency_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Main flows
# ---------------------------------------------------------------------------

def run_real_dataset(model, opt, device, dtype, num_images, warmup):
    ds_name, dataset = _build_test_dataset(opt)
    n = min(num_images, len(dataset))
    print(f"\n--- Real dataset: {ds_name} ({len(dataset)} images, profiling first {n}) ---")
    print(f"{'idx':>3}  {'name':<22}{'LR HxW':<14}{'SR HxW':<14}"
          f"{'weights':<12}{'activs':<12}{'peak':<12}{'reserved':<12}{'latency':<10}")
    print("-" * 110)

    if dtype == torch.float16:
        model = model.half()

    # Warmup with the first sample
    sample0 = dataset[0]
    lq0 = sample0["lq"].unsqueeze(0).to(device).to(dtype)
    gray0 = sample0["gray"].unsqueeze(0).to(device).to(dtype) if "gray" in sample0 else None
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(lq0, gray0)
    torch.cuda.synchronize(device)

    aggregate = {"peak": [], "act": [], "lat": []}
    for i in range(n):
        sample = dataset[i]
        lq = sample["lq"].unsqueeze(0).to(device).to(dtype)
        gray = sample["gray"].unsqueeze(0).to(device).to(dtype) if "gray" in sample else None
        try:
            r = _measure(model, lq, gray, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{i:>3}  OOM at LR {tuple(lq.shape[-2:])}")
            continue

        name = Path(sample.get("lq_path", f"#{i}")).name
        lr = f"{lq.shape[-2]}x{lq.shape[-1]}"
        sr = f"{r['out_shape'][-2]}x{r['out_shape'][-1]}"
        print(f"{i:>3}  {name[:22]:<22}{lr:<14}{sr:<14}"
              f"{_format_mb(r['weights_mb']*1024**2)}  "
              f"{_format_mb(r['activations_mb']*1024**2)}  "
              f"{_format_mb(r['peak_alloc_mb']*1024**2)}  "
              f"{_format_mb(r['peak_reserved_mb']*1024**2)}  "
              f"{r['latency_ms']:>6.1f} ms")
        aggregate["peak"].append(r["peak_alloc_mb"])
        aggregate["act"].append(r["activations_mb"])
        aggregate["lat"].append(r["latency_ms"])

    if aggregate["peak"]:
        peak = aggregate["peak"]
        act = aggregate["act"]
        lat = aggregate["lat"]
        print("-" * 110)
        print(f"     Peak alloc      mean={sum(peak)/len(peak):.1f} MB  "
              f"max={max(peak):.1f} MB  min={min(peak):.1f} MB")
        print(f"     Activations     mean={sum(act)/len(act):.1f} MB  "
              f"max={max(act):.1f} MB")
        print(f"     Latency         mean={sum(lat)/len(lat):.1f} ms  "
              f"max={max(lat):.1f} ms")


def run_synthetic(model, opt, device, dtype, resolutions, warmup):
    print(f"\n--- Synthetic resolutions ({'fp16' if dtype==torch.float16 else 'fp32'}) ---")
    print(f"{'LR HxW':<11}{'SR HxW':<13}{'weights':<12}{'activs':<12}"
          f"{'peak':<12}{'reserved':<12}{'latency':<10}")
    print("-" * 80)

    if dtype == torch.float16:
        model = model.half()

    scale = int(opt.get("scale", 1))
    for h, w in resolutions:
        lq = torch.randn(1, 3, h, w, device=device, dtype=dtype)
        gray = torch.randn(1, 2, h, w, device=device, dtype=dtype)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(lq, gray)
        torch.cuda.synchronize(device)

        try:
            r = _measure(model, lq, gray, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{h}x{w:<7}  OOM")
            continue

        sr = f"{h*scale}x{w*scale}"
        print(f"{h}x{w:<7}{sr:<13}"
              f"{_format_mb(r['weights_mb']*1024**2)}  "
              f"{_format_mb(r['activations_mb']*1024**2)}  "
              f"{_format_mb(r['peak_alloc_mb']*1024**2)}  "
              f"{_format_mb(r['peak_reserved_mb']*1024**2)}  "
              f"{r['latency_ms']:>6.1f} ms")


def _parse_resolutions(specs):
    out = []
    for s in specs:
        if "x" in s.lower():
            h, w = s.lower().split("x")
        elif "," in s:
            h, w = s.split(",")
        else:
            h = w = s
        out.append((int(h), int(w)))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opt", required=True,
                   help="Path to YAML config (uses network_g + datasets + path).")
    p.add_argument("--num-images", type=int, default=8,
                   help="How many real test images to profile (default: 8).")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup forward passes before measurement.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Skip loading pretrained weights (use random init).")
    p.add_argument("--non-strict", action="store_true",
                   help="Use strict=False when loading the checkpoint.")
    p.add_argument("--half", action="store_true",
                   help="Run in fp16 (otherwise fp32).")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip real dataset; use random tensors at --resolutions.")
    p.add_argument("--resolutions", nargs="+",
                   default=["128x128", "256x256", "384x384", "512x512"],
                   help="LR HxW shapes for --synthetic mode.")
    args = p.parse_args()

    _add_repo_root_to_path()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:    {props.name} ({props.total_memory / 1024**3:.2f} GB)")

    opt = _load_opt(args.opt)
    print(f"Config: {args.opt} (scale=x{opt.get('scale', '?')})")

    model = _build_model(opt, device)
    if not args.no_checkpoint:
        _load_checkpoint(model, opt, strict=not args.non_strict)

    stats = _param_stats(model)
    print("\n--- Parameter size ---")
    print(f"  Trainable params: {stats['trainable']:,}  ({stats['trainable']/1e6:.3f} M)")
    print(f"  Total params:     {stats['total']:,}")
    print(f"  Buffers:          {stats['buffers']:,}")
    print(f"  Param size fp32:  {stats['bytes_fp32']/1024**2:.2f} MB")
    print(f"  Param size fp16:  {stats['bytes_fp16']/1024**2:.2f} MB")

    if device.type != "cuda":
        print("\n[CUDA not available — skipping VRAM measurement]")
        return

    dtype = torch.float16 if args.half else torch.float32

    if args.synthetic:
        run_synthetic(model, opt, device, dtype,
                      _parse_resolutions(args.resolutions), args.warmup)
    else:
        run_real_dataset(model, opt, device, dtype, args.num_images, args.warmup)


if __name__ == "__main__":
    main()
