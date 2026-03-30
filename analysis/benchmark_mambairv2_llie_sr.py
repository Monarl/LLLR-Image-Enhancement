import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _register_supported_ops():
    def _numel_from_sizes(sizes):
        if not sizes:
            return 0
        numel = 1
        for d in sizes:
            if d is None:
                return 0
            numel *= int(d)
        return int(numel)

    def _numel_from_value(val):
        try:
            return _numel_from_sizes(val.type().sizes())
        except Exception:
            return 0

    def _out_numel(outputs):
        if outputs is None:
            return 0
        if isinstance(outputs, (list, tuple)):
            return sum(_numel_from_value(o) for o in outputs)
        return _numel_from_value(outputs)

    def elemwise_flop_jit(inputs, outputs):
        return _out_numel(outputs)

    def softmax_flop_jit(inputs, outputs):
        return 5 * _out_numel(outputs)

    def var_flop_jit(inputs, outputs):
        return 3 * _out_numel(outputs)

    def max_pool2d_flop_jit(inputs, outputs):
        out_elems = _out_numel(outputs)
        try:
            k = inputs[1].toIValue()
            if isinstance(k, int):
                kh, kw = k, k
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                kh, kw = int(k[0]), int(k[1])
            else:
                kh, kw = 2, 2
            return out_elems * max(kh * kw - 1, 1)
        except Exception:
            return out_elems

    def pixel_shuffle_flop_jit(inputs, outputs):
        return 0

    def zero_flop_jit(inputs, outputs):
        return 0

    def flops_selective_scan_fn(b=1, l=256, d=768, n=16, with_d=True, with_z=False):
        flops = 9 * b * l * d * n
        if with_d:
            flops += b * d * l
        if with_z:
            flops += b * d * l
        return flops

    def selective_scan_flop_jit(inputs, outputs):
        b, d, l = inputs[0].type().sizes()
        n = inputs[2].type().sizes()[1]
        return flops_selective_scan_fn(b=b, l=l, d=d, n=n, with_d=True, with_z=False)

    return {
        "aten::gelu": elemwise_flop_jit,
        "aten::silu": elemwise_flop_jit,
        "aten::neg": elemwise_flop_jit,
        "aten::exp": elemwise_flop_jit,
        "aten::add": elemwise_flop_jit,
        "aten::add_": elemwise_flop_jit,
        "aten::sub": elemwise_flop_jit,
        "aten::mul": elemwise_flop_jit,
        "aten::mul_": elemwise_flop_jit,
        "aten::div": elemwise_flop_jit,
        "aten::sqrt": elemwise_flop_jit,
        "aten::sigmoid": elemwise_flop_jit,
        "aten::hardsigmoid": elemwise_flop_jit,
        "aten::hardswish_": elemwise_flop_jit,
        "aten::leaky_relu": elemwise_flop_jit,
        "aten::leaky_relu_": elemwise_flop_jit,
        "aten::clamp_min": elemwise_flop_jit,
        "aten::ne": elemwise_flop_jit,
        "aten::log": elemwise_flop_jit,
        "aten::linalg_vector_norm": elemwise_flop_jit,
        "aten::softmax": softmax_flop_jit,
        "aten::log_softmax": softmax_flop_jit,
        "aten::var": var_flop_jit,
        "aten::max_pool2d": max_pool2d_flop_jit,
        "aten::pixel_shuffle": pixel_shuffle_flop_jit,
        "aten::empty_like": zero_flop_jit,
        "aten::fill_": zero_flop_jit,
        "aten::expand_as": zero_flop_jit,
        "aten::scatter_": zero_flop_jit,
        "aten::argmax": zero_flop_jit,
        "aten::sort": zero_flop_jit,
        "aten::repeat": zero_flop_jit,
        "aten::exponential_": zero_flop_jit,
        "aten::flip": None,
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,
    }


def _resolve_state_dict(ckpt, param_key="auto"):
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint is not a dict. Expected a state-dict style checkpoint.")

    if param_key == "auto":
        if "params_ema" in ckpt:
            state = ckpt["params_ema"]
            key_used = "params_ema"
        elif "params" in ckpt:
            state = ckpt["params"]
            key_used = "params"
        else:
            state = ckpt
            key_used = "root"
    elif param_key == "root":
        state = ckpt
        key_used = "root"
    else:
        state = ckpt[param_key] if param_key in ckpt else ckpt
        key_used = param_key if param_key in ckpt else "root"

    clean_state = {}
    for k, v in state.items():
        clean_state[k[7:] if k.startswith("module.") else k] = v
    return clean_state, key_used


def _load_state_dict_flexible(model, state_dict, strict=False):
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped_shape.append(k)
    result = model.load_state_dict(filtered, strict=strict)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    return missing, unexpected, skipped_shape, len(filtered)


def _build_model_from_yaml(opt_path, device):
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from basicsr.archs import build_network

    with open(opt_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    model = build_network(opt["network_g"])
    return model.to(device), opt


def _compute_flops_g(model, h, w, device, pass_gray=False):
    from fvcore.nn.flop_count import flop_count

    model.eval()
    dummy = torch.randn(1, 3, h, w, device=device)
    inputs = (dummy, torch.randn(1, 2, h, w, device=device)) if pass_gray else (dummy,)

    gflops_map, unsupported = flop_count(
        model,
        inputs=inputs,
        supported_ops=_register_supported_ops(),
    )
    return float(sum(gflops_map.values())), unsupported


def _benchmark_latency(model, h, w, device, warmup, repetitions, use_half=False, pass_gray=False):
    model.eval()

    if device.type == "cuda":
        dummy = torch.randn(1, 3, h, w, device=device)
        dummy_gray = torch.randn(1, 2, h, w, device=device) if pass_gray else None

        if use_half:
            model = model.half()
            dummy = dummy.half()
            if dummy_gray is not None:
                dummy_gray = dummy_gray.half()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1), dtype=np.float64)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy, dummy_gray) if pass_gray else model(dummy)

            for rep in range(repetitions):
                starter.record()
                _ = model(dummy, dummy_gray) if pass_gray else model(dummy)
                ender.record()
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)

        mean_ms = float(np.mean(timings))
        std_ms = float(np.std(timings))
        fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        return mean_ms, std_ms, fps

    dummy = torch.randn(1, 3, h, w, device=device)
    dummy_gray = torch.randn(1, 2, h, w, device=device) if pass_gray else None

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy, dummy_gray) if pass_gray else model(dummy)

        times_ms = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = model(dummy, dummy_gray) if pass_gray else model(dummy)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    return mean_ms, std_ms, fps


def main():
    parser = argparse.ArgumentParser(description="Benchmark MambaIRv2 LLIE-SR FLOPs and inference speed.")
    parser.add_argument("--opt", type=str, default="options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x4.yml")
    parser.add_argument("--checkpoint", type=str, default="experiments/pretrained_models/net_g_45000_mobilenet.pth")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--param-key", type=str, default="auto", choices=["auto", "params", "params_ema", "root"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--pass-gray", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, opt = _build_model_from_yaml(args.opt, device)
    semantic_type = opt.get("network_g", {}).get("semantic_extractor", "mini_aspp")
    print(f"Semantic extractor: {semantic_type}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state, key_used = _resolve_state_dict(ckpt, param_key=args.param_key)
    missing, unexpected, skipped_shape, loaded_count = _load_state_dict_flexible(model, state, strict=args.strict)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint key used: {key_used}")
    print(f"Loaded tensors: {loaded_count}")
    if skipped_shape:
        print(f"Skipped shape-mismatch tensors: {len(skipped_shape)}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {num_params / 1e6:.3f} M")

    gflops, unsupported = _compute_flops_g(model, args.height, args.width, device, pass_gray=args.pass_gray)
    print(f"FLOPs: {gflops:.3f} G (input: 1x3x{args.height}x{args.width})")
    print(f"Guidance input mode: {'external gray (2ch)' if args.pass_gray else 'internal compute'}")
    if unsupported:
        print(f"Unsupported ops from fvcore: {dict(unsupported)}")

    mean_ms, std_ms, fps = _benchmark_latency(
        model,
        h=args.height,
        w=args.width,
        device=device,
        warmup=args.warmup,
        repetitions=args.repetitions,
        use_half=args.half,
        pass_gray=args.pass_gray,
    )
    print(
        "Timing: "
        f"Mean {mean_ms:.3f} ms | Std {std_ms:.3f} ms | FPS {fps:.2f} "
        f"(warmup={args.warmup}, reps={args.repetitions})"
    )

    scale = opt.get("scale", "unknown")
    print(f"Config scale: x{scale}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
