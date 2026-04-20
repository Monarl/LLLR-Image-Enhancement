import argparse
import contextlib
import csv
import io
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _short_error(exc: Exception, max_len: int = 240) -> str:
    msg = str(exc).replace("\n", " ").replace("\r", " ")
    msg = " ".join(msg.split())
    if len(msg) > max_len:
        return msg[: max_len - 3] + "..."
    return msg


def _register_supported_ops():
    """Custom FLOPs handlers for ops frequently missing in default counters."""

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

    def selective_scan_flop_jit(inputs, outputs, verbose=False):
        del outputs
        del verbose
        b, d, l = inputs[0].type().sizes()
        n = inputs[2].type().sizes()[1]
        return flops_selective_scan_fn(b=b, l=l, d=d, n=n, with_d=True, with_z=False)

    return {
        "aten::gelu": elemwise_flop_jit,
        "aten::silu": elemwise_flop_jit,
        "aten::neg": elemwise_flop_jit,
        "aten::exp": elemwise_flop_jit,
        "aten::flip": None,
        "aten::add": elemwise_flop_jit,
        "aten::add_": elemwise_flop_jit,
        "aten::sub": elemwise_flop_jit,
        "aten::mul": elemwise_flop_jit,
        "aten::mul_": elemwise_flop_jit,
        "aten::div": elemwise_flop_jit,
        "aten::sqrt": elemwise_flop_jit,
        "aten::sigmoid": elemwise_flop_jit,
        "aten::leaky_relu": elemwise_flop_jit,
        "aten::leaky_relu_": elemwise_flop_jit,
        "aten::clamp_min": elemwise_flop_jit,
        "aten::ne": elemwise_flop_jit,
        "aten::softmax": softmax_flop_jit,
        "aten::log_softmax": softmax_flop_jit,
        "aten::var": var_flop_jit,
        "aten::max_pool2d": max_pool2d_flop_jit,
        "aten::pixel_shuffle": pixel_shuffle_flop_jit,
        "aten::log": elemwise_flop_jit,
        "aten::linalg_vector_norm": elemwise_flop_jit,
        "aten::empty_like": zero_flop_jit,
        "aten::fill_": zero_flop_jit,
        "aten::expand_as": zero_flop_jit,
        "aten::scatter_": zero_flop_jit,
        "aten::argmax": zero_flop_jit,
        "aten::sort": zero_flop_jit,
        "aten::exponential_": zero_flop_jit,
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,
    }


def _load_config(config_path: Path) -> dict:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required for .yml/.yaml config files") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format: {config_path}")


def _resolve_state_dict(ckpt: dict, param_key: str = "auto") -> Tuple[dict, str]:
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")

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

    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean[k[7:]] = v
        else:
            clean[k] = v
    return clean, key_used


def _import_callable(module_name: str, symbol_name: str):
    mod = importlib.import_module(module_name)
    if not hasattr(mod, symbol_name):
        raise AttributeError(f"{symbol_name} not found in {module_name}")
    return getattr(mod, symbol_name)


def _build_network_from_opt_dict(net_opt: dict):
    """Build network from BasicSR-style network_g dict.

    Supports both MambaIR-style `basicsr.archs.build_network` and
    UltraIS-style `basicsr.models.archs.define_network`.
    """
    try:
        from basicsr.archs import build_network
        return build_network(net_opt)
    except Exception:
        from basicsr.models.archs import define_network
        return define_network(net_opt)


def _reset_basicsr_if_needed(expected_root: Path):
    """Reload basicsr package when switching between different repo roots."""
    if not (expected_root / "basicsr").exists():
        return
    loaded = sys.modules.get("basicsr")
    if loaded is None:
        return
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is None:
        return
    try:
        loaded_root = Path(loaded_file).resolve().parents[1]
    except Exception:
        return
    if loaded_root == expected_root.resolve():
        return

    for name in list(sys.modules.keys()):
        if name == "basicsr" or name.startswith("basicsr."):
            sys.modules.pop(name, None)


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def _build_model_from_entry(entry: dict, workspace_root: Path, device: torch.device):
    module_root = entry.get("module_root")
    abs_root = workspace_root.resolve()
    if module_root:
        abs_module_root = (workspace_root / module_root).resolve()
        abs_root = abs_module_root
        _reset_basicsr_if_needed(abs_module_root)
        abs_module_root_str = str(abs_module_root)
        if abs_module_root_str in sys.path:
            sys.path.remove(abs_module_root_str)
        sys.path.insert(0, abs_module_root_str)

    kwargs = entry.get("kwargs", {})
    opt_file = entry.get("opt_file")

    with _pushd(abs_root):
        removed_workspace_paths = []
        ultrais_namespace_basicsr = (abs_root / "basicsr").exists() and not (abs_root / "basicsr" / "__init__.py").exists()
        if ultrais_namespace_basicsr:
            ws_resolved = workspace_root.resolve()
            for p in list(sys.path):
                if not p:
                    continue
                try:
                    if Path(p).resolve() == ws_resolved:
                        sys.path.remove(p)
                        removed_workspace_paths.append(p)
                except Exception:
                    pass

        try:
            opt_cfg = None
            if opt_file:
                opt_path = (workspace_root / opt_file).resolve()
                if not opt_path.exists():
                    raise FileNotFoundError(f"Option file not found: {opt_path}")
                opt_cfg = _load_config(opt_path)
                if "network_g" not in opt_cfg:
                    raise ValueError(f"Option file missing network_g: {opt_path}")
                model = _build_network_from_opt_dict(opt_cfg["network_g"])
            elif "builder" in entry:
                builder = entry["builder"]
                fn = _import_callable(builder["module"], builder["name"])
                model = fn(**kwargs)
            elif "ctor" in entry:
                ctor = entry["ctor"]
                cls = _import_callable(ctor["module"], ctor["name"])
                model = cls(**kwargs)
            else:
                raise ValueError("Model entry must have opt_file or either 'builder' or 'ctor'.")

            model = model.to(device)

            ckpt_rel = entry.get("checkpoint")
            if not ckpt_rel and opt_cfg is not None:
                ckpt_rel = (opt_cfg.get("path") or {}).get("pretrain_network_g")
                if ckpt_rel in (None, "", "~"):
                    ckpt_rel = None
            if ckpt_rel:
                ckpt_path = Path(ckpt_rel)
                if not ckpt_path.is_absolute():
                    ckpt_path = (workspace_root / ckpt_path).resolve()
                    if not ckpt_path.exists():
                        ckpt_path = (abs_root / ckpt_rel).resolve()
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                param_key = entry.get("param_key", None)
                if param_key is None and opt_cfg is not None:
                    param_key = (opt_cfg.get("path") or {}).get("param_key", "auto")
                if param_key is None:
                    param_key = "auto"

                strict_val = entry.get("strict", None)
                if strict_val is None and opt_cfg is not None:
                    strict_val = (opt_cfg.get("path") or {}).get("strict_load_g", False)
                if strict_val is None:
                    strict_val = False

                state, key_used = _resolve_state_dict(ckpt, param_key)
                strict = bool(strict_val)
                missing, unexpected = model.load_state_dict(state, strict=strict)

                return model, {
                    "checkpoint": str(ckpt_path),
                    "param_key_used": key_used,
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                }

            return model, {"checkpoint": "", "param_key_used": "", "missing_keys": 0, "unexpected_keys": 0}
        finally:
            for p in removed_workspace_paths:
                if p not in sys.path:
                    sys.path.append(p)


def _benchmark_latency(
    model,
    input_mode: str,
    h: int,
    w: int,
    device: torch.device,
    warmup: int,
    reps: int,
    half: bool,
    pass_gray: Optional[bool] = None,
):
    model.eval()

    use_gray = (input_mode == "rgb_gray") and (pass_gray is not False)
    x = torch.randn(1, 3, h, w, device=device)
    gray = torch.randn(1, 2, h, w, device=device) if use_gray else None

    if half and device.type == "cuda":
        model = model.half()
        x = x.half()
        if gray is not None:
            gray = gray.half()

    def _forward():
        if use_gray:
            return model(x, gray)
        return model(x)

    with torch.no_grad():
        if device.type == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            times = np.zeros((reps,), dtype=np.float64)

            for _ in range(warmup):
                _ = _forward()

            for i in range(reps):
                starter.record()
                _ = _forward()
                ender.record()
                torch.cuda.synchronize()
                times[i] = starter.elapsed_time(ender)
        else:
            times = np.zeros((reps,), dtype=np.float64)
            for _ in range(warmup):
                _ = _forward()

            for i in range(reps):
                t0 = time.perf_counter()
                _ = _forward()
                t1 = time.perf_counter()
                times[i] = (t1 - t0) * 1000.0

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    return mean_ms, std_ms, fps


def _compute_flops_thop(model, use_gray: bool, h: int, w: int, device: torch.device) -> float:
    from thop import profile

    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    if use_gray:
        gray = torch.randn(1, 2, h, w, device=device)
        flops, _ = profile(model, inputs=(x, gray), verbose=False)
    else:
        flops, _ = profile(model, inputs=(x,), verbose=False)
    return float(flops / 1e9)


def _compute_flops_fvcore(model, use_gray: bool, h: int, w: int, device: torch.device) -> Tuple[float, str]:
    from fvcore.nn.flop_count import flop_count

    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    inputs = (x, torch.randn(1, 2, h, w, device=device)) if use_gray else (x,)
    gflops_map, unsupported = flop_count(model, inputs=inputs, supported_ops=_register_supported_ops())
    unsupported_str = ""
    if unsupported:
        unsupported_str = json.dumps(dict(unsupported), ensure_ascii=True)
    return float(sum(gflops_map.values())), unsupported_str


def _compute_flops_fvcore_quiet(
    model,
    use_gray: bool,
    h: int,
    w: int,
    device: torch.device,
    verbose: bool,
) -> Tuple[float, str]:
    if verbose:
        return _compute_flops_fvcore(model, use_gray, h, w, device)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _compute_flops_fvcore(model, use_gray, h, w, device)


def _compute_flops_optional(
    model,
    input_mode: str,
    h: int,
    w: int,
    device: torch.device,
    backend: str,
    pass_gray: Optional[bool] = None,
    verbose: bool = False,
) -> Tuple[Optional[float], str, str]:
    use_gray = (input_mode == "rgb_gray") and (pass_gray is not False)
    if backend not in {"auto", "fvcore", "thop"}:
        raise ValueError(f"Unsupported FLOPs backend: {backend}")

    # Keep results robust: if fvcore fails (common on some dynamic graphs),
    # fall back to thop so FLOPs are still reported.
    if backend in {"auto", "fvcore"}:
        backends = ("fvcore", "thop")
    else:
        backends = ("thop",)

    errors = []
    for b in backends:
        try:
            if b == "fvcore":
                gflops, unsupported_ops = _compute_flops_fvcore_quiet(
                    model,
                    use_gray,
                    h,
                    w,
                    device,
                    verbose=verbose,
                )
                return gflops, "fvcore", unsupported_ops
            gflops = _compute_flops_thop(model, use_gray, h, w, device)
            if errors:
                return gflops, "thop(fallback)", " | ".join(errors)
            return gflops, "thop", ""
        except Exception as exc:
            errors.append(f"{b}: {_short_error(exc)}")

    return None, "", " | ".join(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified multi-model latency benchmark for LLIE+SR.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config (JSON/YAML).")
    parser.add_argument("--csv", type=str, default="", help="Override output CSV path.")
    parser.add_argument("--flops", action="store_true", help="Also compute FLOPs with thop if available.")
    parser.add_argument("--flops-backend", type=str, default="auto", choices=["auto", "fvcore", "thop"], help="FLOPs backend. 'auto' tries fvcore first, then thop.")
    parser.add_argument("--height", type=int, default=None, help="Override input LR height.")
    parser.add_argument("--width", type=int, default=None, help="Override input LR width.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup iterations.")
    parser.add_argument("--repetitions", type=int, default=None, help="Override measured iterations.")
    parser.add_argument("--pass-gray", action="store_true", help="Force external 2-channel guidance for rgb_gray models.")
    parser.add_argument("--no-pass-gray", action="store_true", help="Force single-input forward for rgb_gray models.")
    parser.add_argument("--verbose-flops", action="store_true", help="Show raw FLOPs backend logs (fvcore tracing output).")
    args = parser.parse_args()

    if args.pass_gray and args.no_pass_gray:
        raise ValueError("--pass-gray and --no-pass-gray cannot be used together.")

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)

    workspace_root = (config_path.parent / cfg.get("workspace_root", "..")).resolve()

    device_name = cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    input_h = int(args.height if args.height is not None else cfg.get("input_height", 128))
    input_w = int(args.width if args.width is not None else cfg.get("input_width", 128))
    warmup = int(args.warmup if args.warmup is not None else cfg.get("warmup", 10))
    reps = int(args.repetitions if args.repetitions is not None else cfg.get("repetitions", 200))
    half = bool(cfg.get("half", False))

    pass_gray_override: Optional[bool] = None
    if args.pass_gray:
        pass_gray_override = True
    elif args.no_pass_gray:
        pass_gray_override = False

    models = [m for m in cfg.get("models", []) if m.get("enabled", True)]
    if not models:
        raise ValueError("No enabled models found in config.")

    rows = []
    for entry in models:
        name = entry.get("name", "model")
        label = entry.get("label", name)
        input_mode = entry.get("input_mode", "rgb")
        entry_pass_gray = pass_gray_override
        if entry_pass_gray is None and "pass_gray" in entry:
            entry_pass_gray = bool(entry.get("pass_gray"))

        print(f"[RUN] {name}")
        try:
            model, load_info = _build_model_from_entry(entry, workspace_root, device)
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            mean_ms, std_ms, fps = _benchmark_latency(
                model,
                input_mode,
                input_h,
                input_w,
                device,
                warmup,
                reps,
                half,
                pass_gray=entry_pass_gray,
            )
            gflops = None
            flops_backend = ""
            unsupported_ops = ""
            if args.flops:
                gflops, flops_backend, unsupported_ops = _compute_flops_optional(
                    model,
                    input_mode,
                    input_h,
                    input_w,
                    device,
                    backend=args.flops_backend,
                    pass_gray=entry_pass_gray,
                    verbose=args.verbose_flops,
                )

            row = {
                "name": name,
                "label": label,
                "input_mode": input_mode,
                "pass_gray_used": (input_mode == "rgb_gray") and (entry_pass_gray is not False),
                "params_m": params_m,
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "fps": fps,
                "gflops": gflops if gflops is not None else "",
                "flops_backend": flops_backend,
                "unsupported_ops": unsupported_ops,
                "checkpoint": load_info.get("checkpoint", ""),
                "param_key_used": load_info.get("param_key_used", ""),
                "missing_keys": load_info.get("missing_keys", 0),
                "unexpected_keys": load_info.get("unexpected_keys", 0),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            err_msg = _short_error(exc, max_len=500)
            row = {
                "name": name,
                "label": label,
                "input_mode": input_mode,
                "pass_gray_used": "",
                "params_m": "",
                "mean_ms": "",
                "std_ms": "",
                "fps": "",
                "gflops": "",
                "flops_backend": "",
                "unsupported_ops": "",
                "checkpoint": entry.get("checkpoint", ""),
                "param_key_used": "",
                "missing_keys": "",
                "unexpected_keys": "",
                "status": "error",
                "error": err_msg,
            }
            print(f"[ERR] {name}: {err_msg}")

        rows.append(row)

    csv_path = Path(args.csv).resolve() if args.csv else (workspace_root / cfg.get("output_csv", "analysis/outputs/latency/latency_report.csv")).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "name",
        "label",
        "input_mode",
        "pass_gray_used",
        "params_m",
        "gflops",
        "flops_backend",
        "unsupported_ops",
        "mean_ms",
        "std_ms",
        "fps",
        "checkpoint",
        "param_key_used",
        "missing_keys",
        "unexpected_keys",
        "status",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] Saved benchmark CSV: {csv_path}")

    print("\n=== Summary (ok only) ===")
    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_rows.sort(key=lambda r: float(r["mean_ms"]))
    for r in ok_rows:
        print(
            f"{r['label']}: {float(r['mean_ms']):.3f} ms | "
            f"{float(r['fps']):.2f} FPS | {float(r['params_m']):.2f} M params"
        )


if __name__ == "__main__":
    main()
