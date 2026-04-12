import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


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


def _build_model_from_entry(entry: dict, workspace_root: Path, device: torch.device):
    module_root = entry.get("module_root")
    if module_root:
        abs_module_root = (workspace_root / module_root).resolve()
        if str(abs_module_root) not in sys.path:
            sys.path.insert(0, str(abs_module_root))

    kwargs = entry.get("kwargs", {})

    if "builder" in entry:
        builder = entry["builder"]
        fn = _import_callable(builder["module"], builder["name"])
        model = fn(**kwargs)
    elif "ctor" in entry:
        ctor = entry["ctor"]
        cls = _import_callable(ctor["module"], ctor["name"])
        model = cls(**kwargs)
    else:
        raise ValueError("Model entry must have either 'builder' or 'ctor'.")

    model = model.to(device)

    ckpt_rel = entry.get("checkpoint")
    if ckpt_rel:
        ckpt_path = (workspace_root / ckpt_rel).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state, key_used = _resolve_state_dict(ckpt, entry.get("param_key", "auto"))
        strict = bool(entry.get("strict", False))
        missing, unexpected = model.load_state_dict(state, strict=strict)

        return model, {
            "checkpoint": str(ckpt_path),
            "param_key_used": key_used,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }

    return model, {"checkpoint": "", "param_key_used": "", "missing_keys": 0, "unexpected_keys": 0}


def _benchmark_latency(model, input_mode: str, h: int, w: int, device: torch.device, warmup: int, reps: int, half: bool):
    model.eval()

    x = torch.randn(1, 3, h, w, device=device)
    gray = torch.randn(1, 2, h, w, device=device) if input_mode == "rgb_gray" else None

    if half and device.type == "cuda":
        model = model.half()
        x = x.half()
        if gray is not None:
            gray = gray.half()

    def _forward():
        if input_mode == "rgb_gray":
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


def _compute_flops_optional(model, input_mode: str, h: int, w: int, device: torch.device) -> Optional[float]:
    try:
        from thop import profile
    except Exception:
        return None

    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    if input_mode == "rgb_gray":
        gray = torch.randn(1, 2, h, w, device=device)
        flops, _ = profile(model, inputs=(x, gray), verbose=False)
    else:
        flops, _ = profile(model, inputs=(x,), verbose=False)
    return float(flops / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified multi-model latency benchmark for LLIE+SR.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config (JSON/YAML).")
    parser.add_argument("--csv", type=str, default="", help="Override output CSV path.")
    parser.add_argument("--flops", action="store_true", help="Also compute FLOPs with thop if available.")
    args = parser.parse_args()

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

    input_h = int(cfg.get("input_height", 128))
    input_w = int(cfg.get("input_width", 128))
    warmup = int(cfg.get("warmup", 10))
    reps = int(cfg.get("repetitions", 200))
    half = bool(cfg.get("half", False))

    models = [m for m in cfg.get("models", []) if m.get("enabled", True)]
    if not models:
        raise ValueError("No enabled models found in config.")

    rows = []
    for entry in models:
        name = entry.get("name", "model")
        label = entry.get("label", name)
        input_mode = entry.get("input_mode", "rgb")

        print(f"[RUN] {name}")
        try:
            model, load_info = _build_model_from_entry(entry, workspace_root, device)
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            mean_ms, std_ms, fps = _benchmark_latency(model, input_mode, input_h, input_w, device, warmup, reps, half)
            gflops = _compute_flops_optional(model, input_mode, input_h, input_w, device) if args.flops else None

            row = {
                "name": name,
                "label": label,
                "input_mode": input_mode,
                "params_m": params_m,
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "fps": fps,
                "gflops": gflops if gflops is not None else "",
                "checkpoint": load_info.get("checkpoint", ""),
                "param_key_used": load_info.get("param_key_used", ""),
                "missing_keys": load_info.get("missing_keys", 0),
                "unexpected_keys": load_info.get("unexpected_keys", 0),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            row = {
                "name": name,
                "label": label,
                "input_mode": input_mode,
                "params_m": "",
                "mean_ms": "",
                "std_ms": "",
                "fps": "",
                "gflops": "",
                "checkpoint": entry.get("checkpoint", ""),
                "param_key_used": "",
                "missing_keys": "",
                "unexpected_keys": "",
                "status": "error",
                "error": str(exc),
            }
            print(f"[ERR] {name}: {exc}")

        rows.append(row)

    csv_path = Path(args.csv).resolve() if args.csv else (workspace_root / cfg.get("output_csv", "analysis/outputs/latency/latency_report.csv")).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "name",
        "label",
        "input_mode",
        "params_m",
        "gflops",
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
