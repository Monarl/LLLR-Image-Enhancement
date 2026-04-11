import argparse
import contextlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


@torch.no_grad()
def compute_illumination_guidance(x: torch.Tensor) -> torch.Tensor:
    """Compute 2-channel illumination guidance from RGB input in [0, 1]."""
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
    if param_key == "auto":
        if "params_ema" in ckpt:
            state = ckpt["params_ema"]
            key = "params_ema"
        elif "params" in ckpt:
            state = ckpt["params"]
            key = "params"
        else:
            state = ckpt
            key = "root"
    elif param_key == "root":
        state = ckpt
        key = "root"
    else:
        state = ckpt[param_key] if param_key in ckpt else ckpt
        key = param_key if param_key in ckpt else "root"

    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean[k[7:]] = v
        else:
            clean[k] = v
    return clean, key


def _import_symbol(module_name: str, symbol_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        raise AttributeError(f"{symbol_name} not found in module {module_name}")
    return getattr(module, symbol_name)


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


def _build_model(entry: dict, workspace_root: Path, device: torch.device):
    module_root = entry.get("module_root")
    abs_root = workspace_root.resolve()
    if module_root:
        abs_root = (workspace_root / module_root).resolve()
        _reset_basicsr_if_needed(abs_root)
        abs_root_str = str(abs_root)
        if abs_root_str in sys.path:
            sys.path.remove(abs_root_str)
        sys.path.insert(0, abs_root_str)

    kwargs = entry.get("kwargs", {})
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
            if "builder" in entry:
                b = entry["builder"]
                fn = _import_symbol(b["module"], b["name"])
                model = fn(**kwargs)
            elif "ctor" in entry:
                c = entry["ctor"]
                cls = _import_symbol(c["module"], c["name"])
                model = cls(**kwargs)
            else:
                raise ValueError("Model entry must contain 'builder' or 'ctor'.")

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
                print(
                    f"[LOAD] {entry.get('name')} ckpt={ckpt_path} key={key_used} "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

                max_missing = entry.get("max_missing_keys")
                max_unexpected = entry.get("max_unexpected_keys")
                if max_missing is not None and len(missing) > int(max_missing):
                    raise RuntimeError(
                        f"Checkpoint mismatch too large for {entry.get('name')}: "
                        f"missing={len(missing)} > max_missing_keys={int(max_missing)}"
                    )
                if max_unexpected is not None and len(unexpected) > int(max_unexpected):
                    raise RuntimeError(
                        f"Checkpoint mismatch too large for {entry.get('name')}: "
                        f"unexpected={len(unexpected)} > max_unexpected_keys={int(max_unexpected)}"
                    )
        finally:
            for p in removed_workspace_paths:
                if p not in sys.path:
                    sys.path.append(p)

    model.eval()
    return model


def _read_image_to_tensor(img_path: Path, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return x.to(device), bgr


def _tensor_to_bgr_u8(y: torch.Tensor) -> np.ndarray:
    y = y.detach().float().cpu().clamp(0.0, 1.0)
    y_np = y[0].permute(1, 2, 0).numpy()
    y_np = (y_np * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(y_np, cv2.COLOR_RGB2BGR)


def _unwrap_model_output(y):
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, (list, tuple)):
        # Common restoration convention: final prediction is the last tensor.
        for item in reversed(y):
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported model output type: {type(y)}")


def _pad_to_multiple(x: torch.Tensor, multiple: int):
    if multiple <= 1:
        return x, (x.shape[-2], x.shape[-1], 0, 0)

    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w, 0, 0)

    x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (h, w, pad_h, pad_w)


def _pad_with_size(x: torch.Tensor, pad_h: int, pad_w: int):
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _crop_output(y: torch.Tensor, orig_hw, output_scale: int):
    h, w = orig_hw
    eh = int(h * output_scale)
    ew = int(w * output_scale)
    return y[..., :eh, :ew]


def _run_ultrais_test_wrapper(entry: dict, workspace_root: Path, device: torch.device, x: torch.Tensor) -> torch.Tensor:
    module_root = entry.get("module_root", "UltraIS-main")
    abs_root = (workspace_root / module_root).resolve()

    _reset_basicsr_if_needed(abs_root)
    abs_root_str = str(abs_root)
    if abs_root_str in sys.path:
        sys.path.remove(abs_root_str)
    sys.path.insert(0, abs_root_str)

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
            model_cls = _import_symbol("basicsr.models.image_restoration_model", "CSDLLSRv4")
            kwargs = entry.get("kwargs", {})
            scale = int(kwargs.get("scale", kwargs.get("upscale", 1)))
            n_feat = int(kwargs.get("n_feat", 64))
            inp_channels = int(kwargs.get("inp_channels", 3))
            out_channels = int(kwargs.get("out_channels", 3))

            ckpt_rel = entry.get("checkpoint")
            if not ckpt_rel:
                raise ValueError("UltraIS test-wrapper inference requires 'checkpoint'.")
            ckpt_path = (workspace_root / ckpt_rel).resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            strict = bool(entry.get("strict", True))
            param_key = entry.get("param_key", "params")
            pad_to = int(entry.get("pad_to", 16))

            opt = {
                "num_gpu": 1 if device.type == "cuda" else 0,
                "is_train": False,
                "dist": False,
                "rank": 0,
                "world_size": 1,
                "scale": scale,
                "network_g": {
                    "type": "CSDLLSRNetv9_7_5",
                    "inp_channels": inp_channels,
                    "out_channels": out_channels,
                    "n_feat": n_feat,
                    "scale": scale,
                },
                "path": {
                    "pretrain_network_g": str(ckpt_path),
                    "strict_load_g": strict,
                    "param_key": param_key,
                },
                "datasets": {
                    "test_1": {
                        "use_grayatten": False,
                        "use_illguidance": True,
                    }
                },
            }

            wrapper = model_cls(opt)
            wrapper.net_g.eval()

            gray = compute_illumination_guidance(x)
            wrapper.feed_data({"lq": x, "gray": gray})
            wrapper.pad_test(pad_to)
            y = wrapper.output
            print(f"[LOAD] {entry.get('name')} ckpt={ckpt_path} key={param_key} strict={strict} via=CSDLLSRv4")
            return y
        finally:
            for p in removed_workspace_paths:
                if p not in sys.path:
                    sys.path.append(p)


def _is_ultrais_entry(entry: dict) -> bool:
    ctor = entry.get("ctor", {})
    mod = str(ctor.get("module", "")).lower()
    cls = str(ctor.get("name", "")).lower()
    name = str(entry.get("name", "")).lower()
    return ("ultrais_arch" in mod) or ("csdllsrnet" in cls) or (name == "ultrais")


def _select_inference_impl(entry: dict) -> str:
    configured = entry.get("inference_impl")
    if configured:
        return str(configured)
    if _is_ultrais_entry(entry):
        # Default to the same wrapper path used by basicsr.test for UltraIS,
        # so single-image analysis matches test-time behavior.
        return "ultrais_test_wrapper"
    return "direct"


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image inference for multiple models.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON/YAML config.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_config(cfg_path)

    workspace_root = (cfg_path.parent / cfg.get("workspace_root", "../..")).resolve()
    image_path = (workspace_root / cfg["input_image"]).resolve()
    output_root = (workspace_root / cfg.get("output_root", "analysis/outputs/single_image_outputs")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device_name = cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        print("[WARN] CUDA not available; fallback to CPU.")
    device = torch.device(device_name)

    x, _ = _read_image_to_tensor(image_path, device)

    models = [m for m in cfg.get("models", []) if m.get("enabled", True)]
    if not models:
        raise ValueError("No enabled models in config.")

    summary = []
    for entry in models:
        name = entry.get("name", "model")
        label = entry.get("label", name)
        input_mode = entry.get("input_mode", "rgb")
        out_subdir = entry.get("output_dir", name)
        save_dir = (output_root / out_subdir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[RUN] {name}")
            inference_impl = _select_inference_impl(entry)

            seed = entry.get("manual_seed", cfg.get("manual_seed"))
            if seed is None and _is_ultrais_entry(entry):
                # UltraIS test options commonly use manual_seed=100.
                seed = 100
            if seed is not None:
                _set_random_seed(int(seed))
                print(f"[SEED] {name} manual_seed={int(seed)}")

            if inference_impl == "ultrais_test_wrapper":
                y = _run_ultrais_test_wrapper(entry, workspace_root, device, x)
            else:
                model = _build_model(entry, workspace_root, device)

                pad_to = int(entry.get("pad_to", 1))
                output_scale = int(entry.get("output_scale", entry.get("kwargs", {}).get("scale", entry.get("kwargs", {}).get("upscale", 1))))

                x_in, (orig_h, orig_w, pad_h, pad_w) = _pad_to_multiple(x, pad_to)

                with torch.no_grad():
                    if input_mode == "rgb_gray":
                        # Match UltraIS test pipeline: guidance is computed from
                        # the unpadded input, then both lq and guidance are padded.
                        gray = compute_illumination_guidance(x)
                        gray_in = _pad_with_size(gray, pad_h, pad_w)
                        y = model(x_in, gray_in)
                    else:
                        y = model(x_in)

                y = _unwrap_model_output(y)
                y = _crop_output(y, (orig_h, orig_w), output_scale)

            pred_bgr = _tensor_to_bgr_u8(y)
            save_name = f"{image_path.stem}_{name}.png"
            save_path = save_dir / save_name
            cv2.imwrite(str(save_path), pred_bgr)
            print(f"[OK] {label} -> {save_path}")
            summary.append({"name": name, "status": "ok", "output": str(save_path)})
        except Exception as exc:
            print(f"[ERR] {name}: {exc}")
            summary.append({"name": name, "status": "error", "error": str(exc)})

    print("\n=== Inference Summary ===")
    for s in summary:
        if s["status"] == "ok":
            print(f"{s['name']}: OK")
        else:
            print(f"{s['name']}: ERROR -> {s['error']}")


if __name__ == "__main__":
    main()
