import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda x: x.name)
    return files


def _norm_stem(stem: str) -> str:
    return stem.lower().strip()


def _find_matching_image(image_stem: str, folder: Path, model_name: Optional[str] = None) -> Optional[Path]:
    stem = _norm_stem(image_stem)
    candidates = _list_images(folder)
    if not candidates:
        return None

    if model_name:
        model_stem = _norm_stem(model_name)
        exact_model_stem = f"{stem}_{model_stem}"
        exact_model = [p for p in candidates if _norm_stem(p.stem) == exact_model_stem]
        if exact_model:
            return exact_model[0]

    exact = [p for p in candidates if _norm_stem(p.stem) == stem]
    if exact:
        return exact[0]

    prefix = [p for p in candidates if _norm_stem(p.stem).startswith(stem + "_")]
    if prefix:
        return prefix[0]

    contain = [p for p in candidates if stem in _norm_stem(p.stem)]
    if contain:
        return contain[0]

    # Allow case names like "00187-3.0" to fallback to base id "00187"
    # when GT/LQ stems do not include exposure suffixes.
    if "-" in stem:
        base_stem = stem.split("-", 1)[0].strip()
        if base_stem:
            base_exact = [p for p in candidates if _norm_stem(p.stem) == base_stem]
            if base_exact:
                return base_exact[0]

            base_prefix = [p for p in candidates if _norm_stem(p.stem).startswith(base_stem + "_")]
            if base_prefix:
                return base_prefix[0]

            base_contain = [p for p in candidates if base_stem in _norm_stem(p.stem)]
            if base_contain:
                return base_contain[0]

    return None


def _psnr_uint8(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float:
    pred = pred_bgr.astype(np.float32)
    gt = gt_bgr.astype(np.float32)
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim_gray_uint8(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float:
    pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(pred, kernel, sigma)
    mu2 = cv2.GaussianBlur(gt, kernel, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(pred * pred, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gt * gt, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(pred * gt, kernel, sigma) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.55, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _extract_patch(img: np.ndarray, xywh: Tuple[int, int, int, int], ref_hw: Tuple[int, int]) -> np.ndarray:
    ref_h, ref_w = ref_hw
    h, w = img.shape[:2]

    x, y, cw, ch = xywh
    sx = int(round(x * w / max(ref_w, 1)))
    sy = int(round(y * h / max(ref_h, 1)))
    scw = max(1, int(round(cw * w / max(ref_w, 1))))
    sch = max(1, int(round(ch * h / max(ref_h, 1))))

    sx = max(0, min(sx, w - 1))
    sy = max(0, min(sy, h - 1))
    ex = max(sx + 1, min(sx + scw, w))
    ey = max(sy + 1, min(sy + sch, h))

    return img[sy:ey, sx:ex]


def _run_model_commands(models: List[dict], cwd: Path) -> None:
    for model in models:
        cmd = model.get("run_cmd")
        if not cmd:
            continue
        print(f"[RUN] {model.get('name', 'model')}: {cmd}")
        subprocess.run(cmd, shell=True, check=True, cwd=str(cwd))


def _build_panel(
    image_name: str,
    lq_img: Optional[np.ndarray],
    gt_img: np.ndarray,
    model_items: List[Tuple[str, np.ndarray]],
    crops: List[dict],
    thumb_size: Tuple[int, int],
    patch_size: int,
    margin: int,
) -> np.ndarray:
    # Column order: Input, model outputs..., Ref.
    columns: List[Tuple[str, np.ndarray]] = []
    if lq_img is not None:
        columns.append(("Input", lq_img.copy()))
    columns.extend(model_items)
    columns.append(("Ref.", gt_img.copy()))

    gt_h, gt_w = gt_img.shape[:2]

    crop_specs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int], str]] = []
    default_colors = [(0, 165, 255), (120, 255, 120), (255, 180, 0), (255, 100, 255)]
    for i, c in enumerate(crops):
        xywh = tuple(c.get("xywh", [0, 0, 32, 32]))
        color = tuple(c.get("color_bgr", default_colors[i % len(default_colors)]))
        name = c.get("name", chr(ord("A") + i))
        crop_specs.append((xywh, color, str(name)))

    tw, th = thumb_size
    top_tiles = []
    for _, img in columns:
        tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_CUBIC)
        for xywh, color, _ in crop_specs:
            x, y, cw, ch = xywh
            rx = int(round(x * tw / max(gt_w, 1)))
            ry = int(round(y * th / max(gt_h, 1)))
            rcw = max(1, int(round(cw * tw / max(gt_w, 1))))
            rch = max(1, int(round(ch * th / max(gt_h, 1))))
            cv2.rectangle(tile, (rx, ry), (min(tw - 1, rx + rcw), min(th - 1, ry + rch)), color, 2)
        top_tiles.append(tile)

    top_row = cv2.hconcat([cv2.copyMakeBorder(t, 0, 0, 0, margin, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in top_tiles])
    top_row = top_row[:, :-margin, :]

    patch_rows = []
    for xywh, color, _ in crop_specs:
        patches = []
        for _, img in columns:
            patch = _extract_patch(img, xywh, (gt_h, gt_w))
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            patch = cv2.copyMakeBorder(patch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color)
            patches.append(patch)
        row = cv2.hconcat([cv2.copyMakeBorder(p, 0, 0, 0, margin, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for p in patches])
        row = row[:, :-margin, :]
        patch_rows.append(row)

    metric_h = 34
    title_h = 32
    panel_w = top_row.shape[1]
    body_h = title_h + top_row.shape[0] + len(patch_rows) * (patch_rows[0].shape[0] + margin) + metric_h + 2 * margin

    panel = np.full((body_h, panel_w, 3), 20, dtype=np.uint8)

    x = 0
    col_w = tw + margin
    for title, _ in columns:
        _draw_text(panel, title, x + 6, 22, scale=0.56, color=(255, 255, 255))
        x += col_w

    y = title_h
    panel[y:y + top_row.shape[0], :, :] = top_row
    y += top_row.shape[0] + margin

    for row in patch_rows:
        if row.shape[1] < panel_w:
            row = cv2.copyMakeBorder(row, 0, 0, 0, panel_w - row.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
        elif row.shape[1] > panel_w:
            row = row[:, :panel_w, :]
        panel[y:y + row.shape[0], :, :] = row
        y += row.shape[0] + margin

    _draw_text(panel, image_name, 6, body_h - 10, scale=0.58, color=(240, 240, 120))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare restored images with crop zoom panels and metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--run-models", action="store_true", help="Run model commands before composing panels.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases. 0 means no limit.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)

    workspace_root = (config_path.parent / cfg.get("workspace_root", "..")).resolve()
    gt_dir = (workspace_root / cfg["gt_dir"]).resolve()
    lq_dir = (workspace_root / cfg["lq_dir"]).resolve() if cfg.get("lq_dir") else None
    out_dir = (workspace_root / cfg.get("output_dir", "analysis/outputs/compare_panels")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = cfg.get("models", [])
    if not models:
        raise ValueError("Config must define at least one model in 'models'.")

    if args.run_models:
        _run_model_commands(models, workspace_root)

    gt_images = _list_images(gt_dir)
    if not gt_images:
        raise RuntimeError(f"No GT images found in: {gt_dir}")

    provided_cases = cfg.get("cases", [])
    cases = []
    if provided_cases:
        for c in provided_cases:
            name = c["name"]
            crops = c.get("crops", [{"name": "A", "xywh": c.get("xywh", [0, 0, 64, 64])}])
            cases.append({"name": name, "crops": crops})
    else:
        # Auto case discovery from GT names and default crop.
        for p in gt_images:
            cases.append({"name": p.stem, "crops": [{"name": "A", "xywh": [0, 0, 64, 64]}]})

    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    thumb_size = tuple(cfg.get("thumb_size", [220, 150]))
    patch_size = int(cfg.get("patch_size", 120))
    margin = int(cfg.get("margin", 8))

    csv_rows = []

    for case in cases:
        name = case["name"]
        crops = case.get("crops", [])

        gt_path = _find_matching_image(name, gt_dir)
        if gt_path is None:
            print(f"[WARN] Missing GT for {name}")
            continue

        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if gt_img is None:
            print(f"[WARN] Cannot read GT: {gt_path}")
            continue

        lq_img = None
        if lq_dir is not None and lq_dir.exists():
            lq_path = _find_matching_image(name, lq_dir)
            if lq_path is not None:
                lq_img = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)

        model_tiles: List[Tuple[str, np.ndarray]] = []
        metric_row = {"image": name}

        for model in models:
            label = model.get("label", model.get("name", "model"))
            output_dir = (workspace_root / model["output_dir"]).resolve()
            pred_path = _find_matching_image(name, output_dir, model.get("name"))
            if pred_path is None:
                print(f"[WARN] Missing prediction for {name} in {output_dir}")
                continue

            pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
            if pred is None:
                print(f"[WARN] Cannot read prediction: {pred_path}")
                continue

            # Keep model image for visualization and normalize size for metrics.
            model_tiles.append((label, pred))

            pred_for_metric = pred
            if pred.shape[:2] != gt_img.shape[:2]:
                pred_for_metric = cv2.resize(pred, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_CUBIC)

            psnr = _psnr_uint8(pred_for_metric, gt_img)
            ssim = _ssim_gray_uint8(pred_for_metric, gt_img)
            metric_row[f"{label}_psnr"] = psnr
            metric_row[f"{label}_ssim"] = ssim

        if not model_tiles:
            print(f"[WARN] No model outputs available for case: {name}")
            continue

        panel = _build_panel(
            image_name=name,
            lq_img=lq_img,
            gt_img=gt_img,
            model_items=model_tiles,
            crops=crops,
            thumb_size=thumb_size,
            patch_size=patch_size,
            margin=margin,
        )

        out_path = out_dir / f"{name}_comparison.png"
        cv2.imwrite(str(out_path), panel)
        print(f"[OK] Saved panel: {out_path}")

        csv_rows.append(metric_row)

    if csv_rows:
        csv_path = out_dir / "metrics_per_image.csv"
        headers = sorted({k for row in csv_rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"[OK] Saved metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
