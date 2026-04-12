import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from basicsr.metrics import calculate_psnr, calculate_ssim

try:
    import lpips
except ImportError:
    lpips = None

try:
    from realDenoising.basicsr.metrics.niqe import calculate_niqe
except Exception:
    calculate_niqe = None


def _list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _to_lpips_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def _loe(pred_bgr: np.ndarray, lq_bgr: np.ndarray, resize_to: int = 50) -> float:
    pred_l = np.max(pred_bgr, axis=2).astype(np.float32)
    lq_l = np.max(lq_bgr, axis=2).astype(np.float32)

    pred_l = cv2.resize(pred_l, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    lq_l = cv2.resize(lq_l, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    p = pred_l.reshape(-1)
    q = lq_l.reshape(-1)
    n = p.shape[0]

    err = 0
    for i in range(n):
        rel_q = q[i] >= q
        rel_p = p[i] >= p
        err += np.count_nonzero(rel_q != rel_p)
    return float(err / n)


def _infer_lq_stem(pred_stem: str, lq_map: dict):
    # Expected basicsr.test naming: <lq_stem>_<exp_name>
    candidate = pred_stem.split("_", 1)[0]
    if candidate in lq_map:
        return candidate

    # Fallback: find longest lq stem that is a prefix of pred stem.
    matches = [k for k in lq_map.keys() if pred_stem.startswith(k)]
    if not matches:
        return None
    return sorted(matches, key=len, reverse=True)[0]


def _infer_gt_key_from_lq_stem(lq_stem: str, gt_map: dict):
    # For RELLISUR direct_gt_match: lq like 00076-3.0 -> gt like 00076
    base_id = lq_stem.split("-", 1)[0]
    if base_id in gt_map:
        return base_id

    # Fallback for datasets where GT keeps full stem.
    if lq_stem in gt_map:
        return lq_stem
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS/RMSE/NIQE/LOE and save CSV.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory of model output images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of GT HR images.")
    parser.add_argument("--lq_dir", type=str, required=True, help="Directory of low-light input images (for LOE).")
    parser.add_argument("--csv_path", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--crop_border", type=int, default=2, help="Crop border for PSNR/SSIM.")
    parser.add_argument("--test_y_channel", action="store_true", help="Use Y-channel for PSNR/SSIM.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for LPIPS.")
    parser.add_argument("--loe_resize", type=int, default=50, help="Resize side length for LOE.")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS even if package is installed.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    lq_dir = Path(args.lq_dir)
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    if not pred_dir.exists():
        missing.append(f"pred_dir not found: {pred_dir}")
    if not gt_dir.exists():
        missing.append(f"gt_dir not found: {gt_dir}")
    if not lq_dir.exists():
        missing.append(f"lq_dir not found: {lq_dir}")
    if missing:
        raise FileNotFoundError("; ".join(missing))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    use_lpips = (not args.skip_lpips) and (lpips is not None)
    if use_lpips:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    else:
        lpips_model = None
        if lpips is None and not args.skip_lpips:
            print("[WARN] lpips package is not installed; LPIPS will be written as NaN.")
            print("[HINT] Install with: pip install lpips")

    use_niqe = calculate_niqe is not None
    if not use_niqe:
        print("[WARN] NIQE function is unavailable; NIQE will be written as NaN.")

    pred_files = _list_images(pred_dir)
    gt_files = _list_images(gt_dir)
    lq_files = _list_images(lq_dir)

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction images found in {pred_dir}. Run basicsr.test first with val.save_img=true."
        )

    gt_map = {p.stem: p for p in gt_files}
    lq_map = {p.stem: p for p in lq_files}

    rows = []
    skipped = 0
    for pred_path in tqdm(pred_files, desc="Evaluating"):
        pred_stem = pred_path.stem

        lq_stem = _infer_lq_stem(pred_stem, lq_map)
        if lq_stem is None:
            skipped += 1
            continue

        gt_key = _infer_gt_key_from_lq_stem(lq_stem, gt_map)
        if gt_key is None:
            skipped += 1
            continue

        gt_path = gt_map[gt_key]
        lq_path = lq_map[lq_stem]

        pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
        if pred is None or gt is None or lq is None:
            skipped += 1
            continue

        if pred.shape[:2] != gt.shape[:2]:
            pred_for_ref = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            pred_for_ref = pred

        if lpips_model is not None:
            lp = lpips_model(_to_lpips_tensor(pred_for_ref, device), _to_lpips_tensor(gt, device))
            lpips_val = float(lp.detach().cpu().item())
        else:
            lpips_val = float("nan")

        niqe_val = (
            float(calculate_niqe(pred_for_ref, crop_border=0, input_order="HWC", convert_to="y"))
            if use_niqe else float("nan")
        )

        row = {
            "imgname": pred_stem,
            "psnr": float(calculate_psnr(pred_for_ref, gt, crop_border=args.crop_border, test_y_channel=args.test_y_channel)),
            "ssim": float(calculate_ssim(pred_for_ref, gt, crop_border=args.crop_border, test_y_channel=args.test_y_channel)),
            "lpips": lpips_val,
            "rmse": _rmse(pred_for_ref, gt),
            "niqe": niqe_val,
            "loe": _loe(pred_for_ref, lq, resize_to=args.loe_resize),
        }
        rows.append(row)

    fields = ["imgname", "psnr", "ssim", "lpips", "rmse", "niqe", "loe"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        avg = {k: float(np.nanmean([r[k] for r in rows])) for k in fields if k != "imgname"}
        print("Averages:")
        for k, v in avg.items():
            print(f"  {k}: {v:.6f}")
    print(f"Saved CSV: {csv_path}")
    if skipped:
        print(f"[WARN] Skipped {skipped} files due to unmatched GT/LQ mapping.")


if __name__ == "__main__":
    main()
