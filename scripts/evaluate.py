"""Evaluate the dehazing pipeline across ablation configurations.

Given a directory of hazy images and a matching directory of ground-truth
haze-free references (same filename in each), compute PSNR and SSIM for each
of the four ablation configurations:

    coarse           — skip refinement AND sky detection
    +sky             — skip refinement only
    +refinement      — skip sky detection only
    full             — full pipeline

Writes per-image and averaged metrics to a CSV file, and a Markdown table
suitable for pasting into the report.

Example:
    python scripts/evaluate.py \\
        --hazy-dir data/hazy \\
        --gt-dir data/gt \\
        --output-csv results/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from src import constants
from src.io_utils import read_image
from src.pipeline import dehaze_image


CONFIGS: list[tuple[str, dict]] = [
    ("coarse",      {"skip_refinement": True,  "skip_sky_detection": True}),
    ("+sky",        {"skip_refinement": True,  "skip_sky_detection": False}),
    ("+refinement", {"skip_refinement": False, "skip_sky_detection": True}),
    ("full",        {"skip_refinement": False, "skip_sky_detection": False}),
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def base_kwargs() -> dict:
    return dict(
        patch_size=constants.PATCH_SIZE,
        omega=constants.OMEGA,
        t_min=constants.T_MIN,
        atmos_top_percent=constants.ATMOS_TOP_PERCENT,
        guided_radius=constants.GUIDED_FILTER_RADIUS,
        guided_eps=constants.GUIDED_FILTER_EPS,
        gamma=constants.GAMMA,
        sky_brightness_threshold=constants.SKY_BRIGHTNESS_THRESHOLD,
        sky_dark_channel_threshold=constants.SKY_DARK_CHANNEL_THRESHOLD,
        sky_transmission=constants.SKY_TRANSMISSION,
    )


def match_gt(hazy_path: Path, gt_dir: Path) -> Path | None:
    """Look for a GT file with the same name, or by stem if names differ."""
    if (gt_dir / hazy_path.name).exists():
        return gt_dir / hazy_path.name
    for gt in gt_dir.iterdir():
        if gt.stem == hazy_path.stem and gt.suffix.lower() in VALID_EXTS:
            return gt
    # RESIDE-style: hazy is "1400_3_0.85.jpg", GT is "1400.jpg"
    stem_prefix = hazy_path.stem.split("_")[0]
    for gt in gt_dir.iterdir():
        if gt.stem == stem_prefix and gt.suffix.lower() in VALID_EXTS:
            return gt
    # Outdoor-style: hazy is "01_outdoor_hazy.jpg", GT is "01_outdoor_GT.jpg"
    for gt in gt_dir.iterdir():
        if gt.stem.split("_")[0] == stem_prefix and gt.suffix.lower() in VALID_EXTS:
            return gt
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hazy-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=Path("results/metrics.csv"))
    p.add_argument("--output-md", type=Path, default=Path("results/metrics.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    kwargs = base_kwargs()

    per_image_rows: list[list] = []
    aggregated: dict[str, dict[str, list[float]]] = {
        name: {"psnr": [], "ssim": []} for name, _ in CONFIGS
    }

    hazy_paths = sorted(p for p in args.hazy_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
    if not hazy_paths:
        raise SystemExit(f"No images found in {args.hazy_dir}")

    for hazy_path in hazy_paths:
        gt_path = match_gt(hazy_path, args.gt_dir)
        if gt_path is None:
            print(f"[skip] no GT match for {hazy_path.name}")
            continue

        hazy = read_image(hazy_path)
        gt = read_image(gt_path)
        if gt.shape != hazy.shape:
            print(f"[skip] shape mismatch {hazy_path.name}: hazy={hazy.shape} gt={gt.shape}")
            continue

        for name, overrides in CONFIGS:
            out = dehaze_image(hazy, **{**kwargs, **overrides})
            out = np.clip(out, 0, 1).astype(np.float64)
            gt_f = gt.astype(np.float64)
            psnr_val = psnr_metric(gt_f, out, data_range=1.0)
            ssim_val = ssim_metric(gt_f, out, data_range=1.0, channel_axis=2)
            per_image_rows.append([hazy_path.name, name, psnr_val, ssim_val])
            aggregated[name]["psnr"].append(psnr_val)
            aggregated[name]["ssim"].append(ssim_val)
            print(f"{hazy_path.name:30s}  {name:12s}  PSNR={psnr_val:6.3f}  SSIM={ssim_val:.4f}")

    # Write per-image CSV.
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "config", "psnr", "ssim"])
        w.writerows(per_image_rows)
        w.writerow([])
        w.writerow(["config", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"])
        for name, _ in CONFIGS:
            psnrs = aggregated[name]["psnr"]
            ssims = aggregated[name]["ssim"]
            if not psnrs:
                continue
            w.writerow([
                name,
                statistics.mean(psnrs),
                statistics.stdev(psnrs) if len(psnrs) > 1 else 0.0,
                statistics.mean(ssims),
                statistics.stdev(ssims) if len(ssims) > 1 else 0.0,
            ])

    # Write a Markdown summary table.
    with open(args.output_md, "w") as f:
        f.write("| Configuration | PSNR ↑ | SSIM ↑ |\n")
        f.write("|---|---|---|\n")
        for name, _ in CONFIGS:
            psnrs = aggregated[name]["psnr"]
            ssims = aggregated[name]["ssim"]
            if not psnrs:
                continue
            f.write(f"| {name} | {statistics.mean(psnrs):.3f} | {statistics.mean(ssims):.4f} |\n")

    print(f"\nWrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
