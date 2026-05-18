"""Measure per-stage runtime of the full pipeline on a representative image.

Reports mean and standard deviation across N repeats. Useful for the
runtime discussion in the report.

Example:
    python scripts/time_stages.py --input data/input/sunset.jpeg --runs 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import constants
from src.atmospheric_light import estimate_atmospheric_light
from src.dark_channel import estimate_dark_channel
from src.io_utils import read_image
from src.postprocessing import clip_image
from src.recovery import gamma_correct, recover_scene_radiance
from src.refinement import refine_transmission
from src.sky_detection import apply_sky_transmission, detect_sky_mask
from src.transmission import estimate_transmission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--runs", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = read_image(args.input)
    h, w = image.shape[:2]
    print(f"Image: {h}x{w}px, runs={args.runs}")
    print()

    stages = [
        "dark_channel",
        "atmospheric_light",
        "transmission",
        "sky_detection",
        "refinement",
        "recovery",
        "postprocess",
    ]
    timings: dict[str, list[float]] = {s: [] for s in stages}

    for _ in range(args.runs):
        t0 = time.perf_counter()
        dc = estimate_dark_channel(image, constants.PATCH_SIZE)
        timings["dark_channel"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        A = estimate_atmospheric_light(image, dc, constants.ATMOS_TOP_PERCENT)
        timings["atmospheric_light"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        t = estimate_transmission(image, A, constants.PATCH_SIZE, constants.OMEGA)
        timings["transmission"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sky = detect_sky_mask(image, dc, constants.SKY_BRIGHTNESS_THRESHOLD, constants.SKY_DARK_CHANNEL_THRESHOLD)
        t = apply_sky_transmission(t, sky, constants.SKY_TRANSMISSION)
        timings["sky_detection"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        t = refine_transmission(image, t, constants.GUIDED_FILTER_RADIUS, constants.GUIDED_FILTER_EPS)
        timings["refinement"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = recover_scene_radiance(image, t, A, constants.T_MIN)
        timings["recovery"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = gamma_correct(clip_image(result), constants.GAMMA)
        timings["postprocess"].append(time.perf_counter() - t0)

    print(f"{'Stage':25s} {'Mean (ms)':>12s} {'Std (ms)':>12s}")
    print("-" * 51)
    total = 0.0
    for s in stages:
        m = float(np.mean(timings[s])) * 1000
        sd = float(np.std(timings[s])) * 1000
        total += m
        print(f"{s:25s} {m:>12.2f} {sd:>12.2f}")
    print("-" * 51)
    print(f"{'TOTAL':25s} {total:>12.2f}")


if __name__ == "__main__":
    main()
