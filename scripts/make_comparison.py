"""Generate a side-by-side qualitative comparison figure for the report.

Produces a single PNG that contains the hazy input and the dehazed result
under each ablation configuration, aligned in a row.

Example:
    python scripts/make_comparison.py \\
        --input data/input/sunset.jpeg \\
        --output results/figures/sunset_comparison.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src import constants
from src.io_utils import read_image
from src.pipeline import dehaze_image


CONFIGS: list[tuple[str, dict | None]] = [
    ("Hazy",         None),
    ("Coarse",       {"skip_refinement": True,  "skip_sky_detection": True}),
    ("+ Sky",        {"skip_refinement": True,  "skip_sky_detection": False}),
    ("+ Refinement", {"skip_refinement": False, "skip_sky_detection": True}),
    ("Full",         {"skip_refinement": False, "skip_sky_detection": False}),
]


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Hazy input image.")
    p.add_argument("--output", type=Path, required=True, help="Output figure PNG.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = base_kwargs()
    hazy = read_image(args.input)

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 4))
    for ax, (name, overrides) in zip(axes, CONFIGS):
        img = hazy if overrides is None else dehaze_image(hazy, **{**kwargs, **overrides})
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
