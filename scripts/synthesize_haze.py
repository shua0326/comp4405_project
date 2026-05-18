"""Synthesise hazy versions of clean images using the atmospheric scattering
model. Useful if you don't have a paired benchmark dataset to hand: you can
take any clean outdoor photo, synthesise haze, and use the original as GT.

I(x) = J(x) * t(x) + A * (1 - t(x))

where t(x) = exp(-beta * d(x)) and d(x) is a synthetic depth proxy (radial
distance from the image centre, normalised to [0, 1]).

Example:
    python scripts/synthesize_haze.py \\
        --input data/clean \\
        --output data/synthetic_hazy \\
        --beta 1.2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.io_utils import ensure_dir, read_image, write_image


def radial_depth(h: int, w: int) -> np.ndarray:
    """Synthetic depth proxy: radial distance from image centre, normalised."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = (h - 1) / 2, (w - 1) / 2
    d = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    return d / d.max()


def synthesize(image: np.ndarray, beta: float, A: float) -> np.ndarray:
    """Apply the atmospheric scattering model with synthetic radial depth."""
    h, w = image.shape[:2]
    d = radial_depth(h, w)
    t = np.exp(-beta * d)[..., None]  # HxWx1
    A_vec = np.array([A, A, A], dtype=np.float32)
    hazy = image * t + A_vec * (1 - t)
    return np.clip(hazy, 0, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Clean image dir.")
    p.add_argument("--output", type=Path, required=True, help="Output hazy dir.")
    p.add_argument("--beta", type=float, default=1.2, help="Scattering coefficient.")
    p.add_argument("--A", type=float, default=0.9, help="Atmospheric light brightness.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    inputs = sorted(p for p in args.input.iterdir() if p.suffix.lower() in valid_exts)
    if not inputs:
        raise SystemExit(f"No images found in {args.input}")
    for p in inputs:
        clean = read_image(p)
        hazy = synthesize(clean, args.beta, args.A)
        out_path = args.output / p.name
        write_image(out_path, hazy)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
