from __future__ import annotations

from pathlib import Path

import numpy as np


def dehaze_image(
    image: np.ndarray,
    patch_size: int,
    omega: float,
    t_min: float,
    atmos_top_percent: float,
    guided_radius: int,
    guided_eps: float,
    gamma: float,
) -> np.ndarray:
    """Run the full dehazing pipeline on a single image.

    Stages:
        1. Dark channel computation
        2. Atmospheric light estimation
        3. Coarse transmission estimation
        4. Transmission refinement (guided filter)
        5. Scene radiance recovery
        6. Postprocessing
    """
    raise NotImplementedError


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    patch_size: int,
    omega: float,
    t_min: float,
    atmos_top_percent: float,
    guided_radius: int,
    guided_eps: float,
    gamma: float,
) -> None:
    """Iterate over all images in input_dir, dehaze them, and write to output_dir."""
    raise NotImplementedError
