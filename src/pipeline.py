from __future__ import annotations

from pathlib import Path

import numpy as np

from .dark_channel import estimate_dark_channel
from .atmospheric_light import estimate_atmospheric_light
from .transmission import estimate_transmission
from .recovery import recover_scene_radiance, gamma_correct
from .postprocessing import clip_image
from .refinement import refine_transmission


def dehaze_image(
    image: np.ndarray,
    patch_size: int,
    omega: float,
    t_min: float,
    atmos_top_percent: float,
    guided_radius: int,
    guided_eps: float,
    gamma: float,
    skip_refinement: bool = False,
) -> np.ndarray:
    """Run the full dehazing pipeline on a single image.

    Stages:
        1. Dark channel computation
        2. Atmospheric light estimation
        3. Coarse transmission estimation
        4. Transmission refinement (guided filter) — skipped if skip_refinement=True
        5. Scene radiance recovery
        6. Postprocessing
    """
    dark_channel = estimate_dark_channel(image, patch_size)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel, atmos_top_percent)
    transmission = estimate_transmission(image, atmospheric_light, patch_size, omega)
    if not skip_refinement:
        transmission = refine_transmission(image, transmission, guided_radius, guided_eps)
    recovered = recover_scene_radiance(image, transmission, atmospheric_light, t_min)
    return gamma_correct(clip_image(recovered), gamma)


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
    skip_refinement: bool = False,
) -> None:
    """Iterate over all images in input_dir, dehaze them, and write to output_dir."""
    raise NotImplementedError
