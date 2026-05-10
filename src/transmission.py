from __future__ import annotations

import numpy as np

from .dark_channel import estimate_dark_channel


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    patch_size: int,
    omega: float,
) -> np.ndarray:
    """Estimate the coarse transmission map t(x) = 1 - omega * dark_channel(I/A).

    image: HxWx3 float in [0, 1]
    atmospheric_light: 3-vector A
    Returns HxW float array.
    """
    normalised = image / atmospheric_light
    dark = estimate_dark_channel(normalised, patch_size)
    return 1 - omega * dark


def clip_transmission(transmission: np.ndarray, t_min: float) -> np.ndarray:
    """Lower-bound the transmission map by t_min to avoid noise amplification."""
    return np.maximum(transmission, t_min)
