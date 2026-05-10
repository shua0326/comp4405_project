from __future__ import annotations

import numpy as np


def detect_sky_mask(
    image: np.ndarray,
    dark_channel: np.ndarray,
    brightness_threshold: float,
    dark_channel_threshold: float,
) -> np.ndarray:
    """Return a boolean mask of sky pixels.

    Sky pixels are typically bright across all channels and have a high dark
    channel value (violating the dark channel prior). Both conditions are
    combined to reduce false positives on bright non-sky regions.

    image: HxWx3 float in [0, 1]
    dark_channel: HxW float
    Returns HxW boolean array, True where sky is detected.
    """
    raise NotImplementedError


def apply_sky_transmission(
    transmission: np.ndarray,
    sky_mask: np.ndarray,
    sky_transmission: float,
) -> np.ndarray:
    """Override transmission values in sky regions with a fixed value.

    Prevents the dark channel prior from under-estimating transmission in
    the sky, which would otherwise cause over-brightening on recovery.

    transmission: HxW float
    sky_mask: HxW boolean
    Returns HxW float transmission map with sky regions corrected.
    """
    raise NotImplementedError
