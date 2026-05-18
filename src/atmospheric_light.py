from __future__ import annotations

import numpy as np


def estimate_atmospheric_light(
    image: np.ndarray,
    dark_channel: np.ndarray,
    top_percent: float,
) -> np.ndarray:
    """Estimate the atmospheric light A from the brightest pixels of the dark channel.

    image: HxWx3 float in [0, 1]
    dark_channel: HxW float
    top_percent: fraction of brightest dark-channel pixels to consider.
    Returns a 3-vector A.
    """

    num_pixels = max(1, int(top_percent * dark_channel.size))
    brightest_pixels = np.argsort(dark_channel, axis=None)[-num_pixels:]
    brightest_pixels = np.unravel_index(brightest_pixels, dark_channel.shape)
    candidates = image[brightest_pixels]
    brightest_idx = np.argmax(np.sum(candidates, axis=1))
    return candidates[brightest_idx]
