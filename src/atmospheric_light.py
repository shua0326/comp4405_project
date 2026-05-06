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
    raise NotImplementedError
