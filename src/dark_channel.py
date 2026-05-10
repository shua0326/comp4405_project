from __future__ import annotations

import numpy as np


from scipy.ndimage import minimum_filter


def estimate_dark_channel(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Compute the dark channel: J_dark(x) = min_{y in Omega(x)} min_c J^c(y).

    image: HxWx3 float in [0, 1]
    Returns HxW float array.
    """
    dark = minimum_filter(np.min(image, axis=2), size=patch_size)
    return dark
