from __future__ import annotations

import numpy as np


def compute_dark_channel(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Compute the dark channel: J_dark(x) = min_{y in Omega(x)} min_c J^c(y).

    image: HxWx3 float in [0, 1]
    Returns HxW float array.
    """
    raise NotImplementedError
