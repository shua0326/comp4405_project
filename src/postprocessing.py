from __future__ import annotations

import numpy as np


def clip_image(image: np.ndarray) -> np.ndarray:
    """Clip an image to [0, 1]."""
    return np.clip(image, 0, 1)

