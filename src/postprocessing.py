from __future__ import annotations

import numpy as np


def clip_image(image: np.ndarray) -> np.ndarray:
    """Clip an image to [0, 1]."""
    raise NotImplementedError


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8."""
    raise NotImplementedError
