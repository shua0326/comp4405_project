from __future__ import annotations

import numpy as np


def to_float(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 image to float32 in [0, 1]."""
    raise NotImplementedError


def resize(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Resize the image to (height, width) if size is provided, else return as-is."""
    raise NotImplementedError


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize per-channel intensities for numerical stability."""
    raise NotImplementedError
