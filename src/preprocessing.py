from __future__ import annotations

import numpy as np
from PIL import Image


def resize(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Resize the image to (height, width) if size is provided, else return as-is."""
    if size is None:
        return image
    return Image.resize(image, size)