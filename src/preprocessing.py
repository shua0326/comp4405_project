from __future__ import annotations

import numpy as np
from PIL import Image


def resize(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Resize the image to (height, width) if size is provided, else return as-is."""
    if size is None:
        return image
    height, width = size
    pil_image = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    resized = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(resized) / 255.0
