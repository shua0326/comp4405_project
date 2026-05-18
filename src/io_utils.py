from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .postprocessing import to_uint8


def list_images(directory: str | Path, extensions: list[str]) -> list[Path]:
    """Return all image paths under `directory` matching the given extensions."""
    return [p for p in Path(directory).rglob("*") if p.suffix.lower() in extensions]


def read_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as an RGB float array in [0, 1]."""
    return np.array(Image.open(path).convert("RGB")) / 255.0


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image (RGB, float in [0, 1] or uint8) to disk."""
    if image.dtype != np.uint8:
        image = to_uint8(image)
    Image.fromarray(image).save(path)


def ensure_dir(path: str | Path) -> Path:
    """Create the directory if it does not exist and return it as a Path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
