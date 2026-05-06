from __future__ import annotations

import numpy as np


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Edge-preserving guided filter.

    guide: HxW or HxWx3 float guidance image (typically the hazy input as grayscale).
    src: HxW float image to be filtered (typically the coarse transmission).
    Returns HxW float refined image.
    """
    raise NotImplementedError


def refine_transmission(
    image: np.ndarray,
    transmission: np.ndarray,
    method: str,
    params: dict,
) -> np.ndarray:
    """Refine a coarse transmission map using the configured method."""
    raise NotImplementedError
