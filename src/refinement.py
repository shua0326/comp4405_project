from __future__ import annotations

import cv2
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
    if guide.ndim == 3:
        guide = cv2.cvtColor(guide.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        guide = guide.astype(np.float32)

    src = src.astype(np.float32)
    radius = max(1, int(radius))
    window_size = (2 * radius + 1, 2 * radius + 1)

    mean_guide = cv2.boxFilter(guide, ddepth=-1, ksize=window_size, normalize=True)
    mean_src = cv2.boxFilter(src, ddepth=-1, ksize=window_size, normalize=True)
    mean_guide_src = cv2.boxFilter(guide * src, ddepth=-1, ksize=window_size, normalize=True)
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(guide * guide, ddepth=-1, ksize=window_size, normalize=True)
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=window_size, normalize=True)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=window_size, normalize=True)
    return mean_a * guide + mean_b


def refine_transmission(
    image: np.ndarray,
    transmission: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Refine a coarse transmission map using the configured method."""
    refined = guided_filter(image, transmission, radius, eps)
    return np.clip(refined, 0, 1)
