from __future__ import annotations

import numpy as np


def recover_scene_radiance(
    image: np.ndarray,
    transmission: np.ndarray,
    atmospheric_light: np.ndarray,
    t_min: float,
) -> np.ndarray:
    """Recover the haze-free image J using the atmospheric scattering model:

        J(x) = (I(x) - A) / max(t(x), t_min) + A
    """
    raise NotImplementedError


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction for tonal adjustment after recovery."""
    raise NotImplementedError
