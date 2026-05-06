from __future__ import annotations

import argparse
from pathlib import Path

from src import constants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COMP4405 Image Dehazing")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output image or directory.")
    parser.add_argument("--patch-size", type=int, default=constants.PATCH_SIZE)
    parser.add_argument("--omega", type=float, default=constants.OMEGA)
    parser.add_argument("--t-min", type=float, default=constants.T_MIN)
    parser.add_argument("--atmos-top-percent", type=float, default=constants.ATMOS_TOP_PERCENT)
    parser.add_argument("--guided-radius", type=int, default=constants.GUIDED_FILTER_RADIUS)
    parser.add_argument("--guided-eps", type=float, default=constants.GUIDED_FILTER_EPS)
    parser.add_argument("--gamma", type=float, default=constants.GAMMA)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint: dispatch to single-image or directory pipeline."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
