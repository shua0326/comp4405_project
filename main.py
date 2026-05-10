from __future__ import annotations

import argparse
from pathlib import Path

from src import constants
from src.io_utils import read_image, write_image, ensure_dir
from src.pipeline import dehaze_image, process_directory


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
    parser.add_argument("--skip-refinement", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint: dispatch to single-image or directory pipeline."""
    args = parse_args()

    kwargs = dict(
        patch_size=args.patch_size,
        omega=args.omega,
        t_min=args.t_min,
        atmos_top_percent=args.atmos_top_percent,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        gamma=args.gamma,
        skip_refinement=args.skip_refinement,
    )

    if args.input.is_dir():
        process_directory(args.input, args.output, **kwargs)
    else:
        ensure_dir(args.output.parent)
        image = read_image(args.input)
        result = dehaze_image(image, **kwargs)
        write_image(args.output, result)


if __name__ == "__main__":
    main()
