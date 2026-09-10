import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from algorithms import nf
from functions import get_function
from logger import DEFAULT_LOG_FILE, make_logger, log_enabled
from visualizer import get_visualizer

DEFAULT_SAVE_DIR = Path('/storage/emulated/0/Download/mathcad-nf-sect')


def parse_start(value: str):
    parts = value.split(',')
    if len(parts) == 1:
        try:
            return (float(parts[0]),)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"'--start' must be a number or 'x,y', got {value!r}")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"'--start' must be two numbers, got {value!r}")
    raise argparse.ArgumentTypeError(
        f"'--start' must be 'x' or 'x,y', got {value!r}")


def build_parser():
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Newton-Raphson + secant on test functions')
    p.add_argument('--function', required=True,
                   choices=['quadratic', 'himmelblau', 'rosenbrock', 'double_well'],
                   help='test function (from functions/)')
    p.add_argument('--viz', default='photo',
                   choices=['photo', 'oned'],
                   help='visualizer implementation')
    p.add_argument('--save-dir', default=None,
                   help='photo save dir; default from .env (VIZ_SAVE_DIR)')
    p.add_argument('--start', type=parse_start, metavar='X,Y',
                   help='start point (required unless --random)')
    p.add_argument('--random', action='store_true',
                   help='run N random start points (no --start needed)')
    p.add_argument('--points', type=int, default=4,
                   help='number of random points (with --random)')
    p.add_argument('--seed', type=int, default=None,
                   help='random seed for reproducible runs')
    return p


def check_args(args):
    if not args.random and args.start is None:
        sys.exit("Error: provide --start X,Y or use --random")
    if args.random and args.points < 1:
        sys.exit("Error: --points must be >= 1")


def print_trajectory(X, func, label=''):
    print(f"\n{'='*70}")
    print(f"  Trajectory: {label}")
    print(f"{'='*70}")
    n = X.shape[1] if X.ndim == 2 else 1
    cols = '  '.join(f"{'x%d' % (j + 1):>20s}" for j in range(n))
    print(f"{'k':>3s}  {cols}  {'f(x)':>12s}  {'|grad|':>12s}")
    print(f"{'-'*70}")
    for k in range(len(X)):
        xv = X[k]
        fv = float(func.f(xv))
        gv = float(np.linalg.norm(func.grad(xv)))
        vals = '  '.join(f"{vv:20.10f}" for vv in np.asarray(xv, float).ravel())
        print(f"{k:3d}  {vals}  {fv:12.6e}  {gv:12.6e}")
    print(f"{'-'*70}")
    print(f"  Total iterations: {len(X)-1}")


def closest_minimum(xf, minima):
    xf = np.asarray(xf, float).ravel()
    dists = [np.linalg.norm(xf - np.asarray(m, float)) for m in minima]
    idx = int(np.argmin(dists))
    return idx, minima[idx], dists[idx]


def main():
    load_dotenv()
    args = build_parser().parse_args()
    check_args(args)

    func = get_function(args.function)
    m = 1000 if func.name == 'rosenbrock' else 500
    if args.random:
        rng = np.random.default_rng(args.seed)
        starts = rng.uniform(func.lo, func.hi, size=(args.points, func.dim))
    else:
        starts = [np.asarray(args.start, float)]
    multi = len(starts) > 1

    if log_enabled():
        mode = (os.getenv('LOG_MODE') or 'no').strip().lower()
        if mode in ('file', 'both'):
            Path(os.getenv('LOG_FILE') or DEFAULT_LOG_FILE).unlink(missing_ok=True)
        for i, x0 in enumerate(starts):
            name = f'{func.name} #{i + 1}' if multi else func.name
            with make_logger(name) as logger:
                nf(func.f, func.grad, func.hess, x0, m=m, logger=logger)
        return

    save_dir = args.save_dir or os.getenv('VIZ_SAVE_DIR') or str(DEFAULT_SAVE_DIR)
    if args.viz == 'photo' and func.dim == 1:
        args.viz = 'oned'
    visualizer = get_visualizer(args.viz, save_dir=save_dir)

    trajectories = []
    print(f"\nFunction: {func.name} ({func.dim}D) | "
          f"{'random (%d points)' % args.points if args.random else 'single start'}")
    for i, x0 in enumerate(starts):
        with make_logger(func.name) as logger:
            X = nf(func.f, func.grad, func.hess, x0, m=m, logger=logger)
        trajectories.append(X)
        xf = X[-1]
        x0f = np.asarray(x0, float).ravel()
        label = f"{func.name} from ({', '.join(f'{v:.3f}' for v in x0f)})"
        print_trajectory(X, func, label) if not args.random else None
        xf_str = '(' + ', '.join(f'{v:.6f}' for v in np.asarray(xf, float).ravel()) + ')'
        if func.minima:
            idx, cmin, dist = closest_minimum(xf, func.minima)
            print(f"  Final: {xf_str}, closest min#{idx} dist={dist:.2e}, iters={len(X)-1}")
        else:
            print(f"  Final: {xf_str}, iters={len(X)-1}")

    tag = 'random' if args.random else 'single'
    filename = f"{func.name}_{tag}"
    visualizer.view(func, trajectories,
                    title=f"{func.name} - "
                          f"{'%d random starts' % args.points if args.random else 'one start'}",
                filename=filename)
    print(f"\nDone. Files saved to {save_dir}")


if __name__ == '__main__':
    main()