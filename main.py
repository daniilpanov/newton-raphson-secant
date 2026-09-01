import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from algorithms import get_method_class
from tests import get_function
from visualizer import get_visualizer

DEFAULT_SAVE_DIR = Path('/storage/emulated/0/Download/mathcad-nf-sect')


def parse_start(value: str):
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"'--start' must be 'x,y', got {value!r}")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'--start' must be two numbers, got {value!r}")


def build_parser():
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Newton-Raphson + secant on test functions')
    p.add_argument('--function', required=True,
                   choices=['quadratic', 'himmelblau', 'rosenbrock'],
                   help='test function (from tests/)')
    p.add_argument('--method', default='modified',
                   choices=['simple', 'modified'],
                   help='optimization implementation')
    p.add_argument('--viz', default='photo',
                   choices=['photo', 'interactive'],
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
    print(f"{'k':>3s}  {'x':>20s}  {'y':>20s}  {'f(x,y)':>12s}  {'|grad|':>12s}")
    print(f"{'-'*70}")
    for k in range(len(X)):
        px, py = X[k]
        fv = float(func.f(px, py))
        gv = float(np.linalg.norm(func.grad(X[k])))
        print(f"{k:3d}  {px:20.10f}  {py:20.10f}  {fv:12.6e}  {gv:12.6e}")
    print(f"{'-'*70}")
    print(f"  Total iterations: {len(X)-1}")


def closest_minimum(xf, minima):
    dists = [np.hypot(xf[0] - m[0], xf[1] - m[1]) for m in minima]
    idx = int(np.argmin(dists))
    return idx, minima[idx], dists[idx]


def main():
    load_dotenv()
    args = build_parser().parse_args()
    check_args(args)

    func = get_function(args.function)
    method_cls = get_method_class(args.method)
    method = method_cls()

    save_dir = args.save_dir or os.getenv('VIZ_SAVE_DIR') or str(DEFAULT_SAVE_DIR)
    visualizer = get_visualizer(args.viz, save_dir=save_dir)

    if args.random:
        rng = np.random.default_rng(args.seed)
        starts = rng.uniform(func.lo, func.hi, size=(args.points, 2))
    else:
        starts = [np.asarray(args.start, float)]

    trajectories = []
    print(f"\nFunction: {func.name} | method: {args.method} | "
          f"{'random (%d points)' % args.points if args.random else 'single start'}")
    for i, x0 in enumerate(starts):
        X = method.nf(func.f, func.grad, func.hess, x0, m=1000 if func.name == 'rosenbrock' else 500)
        trajectories.append(X)
        xf = X[-1]
        label = f"{func.name} from ({x0[0]:.3f}, {x0[1]:.3f})"
        print_trajectory(X, func, label) if not args.random else None
        if func.minima:
            idx, cmin, dist = closest_minimum(xf, func.minima)
            print(f"  Final: ({xf[0]:.6f}, {xf[1]:.6f}), "
                  f"closest min#{idx} dist={dist:.2e}, iters={len(X)-1}")
        else:
            print(f"  Final: ({xf[0]:.6f}, {xf[1]:.6f}), iters={len(X)-1}")

    tag = 'random' if args.random else 'single'
    filename = f"{func.name}_{args.method}_{tag}"
    visualizer.view(func, trajectories,
                    title=f"{func.name} ({args.method}) - "
                          f"{'%d random starts' % args.points if args.random else 'one start'}",
                    filename=filename)
    print(f"\nDone. Files saved to {save_dir}")


if __name__ == '__main__':
    main()