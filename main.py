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
    p.add_argument('--trace', action='store_true',
                   help='print step-by-step trace for each start')
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


def _fmt(v):
    if v is None:
        return '—'
    v = float(v)
    if v == 0:
        return '0'
    a = abs(v)
    if a < 1e-4 or a >= 1e5:
        return f'{v:.3e}'
    return f'{v:.6f}'


def _fmt_vec(v):
    return '(' + ', '.join(_fmt(c) for c in np.asarray(v, float).ravel()) + ')'


def _fmt_mat(m):
    rows = ['[' + ', '.join(_fmt(c) for c in np.asarray(r, float)) + ']'
            for r in np.asarray(m, float)]
    return '[' + ', '.join(rows) + ']'


def format_trace(trace, func_name, method_name):
    if method_name == 'modified':
        meth = 'модифицированный Ньютон (защиты + линейный поиск)'
    else:
        meth = 'канонический Ньютон + секущие'
    lines = []
    for rec in trace:
        kind = rec['kind']
        if kind == 'start':
            lines.append('=' * 78)
            lines.append(f"Функция: {func_name}  |  метод: {meth}")
            lines.append(f"x0 = {_fmt_vec(rec['x'])}  |  eps1 = {_fmt(rec['eps1'])}  |  "
                         f"eps2 = {_fmt(rec['eps2'])}  |  m = {rec['m']}")
            lines.append('=' * 78)
        elif kind == 'iter':
            lines.append('')
            lines.append(f"k={rec['k']}  x({rec['k']}) = {_fmt_vec(rec['x'])}  "
                         f"f = {_fmt(rec['f'])}  |grad| = {_fmt(rec['g_norm'])}")
            lines.append(f"   grad = {_fmt_vec(rec['g'])}")
        elif kind == 'hess':
            ok = 'ДА' if rec['pd_ok'] else 'НЕТ'
            lines.append(f"   H = {_fmt_mat(rec['H'])}")
            lines.append(f"   det(H) = {_fmt(rec['det'])}  |  H00 = {_fmt(rec['h00'])}  |  "
                         f"условие PD (H00>0 ∧ |det|>1e-12): {ok}")
        elif kind == 'direction':
            origin = ('Ньютон (d = -H⁻¹·g)' if rec['origin'] == 'newton'
                      else 'градиентный спуск (fallback d = -g)')
            lines.append(f"   → d = {_fmt_vec(rec['d'])}  |d| = {_fmt(rec['d_norm'])}  "
                         f"d·g = {_fmt(rec['d_dot_g'])}")
            lines.append(f"   выбор: {origin}  —  {rec['reason']}")
        elif kind == 'bracket_start':
            lines.append(f"   брекет: b = 1 → φ(b) = {_fmt(rec['phi_b'])}")
        elif kind == 'bracket':
            lines.append(f"           b = {_fmt(rec['b'])} → φ(b) = {_fmt(rec['phi_b'])} "
                         f"(φ(b) < 0 → b ← 2·b)")
        elif kind == 'sect':
            lines.append(f"   SECT: a = {_fmt(rec['a'])}  b = {_fmt(rec['b'])}  "
                         f"t = {_fmt(rec['t'])}  φ(a) = {_fmt(rec['phi_a'])}  "
                         f"φ(b) = {_fmt(rec['phi_b'])}")
        elif kind == 'sect_denzero':
            lines.append('   SECT: знаменатель φ(b)-φ(a) = 0 → стоп')
        elif kind == 'sect_final':
            lines.append(f"   → t* = {_fmt(rec['t'])}   (φ(t*) = {_fmt(rec['phi_t'])})")
        elif kind == 'backtrack':
            for p, fv in rec['probes']:
                rel = '≥' if fv >= rec['f_x'] else '<'
                lines.append(f"   backtracking: t = {_fmt(p)} → f(x + t·d) = {_fmt(fv)} {rel} "
                             f"f(x) = {_fmt(rec['f_x'])} → t ← t/2")
            lines.append(f"   шаг t = {_fmt(rec['t'])}: f(x+t·d) = {_fmt(rec['f_t'])} ≤ "
                         f"f(x) = {_fmt(rec['f_x'])} → шаг принят")
        elif kind == 'step':
            lines.append(f"   t = {_fmt(rec['t'])}  →  Δx = {_fmt_vec(rec['step'])}")
        elif kind == 'update':
            lines.append(f"   x({rec['k']}+1) = {_fmt_vec(rec['x'])}  f = {_fmt(rec['f'])}  "
                         f"|grad| = {_fmt(rec['g_norm'])}")
        elif kind == 'linerr':
            lines.append('   ПРЕРВАНО: LinAlgError при решении H·d = -g')
        elif kind == 'nan':
            lines.append('   ПРЕРВАНО: NaN/Inf в x')
        elif kind == 'stop':
            lines.append('')
            if rec['reason'] == 'converged':
                lines.append(f"СТОП: |grad| = {_fmt(rec['g_norm'])} ≤ eps1 → СХОДИМОСТЬ к "
                             f"{_fmt_vec(rec['x'])};  итераций: {rec['k']}")
            elif rec['reason'] == 'm_limit':
                lines.append(f"СТОП: достигнут лимит m = {rec['m']} (итераций: {rec['k']})")
            else:
                lines.append(f"СТОП: аварийно (итераций: {rec['k']})")
    return '\n'.join(lines)


def main():
    load_dotenv()
    args = build_parser().parse_args()
    check_args(args)

    func = get_function(args.function)
    method_cls = get_method_class(args.method)
    method = method_cls()

    if args.trace:
        if args.random:
            rng = np.random.default_rng(args.seed)
            starts = rng.uniform(func.lo, func.hi, size=(args.points, 2))
        else:
            starts = [np.asarray(args.start, float)]
        multi = len(starts) > 1
        for i, x0 in enumerate(starts):
            if multi:
                print(f"\n{'#' * 78}\n### Старт #{i + 1}/{len(starts)}: "
                      f"{_fmt_vec(x0)}\n{'#' * 78}", flush=True)
            trace = []
            method.nf(func.f, func.grad, func.hess, x0,
                      m=1000 if func.name == 'rosenbrock' else 500, trace=trace)
            print(format_trace(trace, func.name, args.method))
        return

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