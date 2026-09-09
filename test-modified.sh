#!/usr/bin/env bash
# Automated test suite for the 'modified' Newton method.
# Runs main.py with photo saving (works headless in termux) for:
#   - Himmelblau from (0,0), (-1,-2), (-1,2), (0.5,-0.5)
#   - Rosenbrock from 4 random starts (seed 42)
#   - Quadratic from 2 random starts (seed 7)
# then verifies convergence in-memory and asserts quadratic does it in 1 step.
# Photos are saved to the dir from .env (VIZ_SAVE_DIR) — not overridden here.
# Usage: ./test-modified.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED_ROS=42
SEED_QUAD=7

if [[ ! -f main.py ]]; then
    echo "Error: main.py not found in $ROOT" >&2
    exit 1
fi
if [[ ! -f .env ]]; then
    echo "Warning: .env not found; photos will use the default save dir" >&2
fi

DEFAULT_DIR='/storage/emulated/0/Download/mathcad-nf-sect'
SAVE_DIR="$(grep -E '^VIZ_SAVE_DIR=' .env 2>/dev/null | head -n1 | cut -d= -f2- || true)"
SAVE_DIR="${SAVE_DIR:-$DEFAULT_DIR}"

echo "==> [1/4] Himmelblau (modified), 4 fixed starts, photos -> $SAVE_DIR"
for pt in "0,0" "-1,-2" "-1,2" "0.5,-0.5"; do
    sub="himmelblau_$(echo "$pt" | tr ',-' '__')"
    python main.py --function himmelblau \
        --start="$pt" --viz photo >/dev/null
    # main.py saves all single starts under one name; keep each run's photo
    mkdir -p "$SAVE_DIR/$sub"
    mv -f "$SAVE_DIR/himmelblau_modified_single.png" "$SAVE_DIR/$sub/"
done

echo "==> [2/4] Rosenbrock (modified), 4 random starts (seed $SEED_ROS)"
python main.py --function rosenbrock \
    --random --points 4 --seed "$SEED_ROS" \
    --viz photo >/dev/null

echo "==> [3/4] Quadratic (modified), 2 random starts (seed $SEED_QUAD)"
python main.py --function quadratic \
    --random --points 2 --seed "$SEED_QUAD" \
    --viz photo >/dev/null

echo "==> [4/4] Convergence checks (in-memory)"
python - "$SEED_ROS" "$SEED_QUAD" <<'PY'
import sys

import numpy as np

from algorithms import nf
from tests import get_function

seed_ros, seed_quad = int(sys.argv[1]), int(sys.argv[2])
failed = False


def check(name, cond, detail):
    global failed
    if not cond:
        failed = True
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")


def near_min(func, xf, tol):
    return min(np.hypot(xf[0] - m[0], xf[1] - m[1]) for m in func.minima) < tol


h = get_function('himmelblau')
for pt in [(0, 0), (-1, -2), (-1, 2), (0.5, -0.5)]:
    X = nf(h.f, h.grad, h.hess, np.asarray(pt, float))
    xf = X[-1]
    it = len(X) - 1
    ok = (np.linalg.norm(h.grad(xf)) < 1e-5
          and near_min(h, xf, 1e-4) and it < 500)
    check(f"himmelblau {pt}", ok,
          f"final=({xf[0]:+.4f},{xf[1]:+.4f}) f={float(h.f(*xf)):.1e} "
          f"iters={it}")

r = get_function('rosenbrock')
rpts = np.random.default_rng(seed_ros).uniform(r.lo, r.hi, size=(4, 2))
for pt in rpts:
    X = nf(r.f, r.grad, r.hess, np.asarray(pt, float), m=1000)
    xf = X[-1]
    it = len(X) - 1
    ok = (np.linalg.norm(r.grad(xf)) < 1e-5
          and near_min(r, xf, 1e-4) and it < 1000)
    check(f"rosenbrock ({pt[0]:+.3f},{pt[1]:+.3f})", ok,
          f"final=({xf[0]:+.6f},{xf[1]:+.6f}) f={float(r.f(*xf)):.1e} "
          f"iters={it}")

q = get_function('quadratic')
qpts = np.random.default_rng(seed_quad).uniform(q.lo, q.hi, size=(2, 2))
for pt in qpts:
    X = nf(q.f, q.grad, q.hess, np.asarray(pt, float))
    xf = X[-1]
    it = len(X) - 1
    ok = (it == 1 and near_min(q, xf, 1e-6))  # quadratic must converge in 1 step
    check(f"quadratic ({pt[0]:+.3f},{pt[1]:+.3f})", ok,
          f"final=({xf[0]:+.6f},{xf[1]:+.6f}) f={float(q.f(*xf)):.1e} "
          f"iters={it}")

print()
if failed:
    print("RESULT: FAIL")
    sys.exit(1)
print("RESULT: ALL PASS")
PY

echo
echo "Photos saved to (from .env): $SAVE_DIR"
ls -1 "$SAVE_DIR"/*.png "$SAVE_DIR"/*/*.png 2>/dev/null || true