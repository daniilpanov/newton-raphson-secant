import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path('/storage/emulated/0/Download/mathcad-nf-sect')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

x, y = sp.symbols('x y')

F_QUAD       = (x - 1)**2 + (y + 2)**2
F_HIMMELBLAU = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
F_ROSENBROCK = (1 - x)**2 + 100*(y - x**2)**2


def make_derivatives(expr, syms):
    raw_f = sp.lambdify(syms, expr, 'numpy')
    g = [sp.diff(expr, v) for v in syms]
    H = sp.Matrix([[sp.diff(e, v) for v in syms] for e in g])
    raw_g = sp.lambdify(syms, g, 'numpy')
    raw_H = sp.lambdify(syms, H.tolist(), 'numpy')

    def f(*args):
        if len(args) == 1:
            args = tuple(np.asarray(args[0], float).ravel())
        return raw_f(*args)

    def grad(p):
        return np.asarray(raw_g(*np.asarray(p, float).ravel()), float).ravel()

    def hess(p):
        n = len(syms)
        return np.asarray(raw_H(*np.asarray(p, float).ravel()), float).reshape(n, n)

    return f, grad, hess


def SECT(g, a, b, eps, maxit=500):
    it = 0
    while abs(g(b)) >= eps and it < maxit:
        den = g(b) - g(a)
        if den == 0:
            break
        t = b - g(b) * (b - a) / den
        if g(t) * g(a) > 0:
            a = t
        else:
            b = t
        it += 1
    return b


def NF(f, grad, hess, x0, eps1=1e-6, eps2=1e-6, M=500,
       det_tol=1e-12, d_max=1e6, b_max=100.0):
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    step_types = []
    k = 0
    while np.linalg.norm(grad(x)) > eps1:
        if k >= M:
            break
        g = grad(x)
        H = hess(x)
        d = None
        used_newton = False
        if H[0, 0] > 0 and np.linalg.det(H) > det_tol:
            try:
                d = -np.linalg.solve(H, g)
                used_newton = True
            except np.linalg.LinAlgError:
                d = None
        if d is None or not np.all(np.isfinite(d)) or np.linalg.norm(d) > d_max or d @ g >= 0:
            d = -g
            used_newton = False
        step_types.append('N' if used_newton else 'G')
        dphi = lambda t_val: grad(x + t_val * d) @ d
        a, b = 0.0, 1.0
        while dphi(b) < 0 and b < b_max:
            b *= 2.0
        t = SECT(dphi, a, b, eps2)
        while f(x + t * d) >= f(x) and t > eps2:
            t /= 2.0
        x = x + t * d
        traj.append(x.copy())
        k += 1
    return np.asarray(traj), step_types


HIMM_MIN = [
    (3, 2),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]

HIMM_LABELS = [
    'A (3, 2)',
    'B (-2.805, 3.131)',
    'C (-3.779, -3.283)',
    'D (3.584, -1.848)',
]


def find_closest_minima(xf):
    dists = [np.hypot(xf[0]-m[0], xf[1]-m[1]) for m in HIMM_MIN]
    idx = int(np.argmin(dists))
    return idx, HIMM_MIN[idx], dists[idx]


def print_trajectory(X, step_types, f_func, grad_func, label=''):
    print(f"\n{'='*70}")
    print(f"  Trajectory: {label}")
    print(f"{'='*70}")
    print(f"{'k':>3s}  {'x':>20s}  {'y':>20s}  {'f(x,y)':>12s}  {'|grad|':>12s}  {'step':>4s}")
    print(f"{'-'*70}")
    for k in range(len(X)):
        px, py = X[k]
        fv = float(f_func(px, py))
        gv = float(np.linalg.norm(grad_func(X[k])))
        st = 'init' if k == 0 else step_types[k-1]
        print(f"{k:3d}  {px:20.10f}  {py:20.10f}  {fv:12.6e}  {gv:12.6e}  {st:>4s}")
    print(f"{'-'*70}")
    print(f"  Total iterations: {len(X)-1}")


def test_sect_basic():
    t = SECT(lambda t: 2*(t - 1), 0, 2, 1e-10)
    assert abs(t - 1) < 1e-6, f"SECT failed: got {t}"
    print("  [PASS] SECT basic")


def test_quadratic():
    f, gr, hs = make_derivatives(F_QUAD, (x, y))
    X, st = NF(f, gr, hs, (5, -7))
    xf = X[-1]
    ref = (1, -2)
    err = np.hypot(xf[0]-ref[0], xf[1]-ref[1])
    print(f"  Found:   ({xf[0]:.8f}, {xf[1]:.8f})")
    print(f"  Reference: ({ref[0]}, {ref[1]})")
    print(f"  Error:     {err:.2e}")
    print_trajectory(X, st, f, gr, 'Quadratic from (5,-7)')
    assert np.allclose(xf, ref, atol=1e-5), f"Failed: {xf}"
    assert len(X) <= 3, f"Took {len(X)} steps"
    print("  [PASS] Quadratic")


def test_himmelblau_single(x0, expected_idx):
    f, gr, hs = make_derivatives(F_HIMMELBLAU, (x, y))
    X, st = NF(f, gr, hs, x0)
    xf = X[-1]
    fv = float(f(xf))
    closest_idx, closest_min, dist = find_closest_minima(xf)
    ref = HIMM_MIN[expected_idx]
    ref_err = np.hypot(xf[0]-ref[0], xf[1]-ref[1])

    print(f"\n  Start:    ({x0[0]}, {x0[1]})")
    print(f"  Found:    ({xf[0]:.8f}, {xf[1]:.8f})")
    print(f"  Reference: ({ref[0]}, {ref[1]})  [{HIMM_LABELS[expected_idx]}]")
    print(f"  Error:    {ref_err:.2e}")
    print(f"  f(x*):    {fv:.6e}")
    print(f"  Closest min: #{closest_idx} [{HIMM_LABELS[closest_idx]}], dist={dist:.2e}")

    print_trajectory(X, st, f, gr, f'Himmelblau from {x0}')

    fs = [f(p) for p in X]
    mono = all(fs[i+1] <= fs[i] + 1e-9 for i in range(len(fs)-1))
    print(f"  Monotone f: {'YES' if mono else 'NO'}")

    assert fv < 1e-6, f"f too large: {fv}"
    assert closest_idx == expected_idx, \
        f"Converged to wrong minimum: got #{closest_idx}, expected #{expected_idx}"
    assert mono, "f is not monotonous"
    print(f"  [PASS] Himmelblau from {x0} -> min #{expected_idx}")


def test_himmelblau():
    print("\n--- Himmelblau: all 4 minima ---")
    starts = [
        ((3, 2),    0),   # near A (3, 2)
        ((1, -2),   3),   # -> D (3.584, -1.848)
        ((-2, 1),   1),   # -> B (-2.805, 3.131)
        ((-1, -1),  2),   # -> C (-3.779, -3.283)
    ]
    for x0, eidx in starts:
        test_himmelblau_single(x0, eidx)


def test_rosenbrock():
    f, gr, hs = make_derivatives(F_ROSENBROCK, (x, y))
    ref = (1, 1)
    print("\n--- Rosenbrock from multiple starts ---")
    for x0 in [(-2, -2), (-2, 3), (2, -2)]:
        X, st = NF(f, gr, hs, x0, M=1000)
        xf = X[-1]
        err = np.hypot(xf[0]-ref[0], xf[1]-ref[1])
        print(f"\n  Start:    ({x0[0]}, {x0[1]})")
        print(f"  Found:    ({xf[0]:.8f}, {xf[1]:.8f})")
        print(f"  Reference: ({ref[0]}, {ref[1]})")
        print(f"  Error:    {err:.2e}")
        print(f"  f(x*):    {float(f(xf)):.6e}")
        print_trajectory(X, st, f, gr, f'Rosenbrock from {x0}')
        assert np.allclose(xf, ref, atol=1e-3), f"Failed from {x0}: {xf}"
    print("  [PASS] Rosenbrock")


def run_tests():
    print("="*70)
    print("  TESTS: NF + SECT")
    print("="*70)
    test_sect_basic()
    test_quadratic()
    test_himmelblau()
    test_rosenbrock()
    print("\n" + "="*70)
    print("  ALL TESTS PASSED")
    print("="*70)


def plot3d(f, trajs, lo=-6, hi=6, zmax=None, title='', filename='plot3d.png',
           minima=None):
    xs = np.linspace(lo, hi, 200)
    Xg, Yg = np.meshgrid(xs, xs)
    Zg = f(Xg, Yg)
    if zmax is not None:
        Zg = np.minimum(Zg, zmax)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.55)
    if minima is not None:
        mx = np.asarray([m[0] for m in minima], float)
        my = np.asarray([m[1] for m in minima], float)
        ax.scatter(mx, my, f(mx, my), color='violet', s=110, marker='*',
                   edgecolor='black', depthshade=False, label='real minima')
    for X in trajs:
        zs = f(X[:, 0], X[:, 1])
        ax.plot(X[:, 0], X[:, 1], zs, 'r.-', lw=2, ms=5)
        ax.scatter(*X[-1], f(X[-1]), color='lime', s=90, edgecolor='k')
    ax.set_title(title)
    if minima is not None:
        ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: {SAVE_DIR / filename}")


def plot_contour(f, trajs, lo=-6, hi=6, title='', filename='contour.png',
                 minima=None):
    xs = np.linspace(lo, hi, 300)
    Xg, Yg = np.meshgrid(xs, xs)
    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, f(Xg, Yg), levels=60, cmap='viridis', alpha=0.7)
    if minima is not None:
        plt.plot([m[0] for m in minima], [m[1] for m in minima], '*',
                 color='violet', ms=14, mec='black', label='real minima')
    for X in trajs:
        plt.plot(X[:, 0], X[:, 1], 'r.-', lw=1.5, ms=5)
        plt.plot(*X[-1], 'go', ms=10, mec='k')
    plt.axis('equal')
    plt.colorbar()
    plt.title(title)
    if minima is not None:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: {SAVE_DIR / filename}")


if __name__ == '__main__':
    run_tests()

    rng = np.random.default_rng(42)

    print("\n--- Himmelblau: 4 random starts ---")
    f, gr, hs = make_derivatives(F_HIMMELBLAU, (x, y))
    starts_h = rng.uniform(-6, 6, size=(4, 2))
    trajs_h = []
    finals_h = []
    for x0 in starts_h:
        X, st = NF(f, gr, hs, x0)
        trajs_h.append(X)
        idx, cmin, dist = find_closest_minima(X[-1])
        finals_h.append((X[-1], idx, dist, len(X)-1))
        print(f"  from ({x0[0]:.3f}, {x0[1]:.3f}) -> "
              f"({X[-1][0]:.6f}, {X[-1][1]:.6f}) "
              f"min#{idx} [{HIMM_LABELS[idx]}], error={dist:.2e}, iters={len(X)-1}")
    plot3d(f, trajs_h, -6, 6, title='Himmelblau: NF from 4 random starts',
           filename='himmelblau_4runs_3d.png', minima=HIMM_MIN)
    plot_contour(f, trajs_h, -6, 6,
                 title='Himmelblau contour: NF from 4 random starts',
                 filename='himmelblau_4runs_contour.png', minima=HIMM_MIN)
    for i, fin in enumerate(finals_h):
        print(f"  Run {i}: final={np.round(fin[0],6)}, min#{fin[1]}, "
              f"error={fin[2]:.2e}, iters={fin[3]}")

    print("\n--- Rosenbrock: 4 random starts ---")
    f, gr, hs = make_derivatives(F_ROSENBROCK, (x, y))
    starts_r = rng.uniform(-4, 4, size=(4, 2))
    trajs_r = []
    ROS_MIN = [(1.0, 1.0)]
    for x0 in starts_r:
        X, st = NF(f, gr, hs, x0, M=1000)
        trajs_r.append(X)
        err = np.hypot(X[-1][0]-1, X[-1][1]-1)
        print(f"  from ({x0[0]:.3f}, {x0[1]:.3f}) -> "
              f"({X[-1][0]:.6f}, {X[-1][1]:.6f}), error={err:.2e}, iters={len(X)-1}")
    plot3d(f, trajs_r, -2.5, 2.5, zmax=300,
           title='Rosenbrock: NF from 4 random starts',
           filename='rosenbrock_4runs_3d.png', minima=ROS_MIN)
    plot_contour(f, trajs_r, -2.5, 2.5,
                 title='Rosenbrock contour: NF from 4 random starts',
                 filename='rosenbrock_4runs_contour.png', minima=ROS_MIN)

    print("\nDone! All files saved to", SAVE_DIR)
