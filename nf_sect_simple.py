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


def SECT(phi_prime, L, R, eps):
    z = L
    while abs(phi_prime(z)) > eps:
        den = phi_prime(R) - phi_prime(L)
        if den == 0:
            print("  SECT: denominator = 0, stopping")
            break
        newz = L - phi_prime(L) * (R - L) / den
        if phi_prime(newz) * phi_prime(L) > 0:
            L = newz
        else:
            R = newz
        z = newz
    return z


def NF(f, grad, hess, x0, eps=1e-6, eps2=1e-6, M=500):
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    k = 0
    while np.linalg.norm(grad(x)) > eps:
        if k >= M:
            print(f"  NF: max iterations M={M} reached")
            break
        g = grad(x)
        H = hess(x)
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print(f"  NF: degenerate hessian at k={k}, x={x}")
            break
        dphi = lambda t_val: grad(x + t_val * d) @ d
        t = SECT(dphi, 0, 1, eps2)
        x = x + t * d
        if not np.all(np.isfinite(x)):
            print(f"  NF: x became non-finite at k={k}, x={x}")
            break
        traj.append(x.copy())
        k += 1
    return np.asarray(traj)


HIMM_MIN = [
    (3, 2),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]


def print_trajectory(X, f_func, grad_func, label=''):
    print(f"\n{'='*70}")
    print(f"  Trajectory: {label}")
    print(f"{'='*70}")
    print(f"{'k':>3s}  {'x':>20s}  {'y':>20s}  {'f(x,y)':>12s}  {'|grad|':>12s}")
    print(f"{'-'*70}")
    for k in range(len(X)):
        px, py = X[k]
        fv = float(f_func(px, py))
        gv = float(np.linalg.norm(grad_func(X[k])))
        print(f"{k:3d}  {px:20.10f}  {py:20.10f}  {fv:12.6e}  {gv:12.6e}")
    print(f"{'-'*70}")
    print(f"  Total iterations: {len(X)-1}")


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


def test_sect():
    print("\n--- SECT: 2*(t-1)=0 on [0, 2] ---")
    t = SECT(lambda t: 2*(t - 1), 0, 2, 1e-10)
    print(f"  Found: {t}, expected: 1.0, error: {abs(t-1):.2e}")
    assert abs(t - 1) < 1e-6
    print("  [PASS]")


if __name__ == '__main__':
    print("="*70)
    print("  PURE Newton-Raphson + Secant (no modifications)")
    print("="*70)

    test_sect()

    rng = np.random.default_rng()   # без seed — случайные стартовые точки каждый запуск

    # --- QUADRATIC: should converge in 1 step ---
    print("\n--- Quadratic: NF from (5, -7) ---")
    f, gr, hs = make_derivatives(F_QUAD, (x, y))
    X = NF(f, gr, hs, (5, -7))
    xf = X[-1]
    err = np.hypot(xf[0]-1, xf[1]+2)
    print(f"  Found: ({xf[0]:.8f}, {xf[1]:.8f})")
    print(f"  Reference: (1, -2), error: {err:.2e}")
    print_trajectory(X, f, gr, 'Quadratic from (5,-7)')

    # --- HIMMELBLAU: expect divergence/oscillation ---
    print("\n--- Himmelblau: NF from 4 random starts ---")
    f, gr, hs = make_derivatives(F_HIMMELBLAU, (x, y))
    starts_h = rng.uniform(-6, 6, size=(4, 2))
    trajs_h = []
    finals_h = []
    for x0 in starts_h:
        print(f"\n  === Start ({x0[0]:.3f}, {x0[1]:.3f}) ===")
        try:
            X = NF(f, gr, hs, x0)
            trajs_h.append(X)
            xf = X[-1]
            dists = [np.hypot(xf[0]-m[0], xf[1]-m[1]) for m in HIMM_MIN]
            closest = int(np.argmin(dists))
            fv = float(f(xf))
            finals_h.append((xf, closest, dists[closest], len(X)-1))
            print(f"  Final: ({xf[0]:.6f}, {xf[1]:.6f}), f={fv:.6e}")
            print(f"  Closest min: #{closest}, dist={dists[closest]:.2e}")
            print_trajectory(X, f, gr, f'Himmelblau from {tuple(np.round(x0,3))}')
            if fv > 1e-2:
                print(f"  >>> EXPECTED: method did NOT converge (or converged poorly)")
            else:
                print(f"  >>> method happened to converge to min #{closest}")
        except np.linalg.LinAlgError as e:
            finals_h.append((None, None, None, np.nan))
            print(f"  CRASHED: {e} — degenerate hessian")
            print(f"  >>> EXPECTED: this is normal for pure Newton on Himmelblau")
    if trajs_h:
        plot3d(f, trajs_h, -8, 8, title='Himmelblau: pure NF (4 random starts)',
               filename='simple_himmelblau_3d.png', minima=HIMM_MIN)
        plot_contour(f, trajs_h, -8, 8,
                     title='Himmelblau contour: pure NF',
                     filename='simple_himmelblau_contour.png', minima=HIMM_MIN)
    for i, fin in enumerate(finals_h):
        if fin[0] is None:
            print(f"  Run {i}: CRASHED")
        else:
            print(f"  Run {i}: final={np.round(fin[0],6)}, min#{fin[1]}, "
                  f"error={fin[2]:.2e}, iters={fin[3]}")

    # --- ROSENBROCK: expect slow convergence or divergence ---
    print("\n--- Rosenbrock: NF from 4 random starts ---")
    f, gr, hs = make_derivatives(F_ROSENBROCK, (x, y))
    starts_r = rng.uniform(-4, 4, size=(4, 2))
    ROS_MIN = [(1.0, 1.0)]
    trajs_r = []
    finals_r = []
    for x0 in starts_r:
        print(f"\n  === Start ({x0[0]:.3f}, {x0[1]:.3f}) ===")
        try:
            X = NF(f, gr, hs, x0, M=1000)
            trajs_r.append(X)
            xf = X[-1]
            err = np.hypot(xf[0]-1, xf[1]-1)
            finals_r.append((xf, err, len(X)-1))
            print(f"  Final: ({xf[0]:.6f}, {xf[1]:.6f}), error: {err:.2e}")
            print_trajectory(X, f, gr, f'Rosenbrock from {tuple(np.round(x0,3))}')
            if err > 0.01:
                print(f"  >>> method did NOT converge well")
            else:
                print(f"  >>> method converged (lucky)")
        except np.linalg.LinAlgError as e:
            finals_r.append((None, np.nan, np.nan))
            print(f"  CRASHED: {e}")
            print(f"  >>> EXPECTED: degenerate hessian on Rosenbrock valley")
    if trajs_r:
        plot3d(f, trajs_r, -3, 3, zmax=500,
               title='Rosenbrock: pure NF (4 random starts)',
               filename='simple_rosenbrock_3d.png', minima=ROS_MIN)
        plot_contour(f, trajs_r, -3, 3,
                     title='Rosenbrock contour: pure NF',
                     filename='simple_rosenbrock_contour.png', minima=ROS_MIN)
    for i, fin in enumerate(finals_r):
        if fin[0] is None:
            print(f"  Run {i}: CRASHED")
        else:
            print(f"  Run {i}: final={np.round(fin[0],6)}, "
                  f"error={fin[1]:.2e}, iters={fin[2]}")

    print("\n" + "="*70)
    print("  SUMMARY: pure Newton-Raphson without modifications")
    print("  - Quadratic: converges in 1 step (exact Newton)")
    print("  - Himmelblau: diverges / oscillates (expected)")
    print("  - Rosenbrock: slow or no convergence (expected)")
    print("="*70)
    print(f"\nAll PNG saved to {SAVE_DIR}")
