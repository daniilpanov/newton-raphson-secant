import numpy as np


class ModifiedNewton:
    """Newton-Raphson with safeguards:
    - descent direction check, cap on |d|, fallback to gradient descent
    - backtracking, step via secant
    """

    def __init__(self, det_tol=1e-12, d_max=1e6, b_max=100.0):
        self.det_tol = det_tol
        self.d_max = d_max
        self.b_max = b_max

    def sect(self, g, a, b, eps, maxit=500):
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

    def nf(self, f, grad, hess, x0, eps1=1e-6, eps2=1e-6, m=500):
        x = np.asarray(x0, float).copy()
        traj = [x.copy()]
        k = 0
        while np.linalg.norm(grad(x)) > eps1:
            if k >= m:
                break
            g = grad(x)
            H = hess(x)
            d = None
            if H[0, 0] > 0 and np.linalg.det(H) > self.det_tol:
                try:
                    d = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    d = None
            if d is None or not np.all(np.isfinite(d)) or np.linalg.norm(d) > self.d_max or d @ g >= 0:
                d = -g
            dphi = lambda t_val: grad(x + t_val * d) @ d
            a, b = 0.0, 1.0
            while dphi(b) < 0 and b < self.b_max:
                b *= 2.0
            t = self.sect(dphi, a, b, eps2)
            while f(x + t * d) >= f(x) and t > eps2:
                t /= 2.0
            x = x + t * d
            traj.append(x.copy())
            k += 1
        return np.asarray(traj)