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

    def sect(self, g, a, b, eps, maxit=500, trace=None):
        it = 0
        while abs(g(b)) >= eps and it < maxit:
            den = g(b) - g(a)
            if den == 0:
                if trace is not None:
                    trace.append({'kind': 'sect_denzero',
                                  'a': a, 'b': b,
                                  'phi_a': g(a), 'phi_b': g(b)})
                break
            t = b - g(b) * (b - a) / den
            if trace is not None:
                trace.append({'kind': 'sect', 'a': a, 'b': b, 't': t,
                              'phi_a': g(a), 'phi_b': g(b)})
            if g(t) * g(a) > 0:
                a = t
            else:
                b = t
            it += 1
        return a if abs(g(a)) < abs(g(b)) else b

    def nf(self, f, grad, hess, x0, eps1=1e-6, eps2=1e-6, m=500, trace=None):
        x = np.asarray(x0, float).copy()
        traj = [x.copy()]
        k = 0
        if trace is not None:
            trace.append({'kind': 'start', 'x': x.copy(), 'eps1': eps1,
                          'eps2': eps2, 'm': m})
        while np.linalg.norm(grad(x)) > eps1:
            if k >= m:
                break
            g = grad(x)
            H = hess(x)
            detH = float(np.linalg.det(H))
            pd_ok = H[0, 0] > 0 and detH > self.det_tol
            if trace is not None:
                trace.append({'kind': 'iter', 'k': k, 'x': x.copy(),
                              'f': float(f(x)), 'g': g.copy(),
                              'g_norm': float(np.linalg.norm(g))})
                trace.append({'kind': 'hess', 'k': k, 'H': H.copy(),
                              'det': detH, 'h00': float(H[0, 0]),
                              'pd_ok': pd_ok})
            d = None
            if pd_ok:
                try:
                    d = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    d = None
            use_newton = (d is not None and np.all(np.isfinite(d))
                          and np.linalg.norm(d) <= self.d_max and d @ g < 0)
            d_final = d if use_newton else -g
            if trace is not None:
                if use_newton:
                    reason = 'PD выполнен, d·g < 0 и |d| ≤ 1e6 → ньютоновский шаг'
                elif pd_ok:
                    reason = 'PD выполнен, но проверки не прошли (|d| ≤ 1e6, d·g < 0) → сброс на -g'
                else:
                    reason = 'H00 ≤ 0 или |det| ≤ 1e-12 → ньютоновский шаг не строится'
                trace.append({'kind': 'direction', 'k': k, 'd': d_final.copy(),
                              'd_norm': float(np.linalg.norm(d_final)),
                              'd_dot_g': float(d_final @ g),
                              'origin': 'newton' if use_newton else 'fallback(-g)',
                              'reason': reason})
            dphi = lambda t_val: grad(x + t_val * d_final) @ d_final
            a, b = 0.0, 1.0
            if trace is not None:
                trace.append({'kind': 'bracket_start', 'k': k, 'b': b,
                              'phi_b': float(dphi(b))})
            while dphi(b) < 0 and b < self.b_max:
                b *= 2.0
                if trace is not None:
                    trace.append({'kind': 'bracket', 'k': k, 'b': b,
                                  'phi_b': float(dphi(b))})
            t = self.sect(dphi, a, b, eps2, trace=trace)
            if trace is not None:
                trace.append({'kind': 'sect_final', 'k': k, 't': t,
                              'phi_t': float(dphi(t))})
            probes = []
            while f(x + t * d_final) >= f(x) and t > eps2:
                t /= 2.0
                probes.append((float(t), float(f(x + t * d_final))))
            if trace is not None:
                trace.append({'kind': 'backtrack', 'k': k, 't': t,
                              'f_x': float(f(x)),
                              'f_t': float(f(x + t * d_final)),
                              'probes': probes})
            x = x + t * d_final
            traj.append(x.copy())
            if trace is not None:
                trace.append({'kind': 'update', 'k': k, 'x': x.copy(),
                              'f': float(f(x)),
                              'g_norm': float(np.linalg.norm(grad(x)))})
            k += 1
        if trace is not None:
            conv = float(np.linalg.norm(grad(x))) <= eps1
            trace.append({'kind': 'stop',
                          'reason': 'converged' if conv else 'm_limit',
                          'k': k, 'x': x.copy(),
                          'g_norm': float(np.linalg.norm(grad(x)))})
        return np.asarray(traj)