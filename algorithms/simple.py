import numpy as np


class SimpleNewton:
    """Canonical Newton-Raphson + secant, no modifications."""

    def sect(self, phi_prime, L, R, eps, maxit=500, trace=None):
        z = L
        it = 0
        while abs(phi_prime(z)) > eps and it < maxit:
            den = phi_prime(R) - phi_prime(L)
            if den == 0:
                if trace is not None:
                    trace.append({'kind': 'sect_denzero', 'a': L, 'b': R,
                                  'phi_a': phi_prime(L), 'phi_b': phi_prime(R)})
                break
            newz = L - phi_prime(L) * (R - L) / den
            if trace is not None:
                trace.append({'kind': 'sect', 'a': L, 'b': R, 't': newz,
                              'phi_a': phi_prime(L), 'phi_b': phi_prime(R),
                              'phi_t': phi_prime(newz)})
            if phi_prime(newz) * phi_prime(L) > 0:
                L = newz
            else:
                R = newz
            z = newz
            it += 1
        return L if abs(phi_prime(L)) < abs(phi_prime(R)) else R

    def nf(self, f, grad, hess, x0, eps1=1e-6, eps2=1e-6, m=500, trace=None):
        x = np.asarray(x0, float).copy()
        traj = [x.copy()]
        k = 0
        if trace is not None:
            trace.append({'kind': 'start', 'x': x.copy(), 'eps1': eps1,
                          'eps2': eps2, 'm': m})
        aborted = False
        while np.linalg.norm(grad(x)) > eps1:
            if k >= m:
                break
            g = grad(x)
            H = hess(x)
            if trace is not None:
                trace.append({'kind': 'iter', 'k': k, 'x': x.copy(),
                              'f': float(f(x)), 'g': g.copy(),
                              'g_norm': float(np.linalg.norm(g))})
            try:
                d = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                if trace is not None:
                    trace.append({'kind': 'linerr', 'k': k})
                aborted = True
                break
            if trace is not None:
                trace.append({'kind': 'direction', 'k': k, 'd': d.copy(),
                              'd_norm': float(np.linalg.norm(d)),
                              'd_dot_g': float(d @ g), 'origin': 'newton',
                              'reason': 'd = -H⁻¹·g — канонический шаг Ньютона'})
            dphi = lambda t_val: grad(x + t_val * d) @ d
            t = self.sect(dphi, 0, 1, eps2, trace=trace)
            if trace is not None:
                trace.append({'kind': 'step', 'k': k, 't': t,
                              'step': (t * d).copy()})
            x = x + t * d
            if not np.all(np.isfinite(x)):
                if trace is not None:
                    trace.append({'kind': 'nan', 'k': k})
                aborted = True
                break
            traj.append(x.copy())
            if trace is not None:
                trace.append({'kind': 'update', 'k': k, 'x': x.copy(),
                              'f': float(f(x)),
                              'g_norm': float(np.linalg.norm(grad(x)))})
            k += 1
        if trace is not None:
            if float(np.linalg.norm(grad(x))) <= eps1:
                reason = 'converged'
            elif k >= m:
                reason = 'm_limit'
            else:
                reason = 'abort' if aborted else '???'
            trace.append({'kind': 'stop', 'reason': reason, 'k': k,
                          'x': x.copy(),
                          'g_norm': float(np.linalg.norm(grad(x)))})
        return np.asarray(traj)