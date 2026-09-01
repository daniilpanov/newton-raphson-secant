import numpy as np


class SimpleNewton:
    """Canonical Newton-Raphson + secant, no modifications."""

    def sect(self, phi_prime, L, R, eps, maxit=500):
        z = L
        it = 0
        while abs(phi_prime(z)) > eps and it < maxit:
            den = phi_prime(R) - phi_prime(L)
            if den == 0:
                break
            newz = L - phi_prime(L) * (R - L) / den
            if phi_prime(newz) * phi_prime(L) > 0:
                L = newz
            else:
                R = newz
            z = newz
            it += 1
        return z

    def nf(self, f, grad, hess, x0, eps1=1e-6, eps2=1e-6, m=500):
        x = np.asarray(x0, float).copy()
        traj = [x.copy()]
        k = 0
        while np.linalg.norm(grad(x)) > eps1:
            if k >= m:
                break
            g = grad(x)
            H = hess(x)
            try:
                d = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            dphi = lambda t_val: grad(x + t_val * d) @ d
            t = self.sect(dphi, 0, 1, eps2)
            x = x + t * d
            if not np.all(np.isfinite(x)):
                break
            traj.append(x.copy())
            k += 1
        return np.asarray(traj)