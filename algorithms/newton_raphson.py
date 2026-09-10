import numpy as np

from .secant import sect


def nf(f, grad, hess, x0, eps1=1e-6, eps2=1e-6, m=500, det_tol=1e-12,
       d_max=1e6, b_max=100.0, *, logger):
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    k = 0
    logger.log('start', {'x': x, 'eps1': eps1, 'eps2': eps2, 'm': m})
    g = np.asarray(grad(x), float)
    g_norm = float(np.linalg.norm(g))
    while g_norm > eps1 and k < m:
        f_x = f(x)
        logger.log('iter', {'k': k, 'x': x, 'f': f_x, 'g': g, 'g_norm': g_norm})
        H = hess(x)
        detH = float(np.linalg.det(H))
        pd_ok = H[0, 0] > 0 and detH > det_tol
        logger.log('hess', {'k': k, 'H': H, 'det': detH,
                            'h00': float(H[0, 0]), 'pd_ok': pd_ok})
        d = None
        if pd_ok:
            try:
                d = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                d = None
        use_newton = (d is not None and np.all(np.isfinite(d))
                      and np.linalg.norm(d) <= d_max and d @ g < 0)
        d_final = d if use_newton else -g
        if use_newton:
            reason = 'PD выполнен, d·g < 0 и |d| ≤ 1e6 → ньютоновский шаг'
        elif pd_ok:
            reason = 'PD выполнен, но проверки не прошли (|d| ≤ 1e6, d·g < 0) → сброс на -g'
        else:
            reason = 'H00 ≤ 0 или |det| ≤ 1e-12 → ньютоновский шаг не строится'
        logger.log('direction', {'k': k, 'd': d_final,
                                 'd_norm': float(np.linalg.norm(d_final)),
                                 'd_dot_g': float(d_final @ g),
                                 'origin': 'newton' if use_newton else 'fallback(-g)',
                                 'reason': reason})
        dphi = lambda t_val: float(grad(x + t_val * d_final) @ d_final)
        a, b = 0.0, 1.0
        phi_b = dphi(b)
        logger.log('bracket_start', {'k': k, 'b': b, 'phi_b': phi_b})
        while phi_b < 0 and b < b_max:
            b *= 2.0
            phi_b = dphi(b)
            logger.log('bracket', {'k': k, 'b': b, 'phi_b': phi_b})
        t = sect(dphi, a, b, eps2, logger=logger)
        phi_t = dphi(t)
        logger.log('sect_final', {'k': k, 't': t, 'phi_t': phi_t})
        f_t = f(x + t * d_final)
        probes = []
        while f_t >= f_x and t > eps2:
            t /= 2.0
            f_t = f(x + t * d_final)
            probes.append((float(t), float(f_t)))
        logger.log('backtrack', {'k': k, 't': t, 'f_x': f_x,
                                 'f_t': f_t, 'probes': probes})
        x = x + t * d_final
        traj.append(x.copy())
        g = np.asarray(grad(x), float)
        g_norm = float(np.linalg.norm(g))
        logger.log('update', {'k': k, 'x': x, 'f': f_t, 'g_norm': g_norm})
        k += 1
    conv = g_norm <= eps1
    logger.log('stop', {'reason': 'converged' if conv else 'm_limit',
                        'k': k, 'x': x, 'g_norm': g_norm, 'm': m})
    return np.asarray(traj)