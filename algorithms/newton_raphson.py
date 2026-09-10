import numpy as np

from .secant import sect


def nf(f, grad, hess, x0, eps1=1e-6, eps2=1e-6, m=500,
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
        evals, evecs = np.linalg.eigh(H)
        lam_min = float(evals[0])
        pd_ok = lam_min > 0
        logger.log('hess', {'k': k, 'H': H, 'eigenvalues': list(map(float, evals)),
                            'lam_min': lam_min, 'pd_ok': pd_ok})
        d = None
        if pd_ok:
            try:
                d = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                d = None
        use_newton = (d is not None and np.all(np.isfinite(d))
                      and np.linalg.norm(d) <= d_max and d @ g < 0)
        if use_newton:
            d_final = d
            origin = 'newton'
            reason = 'PD (λmin>0), d·g<0 и |d|≤1e6 → ньютоновский шаг'
        elif pd_ok:
            d_final = -g
            origin = 'fallback(-g)'
            reason = ('PD (λmin>0), но проверки не прошли (|d|≤1e6, d·g<0)'
                      ' → сброс на -g')
        else:
            v = evecs[:, 0]
            if float(g @ v) > 0:
                v = -v
            d_final = v
            origin = 'negative curvature'
            reason = ('λmin≤0 → гессиан не PD; направление отрицательной '
                      'кривизны (собственный вектор λmin), ориентировано g·v≤0,'
                      ' чтобы уйти от седла')
        d_final = np.asarray(d_final, float).ravel()
        logger.log('direction', {'k': k, 'd': d_final,
                                 'd_norm': float(np.linalg.norm(d_final)),
                                 'd_dot_g': float(d_final @ g),
                                 'origin': origin, 'reason': reason})
        dphi = lambda t_val: float(grad(x + t_val * d_final) @ d_final)
        a, b = 0.0, 1.0
        t_star = None
        guard = 0
        logger.log('bracket_start', {'k': k, 'b': b, 'phi_b': dphi(b)})
        while b < b_max and guard < 20:
            guard += 1
            while dphi(b) < 0 and b < b_max:
                b *= 2.0
                logger.log('bracket', {'k': k, 'b': b, 'phi_b': dphi(b)})
            if dphi(a) * dphi(b) <= 0.0:
                t = sect(dphi, a, b, eps2, logger=logger)
                curv = float(d_final @ hess(x + t * d_final) @ d_final)
                logger.log('curv', {'k': k, 't': t, 'curv': curv,
                                    'accepted': curv > 0})
                if curv > 0:      # a MINIMUM along d, not a max/saddle
                    t_star = t
                    break
            a = b
            b *= 2.0
        if t_star is None:
            t = 1.0
            while f(x + t * d_final) >= f_x and t > eps2:
                t /= 2.0
            t_star = t if f(x + t * d_final) < f_x else 0.0
            logger.log('curv_fallback', {'k': k, 't': t_star,
                                         'f_x': f_x,
                                         'f_t': float(f(x + t_star * d_final))})
        logger.log('sect_final', {'k': k, 't': t_star, 'phi_t': dphi(t_star)})
        probes = []
        while f(x + t_star * d_final) >= f_x and t_star > eps2:
            t_star /= 2.0
            probes.append((float(t_star), float(f(x + t_star * d_final))))
        logger.log('backtrack', {'k': k, 't': t_star, 'f_x': f_x,
                                 'f_t': float(f(x + t_star * d_final)),
                                 'probes': probes})
        x = x + t_star * d_final
        traj.append(x.copy())
        g = np.asarray(grad(x), float)
        g_norm = float(np.linalg.norm(g))
        logger.log('update', {'k': k, 'x': x, 'f': float(f(x)),
                              'g_norm': g_norm})
        k += 1
    conv = g_norm <= eps1
    logger.log('stop', {'reason': 'converged' if conv else 'm_limit',
                        'k': k, 'x': x, 'g_norm': g_norm, 'm': m})
    return np.asarray(traj)