import numpy as np


def sect(g, a, b, eps, maxit=100, *, logger):
    """Protected secant method on [a, b] (caller guarantees a sign change).

    The plain secant iterate may leave the bracket on a strongly nonlinear
    function; then the midpoint of the bracket is used instead.
    """
    fa = g(a)
    fb = g(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(maxit):
        den = fb - fa
        if abs(den) < 1e-30:
            z = 0.5 * (a + b)
            logger.log('sect_denzero', {'a': a, 'b': b,
                                        'phi_a': fa, 'phi_b': fb})
        else:
            z = b - fb * (b - a) / den
            if not np.isfinite(z) or z <= a or z >= b:
                z = 0.5 * (a + b)
        logger.log('sect', {'a': a, 'b': b, 't': z,
                            'phi_a': fa, 'phi_b': fb})
        fz = g(z)
        if abs(fz) < eps:
            return z
        if fa * fz < 0:
            b, fb = z, fz
        else:
            a, fa = z, fz
    return 0.5 * (a + b)