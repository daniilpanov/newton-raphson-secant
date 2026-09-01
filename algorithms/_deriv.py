import numpy as np
import sympy as sp


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