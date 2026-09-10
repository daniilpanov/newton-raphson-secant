import sympy as sp

from ._deriv import make_derivatives
from .protocol import BaseFunction

x, y = sp.symbols('x y')

EXPR = (x - 1)**2 + (y + 2)**2

f, grad, hess = make_derivatives(EXPR, (x, y))

FUNCTION = BaseFunction(
    name='quadratic',
    f=f,
    grad=grad,
    hess=hess,
    lo=-6.0,
    hi=6.0,
    zmax=None,
    minima=[(1.0, -2.0)],
)