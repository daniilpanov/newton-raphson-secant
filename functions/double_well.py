import sympy as sp

from ._deriv import make_derivatives
from .protocol import BaseFunction

x = sp.Symbol('x')

EXPR = x**4 - 4*x**2 + sp.Rational(3, 2)*x

f, grad, hess = make_derivatives(EXPR, (x,))

FUNCTION = BaseFunction(
    name='double_well',
    f=f,
    grad=grad,
    hess=hess,
    lo=-4.0,
    hi=4.0,
    zmax=None,
    minima=[],
    dim=1,
)
