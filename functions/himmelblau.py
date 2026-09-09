import sympy as sp

from ._deriv import make_derivatives
from .protocol import BaseFunction

x, y = sp.symbols('x y')

EXPR = (x**2 + y - 11)**2 + (x + y**2 - 7)**2

f, grad, hess = make_derivatives(EXPR, (x, y))

FUNCTION = BaseFunction(
    name='himmelblau',
    f=f,
    grad=grad,
    hess=hess,
    lo=-6.0,
    hi=6.0,
    zmax=None,
    minima=[
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ],
)