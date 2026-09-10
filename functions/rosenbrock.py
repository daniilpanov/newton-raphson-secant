import sympy as sp

from ._deriv import make_derivatives
from .protocol import BaseFunction

x, y = sp.symbols('x y')

EXPR = (1 - x)**2 + 100*(y - x**2)**2

f, grad, hess = make_derivatives(EXPR, (x, y))

FUNCTION = BaseFunction(
    name='rosenbrock',
    f=f,
    grad=grad,
    hess=hess,
    lo=-4.0,
    hi=4.0,
    zmax=300.0,
    minima=[(1.0, 1.0)],
)