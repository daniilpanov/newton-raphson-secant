from .himmelblau import FUNCTION as HIMMELBLAU
from .protocol import TestFunction
from .quadratic import FUNCTION as QUADRATIC
from .rosenbrock import FUNCTION as ROSENBROCK

FUNCTIONS = {
    'quadratic': QUADRATIC,
    'himmelblau': HIMMELBLAU,
    'rosenbrock': ROSENBROCK,
}


def get_function(name: str) -> TestFunction:
    if name not in FUNCTIONS:
        raise KeyError(f"Unknown test function {name!r}; choose from {sorted(FUNCTIONS)}")
    return FUNCTIONS[name]


__all__ = [
    'TestFunction', 'QUADRATIC', 'HIMMELBLAU', 'ROSENBROCK',
    'FUNCTIONS', 'get_function',
]