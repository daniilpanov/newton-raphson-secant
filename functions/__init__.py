from .double_well import FUNCTION as DOUBLE_WELL
from .himmelblau import FUNCTION as HIMMELBLAU
from .protocol import BaseFunction, TestFunction
from .quadratic import FUNCTION as QUADRATIC
from .rosenbrock import FUNCTION as ROSENBROCK

FUNCTIONS = {
    'double_well': DOUBLE_WELL,
    'quadratic': QUADRATIC,
    'himmelblau': HIMMELBLAU,
    'rosenbrock': ROSENBROCK,
}


def get_function(name: str) -> TestFunction:
    if name not in FUNCTIONS:
        raise KeyError(f"Unknown test function {name!r}; choose from {sorted(FUNCTIONS)}")
    return FUNCTIONS[name]


__all__ = [
    'TestFunction', 'BaseFunction', 'DOUBLE_WELL', 'QUADRATIC', 'HIMMELBLAU',
    'ROSENBROCK', 'FUNCTIONS', 'get_function',
]