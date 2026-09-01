from .modified import ModifiedNewton
from .protocol import Optimizer
from .simple import SimpleNewton

METHODS = {
    'simple': SimpleNewton,
    'modified': ModifiedNewton,
}


def get_method_class(name: str) -> type:
    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}; choose from {sorted(METHODS)}")
    return METHODS[name]


__all__ = ['Optimizer', 'SimpleNewton', 'ModifiedNewton', 'METHODS', 'get_method_class']