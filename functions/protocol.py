from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple


class TestFunction(Protocol):
    name: str
    f: Callable
    grad: Callable
    hess: Callable
    lo: float
    hi: float
    zmax: Optional[float]
    minima: List[Tuple[float, ...]]
    dim: int


@dataclass
class BaseFunction:
    name: str
    f: Callable
    grad: Callable
    hess: Callable
    lo: float = -6.0
    hi: float = 6.0
    zmax: Optional[float] = None
    minima: List[Tuple[float, ...]] = field(default_factory=list)
    dim: int = 2