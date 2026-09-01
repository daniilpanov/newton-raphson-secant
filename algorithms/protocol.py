from typing import Callable, Protocol

import numpy as np


class Optimizer(Protocol):
    def sect(
        self,
        g: Callable[[float], float],
        a: float,
        b: float,
        eps: float,
        maxit: int = 500,
    ) -> float:
        ...

    def nf(
        self,
        f: Callable,
        grad: Callable,
        hess: Callable,
        x0,
        eps1: float = 1e-6,
        eps2: float = 1e-6,
        m: int = 500,
    ) -> np.ndarray:
        ...