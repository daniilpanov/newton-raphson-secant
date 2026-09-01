from typing import List, Optional, Protocol

import numpy as np


class Visualizer(Protocol):
    def view(
        self,
        func,
        trajectories: List[np.ndarray],
        *,
        title: str = '',
        filename: Optional[str] = None,
    ) -> None:
        ...