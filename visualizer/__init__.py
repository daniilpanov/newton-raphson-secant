from .oned import OneDVisualizer
from .photo import PhotoVisualizer
from .protocol import Visualizer

VISUALIZERS = {
    'photo': PhotoVisualizer,
    'oned': OneDVisualizer,
}


def get_visualizer(name: str, save_dir=None) -> Visualizer:
    if name not in VISUALIZERS:
        raise KeyError(f"Unknown visualizer {name!r}; choose from {sorted(VISUALIZERS)}")
    return VISUALIZERS[name](save_dir)


__all__ = ['Visualizer', 'PhotoVisualizer', 'OneDVisualizer',
           'VISUALIZERS', 'get_visualizer']