from .interactive import InteractiveViewer
from .oned import OneDVisualizer
from .photo import PhotoVisualizer
from .protocol import Visualizer

VISUALIZERS = {
    'photo': PhotoVisualizer,
    'interactive': InteractiveViewer,
    'oned': OneDVisualizer,
}


def get_visualizer(name: str, save_dir=None) -> Visualizer:
    if name not in VISUALIZERS:
        raise KeyError(f"Unknown visualizer {name!r}; choose from {sorted(VISUALIZERS)}")
    if name == 'interactive':
        return VISUALIZERS[name]()
    return VISUALIZERS[name](save_dir)


__all__ = ['Visualizer', 'PhotoVisualizer', 'InteractiveViewer', 'OneDVisualizer',
           'VISUALIZERS', 'get_visualizer']