from .interactive import InteractiveViewer
from .photo import PhotoVisualizer
from .protocol import Visualizer

VISUALIZERS = {
    'photo': PhotoVisualizer,
    'interactive': InteractiveViewer,
}


def get_visualizer(name: str, save_dir=None) -> Visualizer:
    if name not in VISUALIZERS:
        raise KeyError(f"Unknown visualizer {name!r}; choose from {sorted(VISUALIZERS)}")
    if name == 'photo':
        return VISUALIZERS[name](save_dir)
    return VISUALIZERS[name]()


__all__ = ['Visualizer', 'PhotoVisualizer', 'InteractiveViewer', 'VISUALIZERS', 'get_visualizer']