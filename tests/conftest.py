from pathlib import Path

import pytest
import numpy as np

from logger import make_logger
from visualizer import get_visualizer

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'


@pytest.fixture
def output():
    """Directory where photo artifacts are saved."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture
def log(output):
    """Creates a file-backed logger for a run.

    Log file is written to output/ with the same base name as the photo,
    e.g. '<name>.png' / '<name>.ndjson'. The file is truncated first so
    repeated runs don't accumulate (handlers append).
    """
    def _make(name):
        path = output / f'{name}.ndjson'
        path.unlink(missing_ok=True)
        env = {
            'LOG_MODE': 'file',
            'LOG_FILE': str(path),
        }
        return make_logger(name, env=env)
    return _make


@pytest.fixture
def save_photo(output):
    """Fixture that saves optimization trajectories as PNG photos into output/."""
    def _save(func, trajectories, filename, title=''):
        viz_name = 'oned' if func.dim == 1 else 'photo'
        visualizer = get_visualizer(viz_name, save_dir=output)
        visualizer.view(func, trajectories, title=title, filename=filename)
    return _save


@pytest.fixture
def label():
    def _label(x0):
        return '_'.join(
            f'{v:.1f}'.replace('.', '_').replace('-', 'm')
            for v in np.asarray(x0, float).ravel()
        )

    return _label
