from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ._plots import draw_plots


class PhotoVisualizer:
    """Renders plots and saves them as PNG files into a directory."""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)

    def view(self, func, trajectories, *, title='', filename=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{func.name}"
        base = self.save_dir / filename

        fig = plt.figure(figsize=(16, 7))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        draw_plots(ax3d, ax2d, func, trajectories, title)
        fig.tight_layout()
        fig.savefig(base.with_suffix('.png'), dpi=150)
        plt.close(fig)
        print(f"Saved: {base.with_suffix('.png')}")