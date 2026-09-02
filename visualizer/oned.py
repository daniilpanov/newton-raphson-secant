from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class OneDVisualizer:
    """Renders 1D function plots with optimization trajectories."""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)

    def view(self, func, trajectories, *, title='', filename=None, x_range=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{func.name}_1d"
        base = self.save_dir / filename

        lo = x_range[0] if x_range else func.lo
        hi = x_range[1] if x_range else func.hi

        fig, ax = plt.subplots(figsize=(12, 6))

        xs = np.linspace(lo, hi, 500)
        ys = np.array([func.f(np.array([x])) for x in xs])
        ax.plot(xs, ys, 'k-', lw=2, label=func.name)

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))
        for i, X in enumerate(trajectories):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            ys_traj = np.array([func.f(X[j]) for j in range(len(X))])
            ax.plot(X[:, 0], ys_traj, 'o-', color=colors[i], ms=5, lw=1.5,
                    label=f'trajectory {i}', zorder=4)
            ax.plot(X[-1, 0], ys_traj[-1], '*', color=colors[i], ms=15,
                    mec='k', zorder=5)

        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(base.with_suffix('.png'), dpi=150)
        plt.close(fig)
        print(f"Saved: {base.with_suffix('.png')}")
