import matplotlib.pyplot as plt

from ._plots import draw_plots


class InteractiveViewer:
    """Opens interactive matplotlib windows (zoom/rotate); no files saved."""

    def view(self, func, trajectories, *, title='', filename=None):
        fig = plt.figure(figsize=(16, 7))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        draw_plots(ax3d, ax2d, func, trajectories, title)
        fig.tight_layout()
        fig.show()
        print("Interactive window open - close it to continue.")
        plt.show(block=True)