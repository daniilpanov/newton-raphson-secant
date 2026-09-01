import numpy as np


def build_surface(func, lo, hi, n=200):
    xs = np.linspace(lo, hi, n)
    Xg, Yg = np.meshgrid(xs, xs)
    Zg = func.f(Xg, Yg)
    if func.zmax is not None:
        Zg = np.minimum(Zg, func.zmax)
    return Xg, Yg, Zg


def draw_plots(ax3d, ax2d, func, trajectories, title):
    Xg, Yg, Zg = build_surface(func, func.lo, func.hi)
    ax3d.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.55)
    if getattr(func, 'minima', None):
        mx = np.asarray([m[0] for m in func.minima], float)
        my = np.asarray([m[1] for m in func.minima], float)
        ax3d.scatter(mx, my, func.f(mx, my), color='violet', s=110, marker='*',
                     edgecolor='black', depthshade=False, label='real minima')
        ax2d.plot(mx, my, '*', color='violet', ms=14, mec='black',
                  label='real minima')
    for X in trajectories:
        zs = func.f(X[:, 0], X[:, 1])
        ax3d.plot(X[:, 0], X[:, 1], zs, 'r.-', lw=2, ms=5)
        ax3d.scatter(*X[-1], zs[-1], color='lime', s=90, edgecolor='k')
        ax2d.plot(X[:, 0], X[:, 1], 'r.-', lw=1.5, ms=5)
        ax2d.plot(*X[-1], 'go', ms=10, mec='k')
    ax2d.contourf(Xg, Yg, func.f(Xg, Yg)
                  if func.zmax is None
                  else np.minimum(func.f(Xg, Yg), func.zmax),
                  levels=60, cmap='viridis', alpha=0.7)
    ax2d.axis('equal')
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('f')
    ax2d.set_xlabel('x'); ax2d.set_ylabel('y')
    if title:
        ax3d.set_title(title)
    if getattr(func, 'minima', None):
        ax2d.legend(loc='upper right')
        ax3d.legend()