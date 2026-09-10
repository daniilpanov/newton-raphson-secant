import numpy as np

from algorithms import nf
from functions import get_function
from logger import make_logger

FUNCTION = get_function('himmelblau')
STARTS = [np.asarray(pt, float) for pt in [(0, 0), (-1, -2), (-1, 2), (0.5, -0.5)]]
GRAD_TOL = 1e-5
MIN_TOL = 1e-4
MAX_ITERS = 500


def _run_starts(save, label, log):
    trajectories = []

    for x0 in STARTS:
        name = f'{FUNCTION.name}_{label(x0)}'
        with log(name) as logger:
            X = nf(
                FUNCTION.f,
                FUNCTION.grad,
                FUNCTION.hess,
                x0,
                m=MAX_ITERS,
                logger=logger,
            )
        xf = X[-1]
        iters = len(X) - 1
        grad_norm = float(np.linalg.norm(FUNCTION.grad(xf)))
        dist = min(np.linalg.norm(xf - np.asarray(m, float)) for m in FUNCTION.minima)
        where = f'final=({", ".join(f"{v:.4f}" for v in xf)}) iters={iters}'

        assert grad_norm < GRAD_TOL, f'{FUNCTION.name} from {x0}: |grad|={grad_norm:.2e} {where}'
        assert dist < MIN_TOL, f'{FUNCTION.name} from {x0}: min dist {dist:.2e} {where}'

        save(
            FUNCTION,
            [X],
            name,
            title=f'{FUNCTION.name} from ({", ".join(f"{v:.1f}" for v in x0)})',
        )

        trajectories.append(X)

    return trajectories


def test_himmelblau_converges(save_photo, label, log):
    _run_starts(save_photo, label, log)
