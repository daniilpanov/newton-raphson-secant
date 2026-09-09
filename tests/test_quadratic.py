import numpy as np

from algorithms import nf
from functions import get_function

FUNCTION = get_function('quadratic')
STARTS = np.random.default_rng(7).uniform(FUNCTION.lo, FUNCTION.hi, size=(2, FUNCTION.dim))
TOL = 1e-6


def _run_starts(save, label):
    trajectories = []

    for x0 in STARTS:
        X = nf(FUNCTION.f, FUNCTION.grad, FUNCTION.hess, np.asarray(x0, float))
        xf = X[-1]
        iters = len(X) - 1

        dist = min(
            np.linalg.norm(np.asarray(xf, float) - np.asarray(m, float))
            for m in FUNCTION.minima
        )

        assert iters == 1, f'{FUNCTION.name} from {x0}: expected 1 step, got {iters}'
        assert dist < TOL, f'{FUNCTION.name} from {x0}: min dist {dist:.2e} >= {TOL}'

        save(
            FUNCTION,
            [X],
            f'{FUNCTION.name}_{label(x0)}',
            title=f'{FUNCTION.name} from ({", ".join(f"{v:.1f}" for v in x0)})',
        )

        trajectories.append(X)

    return trajectories


def test_quadratic_converges_in_one_step(save_photo, label):
    _run_starts(save_photo, label)
