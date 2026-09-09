import numpy as np

from algorithms import nf
from functions import get_function

FUNCTION = get_function('double_well')
STARTS = [np.asarray([x], float) for x in [-2.0, -0.5, 1.0, 3.0]]
GRAD_TOL = 1e-5
MAX_ITERS = 500


def _stationary_points():
    s5 = np.sqrt(5)
    return [np.asarray([-1.5]), np.asarray([(3 - s5) / 4]), np.asarray([(3 + s5) / 4])]


def _run_starts(save, label):
    trajectories = []

    for x0 in STARTS:
        X = nf(FUNCTION.f, FUNCTION.grad, FUNCTION.hess, x0, m=MAX_ITERS)

        xf = X[-1]
        iters = len(X) - 1
        grad_norm = float(np.linalg.norm(FUNCTION.grad(xf)))
        dist = min(np.linalg.norm(xf - np.asarray(m, float)) for m in _stationary_points())
        where = f'final=({", ".join(f"{v:.6f}" for v in xf)}) iters={iters}'

        assert grad_norm < GRAD_TOL, f'{FUNCTION.name} from {x0}: |grad|={grad_norm:.2e} {where}'
        assert dist < 1e-3, f'{FUNCTION.name} from {x0}: stationary dist {dist:.2e} {where}'

        save(
            FUNCTION,
            [X],
            f'{FUNCTION.name}_{label(x0)}',
            title=f'{FUNCTION.name} from ({", ".join(f"{v:.1f}" for v in x0)})',
        )

        trajectories.append(X)

    return trajectories


def test_double_well_converges(save_photo, label):
    _run_starts(save_photo, label)
