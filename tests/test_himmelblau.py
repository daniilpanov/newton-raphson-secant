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


_SADDLE_START = np.asarray([0.0, 2.0], float)
MAX_ITERS_SADDLE = 4


def _run_start_0_2(save, log):
    x0 = _SADDLE_START
    name = f'{FUNCTION.name}_0_2'
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
    dist = float(np.linalg.norm(xf - np.asarray([3.0, 2.0], float)))
    where = f'final=({", ".join(f"{v:.6f}" for v in xf)}) iters={iters}'

    assert iters <= MAX_ITERS_SADDLE, (
        f'{FUNCTION.name} from (0,2): expected <= {MAX_ITERS_SADDLE} steps, '
        f'got {iters} {where}'
    )
    assert grad_norm < GRAD_TOL, (
        f'{FUNCTION.name} from (0,2): |grad|={grad_norm:.2e} {where}'
    )
    assert dist < MIN_TOL, (
        f'{FUNCTION.name} from (0,2): (3,2) dist {dist:.2e} {where}'
    )

    return save(
        FUNCTION,
        [X],
        name,
        title=f'{FUNCTION.name} from (0, 2)',
    )


def test_himmelblau_from_0_2(save_photo, log):
    # The problem was: from (0,2) point the algorithm went to the saddle point
    # and converged to a point in 10-11 steps by zigzags
    # Expected only 3–4 steps without zigzags.
    _run_start_0_2(save_photo, log)


def test_himmelblau_converges(save_photo, label, log):
    _run_starts(save_photo, label, log)
