"""Instance generation, metrics, sweeps, and worked examples for reconstruction attacks."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
from numpy.random import Generator

from libdpy.attacks.reconstruction.solvers import lin_reg_reconstruction


def make_instance(
    n: int,
    m: int,
    alpha: float,
    rng: Generator,
    p_secret: float = 0.5,
    p_query: float = 0.5,
    noise: str = "uniform",
    laplace_scale: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a random reconstruction attack instance.

    Args:
        n: Database size.
        m: Number of queries.
        alpha: Accuracy scale (bound for uniform noise; variance reference otherwise).
        rng: NumPy random generator.
        p_secret: Bernoulli probability for each secret bit.
        p_query: Bernoulli probability for each query matrix entry.
        noise: Noise model: ``"uniform"``, ``"laplace"``, or ``"gaussian"``.
        laplace_scale: Optional Laplace scale; defaults to variance-matched ``alpha/sqrt(6)``.

    Returns:
        Tuple ``(b, Q, r, eta)`` where ``b`` is the secret, ``Q`` the query matrix,
        ``r`` the noisy answers, and ``eta`` the noise vector.

    Raises:
        ValueError: If ``m < n`` (need at least as many queries as records).
    """
    if m < n:
        raise ValueError(
            f"Number of queries m={m} must be at least the sample size n={n}"
        )

    b = rng.binomial(1, p_secret, size=n).astype(int)
    Q = rng.binomial(1, p_query, size=(m, n)).astype(float)

    if noise == "uniform":
        eta = rng.uniform(-alpha, alpha, size=m)
    elif noise == "laplace":
        if laplace_scale is None:
            laplace_scale = alpha / np.sqrt(6)
        eta = rng.laplace(0.0, laplace_scale, size=m)
    elif noise == "gaussian":
        sigma = alpha / np.sqrt(3)
        eta = rng.normal(0.0, sigma, size=m)
    else:
        raise ValueError(f"Unknown noise type: {noise}")

    r = Q @ b + eta
    return b, Q, r, eta


def enumerate_binary_vectors(n: int) -> np.ndarray:
    """
    Enumerate all binary vectors in ``{0,1}^n``.

    Args:
        n: Dimension; must satisfy ``n <= 22``.

    Returns:
        Array of shape ``(2**n, n)`` with all binary vectors.
    """
    assert n <= 22, "Enumeration creates 2^n rows; choose n <= 22."
    arr = np.arange(2**n, dtype=np.uint64)[:, None]
    bits = ((arr >> np.arange(n, dtype=np.uint64)) & 1).astype(int)
    return bits


def make_all_accurate_out_of_range_example(
    alpha: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OLS downside 2: all answers alpha-accurate but OLS leaves ``[0, 1]^3``.

    The instance is the bounded-accuracy counterexample
    ``b = (1, 0, 0)``,
    ``Q = [[0, 1, 1], [1, 0, 0], [1, 0, 1]]``, and
    ``r = (alpha, 1 + alpha, 1 - alpha)``. Since ``Q`` is invertible,
    unconstrained OLS fits ``Qx = r`` exactly and returns
    ``(1 + alpha, 3 * alpha, -2 * alpha)``.

    Args:
        alpha: Uniform accuracy bound satisfied by every answer.

    Returns:
        Tuple ``(b, Q, r)`` for the 3-dimensional counterexample.
    """
    Q = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=float,
    )

    b = np.array([1, 0, 0], dtype=int)

    eta = np.array([alpha, alpha, -alpha], dtype=float)
    r = Q @ b + eta

    return b, Q, r


def corner_feasible(
    c: Sequence[float],
    queries: Sequence[Sequence[float]],
    b: np.ndarray,
    alpha: float,
) -> bool:
    """
    Check whether a candidate corner satisfies all query accuracy slabs.

    Args:
        c: Candidate point in ``[0, 1]^n``.
        queries: Iterable of query vectors.
        b: True secret database vector.
        alpha: Uniform accuracy bound for every query.

    Returns:
        True if every query answer on ``c`` is within ``alpha`` of the true answer.
    """
    c_arr = np.asarray(c)
    b_arr = np.asarray(b)
    return all(
        abs(np.asarray(q) @ c_arr - np.asarray(q) @ b_arr) <= alpha for q in queries
    )


def compute_2d_slab_region(
    queries: Sequence[Sequence[float]],
    b: np.ndarray,
    alpha: float,
    grid_resolution: int = 400,
    cube: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """
    Compute the 2D feasible region inside query accuracy slabs.

    Args:
        queries: Query vectors for the slab intersection picture.
        b: True secret vector.
        alpha: Uniform accuracy bound for each query.
        grid_resolution: Number of grid points per axis.
        cube: Lower and upper bounds for each coordinate (default ``[0, 1]``).

    Returns:
        Dictionary with mesh arrays ``X``, ``Y``, boolean mask ``feasible``,
        per-query contour levels ``contour_levels``, and ``cube`` bounds.
    """
    b_arr = np.asarray(b, dtype=float)
    lo_bound, hi_bound = cube
    grid = np.linspace(lo_bound, hi_bound, grid_resolution)
    X, Y = np.meshgrid(grid, grid)
    feasible = np.ones_like(X, dtype=bool)
    contour_levels: list[tuple[float, float]] = []

    for q in queries:
        q_arr = np.asarray(q, dtype=float)
        lo = float(q_arr @ b_arr - alpha)
        hi = float(q_arr @ b_arr + alpha)
        values = q_arr[0] * X + q_arr[1] * Y
        feasible &= (values >= lo) & (values <= hi)
        contour_levels.append((lo, hi))

    return {
        "X": X,
        "Y": Y,
        "feasible": feasible,
        "contour_levels": contour_levels,
        "cube": cube,
    }


def make_candidate_elimination_demo(
    seed: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, int | float]]:
    """
    Fixed small-n instance for the candidate-enumeration lecture demo.

    Parameters are tuned so several wrong nearby candidates remain feasible
    while distant candidates are eliminated by the accuracy slabs.
    """
    n, m = 12, 22
    alpha = 1.75
    rng = np.random.default_rng(seed)
    b, Q, r, _eta = make_instance(n, m, alpha, rng)
    meta: dict[str, int | float] = {"n": n, "m": m, "alpha": alpha, "seed": seed}
    return b, Q, r, alpha, meta


def compute_candidate_elimination(
    b: np.ndarray,
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """
    Enumerate Boolean candidates and score residuals for candidate elimination.

    Args:
        b: True secret binary vector.
        Q: Query matrix.
        r: Released responses.
        alpha: Uniform accuracy bound.

    Returns:
        Dictionary with ``n``, ``true_index``, ``hamming_dist``, ``max_residual``,
        ``feasible``, ``distances``, ``total_by_dist``, and ``feasible_by_dist``.
    """
    n = len(b)
    candidates = enumerate_binary_vectors(n)
    candidate_answers = candidates @ Q.T
    residuals = np.abs(candidate_answers - r[None, :])

    max_residual = residuals.max(axis=1)
    hamming_dist = np.abs(candidates - b[None, :]).sum(axis=1)
    feasible = max_residual <= alpha

    distances = np.arange(n + 1)
    total_by_dist = np.array([np.sum(hamming_dist == k) for k in distances])
    feasible_by_dist = np.array(
        [np.sum((hamming_dist == k) & feasible) for k in distances]
    )

    return {
        "n": n,
        "true_index": int(np.flatnonzero(hamming_dist == 0)[0]),
        "hamming_dist": hamming_dist,
        "max_residual": max_residual,
        "feasible": feasible,
        "distances": distances,
        "total_by_dist": total_by_dist,
        "feasible_by_dist": feasible_by_dist,
    }


def compare_ols_downside2_estimators(
    alpha: float,
) -> dict[str, Any]:
    """
    Build the OLS out-of-cube counterexample and the OLS attack pipeline.

    Args:
        alpha: Uniform accuracy bound satisfied by every answer.

    Returns:
        Dictionary with the instance ``(b, Q, r)``, the continuous OLS solution
        and its clip to the cube, the rounded OLS bits, and the bounded-accuracy
        LP feasible segment.
    """
    b, Q, r = make_all_accurate_out_of_range_example(alpha=alpha)

    # Unconstrained least squares - the same fit the LR attack uses internally.
    x_ols = np.linalg.lstsq(Q, r, rcond=None)[0]
    x_ols_clipped = np.clip(x_ols, 0.0, 1.0)
    b_ols = lin_reg_reconstruction(Q, r)

    # The bounded-accuracy LP feasible region is the segment (1, s, 0) with
    # 0 <= s <= min(2 * alpha, 1); every point of it rounds to the true b = (1, 0, 0).
    lp_upper_x2 = min(2 * alpha, 1.0)
    lp_segment = np.array([[1.0, 0.0, 0.0], [1.0, lp_upper_x2, 0.0]])

    return {
        "b": b,
        "Q": Q,
        "r": r,
        "alpha": alpha,
        "x_ols": x_ols,
        "x_ols_clipped": x_ols_clipped,
        "b_ols": b_ols,
        "lp_segment": lp_segment,
    }


