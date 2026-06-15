"""Geometry helpers for reconstruction-attack lecture visualizations."""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple

import numpy as np
from numpy.random import Generator

from libdpy.attacks.reconstruction.instances import (
    enumerate_binary_vectors,
)

__all__ = [
    "classify_corners_under_slabs",
    "cube_corners",
    "grid_points",
    "plane_polygon",
    "sample_feasible_cloud",
    "slab_polygons",
]


def cube_corners(n: int = 3) -> np.ndarray:
    """
    Return all Boolean corners of ``[0, 1]^n``.

    Args:
        n: Dimension (typically 3 for lecture visuals).

    Returns:
        Array of shape ``(2**n, n)`` with corners in ``{0, 1}^n``.
    """
    return enumerate_binary_vectors(n).astype(float)


def classify_corners_under_slabs(
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """
    Classify each Boolean corner as feasible or infeasible under all slabs.

    Args:
        Q: Query matrix of shape ``(m, n)``.
        r: Released responses of length ``m``.
        alpha: Uniform accuracy bound.

    Returns:
        Dictionary with ``corners`` (``2^n x n``) and boolean ``feasible`` mask.
    """
    n = Q.shape[1]
    corners = cube_corners(n)
    answers = corners @ Q.T
    max_residual = np.abs(answers - r).max(axis=1)
    return {
        "corners": corners,
        "feasible": max_residual <= alpha,
    }


def _cube_edge_pairs(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Enumerate undirected edges of the ``n``-dimensional unit hypercube."""
    corners = list(itertools.product([0.0, 1.0], repeat=n))
    edges: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for corner in corners:
        for dim in range(n):
            neighbor = list(corner)
            neighbor[dim] = 1.0 - neighbor[dim]
            edge = tuple(sorted((corner, tuple(neighbor))))
            edges.add(edge)
    return [(np.array(a), np.array(b)) for a, b in edges]


def plane_polygon(
    q: Sequence[float],
    c: float,
    cube: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray | None:
    """
    Vertices of the plane ``q @ x = c`` clipped to ``[cube_lo, cube_hi]^n``.

    Args:
        q: Normal/query vector.
        c: Plane offset.
        cube: Lower and upper bounds for each coordinate.

    Returns:
        Ordered polygon vertices as ``(k, n)`` array, or ``None`` if empty.
    """
    q_arr = np.asarray(q, dtype=float)
    n = len(q_arr)
    lo, hi = cube
    points: list[np.ndarray] = []

    for p1, p2 in _cube_edge_pairs(n):
        p1s = lo + (hi - lo) * p1
        p2s = lo + (hi - lo) * p2
        v1 = float(q_arr @ p1s)
        v2 = float(q_arr @ p2s)
        if abs(v2 - v1) < 1e-12:
            if abs(v1 - c) < 1e-9:
                points.extend([p1s, p2s])
            continue
        t = (c - v1) / (v2 - v1)
        if -1e-9 <= t <= 1.0 + 1e-9:
            pt = p1s + t * (p2s - p1s)
            if np.all(pt >= lo - 1e-9) and np.all(pt <= hi + 1e-9):
                points.append(pt)

    if not points:
        return None

    uniq: list[np.ndarray] = []
    for pt in points:
        if not any(np.linalg.norm(pt - u) < 1e-8 for u in uniq):
            uniq.append(pt)

    if len(uniq) < 3:
        return np.array(uniq)

    return _order_polygon_vertices(np.array(uniq), q_arr)


def _order_polygon_vertices(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Order coplanar points into a simple polygon using 2D projection."""
    q_norm = q / (np.linalg.norm(q) + 1e-15)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, q_norm))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(q_norm, ref)
    u /= np.linalg.norm(u) + 1e-15
    v = np.cross(q_norm, u)
    center = points.mean(axis=0)
    rel = points - center
    angles = np.arctan2(rel @ v, rel @ u)
    order = np.argsort(angles)
    return points[order]


def slab_polygons(
    q: Sequence[float],
    lo: float,
    hi: float,
    cube: Tuple[float, float] = (0.0, 1.0),
) -> dict[str, np.ndarray | None]:
    """
    Polygons for the two parallel faces of a slab ``lo <= q @ x <= hi``.

    Args:
        q: Query vector.
        lo: Lower slab boundary ``q @ x = lo``.
        hi: Upper slab boundary ``q @ x = hi``.
        cube: Cube bounds.

    Returns:
        Dictionary with ``lower`` and ``upper`` polygon vertex arrays.
    """
    return {
        "lower": plane_polygon(q, lo, cube=cube),
        "upper": plane_polygon(q, hi, cube=cube),
    }


def sample_feasible_cloud(
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
    n_samples: int,
    rng: Generator,
    max_attempts: int | None = None,
) -> np.ndarray:
    """
    Rejection-sample points in ``[0, 1]^n`` satisfying all query slabs.

    Args:
        Q: Query matrix.
        r: Released responses.
        alpha: Uniform accuracy bound.
        n_samples: Target number of accepted samples.
        rng: NumPy random generator.
        max_attempts: Optional cap on rejection trials (defaults to ``50 * n_samples``).

    Returns:
        Array of shape ``(k, n)`` with ``k <= n_samples`` accepted points.
    """
    m, n = Q.shape
    if max_attempts is None:
        max_attempts = max(50 * n_samples, 1000)

    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < n_samples and attempts < max_attempts:
        x = rng.uniform(0.0, 1.0, size=n)
        residual = np.abs(Q @ x - r).max()
        if residual <= alpha + 1e-12:
            accepted.append(x)
        attempts += 1

    if not accepted:
        return np.empty((0, n))
    return np.array(accepted)


def grid_points(level: int | str, n: int = 3) -> np.ndarray | None:
    """
    Discrete grid inside ``[0, 1]^n`` for hypergrid relaxation visuals.

    Args:
        level: Grid resolution — ``1`` gives ``{0, 1}``, ``2`` adds midpoints,
            ``4`` quarter-grid, ``8`` eighth-grid, or ``"continuous"`` for none.
        n: Dimension.

    Returns:
        Array of shape ``(level+1)^n, n`` or ``None`` when ``level == "continuous"``.
    """
    if level == "continuous":
        return None
    level_int = int(level)
    coords = np.linspace(0.0, 1.0, level_int + 1)
    return np.array(list(itertools.product(coords, repeat=n)), dtype=float)


def slab_membership(
    points: np.ndarray,
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Check slab feasibility for a batch of points.

    Args:
        points: Array of shape ``(k, n)``.
        Q: Query matrix.
        r: Released responses.
        alpha: Accuracy bound.

    Returns:
        Boolean mask of length ``k``.
    """
    if len(points) == 0:
        return np.array([], dtype=bool)
    residuals = np.abs(points @ Q.T - r).max(axis=1)
    return residuals <= alpha + 1e-12


