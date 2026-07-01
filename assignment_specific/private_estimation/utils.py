"""Dataset helpers, mechanisms, witness search, and audit wrappers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import Any, Callable

import numpy as np
import pandas as pd

from libdpy.assignment_specific.privacy_auditing.utils import (
    audit_point,
    empirical_roc,
    epsilon_from_roc_point,
    selected_threshold_from_empirical_roc,
)
from libdpy.privacy_mechanisms.above_threshold import above_threshold
# ---------------------------------------------------------------------------
# Public-menu constants (fixed before any private data is touched)
# ---------------------------------------------------------------------------

# Fulton income: p99 ≈ 322k, max ≈ 717k; grid covers 99.5th percentile with headroom.
_INCOME_GRID_MAX = 350_000
PUBLIC_VALUE_LOWER: float = 0.0
PUBLIC_VALUE_UPPER: float = float(_INCOME_GRID_MAX)
# Coarse dollar menu for other lecture sections (not inverse-sensitivity quantiles).
PUBLIC_INCOME_CANDIDATES: np.ndarray = np.arange(0, _INCOME_GRID_MAX + 1, 2_500, dtype=float)
# Public output range for the grid-free exponential-mechanism quantile (S5–S6).
PUBLIC_QUANTILE_LOWER: float = 0.0
PUBLIC_QUANTILE_UPPER: float = 10_000_000.0
# Public accuracy floor for the interval exponential mechanism: points are nudged
# this far (in dollars) toward the bounds so the central interval has positive
# width even under heavy ties. Announced before any private data is touched.
DEFAULT_EM_GRANULARITY: float = 1.0
# Display-only score grid for the S5 rank-loss/probability landscape plots. NOT a
# mechanism input — the quantile mechanism is grid-free (see private_quantile_exponential).
PUBLIC_INVERSE_SENSITIVITY_N_THRESHOLDS: int = 1025
PUBLIC_INVERSE_SENSITIVITY_THRESHOLDS: np.ndarray = np.linspace(
    PUBLIC_VALUE_LOWER,
    PUBLIC_VALUE_UPPER,
    PUBLIC_INVERSE_SENSITIVITY_N_THRESHOLDS,
    dtype=float,
)
PUBLIC_INCOME_BIN_EDGES: np.ndarray = np.arange(0, _INCOME_GRID_MAX + 10_000, 10_000, dtype=float)
# Public shift for log-income view: min observed income ≈ −10k; shift is announced before data.
PUBLIC_LOG1P_SHIFT: float = 15_000.0
PUBLIC_LOG1P_BIN_EDGES: np.ndarray = np.arange(0, 14.5, 0.25, dtype=float)
# Public scales for bounded feature scores (PUMS schema bounds — not derived from private data).
PUBLIC_TARGET_UPPER: float = _INCOME_GRID_MAX
PUBLIC_AGE_SCALE: float = 100.0
# Pairwise-difference and location grids for Gaussian localization (S8).
PUBLIC_SCALE_DIFF_BIN_EDGES: np.ndarray = np.linspace(0, 80_000, 41, dtype=float)
GAUSSIAN_LOCATION_GRID_SPAN: float = 6.0
GAUSSIAN_LOCATION_N_BINS: int = 25
# median(|X−Y|) for X,Y ~ N(0,σ²) equals σ·√2·Φ⁻¹(0.75) ≈ 0.954·σ
HALF_NORMAL_MEDIAN_FACTOR: float = float(np.sqrt(2) * 0.674489750196082)
DEFAULT_AUDIT_EPS_MARGIN: float = 0.15
# Ordered income floors for AboveThreshold (public plausibility: low → high).
PUBLIC_ORDERED_INCOME_THRESHOLDS: np.ndarray = np.array(
    [5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000, 75_000, 100_000],
    dtype=float,
)
# High → low: find the highest floor whose count still clears τ (meaningful sparse-vector search).
PUBLIC_ORDERED_INCOME_THRESHOLDS_HIGH_TO_LOW: np.ndarray = PUBLIC_ORDERED_INCOME_THRESHOLDS[::-1]
# Finer high→low floors for AboveThreshold teaching: crossing near τ ≈ n/2 spreads halt indices.
PUBLIC_ABOVE_THRESHOLD_FLOORS: np.ndarray = np.array(
    [
        100_000,
        75_000,
        50_000,
        40_000,
        30_000,
        25_000,
        22_000,
        20_000,
        19_000,
        18_000,
        17_000,
        16_000,
        15_000,
        10_000,
        5_000,
    ],
    dtype=float,
)
# Public age bins for the minimal baseline predictor (PUMS schema bounds, not from private data).
PUBLIC_AGE_BIN_EDGES: np.ndarray = np.array([0, 30, 50, 100], dtype=float)
MIN_CLIP_WIDTH: float = 5_000.0  # dollars — public floor on clipping interval width

DEFAULT_SEED: int = 42
DEFAULT_N: int = 1000
DEFAULT_EPS_TOTAL: float = 1.0
DEFAULT_K_SIGMA: float = 4.0
DEFAULT_N_DATASETS: int = 5_000
DEFAULT_N_RUNS: int = 1
DEFAULT_N_TRIALS_AUDIT: int = 5_000  # notebook default — smooth teaching audits
DEFAULT_DELTA: float = 1e-2
# Figure-freeze / CI: smaller trials keep ``test_private_estimation.py`` under ~2 minutes.
FIGURE_FREEZE_N_TRIALS_AUDIT: int = 120
FIGURE_FREEZE_REPAIR_N_SEEDS: int = 2
FIGURE_FREEZE_SELECTION_REPEATS: int = 40
REPAIR_AUDIT_N_SEEDS: int = 3
REPAIR_AUDIT_MARGIN: float = 0.5
REPAIR_AUDIT_ALPHA: float = 0.05
# Repair-freeze audit calibration (R8): median eps_plug over REPAIR_AUDIT_N_SEEDS independent
# audit seeds; margin REPAIR_AUDIT_MARGIN chosen so correctly private ε=1 mechanisms pass in
# calibration runs at N_TRIALS_AUDIT=5_000 while still catching leaky prototypes. When
# alpha_total=REPAIR_AUDIT_ALPHA, also checks Clopper–Pearson eps_audit upper bound.
REPAIR_AUDIT_CALIBRATION = (
    "Repair-freeze: median eps_plug over "
    f"{REPAIR_AUDIT_N_SEEDS} seeds, margin={REPAIR_AUDIT_MARGIN}, "
    f"alpha={REPAIR_AUDIT_ALPHA}, trials={DEFAULT_N_TRIALS_AUDIT} (notebook) / "
    f"{FIGURE_FREEZE_N_TRIALS_AUDIT} (figure-freeze tests)."
)


def _normal_cdf(value: float) -> float:
    """Scalar standard-normal CDF without depending on scipy at runtime."""

    return 0.5 * math.erfc(-float(value) / math.sqrt(2.0))


def gaussian_delta_for_sensitivity_std(
    sensitivity: float,
    std: float,
    epsilon: float,
) -> float:
    """Exact Gaussian-mechanism delta for neighboring mean shift ``sensitivity``."""

    sensitivity = abs(float(sensitivity))
    std = float(std)
    epsilon = float(epsilon)
    if sensitivity == 0:
        return 0.0
    if std <= 0:
        return float("inf")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    a = sensitivity / std
    return max(
        0.0,
        _normal_cdf(a / 2.0 - epsilon / a)
        - math.exp(epsilon) * _normal_cdf(-a / 2.0 - epsilon / a),
    )


def gaussian_noise_std(
    sensitivity: float,
    epsilon: float,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Return the Gaussian std giving ``(epsilon, delta)`` for a scalar query."""

    sensitivity = abs(float(sensitivity))
    epsilon = float(epsilon)
    delta = float(delta)
    if sensitivity == 0:
        return 0.0
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")

    hi = max(sensitivity, 1e-12)
    while gaussian_delta_for_sensitivity_std(sensitivity, hi, epsilon) > delta:
        hi *= 2.0
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gaussian_delta_for_sensitivity_std(sensitivity, mid, epsilon) > delta:
            lo = mid
        else:
            hi = mid
    return float(hi)


def clipped_noisy_mean(
    x: np.ndarray,
    L: float,
    U: float,
    epsilon: float,
    rng: np.random.Generator,
    *,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Clip to ``[L, U]``, calibrate Gaussian noise to sensitivity, return noisy mean."""

    clipped = np.clip(x, L, U)
    n = len(x)
    sensitivity = (U - L) / n if U > L else 1.0 / n
    std = gaussian_noise_std(sensitivity, epsilon, delta)
    return float(np.mean(clipped) + rng.normal(0.0, std))


class FigureMode(str, Enum):
    """Structured figure labels for proposal vs audit teaching modes."""

    ANALYTICAL_EXPLANATION = "analytical proposal explanation"
    ORACLE_DIAGNOSTIC = "oracle/non-private diagnostic"
    EMPIRICAL_AUDIT = "empirical audit"


def figure_mode_suffix(mode: FigureMode) -> str:
    return f" [{mode.value}]"


class DataProvenance(str, Enum):
    """Controlled labels for figure/table data sources (R11)."""

    REAL_DATA = "real data"
    EXTREME_REAL = "extreme real record"
    CONSTRUCTED_WITNESS = "constructed real-data witness"
    SYNTHETIC_WITNESS = "synthetic witness"
    SYNTHETIC_MODEL = "synthetic model"
    ORACLE_DIAGNOSTIC = "oracle diagnostic"
    PRIVATE_RUN = "private run"


def neighbor_witness_table(
    *,
    section: str,
    d_description: str,
    d_prime_description: str,
    adjacency: str,
    n: int,
    algorithm_effect: str,
    audit_target: str,
    provenance: DataProvenance,
) -> list[dict]:
    """Compact neighbor/witness table for notebook display (R10)."""

    return [
        {"field": "section", "value": section},
        {"field": "D", "value": d_description},
        {"field": "D′", "value": d_prime_description},
        {"field": "adjacency", "value": adjacency},
        {"field": "n", "value": n},
        {"field": "algorithm effect", "value": algorithm_effect},
        {"field": "audit target", "value": audit_target},
        {"field": "data provenance", "value": provenance.value},
    ]


def raw_mean_perturbation_trace(
    D: np.ndarray,
    D_prime: np.ndarray,
    out_idx: int,
    noise_std: float,
) -> list[dict]:
    """One-row before/after trace for raw noisy mean (R12)."""

    mu_d = float(np.mean(D))
    mu_dp = float(np.mean(D_prime))
    return [
        {"quantity": "swapped row index", "D": out_idx, "D′": out_idx},
        {"quantity": "row value", "D": float(D[out_idx]), "D′": float(D_prime[out_idx])},
        {"quantity": "exact mean", "D": mu_d, "D′": mu_dp},
        {"quantity": "mean shift |Δμ|", "D": "—", "D′": abs(mu_d - mu_dp)},
        {"quantity": "Gaussian noise std", "D": noise_std, "D′": noise_std},
        {"quantity": "|Δμ|/noise std", "D": "—", "D′": abs(mu_d - mu_dp) / noise_std},
    ]


def quantile_clipping_perturbation_trace(
    D: np.ndarray,
    D_prime: np.ndarray,
    out_idx: int,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> list[dict]:
    """One-row before/after trace for empirical quantile clipping (R12)."""

    l_d, u_d = float(np.quantile(D, low_q)), float(np.quantile(D, high_q))
    l_dp, u_dp = float(np.quantile(D_prime, low_q)), float(np.quantile(D_prime, high_q))
    clipped_d = float(np.clip(D[out_idx], l_d, u_d))
    clipped_dp = float(np.clip(D_prime[out_idx], l_dp, u_dp))
    mean_d = float(np.mean(np.clip(D, l_d, u_d)))
    mean_dp = float(np.mean(np.clip(D_prime, l_dp, u_dp)))
    n = len(D)
    return [
        {"quantity": "swapped row value", "D": float(D[out_idx]), "D′": float(D_prime[out_idx])},
        {"quantity": "empirical L (non-private)", "D": l_d, "D′": l_dp},
        {"quantity": "empirical U (non-private)", "D": u_d, "D′": u_dp},
        {"quantity": "clipped row value", "D": clipped_d, "D′": clipped_dp},
        {"quantity": "clipped mean (no noise)", "D": mean_d, "D′": mean_dp},
        {
            "quantity": "sensitivity (U−L)/n",
            "D": (u_d - l_d) / n if u_d > l_d else 0.0,
            "D′": (u_dp - l_dp) / n if u_dp > l_dp else 0.0,
        },
    ]


def mu_sigma_perturbation_trace(
    D: np.ndarray,
    D_prime: np.ndarray,
    out_idx: int,
    k: float = DEFAULT_K_SIGMA,
) -> list[dict]:
    """One-row before/after trace for empirical μ±kσ clipping (R12)."""

    def bounds(x: np.ndarray) -> tuple[float, float, float, float]:
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        return mu, sigma, mu - k * sigma, mu + k * sigma

    mu_d, sig_d, l_d, u_d = bounds(D)
    mu_dp, sig_dp, l_dp, u_dp = bounds(D_prime)
    mean_d = float(np.mean(np.clip(D, l_d, u_d)))
    mean_dp = float(np.mean(np.clip(D_prime, l_dp, u_dp)))
    return [
        {"quantity": "swapped row value", "D": float(D[out_idx]), "D′": float(D_prime[out_idx])},
        {"quantity": "empirical μ", "D": mu_d, "D′": mu_dp},
        {"quantity": "empirical σ", "D": sig_d, "D′": sig_dp},
        {"quantity": "clip L = μ−kσ", "D": l_d, "D′": l_dp},
        {"quantity": "clip U = μ+kσ", "D": u_d, "D′": u_dp},
        {"quantity": "clip width", "D": u_d - l_d, "D′": u_dp - l_dp},
        {"quantity": "clipped mean (no noise)", "D": mean_d, "D′": mean_dp},
    ]


def empirical_mu_sigma_bounds(
    x: np.ndarray,
    k: float = DEFAULT_K_SIGMA,
) -> tuple[float, float, float, float]:
    """Return empirical ``mu, sigma, L, U`` for the broken ``mu ± k sigma`` rule."""

    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return mu, sigma, mu - k * sigma, mu + k * sigma


def empirical_mu_sigma_clipped_stat(x: np.ndarray, k: float = DEFAULT_K_SIGMA) -> float:
    """Deterministic clipped mean under empirical ``mu ± k sigma`` bounds."""

    _, _, L, U = empirical_mu_sigma_bounds(x, k=k)
    return float(np.mean(np.clip(x, L, U)))


def empirical_mu_sigma_noise_scale(
    x: np.ndarray,
    epsilon: float,
    k: float = DEFAULT_K_SIGMA,
) -> float:
    """Gaussian std used by the broken empirical ``mu ± k sigma`` proposal."""

    _, _, L, U = empirical_mu_sigma_bounds(x, k=k)
    sensitivity = (U - L) / len(x) if U > L else 1.0 / len(x)
    return gaussian_noise_std(sensitivity, epsilon)


def empirical_mu_sigma_output_law(
    x: np.ndarray,
    epsilon: float,
    k: float = DEFAULT_K_SIGMA,
) -> tuple[float, float]:
    """Return ``(loc, std)`` for the proposal's Gaussian output law on ``x``."""

    return (
        empirical_mu_sigma_clipped_stat(x, k=k),
        empirical_mu_sigma_noise_scale(x, epsilon=epsilon, k=k),
    )


def select_typical_mu_sigma_pair(
    D: np.ndarray,
    pool: np.ndarray,
    *,
    k: float = DEFAULT_K_SIGMA,
    seed: int = DEFAULT_SEED,
    n_candidates: int = 120,
    target_shift_range: tuple[float, float] = (1.0, 50.0),
) -> tuple[float, int, float, np.ndarray]:
    """Pick an ordinary substitution pair where empirical ``mu±kσ`` barely moves."""

    rng = np.random.default_rng(seed)
    D = np.asarray(D, dtype=float)
    pool = np.asarray(pool, dtype=float)
    base = empirical_mu_sigma_clipped_stat(D, k=k)
    best: tuple[float, int, float, np.ndarray] | None = None
    low, high = target_shift_range
    for _ in range(n_candidates):
        out_idx = int(rng.integers(0, len(D)))
        in_value = float(rng.choice(pool))
        D_prime = replace_one_row(D, out_idx, in_value)
        shift = abs(empirical_mu_sigma_clipped_stat(D_prime, k=k) - base)
        candidate = (shift, out_idx, in_value, D_prime)
        if low <= shift <= high:
            return candidate
        if best is None or shift < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Could not construct a typical mu-sigma substitution pair.")
    return best


def leave_one_out_influences(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Per-row leave-one-out influence on *statistic* (R13 diagnostic, not a DP proof)."""

    x = np.asarray(x, dtype=float)
    base = statistic(x)
    return np.array([abs(statistic(np.delete(x, i)) - base) for i in range(len(x))])


def mean_removal_influences(x: np.ndarray) -> np.ndarray:
    """Distribution of |mean(D) − mean(D \\ i)| — removal diagnostic."""

    return leave_one_out_influences(x, lambda arr: float(np.mean(arr)))


def replacement_swap_influences(
    D: np.ndarray,
    pool: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_swaps: int = 200,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Typical-data stability probe under substitution adjacency.

    This is a realistic row-swap diagnostic, not a DP proof: it asks how much a
    statistic moves for ordinary swaps drawn from a held-out pool.
    """

    rng = np.random.default_rng(seed)
    D = np.asarray(D, dtype=float)
    pool = np.asarray(pool, dtype=float)
    base = float(statistic(D))
    n_swaps = min(n_swaps, len(pool))
    out_indices = rng.integers(0, len(D), size=n_swaps)
    in_rows = rng.choice(pool, size=n_swaps, replace=False)
    shifts = []
    for out_idx, in_row in zip(out_indices, in_rows):
        D_prime = replace_one_row(D, int(out_idx), float(in_row))
        shifts.append(abs(float(statistic(D_prime)) - base))
    return np.asarray(shifts, dtype=float)


def clipped_mean_removal_influences(
    x: np.ndarray,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> np.ndarray:
    """Leave-one-out influence on empirical-quantile clipped mean."""

    def stat(arr: np.ndarray) -> float:
        l_val = float(np.quantile(arr, low_q))
        u_val = float(np.quantile(arr, high_q))
        return float(np.mean(np.clip(arr, l_val, u_val)))

    return leave_one_out_influences(x, stat)


def median_removal_influences(x: np.ndarray) -> np.ndarray:
    """Leave-one-out influence on the median — removal diagnostic."""

    return leave_one_out_influences(x, lambda arr: float(np.median(arr)))


def mu_sigma_removal_influences(x: np.ndarray, k: float = DEFAULT_K_SIGMA) -> np.ndarray:
    """Leave-one-out influence on μ±kσ clip-interval width."""

    return leave_one_out_influences(x, lambda arr: empirical_mu_sigma_clip_width(arr, k=k))


def quantile_bound_removal_influences(
    x: np.ndarray,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out influence on empirical L and U (returns L_infl, U_infl)."""

    def l_stat(arr: np.ndarray) -> float:
        return float(np.quantile(arr, low_q))

    def u_stat(arr: np.ndarray) -> float:
        return float(np.quantile(arr, high_q))

    return leave_one_out_influences(x, l_stat), leave_one_out_influences(x, u_stat)


def median_perturbation_trace(
    D: np.ndarray,
    D_prime: np.ndarray,
    out_idx: int,
) -> list[dict]:
    """One-row before/after trace for median value sensitivity (R12)."""

    med_d = float(np.median(D))
    med_dp = float(np.median(D_prime))
    rank_d = int(np.sum(D <= D[out_idx]))
    rank_dp = int(np.sum(D_prime <= D_prime[out_idx]))
    return [
        {"quantity": "swapped row value", "D": float(D[out_idx]), "D′": float(D_prime[out_idx])},
        {"quantity": "exact median", "D": med_d, "D′": med_dp},
        {"quantity": "median gap |Δ|", "D": "—", "D′": abs(med_d - med_dp)},
        {"quantity": "rank of swapped row", "D": rank_d, "D′": rank_dp},
    ]


def select_typical_median_pair(
    D: np.ndarray,
    pool: np.ndarray,
    *,
    max_shift: float = 1_000.0,
    seed: int = DEFAULT_SEED,
) -> tuple[float, int, float, np.ndarray]:
    """Pick a real-data substitution pair with a small nonzero median shift."""

    D = np.asarray(D, dtype=float)
    pool = np.asarray(pool, dtype=float)
    base_median = float(np.median(D))
    rng = np.random.default_rng(seed)

    near_pool_order = np.argsort(np.abs(pool - base_median))
    pool_candidates = np.unique(
        np.concatenate(
            [
                pool[near_pool_order[: min(500, len(pool))]],
                np.sort(pool)[: min(100, len(pool))],
                np.sort(pool)[-min(100, len(pool)) :],
            ]
        )
    )
    sorted_indices = np.argsort(D)
    near_d_order = np.argsort(np.abs(D - base_median))
    out_candidates = np.unique(
        np.concatenate(
            [
                sorted_indices[: min(200, len(D))],
                sorted_indices[-min(200, len(D)) :],
                near_d_order[: min(200, len(D))],
            ]
        )
    )
    rng.shuffle(out_candidates)

    best: tuple[float, int, float, np.ndarray] | None = None
    for out_idx in out_candidates:
        for in_value in pool_candidates:
            D_prime = replace_one_row(D, int(out_idx), float(in_value))
            shift = abs(float(np.median(D_prime)) - base_median)
            if shift == 0:
                continue
            candidate = (shift, int(out_idx), float(in_value), D_prime)
            if best is None or shift < best[0]:
                best = candidate
            if shift <= max_shift:
                return candidate
    if best is None:
        return 0.0, int(out_candidates[0]), float(pool_candidates[0]), replace_one_row(
            D, int(out_candidates[0]), float(pool_candidates[0])
        )
    return best


def above_threshold_perturbation_trace(
    D: np.ndarray,
    D_prime: np.ndarray,
    out_idx: int,
    floors: np.ndarray,
    count_threshold: float,
    *,
    n_crossing: int = 5,
) -> list[dict]:
    """Before/after exact counts near τ for AboveThreshold (R12)."""

    rows: list[dict] = []
    floors_htl = floors[::-1]
    for floor in floors_htl[:n_crossing]:
        c_d = count_above_floor(D, float(floor))
        c_dp = count_above_floor(D_prime, float(floor))
        rows.append(
            {
                "floor": float(floor),
                "exact_count_D": c_d,
                "exact_count_D_prime": c_dp,
                "threshold_tau": count_threshold,
                "clears_tau_D": c_d >= count_threshold,
                "clears_tau_D_prime": c_dp >= count_threshold,
            }
        )
    rows.insert(
        0,
        {
            "quantity": "swapped row",
            "D": float(D[out_idx]),
            "D′": float(D_prime[out_idx]),
        },
    )
    return rows


def bounded_score_perturbation_trace(
    feature: np.ndarray,
    target: np.ndarray,
    target_prime: np.ndarray,
    out_idx: int,
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]],
    *,
    target_upper: float = PUBLIC_TARGET_UPPER,
) -> list[dict]:
    """Before/after bounded candidate scores for report-noisy-max (R12)."""

    rows: list[dict] = []
    rows.append(
        {
            "quantity": "swapped income row",
            "D": float(target[out_idx]),
            "D′": float(target_prime[out_idx]),
        }
    )
    for name, phi in transforms.items():
        s_d = bounded_feature_score(feature, target, phi, target_upper=target_upper)
        s_dp = bounded_feature_score(feature, target_prime, phi, target_upper=target_upper)
        rows.append({"transform": name, "exact_score_D": s_d, "exact_score_D_prime": s_dp})
    return rows


def histogram_replacement_trace(
    values: np.ndarray,
    bin_edges: np.ndarray,
    out_idx: int,
    in_value: float,
) -> list[dict]:
    """Replacement effect on fixed-bin counts (R12 / R13)."""

    d = np.asarray(values, dtype=float)
    d_prime = replace_one_row(d, out_idx, in_value)
    counts_d, _ = np.histogram(d, bins=bin_edges)
    counts_dp, _ = np.histogram(d_prime, bins=bin_edges)
    changed = np.where(counts_d != counts_dp)[0]
    rows = [
        {
            "quantity": "swapped value",
            "D": float(d[out_idx]),
            "D′": float(in_value),
        },
        {
            "quantity": "bins changed under substitution",
            "D": "—",
            "D′": len(changed),
        },
    ]
    for idx in changed[:4]:
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        rows.append(
            {
                "bin": f"[{lo:.0f}, {hi:.0f})",
                "count_D": int(counts_d[idx]),
                "count_D_prime": int(counts_dp[idx]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_fulton() -> pd.DataFrame:
    """Load the Fulton PUMS microdata table from package resources."""

    csv_path = files("libdpy.resources").joinpath("FultonPUMS5full.csv")
    return pd.read_csv(csv_path)


def extract_income(
    df: pd.DataFrame,
    *,
    keep_zero: bool = True,
    transform: str | None = None,
) -> np.ndarray:
    """Return the income column as a float array.

    Parameters
    ----------
    keep_zero:
        When False, raises — filtering on income *values* is data-dependent and
        must not be done silently.  Use an explicit audited proposal instead.
    transform:
        ``None`` for raw income, or ``"log1p"`` for ``log(1 + income + PUBLIC_LOG1P_SHIFT)``.
        The public shift handles negative incomes without silent filtering.
    """

    if not keep_zero:
        raise ValueError(
            "Dropping nonpositive incomes is data-dependent preprocessing; "
            "use an explicit audited proposal instead."
        )
    income = df["income"].to_numpy(dtype=float)
    if transform is None:
        return income
    if transform == "log1p":
        return np.log1p(income + PUBLIC_LOG1P_SHIFT)
    raise ValueError(f"Unknown transform: {transform!r}")


def split_base_and_pool(
    x: np.ndarray,
    n_base: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split *x* into a base sample *D* and a held-out witness candidate pool."""

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    base_idx = indices[:n_base]
    pool_idx = indices[n_base:]
    return x[base_idx].copy(), x[pool_idx].copy()


def replace_one_row(
    D: np.ndarray,
    out_index: int,
    in_row: float,
) -> np.ndarray:
    """Return a neighboring dataset under substitution adjacency."""

    D_prime = D.copy()
    D_prime[out_index] = in_row
    return D_prime


def construct_quantile_gap_subset(
    x: np.ndarray,
    *,
    n: int = DEFAULT_N,
    bulk_low: float = 20_000.0,
    bulk_high: float = 30_000.0,
    tail_value: float = 120_000.0,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Build a size-``n`` witness where the 99% quantile sits on a bulk/tail gap.

    The lowest 99% of rows form a ramp up to ``bulk_high``; the remaining rows sit
    on a flat upper tail at ``tail_value``. Replacing the top bulk order statistic
    (index ``bulk_count - 1``) by a witness above the tail jumps the learned upper
    clip by roughly ``tail_value - bulk_high``.
    """

    _ = (x, seed)
    bulk_count = max(2, int(round(0.99 * n)))
    tail_count = n - bulk_count
    bulk = np.linspace(bulk_low, bulk_high, bulk_count)
    tail = np.full(tail_count, tail_value)
    return np.concatenate([bulk, tail])


def build_engineered_quantile_neighbor_pair(
    n: int = DEFAULT_N,
    *,
    witness: float = 200_000.0,
    bulk_low: float = 20_000.0,
    bulk_high: float = 30_000.0,
    tail_value: float = 120_000.0,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return ``(D, D', out_idx, witness)`` for the quantile-clipping leverage pair."""

    D = construct_quantile_gap_subset(
        np.array([]),
        n=n,
        bulk_low=bulk_low,
        bulk_high=bulk_high,
        tail_value=tail_value,
    )
    bulk_count = max(2, int(round(0.99 * n)))
    out_idx = bulk_count - 1
    D_prime = replace_one_row(D, out_idx, witness)
    return D, D_prime, out_idx, witness


def construct_median_gap_subset(
    x: np.ndarray,
    *,
    n: int = DEFAULT_N,
    gap_low: float = 40_000.0,
    gap_high: float = 80_000.0,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Build a teaching subset where one row swap can jump the median across a gap."""

    if n < 2:
        raise ValueError("Median-gap witness requires at least two rows.")
    rng = np.random.default_rng(seed)
    below_count = n // 2 + 1
    above_count = n - below_count
    below = rng.uniform(max(0.0, gap_low - 5_000.0), gap_low, size=below_count)
    above = rng.uniform(gap_high, gap_high + 5_000.0, size=above_count)
    subset = np.concatenate([below, above])
    rng.shuffle(subset)
    return subset


# ---------------------------------------------------------------------------
# Audit panel
# ---------------------------------------------------------------------------


@dataclass
class AuditPanel:
    """Result of auditing a mechanism on neighboring datasets."""

    samples_neg: np.ndarray
    samples_pos: np.ndarray
    tau_star: float
    governing_fpr: float
    governing_tpr: float
    eps_plug: float
    audit_result: Any
    adversary_statistic: str
    separates: bool
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    thresholds: np.ndarray = field(repr=False)


def audit_panel(
    mechanism: Callable[..., Any],
    D: np.ndarray,
    D_prime: np.ndarray,
    n_trials: int,
    delta: float,
    *,
    extractor: Callable[[Any], float] = lambda z: float(z),
    seed: int = DEFAULT_SEED,
    adversary_statistic: str = "mechanism output",
    alpha_total: float | None = None,
    mechanism_kwargs: dict | None = None,
    claimed_eps: float | None = None,
    margin: float = DEFAULT_AUDIT_EPS_MARGIN,
) -> AuditPanel:
    """Run a mechanism on *D* and *D'* and build an audit panel.

  The *extractor* reduces vector-valued outputs to the scalar event audited by
  ``empirical_roc`` / ``audit_point``.
    """

    mechanism_kwargs = mechanism_kwargs or {}
    rng = np.random.default_rng(seed)
    s_neg = np.array(
        [extractor(mechanism(D, rng=rng, **mechanism_kwargs)) for _ in range(n_trials)],
        dtype=float,
    )
    s_pos = np.array(
        [extractor(mechanism(D_prime, rng=rng, **mechanism_kwargs)) for _ in range(n_trials)],
        dtype=float,
    )
    fpr, tpr, thresholds = empirical_roc(s_neg, s_pos)
    tau_star, (gov_fpr, gov_tpr), eps_plug = selected_threshold_from_empirical_roc(
        s_neg, s_pos, delta
    )
    result = audit_point(s_neg, s_pos, tau_star, delta, alpha_total=alpha_total)
    if claimed_eps is not None:
        separates = eps_plug > claimed_eps + margin
    else:
        separates = eps_plug > 0 and not result.no_positive_evidence
    return AuditPanel(
        samples_neg=s_neg,
        samples_pos=s_pos,
        tau_star=tau_star,
        governing_fpr=gov_fpr,
        governing_tpr=gov_tpr,
        eps_plug=eps_plug,
        audit_result=result,
        adversary_statistic=adversary_statistic,
        separates=separates,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def evaluate_audit(
    mechanism: Callable[..., Any],
    D: np.ndarray,
    D_prime: np.ndarray,
    extractor: Callable[[Any], float] = lambda z: float(z),
    *,
    n_trials: int = DEFAULT_N_TRIALS_AUDIT,
    delta: float = DEFAULT_DELTA,
    seed: int = DEFAULT_SEED,
    adversary_statistic: str = "mechanism output",
    **audit_kwargs: Any,
) -> AuditPanel:
    """Sample a mechanism on a neighboring pair and return the shared audit panel."""

    return audit_panel(
        mechanism,
        D,
        D_prime,
        n_trials,
        delta,
        extractor=extractor,
        seed=seed,
        adversary_statistic=adversary_statistic,
        **audit_kwargs,
    )


def _extract_estimate(output: Any) -> float:
    """Default estimator extractor for scalar or dict-returning mechanisms."""

    if isinstance(output, dict):
        return float(output.get("estimate", output.get("median", output.get("value"))))
    return float(output)


def _population_rank(population: np.ndarray, released: float) -> float:
    """Full-population CDF value at a released candidate."""

    return float(np.mean(np.asarray(population, dtype=float) <= released))


def population_rank_cdf(population: np.ndarray, released: float) -> float:
    """Reference CDF value used for median/rank accuracy targets."""

    return _population_rank(population, released)


def stability_summary(shifts: np.ndarray, label: str) -> pd.DataFrame:
    """Compact summary for typical substitution-shift probes."""

    shifts = np.asarray(shifts, dtype=float)
    return pd.DataFrame(
        [
            {
                "probe": label,
                "median_shift": np.median(shifts),
                "p90_shift": np.quantile(shifts, 0.9),
                "max_shift": np.max(shifts),
            }
        ]
    )


def evaluate_accuracy(
    mechanism: Callable[..., Any],
    population: np.ndarray,
    targets: dict[str, Callable[..., float]],
    *,
    n: int = DEFAULT_N,
    n_datasets: int = DEFAULT_N_DATASETS,
    n_runs: int = DEFAULT_N_RUNS,
    seed: int = DEFAULT_SEED,
    method: str = "mechanism",
    privacy_status: str = "valid",
    estimate_target: str = "mean",
    epsilon_total: float | None = DEFAULT_EPS_TOTAL,
    notes: str = "",
    extractor: Callable[[Any], float] = _extract_estimate,
) -> pd.DataFrame:
    """Evaluate accuracy on ordinary size-*n* draws from a fixed population.

    Each mechanism run is scored against two references:
    ``reference='sample'`` isolates mechanism utility relative to the drawn
    sample's non-private statistic, while ``reference='population'`` includes
    the sampling error relative to the full Fulton population.

    For ``error_metric='rank'``, each reference uses its own empirical CDF at
    the released value: ``rank_sample(released) / n - 0.5`` for the sample
    reference and ``rank_population(released) / |population| - 0.5`` for the
    population reference.
    """

    population = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for dataset_id in range(n_datasets):
        sample = rng.choice(population, size=n, replace=False)
        for run_id in range(n_runs):
            run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            estimate = extractor(mechanism(sample, rng=run_rng))
            for metric_name, target_fn in targets.items():
                for reference, reference_data in (
                    ("sample", sample),
                    ("population", population),
                ):
                    if metric_name == "rank":
                        rank_denominator = len(reference_data)
                        rank_numerator = int(np.sum(reference_data <= estimate))
                        cdf_value = rank_numerator / rank_denominator
                        target = 0.5
                        metric_estimate = cdf_value
                        error = cdf_value - target
                    else:
                        rank_numerator = None
                        rank_denominator = None
                        target = float(target_fn(reference_data))
                        metric_estimate = estimate
                        error = metric_estimate - target
                    rows.append(
                        {
                            "method": method,
                            "privacy_status": privacy_status,
                            "estimate_target": estimate_target,
                            "error_metric": metric_name,
                            "reference": reference,
                            "epsilon_total": epsilon_total,
                            "dataset_id": dataset_id,
                            "run_id": run_id,
                            "estimate": metric_estimate,
                            "target": target,
                            "error": error,
                            "abs_error": abs(error),
                            "squared_error": error * error,
                            "rank_numerator": rank_numerator,
                            "rank_denominator": rank_denominator,
                            "notes": notes,
                        }
                    )
    return pd.DataFrame(rows)


def accuracy_summary_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Median and 90th-percentile absolute error per leaderboard facet."""

    if leaderboard.empty:
        return pd.DataFrame()
    grouped = leaderboard.groupby(
        [
            "method",
            "privacy_status",
            "estimate_target",
            "error_metric",
            "reference",
        ],
        dropna=False,
    )["abs_error"]
    return grouped.agg(median_abs_error="median", p90_abs_error=lambda s: s.quantile(0.9)).reset_index()


def proposals_diagnoses_repairs_from_leaderboard(leaderboard: pd.DataFrame) -> list[dict]:
    """Summary table with a typical-accuracy column derived from leaderboard rows."""

    summary = accuracy_summary_table(leaderboard)
    sample = summary[summary["reference"] == "sample"] if not summary.empty else summary
    med_by_method = {
        row["method"]: row["median_abs_error"] for _, row in sample.iterrows()
    }
    rows = proposals_diagnoses_repairs_table()
    method_aliases = {
        "Raw noisy mean": "raw noisy mean",
        "Empirical quantile clipping": "empirical quantile clipping",
        "Empirical μ±kσ clipping": "empirical mu±k sigma clipping",
        "Median value+noise": "median value+noise",
        "Log-sigma histogram mean": "log-sigma histogram mean",
    }
    for row in rows:
        key = method_aliases.get(row["proposal"])
        value = med_by_method.get(key)
        row["typical_median_abs_error"] = value if value is not None else "not scored"
    return rows


# ---------------------------------------------------------------------------
# Witness search
# ---------------------------------------------------------------------------


def search_witness(
    D: np.ndarray,
    pool: np.ndarray,
    deterministic_core: Callable[[np.ndarray, int, float], float],
    objective: Callable[[float], float],
    *,
    out_index: int | None = None,
    max_candidates: int | None = 200,
    verbose: bool = True,
) -> tuple[int, float, float]:
    """Find a witness row from *pool* that maximizes *objective*.

    Returns ``(out_index, witness_value, objective_value)``.
    Witness search is an audit convenience, not part of the mechanism.
    """

    if out_index is None:
        out_index = int(np.argmax(np.abs(D - np.median(D))))

    candidates = pool if max_candidates is None else pool[: max_candidates]
    best_val = -np.inf
    best_row = float(candidates[0])
    for row in candidates:
        score = objective(deterministic_core(D, out_index, float(row)))
        if score > best_val:
            best_val = score
            best_row = float(row)

    if verbose:
        print(
            f"Witness: replace D[{out_index}]={D[out_index]:.4g} "
            f"with {best_row:.4g} (objective={best_val:.4g})"
        )
    return out_index, best_row, best_val


def verify_witness_separates(
    mechanism: Callable[..., Any],
    D: np.ndarray,
    D_prime: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS_AUDIT,
    delta: float = DEFAULT_DELTA,
    *,
    claimed_eps: float | None = None,
    margin: float = DEFAULT_AUDIT_EPS_MARGIN,
    **audit_kwargs: Any,
) -> AuditPanel:
    """Re-run ``audit_panel`` and assert the witness exceeds the claimed ε budget."""

    panel = audit_panel(
        mechanism,
        D,
        D_prime,
        n_trials,
        delta,
        claimed_eps=claimed_eps,
        margin=margin,
        **audit_kwargs,
    )
    if claimed_eps is not None:
        if not (panel.eps_plug > claimed_eps + margin):
            raise ValueError(
                f"Witness does not exceed claimed ε "
                f"(eps_plug={panel.eps_plug}, claimed={claimed_eps}, "
                f"statistic={panel.adversary_statistic})"
            )
    elif not panel.separates:
        raise ValueError(
            f"Witness does not separate D from D' "
            f"(eps_plug={panel.eps_plug}, statistic={panel.adversary_statistic})"
        )
    return panel


def verify_repair_within_claimed_eps(
    mechanism: Callable[..., Any],
    D: np.ndarray,
    D_prime: np.ndarray,
    claimed_eps: float,
    n_trials: int = DEFAULT_N_TRIALS_AUDIT,
    delta: float = DEFAULT_DELTA,
    *,
    margin: float = REPAIR_AUDIT_MARGIN,
    n_seeds: int = REPAIR_AUDIT_N_SEEDS,
    seed: int = DEFAULT_SEED,
    **audit_kwargs: Any,
) -> AuditPanel:
    """Assert a repaired mechanism's audit ε stays within the claimed budget.

    Uses the **median** ``eps_plug`` over ``n_seeds`` independent audit seeds (reduces
    boundary flakiness).  When ``alpha_total=REPAIR_AUDIT_ALPHA`` is passed via
    ``audit_kwargs``, also checks the Clopper–Pearson ``eps_audit`` upper bound.
    """

    audit_kwargs.setdefault("alpha_total", REPAIR_AUDIT_ALPHA)
    panels: list[AuditPanel] = []
    eps_plugs: list[float] = []
    eps_audits: list[float] = []
    for offset in range(n_seeds):
        panel = audit_panel(
            mechanism,
            D,
            D_prime,
            n_trials,
            delta,
            seed=seed + offset,
            claimed_eps=claimed_eps,
            margin=margin,
            **audit_kwargs,
        )
        panels.append(panel)
        eps_plugs.append(panel.eps_plug)
        if panel.audit_result.eps_audit is not None and math.isfinite(
            panel.audit_result.eps_audit
        ):
            eps_audits.append(panel.audit_result.eps_audit)

    median_eps = float(np.median(eps_plugs))
    representative = panels[int(np.argmin(np.abs(np.array(eps_plugs) - median_eps)))]
    if median_eps > claimed_eps + margin:
        raise ValueError(
            f"Repair exceeds claimed ε "
            f"(median eps_plug={median_eps:.3g} over {n_seeds} seeds, "
            f"claimed={claimed_eps}, statistic={representative.adversary_statistic})"
        )
    if eps_audits and float(np.median(eps_audits)) > claimed_eps + margin:
        raise ValueError(
            f"Repair exceeds claimed ε "
            f"(median eps_audit={float(np.median(eps_audits)):.3g}, "
            f"claimed={claimed_eps}, statistic={representative.adversary_statistic})"
        )
    return representative


def build_raw_mean_audit_datasets(
    n_base: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Witness for raw-mean audit: swap the minimum row for the pool maximum."""

    income = extract_income(load_fulton())
    D, pool = split_base_and_pool(income, n_base, seed=seed)
    out_idx = int(np.argmin(D))
    witness = float(np.max(pool))
    D_prime = replace_one_row(D, out_idx, witness)
    return D, D_prime, out_idx, witness


# ---------------------------------------------------------------------------
# Broken mechanisms (invalid — prefix signals audit demos only)
# ---------------------------------------------------------------------------


def raw_noisy_mean(
    x: np.ndarray,
    epsilon: float,
    noise_scale_override: float,
    rng: np.random.Generator,
) -> float:
    """Noisy mean with an analyst-chosen scale — no valid sensitivity calibration."""

    return float(np.mean(x) + rng.normal(0.0, noise_scale_override))


def public_range_noisy_mean(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    public_range: float,
    epsilon: float,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Noisy mean calibrated to a public bounded-domain range."""

    std = gaussian_noise_std(float(public_range) / len(x), epsilon, delta)
    return float(np.mean(x) + rng.normal(0.0, std))


def empirical_quantile_clipped_mean(
    x: np.ndarray,
    epsilon: float,
    low_q: float,
    high_q: float,
    rng: np.random.Generator,
) -> float:
    """Clip at empirical quantiles, then add noise — thresholds are non-private."""

    L = float(np.quantile(x, low_q))
    U = float(np.quantile(x, high_q))
    return clipped_noisy_mean(x, L, U, epsilon, rng)


def empirical_quantile_clipped_stat(
    x: np.ndarray,
    low_q: float,
    high_q: float,
) -> float:
    """Deterministic clipped mean using empirical quantile bounds."""

    L = float(np.quantile(x, low_q))
    U = float(np.quantile(x, high_q))
    return float(np.mean(np.clip(x, L, U)))


def empirical_quantile_diagnostics(
    label: str,
    x: np.ndarray,
    low_q: float,
    high_q: float,
    epsilon: float,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """Diagnostics for empirical quantile clipping and its Gaussian scale."""

    L = float(np.quantile(x, low_q))
    U = float(np.quantile(x, high_q))
    clipped = np.clip(x, L, U)
    sensitivity = (U - L) / len(x)
    std = gaussian_noise_std(sensitivity, epsilon, delta)
    return {
        "dataset": label,
        "n": len(x),
        "empirical_L": L,
        "empirical_U": U,
        "clipped_mean_no_noise": float(np.mean(clipped)),
        "Gaussian_std": std,
        "output_variance": std**2,
    }


def empirical_quantile_output_law(
    x: np.ndarray,
    low_q: float,
    high_q: float,
    epsilon: float,
    delta: float = DEFAULT_DELTA,
) -> tuple[float, float]:
    """Return deterministic clipped mean and Gaussian std for empirical quantile clipping."""

    diagnostics = empirical_quantile_diagnostics("", x, low_q, high_q, epsilon, delta)
    return diagnostics["clipped_mean_no_noise"], diagnostics["Gaussian_std"]


def quantile_roc_title(
    prefix: str,
    mu: float,
    scale: float,
    mu_prime: float,
    scale_prime: float,
    *,
    delta: float = DEFAULT_DELTA,
) -> str:
    """Standard title for quantile-clipping analytical ROC figures."""

    del scale_prime
    return (
        f"{prefix}: epsilon-1 calibrated Gaussian ROC; "
        f"loc shift={abs(mu - mu_prime):,.1f}, required std={scale:,.1f}; delta={delta:g}"
    )


def median_value_plus_noise(
    x: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    target_lower: float = 0.0,
    target_upper: float = PUBLIC_TARGET_UPPER,
    public_range: float | None = None,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Release median value + Gaussian noise calibrated to the public value range."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if public_range is not None:
        sensitivity = float(public_range)
    else:
        if target_upper <= target_lower:
            raise ValueError("target_upper must exceed target_lower")
        sensitivity = target_upper - target_lower
    std = gaussian_noise_std(sensitivity, epsilon, delta)
    return float(np.median(x) + rng.normal(0.0, std))


def median_value_plus_noise_with_range(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    public_range: float,
    epsilon: float,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Backward-compatible alias for :func:`median_value_plus_noise`."""

    return median_value_plus_noise(
        x, epsilon, rng, public_range=public_range, delta=delta
    )


def empirical_mu_sigma_clipped_mean(
    x: np.ndarray,
    epsilon: float,
    k: float,
    rng: np.random.Generator,
) -> float:
    """Clip to empirical mean ± k·σ — center and scale are non-private."""

    mu = float(np.mean(x))
    sigma = float(np.std(x))
    L, U = mu - k * sigma, mu + k * sigma
    return clipped_noisy_mean(x, L, U, epsilon, rng)


def private_noisy_mu_sigma_step(
    x: np.ndarray,
    L: float,
    U: float,
    eps_mu: float,
    eps_var: float,
    rng: np.random.Generator,
    *,
    k: float = DEFAULT_K_SIGMA,
    public_lower: float = 0.0,
    public_upper: float = PUBLIC_TARGET_UPPER,
    min_private_sigma: float = 2_500.0,
) -> dict:
    """One private truncated-moment refinement step for ``mu±kσ`` clipping."""

    L, U = _enforce_clip_bounds(float(L), float(U))
    L = max(public_lower, L)
    U = min(public_upper, U)
    if U <= L:
        L, U = public_lower, public_upper
    clipped = np.clip(x, L, U)
    width = U - L
    midpoint = 0.5 * (L + U)
    centered = clipped - midpoint
    n = len(x)

    noisy_centered_mean = float(
        np.mean(centered)
        + rng.normal(0.0, gaussian_noise_std(width / n, eps_mu))
    )
    centered_square_range = (0.5 * width) ** 2
    noisy_second_moment = float(
        np.mean(centered**2)
        + rng.normal(0.0, gaussian_noise_std(centered_square_range / n, eps_var))
    )
    noisy_variance = max(noisy_second_moment - noisy_centered_mean**2, min_private_sigma**2)
    mu_hat = midpoint + noisy_centered_mean
    sigma_hat = float(np.sqrt(noisy_variance))
    next_L = max(public_lower, mu_hat - k * sigma_hat)
    next_U = min(public_upper, mu_hat + k * sigma_hat)
    next_L, next_U = _enforce_clip_bounds(next_L, next_U)
    next_L = max(public_lower, next_L)
    next_U = min(public_upper, next_U)
    return {
        "input_L": L,
        "input_U": U,
        "mu_hat": float(mu_hat),
        "sigma_hat": sigma_hat,
        "L": next_L,
        "U": next_U,
        "epsilon_mu": eps_mu,
        "epsilon_variance": eps_var,
    }


def private_mu_sigma_clipped_mean(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    eps_localize: float,
    eps_mean: float,
    n_rounds: int = 1,
    k: float = DEFAULT_K_SIGMA,
    public_lower: float = 0.0,
    public_upper: float = PUBLIC_TARGET_UPPER,
    min_private_sigma: float = 2_500.0,
) -> dict:
    """Private iterative ``mu/sigma`` localization, clipping, and noisy mean."""

    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    L, U = public_lower, public_upper
    trace = []
    eps_round = eps_localize / n_rounds
    for round_id in range(n_rounds):
        step = private_noisy_mu_sigma_step(
            x,
            L,
            U,
            eps_round / 2.0,
            eps_round / 2.0,
            rng,
            k=k,
            public_lower=public_lower,
            public_upper=public_upper,
            min_private_sigma=min_private_sigma,
        )
        step["round"] = round_id + 1
        trace.append(step)
        L, U = step["L"], step["U"]
    estimate = clipped_noisy_mean(x, L, U, eps_mean, rng)
    mean_sensitivity = (U - L) / len(x)
    return {
        "estimate": estimate,
        "L": L,
        "U": U,
        "mu_hat": trace[-1]["mu_hat"],
        "sigma_hat": trace[-1]["sigma_hat"],
        "trace": trace,
        "epsilon": {"mu_sigma_rounds": eps_localize, "mean": eps_mean},
        "sensitivity": {"final_mean": mean_sensitivity},
        "public_choices": ["public_truncation_range", "k_sigma", "min_private_sigma"],
    }


def private_mu_sigma_summary(result: dict, label: str) -> dict:
    """Compact table row for a private ``mu/sigma`` repair run."""

    return {
        "method": label,
        "estimate": result["estimate"],
        "final_L": result["L"],
        "final_U": result["U"],
        "private_mu_hat": result["mu_hat"],
        "private_sigma_hat": result["sigma_hat"],
        "final_mean_sensitivity": result["sensitivity"]["final_mean"],
    }


def empirical_mu_sigma_clip_width(x: np.ndarray, k: float = DEFAULT_K_SIGMA) -> float:
    """Deterministic clip-interval width from empirical μ±kσ (for witness search)."""

    return 2.0 * k * float(np.std(np.asarray(x, dtype=float)))


def build_mu_sigma_clipping_witness(
    n_base: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    *,
    k: float = DEFAULT_K_SIGMA,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Witness for μ±kσ audit: maximize clip-interval width shift (variance leverage)."""

    income = extract_income(load_fulton())
    D, pool = split_base_and_pool(income, n_base, seed=seed)
    out_idx, witness, _ = search_witness(
        D,
        pool,
        lambda d, i, row: empirical_mu_sigma_clip_width(replace_one_row(d, i, row), k=k),
        lambda width: abs(width - empirical_mu_sigma_clip_width(D, k=k)),
        verbose=False,
    )
    return D, replace_one_row(D, out_idx, witness), out_idx, witness


# ---------------------------------------------------------------------------
# Repaired mechanisms
# ---------------------------------------------------------------------------


def rank_utility(database: list, candidate: float, alpha: float) -> float:
    """Rank-distance utility for quantile selection; sensitivity = 1 under substitution."""

    n = len(database)
    target_rank = alpha * n
    r = sum(1 for v in database if v <= candidate)
    return -abs(r - target_rank)


def private_quantile(
    x: np.ndarray,
    candidates: np.ndarray | list,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> float:
    """Select a quantile candidate via the exponential mechanism (sensitivity = 1).

    Vectorized equivalent of the exponential mechanism with
    :func:`rank_utility`: the rank of every candidate is read off the sorted
    sample with ``searchsorted`` (``side='right'`` counts ``v <= candidate``),
    so cost is ``O(n log n + m log n)`` instead of ``O(m·n)`` over ``m``
    candidates. Same scores, same softmax, same single ``rng.choice`` draw.
    """

    x_sorted = np.sort(np.asarray(x, dtype=float))
    cand = np.asarray(candidates, dtype=float)
    ranks = np.searchsorted(x_sorted, cand, side="right")
    scores = -np.abs(ranks - alpha * len(x_sorted))
    weights = np.exp(epsilon * scores / 2.0)  # sensitivity = 1
    probabilities = weights / weights.sum()
    return float(cand[rng.choice(len(cand), p=probabilities)])


def private_quantile_exponential(
    x: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    epsilon: float,
    rng: np.random.Generator,
    *,
    quantile: float = 0.5,
    granularity: float = DEFAULT_EM_GRANULARITY,
) -> float:
    """Grid-free exponential mechanism for a quantile on bounded data.

    Port of the interval exponential mechanism (Drechsler et al.; see
    ``ExpMech_Drechsler.dpMedianExponential``). The public output range
    ``[lower_bound, upper_bound]`` is partitioned at the sorted sample; each
    interval between adjacent order statistics shares one rank-distance score
    ``-(eps/2)·|rank - quantile·n|``, additively weighted by
    ``log(interval_width)`` so wider intervals are proportionally more likely.
    One interval is drawn by Gumbel-max (exponential-mechanism sampling) and the
    release is uniform inside it, so the estimate is continuous and needs no
    candidate grid. Rank sensitivity is 1 under substitution.

    ``granularity`` is a public accuracy floor: points are nudged that far toward
    the bounds away from the target rank, guaranteeing the central interval has
    positive width even under heavy ties.
    """

    if lower_bound > upper_bound:
        raise ValueError("lower_bound must not exceed upper_bound")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")

    z = np.sort(np.clip(np.asarray(x, dtype=float), lower_bound, upper_bound))
    n = len(z)
    if n < 2:
        return float(z[0]) if n == 1 else float(0.5 * (lower_bound + upper_bound))

    # Nudge points away from the target rank toward the public bounds (ties → width).
    quantile_rank = int(math.floor(n * quantile))
    end = min(n, quantile_rank + 1)
    z[:end] = np.maximum(lower_bound, z[:end] - granularity)
    z[quantile_rank + 1:] = np.minimum(z[quantile_rank + 1:] + granularity, upper_bound)

    # Interval k = (z[k], z[k+1]) has exactly k+1 points at or below it.
    lengths = z[1:] - z[:-1]
    with np.errstate(divide="ignore"):
        log_lengths = np.where(lengths > 0, np.log(lengths), -np.inf)
    ranks = np.arange(1, n)
    rung = np.abs(ranks - n * quantile)
    scores = -(epsilon / 2.0) * rung + log_lengths

    # Gumbel-max == sampling the interval from the exponential mechanism.
    winner = int(np.argmax(scores + rng.gumbel(0.0, 1.0, size=scores.size)))
    return float(rng.uniform(z[winner], z[winner + 1]))


def private_inverse_sensitivity_quantile(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    lower_bound: float = PUBLIC_QUANTILE_LOWER,
    upper_bound: float = PUBLIC_QUANTILE_UPPER,
    alpha: float = 0.5,
    epsilon: float = DEFAULT_EPS_TOTAL,
    granularity: float = DEFAULT_EM_GRANULARITY,
) -> float:
    """Private quantile/median via the grid-free interval exponential mechanism.

    The rank-distance score is the inverse-sensitivity score for quantiles
    (distance, in row changes, from making a value the exact quantile), realized
    continuously over the public range rather than on a fixed candidate grid.
    """

    return private_quantile_exponential(
        x, lower_bound, upper_bound, epsilon, rng, quantile=alpha, granularity=granularity
    )


def private_rank_quantile_public_grid(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    lower_bound: float = PUBLIC_QUANTILE_LOWER,
    upper_bound: float = PUBLIC_QUANTILE_UPPER,
    alpha: float = 0.5,
    epsilon: float = DEFAULT_EPS_TOTAL,
    granularity: float = DEFAULT_EM_GRANULARITY,
) -> float:
    """Alias for :func:`private_inverse_sensitivity_quantile`."""

    return private_inverse_sensitivity_quantile(
        x,
        rng,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        alpha=alpha,
        epsilon=epsilon,
        granularity=granularity,
    )


def enforce_clip_bounds(L: float, U: float) -> tuple[float, float]:
    """Order bounds and enforce the public minimum clipping width."""

    return _enforce_clip_bounds(L, U)


def _enforce_clip_bounds(L: float, U: float) -> tuple[float, float]:
    """Order bounds and enforce the public minimum clipping width."""

    if L > U:
        L, U = U, L
    if U - L < MIN_CLIP_WIDTH:
        mid = 0.5 * (L + U)
        L = mid - 0.5 * MIN_CLIP_WIDTH
        U = mid + 0.5 * MIN_CLIP_WIDTH
    if L == U:
        U = L + MIN_CLIP_WIDTH
    return L, U


def private_quantile_clipped_mean(
    x: np.ndarray,
    candidates: np.ndarray | list,
    low_q: float,
    high_q: float,
    eps_low: float,
    eps_high: float,
    eps_mean: float,
    rng: np.random.Generator,
) -> dict:
    """Private quantiles → clip → noisy clipped mean with explicit budget split."""

    L = private_quantile(x, candidates, low_q, eps_low, rng)
    U = private_quantile(x, candidates, high_q, eps_high, rng)
    L, U = _enforce_clip_bounds(L, U)
    estimate = clipped_noisy_mean(x, L, U, eps_mean, rng)
    sensitivity = (U - L) / len(x)
    return {
        "estimate": estimate,
        "L": L,
        "U": U,
        "epsilon": {"low_q": eps_low, "high_q": eps_high, "mean": eps_mean},
        "sensitivity": {"quantile": 1, "mean": sensitivity},
        "public_choices": ["candidate_grid", "MIN_CLIP_WIDTH"],
    }


def fixed_bin_noisy_histogram(
    values: np.ndarray,
    bin_edges: np.ndarray,
    epsilon: float,
    adjacency: str = "replacement",
    rng: np.random.Generator | None = None,
) -> dict:
    """Release a noisy count vector for fixed public bin edges.

    Under **replacement** adjacency the L2 sensitivity of the count vector is
    ``sqrt(2)`` (one bin −1, another +1).  Calibrate Gaussian noise to that
    vector sensitivity.
    """

    if rng is None:
        rng = np.random.default_rng()
    if adjacency != "replacement":
        raise ValueError("Only substitution adjacency is supported.")
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bin_edges)
    std = gaussian_noise_std(math.sqrt(2.0), epsilon)
    noisy_counts = counts.astype(float) + rng.normal(0.0, std, size=len(counts))
    return {
        "counts": noisy_counts,
        "bin_edges": edges,
        "epsilon": epsilon,
        "sensitivity_l2": math.sqrt(2.0),
        "adjacency": adjacency,
    }


def _disjoint_pairwise_abs_diffs(
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One |difference| per disjoint pair — L1 histogram sensitivity 2.

    Pairing is a permutation of row *indices* (not values), so one replacement
    changes at most one pair's difference.
    """

    x = np.asarray(x, dtype=float)
    n = len(x)
    perm = rng.permutation(n)
    idx_pairs = perm[: 2 * (n // 2)].reshape(-1, 2)
    return np.abs(x[idx_pairs[:, 0]] - x[idx_pairs[:, 1]])


def _noisy_histogram_median(centers: np.ndarray, noisy_counts: np.ndarray) -> float:
    """Median bin center from a noisy histogram via cumulative counts."""

    cumulative = np.cumsum(noisy_counts)
    half = cumulative[-1] / 2.0
    idx = int(np.searchsorted(cumulative, half))
    idx = min(idx, len(centers) - 1)
    return float(centers[idx])


def public_location_bin_edges(
    sigma_tilde: float,
    center: float = 0.0,
) -> np.ndarray:
    """Public location grid in original units, resolution proportional to σ̃."""

    half_width = GAUSSIAN_LOCATION_GRID_SPAN * sigma_tilde
    return np.linspace(
        center - half_width,
        center + half_width,
        GAUSSIAN_LOCATION_N_BINS + 1,
        dtype=float,
    )


def private_scale_from_pairwise_diffs(
    x: np.ndarray,
    epsilon: float,
    bin_edges: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Estimate scale from a noisy histogram of disjoint pairwise |differences|.

    Each row appears in at most one pair, so the count-vector L1 sensitivity is 2.
    Scale is inferred from the median difference (half-normal calibration), not argmax.
    """

    pairwise = _disjoint_pairwise_abs_diffs(x, rng)
    hist = fixed_bin_noisy_histogram(pairwise, bin_edges, epsilon, rng=rng)
    centers = 0.5 * (hist["bin_edges"][:-1] + hist["bin_edges"][1:])
    median_diff = _noisy_histogram_median(centers, hist["counts"])
    sigma_tilde = median_diff / HALF_NORMAL_MEDIAN_FACTOR
    if sigma_tilde <= 0:
        sigma_tilde = 1.0
    return {
        "sigma_tilde": float(sigma_tilde),
        "median_diff": float(median_diff),
        "histogram": hist,
        "epsilon": epsilon,
    }


def private_location_from_scale(
    x: np.ndarray,
    sigma_tilde: float,
    epsilon: float,
    coarse_bin_edges: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Localize center via noisy histogram in original units (center, then refine).

    Step 1: coarse histogram on a wide public grid → coarse center.
    Step 2: fine histogram on a σ̃-spaced grid around the coarse center → μ̂.
    """

    if sigma_tilde <= 0:
        sigma_tilde = 1.0
    x = np.asarray(x, dtype=float)
    eps_coarse = epsilon / 2.0
    eps_fine = epsilon / 2.0
    coarse_hist = fixed_bin_noisy_histogram(x, coarse_bin_edges, eps_coarse, rng=rng)
    coarse_centers = 0.5 * (coarse_hist["bin_edges"][:-1] + coarse_hist["bin_edges"][1:])
    coarse_peak = int(np.argmax(coarse_hist["counts"]))
    coarse_center = float(coarse_centers[coarse_peak])
    loc_edges = public_location_bin_edges(sigma_tilde, center=coarse_center)
    loc_hist = fixed_bin_noisy_histogram(x, loc_edges, eps_fine, rng=rng)
    loc_centers = 0.5 * (loc_hist["bin_edges"][:-1] + loc_hist["bin_edges"][1:])
    loc_peak = int(np.argmax(loc_hist["counts"]))
    mu_hat = float(loc_centers[loc_peak])
    return {
        "mu_hat": mu_hat,
        "coarse_center": coarse_center,
        "histogram": loc_hist,
        "coarse_histogram": coarse_hist,
        "epsilon": epsilon,
    }


def gaussian_histogram_mean(
    x: np.ndarray,
    eps_scale: float,
    eps_location: float,
    eps_mean: float,
    scale_bin_edges: np.ndarray,
    coarse_bin_edges: np.ndarray,
    rng: np.random.Generator,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """KV18-style Gaussian localization: private scale → location → clipped noisy mean.

    Scale is localized from a noisy histogram of disjoint **pairwise absolute
    differences** ``|X_i - X_j|`` (each ``|N(0, 2σ²)|``, so the unknown mean
    cancels), following Karwa & Vadhan (2018, arXiv:1711.03908).
    """

    scale_result = private_scale_from_pairwise_diffs(x, eps_scale, scale_bin_edges, rng)
    sigma = scale_result["sigma_tilde"]
    loc_result = private_location_from_scale(
        x, sigma, eps_location, coarse_bin_edges, rng
    )
    mu = loc_result["mu_hat"]
    # Clip to [mu - 3σ, mu + 3σ] using public width rule
    L, U = mu - 3 * sigma, mu + 3 * sigma
    L, U = _enforce_clip_bounds(L, U)
    estimate = clipped_noisy_mean(x, L, U, eps_mean, rng, delta=delta)
    mean_sensitivity = (U - L) / len(x)
    return {
        "estimate": estimate,
        "mu_hat": mu,
        "sigma_tilde": sigma,
        "L": L,
        "U": U,
        "scale": scale_result,
        "scale_histogram": scale_result["histogram"],
        "location": loc_result,
        "epsilon": {
            "scale": eps_scale,
            "location": eps_location,
            "mean": eps_mean,
        },
        "public_choices": [
            "scale_bin_edges",
            "coarse_bin_edges",
            "GAUSSIAN_LOCATION_GRID_SPAN",
            "MIN_CLIP_WIDTH",
        ],
    }


def report_noisy_max(
    candidates: list,
    score_fn: Callable[[Any], float],
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> Any:
    """Generalized noisy max with Gaussian score noise."""

    noisy_scores = [
        score_fn(c) + rng.normal(0.0, gaussian_noise_std(sensitivity, epsilon))
        for c in candidates
    ]
    return candidates[int(np.argmax(noisy_scores))]


def bounded_feature_score(
    feature: np.ndarray,
    target: np.ndarray,
    phi_fn: Callable[[np.ndarray], np.ndarray],
    *,
    target_upper: float,
) -> float:
    """Canonical bounded score: ``|(1/n) Σ φ_j(x_i) · y_i|`` with y_i ∈ [0, 1].

    Per-row sensitivity under substitution is ``2/n`` (passed to report-noisy-max).
    """

    y = np.clip(target / target_upper, 0.0, 1.0)
    phi = np.clip(phi_fn(feature), -1.0, 1.0)
    return float(abs(np.mean(phi * y)))


def selection_score_sensitivity(n: int) -> float:
    """Sensitivity of the canonical bounded feature score under substitution."""

    return 2.0 / n


def private_group_means(
    feature: np.ndarray,
    clipped_target: np.ndarray,
    groups: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    target_range: float,
) -> dict:
    """Release noisy per-group counts and sums; means are post-processing.

    Under substitution the count vector has L2 sensitivity ``sqrt(2)``; the sum
    vector has L2 sensitivity ≤ ``sqrt(2) * target_range``.
    """

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    counts = np.array([np.sum(groups == g) for g in unique_groups], dtype=float)
    sums = np.array(
        [np.sum(clipped_target[groups == g]) for g in unique_groups], dtype=float
    )
    count_std = gaussian_noise_std(math.sqrt(2.0), epsilon)
    sum_std = gaussian_noise_std(math.sqrt(2.0) * target_range, epsilon)
    noisy_counts = counts + rng.normal(0.0, count_std, size=n_groups)
    noisy_sums = sums + rng.normal(0.0, sum_std, size=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(noisy_counts > 0, noisy_sums / noisy_counts, np.nan)
    return {
        "groups": unique_groups,
        "noisy_counts": noisy_counts,
        "noisy_sums": noisy_sums,
        "means": means,
        "epsilon": epsilon,
        "sensitivity": {
            "count_l2": math.sqrt(2.0),
            "sum_l2": math.sqrt(2.0) * target_range,
        },
    }


# ---------------------------------------------------------------------------
# Sensitivity verification (self-tests, not privacy proofs)
# ---------------------------------------------------------------------------


def max_rank_utility_change(
    x: np.ndarray,
    candidates: np.ndarray,
    alpha: float,
    *,
    n_random: int = 50,
    seed: int = 0,
) -> float:
    """Empirical max per-row change in rank utility under substitution."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_delta = 0.0
    for _ in range(n_random):
        i = rng.integers(0, n)
        x_out = x[i]
        x_in = rng.uniform(x.min(), x.max())
        D = list(x)
        D_prime = replace_one_row(x, i, x_in).tolist()
        for c in candidates:
            u0 = rank_utility(D, float(c), alpha)
            u1 = rank_utility(D_prime, float(c), alpha)
            max_delta = max(max_delta, abs(u1 - u0))
    return max_delta


def max_disjoint_pairwise_histogram_change(
    x: np.ndarray,
    bin_edges: np.ndarray,
    *,
    n_random: int = 100,
    seed: int = 0,
) -> float:
    """Empirical max L1 change in disjoint-pairwise-diff histogram under substitution."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_l1 = 0.0
    for _ in range(n_random):
        i = rng.integers(0, n)
        x_in = rng.uniform(x.min(), x.max())
        perm = rng.permutation(n)
        idx_pairs = perm[: 2 * (n // 2)].reshape(-1, 2)

        def _pairwise_diffs(arr: np.ndarray) -> np.ndarray:
            return np.abs(arr[idx_pairs[:, 0]] - arr[idx_pairs[:, 1]])

        c0, _ = np.histogram(_pairwise_diffs(x), bins=bin_edges)
        c1, _ = np.histogram(_pairwise_diffs(replace_one_row(x, i, x_in)), bins=bin_edges)
        max_l1 = max(max_l1, float(np.sum(np.abs(c1 - c0))))
    return max_l1


def max_histogram_count_change(
    x: np.ndarray,
    bin_edges: np.ndarray,
    *,
    n_random: int = 100,
    seed: int = 0,
) -> float:
    """Empirical max L1 change in bin counts under substitution."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_l1 = 0.0
    for _ in range(n_random):
        i = rng.integers(0, n)
        x_in = rng.uniform(x.min(), x.max())
        c0, _ = np.histogram(x, bins=bin_edges)
        c1, _ = np.histogram(replace_one_row(x, i, x_in), bins=bin_edges)
        max_l1 = max(max_l1, float(np.sum(np.abs(c1 - c0))))
    return max_l1


def max_clipped_mean_change(
    x: np.ndarray,
    L: float,
    U: float,
    *,
    n_random: int = 100,
    seed: int = 0,
) -> float:
    """Empirical max change in clipped mean under substitution."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_delta = 0.0
    for _ in range(n_random):
        i = rng.integers(0, n)
        x_in = rng.uniform(L, U)
        m0 = float(np.mean(np.clip(x, L, U)))
        m1 = float(np.mean(np.clip(replace_one_row(x, i, x_in), L, U)))
        max_delta = max(max_delta, abs(m1 - m0))
    return max_delta


def max_bounded_score_change(
    feature: np.ndarray,
    target: np.ndarray,
    phi_fn: Callable[[np.ndarray], np.ndarray],
    target_upper: float,
    *,
    n_random: int = 100,
    seed: int = 0,
) -> float:
    """Empirical max change in bounded feature score under substitution."""

    rng = np.random.default_rng(seed)
    n = len(feature)
    max_delta = 0.0
    for _ in range(n_random):
        i = rng.integers(0, n)
        s0 = bounded_feature_score(feature, target, phi_fn, target_upper=target_upper)
        f_prime = feature.copy()
        t_prime = target.copy()
        f_prime[i] = rng.uniform(feature.min(), feature.max())
        t_prime[i] = rng.uniform(0, target_upper)
        s1 = bounded_feature_score(f_prime, t_prime, phi_fn, target_upper=target_upper)
        max_delta = max(max_delta, abs(s1 - s0))
    return max_delta


def count_above_floor(income: np.ndarray, floor: float) -> float:
    """Sensitivity-1 query: count of rows with income ≥ *floor*."""

    return float(np.sum(np.asarray(income, dtype=float) >= floor))


def income_fraction_above_threshold(income: np.ndarray, floor: float) -> float:
    """Exact fraction above floor — teaching/oracle only (sensitivity 1/n, not for AboveThreshold)."""

    return float(np.mean(np.asarray(income, dtype=float) >= floor))


def above_threshold_halt_index(
    income: np.ndarray,
    income_floors: np.ndarray | None = None,
    epsilon: float = 0.5,
    rng: np.random.Generator | None = None,
    *,
    count_threshold_fraction: float = 0.5,
    descending: bool = False,
) -> float:
    """Run AboveThreshold on ordered income-floor count queries; return halt index.

    Uses count queries (sensitivity 1) with threshold τ = fraction · n.  This
    helper delegates to the course's AboveThreshold primitive.

    When *descending* is True, floors are queried high → low so the mechanism finds
    the highest floor whose count still clears τ (a meaningful sparse-vector search).

    Defaults to ``PUBLIC_ABOVE_THRESHOLD_FLOORS`` (finer spacing near τ) for teaching.
    """

    if rng is None:
        rng = np.random.default_rng()
    if income_floors is None:
        income_floors = PUBLIC_ABOVE_THRESHOLD_FLOORS
    n = len(income)
    count_threshold = count_threshold_fraction * n
    floors = income_floors[::-1] if descending else income_floors
    queries = [
        lambda db, floor=float(floor): count_above_floor(np.asarray(db), floor)
        for floor in floors
    ]
    responses = above_threshold(
        list(np.asarray(income, dtype=float)),
        queries,
        count_threshold,
        epsilon,
        rng,
    )
    return float(len(responses))


def assign_public_age_groups(age: np.ndarray) -> np.ndarray:
    """Bin ages into public groups defined by ``PUBLIC_AGE_BIN_EDGES``."""

    age = np.asarray(age, dtype=float)
    return np.digitize(age, PUBLIC_AGE_BIN_EDGES[1:-1], right=False)


def private_baseline_predictor(
    feature: np.ndarray,
    target: np.ndarray,
    transform_names: list[str],
    phi_fns: dict[str, Callable[[np.ndarray], np.ndarray]],
    *,
    candidates: np.ndarray | list,
    eps_preprocess: tuple[float, float, float],
    eps_select: float,
    eps_groups: float,
    rng: np.random.Generator,
    target_upper: float = PUBLIC_TARGET_UPPER,
) -> dict:
    """Minimal one-feature private baseline: clip → select φ → noisy group means.

    Pipeline (all public-menu choices fixed before data):
    1. ``private_quantile_clipped_mean`` on the target (preprocessing).
    2. ``report_noisy_max`` over public transform candidates (feature selection).
    3. ``private_group_means`` on public age bins (stump predictor aggregates).

    Returns accounting metadata alongside outputs for the composition slide.
    """

    eps_low, eps_high, eps_mean = eps_preprocess
    low_q, high_q = 0.01, 0.99
    pre = private_quantile_clipped_mean(
        target,
        candidates,
        low_q,
        high_q,
        eps_low,
        eps_high,
        eps_mean,
        rng,
    )
    clipped = np.clip(target, pre["L"], pre["U"])
    n = len(target)
    sens = selection_score_sensitivity(n)
    selected = report_noisy_max(
        transform_names,
        lambda name: bounded_feature_score(
            feature, clipped, phi_fns[name], target_upper=target_upper
        ),
        eps_select,
        sens,
        rng,
    )
    groups = assign_public_age_groups(feature)
    group_result = private_group_means(
        feature,
        clipped,
        groups,
        eps_groups,
        rng,
        target_range=pre["U"] - pre["L"],
    )
    return {
        "selected_transform": selected,
        "group_means": group_result["means"],
        "groups": group_result["groups"],
        "preprocess": pre,
        "group_release": group_result,
        "epsilon": {
            **pre["epsilon"],
            "select": eps_select,
            "groups": eps_groups,
        },
        "public_choices": [
            "candidate_grid",
            "transform_menu",
            "PUBLIC_AGE_BIN_EDGES",
            "MIN_CLIP_WIDTH",
        ],
    }


def above_threshold_oracle_table(
    income: np.ndarray,
    floors: np.ndarray | None = None,
    *,
    count_threshold_fraction: float = 0.5,
) -> list[dict]:
    """Oracle transcript for AboveThreshold: floor | exact count | τ | clears τ?"""

    if floors is None:
        floors = PUBLIC_ABOVE_THRESHOLD_FLOORS
    n = len(income)
    tau = count_threshold_fraction * n
    rows: list[dict] = []
    for floor in floors[::-1]:
        exact_count = count_above_floor(income, float(floor))
        rows.append(
            {
                "floor": float(floor),
                "exact_count": exact_count,
                "threshold_tau": tau,
                "clears_tau_exact": exact_count >= tau,
            }
        )
    return rows


def proposals_diagnoses_repairs_table() -> list[dict]:
    """Summary table for Lecture 5 (``tbl-proposals-diagnoses-repairs``)."""

    return [
        {
            "proposal": "Exact histogram EDA",
            "why_tempting": "understand the distribution",
            "witness": "row across a bin edge",
            "diagnosis": "non-private EDA",
            "repair": "fixed-bin noisy histogram (S7)",
        },
        {
            "proposal": "Raw noisy mean",
            "why_tempting": "the mean is the target",
            "witness": "extreme row",
            "diagnosis": "unbounded contribution",
            "repair": "clip / localize first",
        },
        {
            "proposal": "Empirical quantile clipping",
            "why_tempting": "bound the tails robustly",
            "witness": "constructed boundary/gap row",
            "diagnosis": "non-private thresholds",
            "repair": "private quantiles",
        },
        {
            "proposal": "Median value+noise",
            "why_tempting": "robust to outliers",
            "witness": "constructed median-gap row",
            "diagnosis": "full-range value sensitivity",
            "repair": "rank-error selection",
        },
        {
            "proposal": "Empirical μ±kσ clipping",
            "why_tempting": "familiar scale rule",
            "witness": "leverage row",
            "diagnosis": "non-private center/scale",
            "repair": "private truncated moments or log-sigma localization",
        },
        {
            "proposal": "Log-sigma histogram mean",
            "why_tempting": "exploit normal structure",
            "witness": "near-tie bin row",
            "diagnosis": "utility/model caveat",
            "repair": "account composition; validate assumption",
        },
    ]


def budget_ledger_table(ledger: dict) -> list[dict]:
    """Turn a pipeline budget dict into rows for a composition slide table."""

    eps = ledger.get("epsilon", {})
    rows = []
    for step, value in eps.items():
        rows.append({"step": step, "epsilon": value})
    rows.append({"step": "total", "epsilon": sum(eps.values())})
    return rows
