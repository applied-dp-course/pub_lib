from functools import lru_cache
from typing import Callable

import math
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from sklearn.metrics import roc_curve

from libdpy.hypothesis_testing import (
    FPR_from_threshold,
    TPR_from_threshold,
    threshold_from_FPR,
)
from libdpy.privacy_mechanisms.noise import sample_outputs, two_logistic_noise


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """
    This function returns a lower bound of the success probability using Clopper Pearson method.
    :param successes: number of successes
    :param trials: number of trials
    :param alpha: the probability not to get a lower bound
    :return: the lower bound of the success probability
    """
    return beta.ppf(alpha, successes, trials - successes + 1)


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """This function returns an upper bound of the success probability using Clopper Pearson method."""
    return beta.ppf(1 - alpha, successes + 1, trials - successes)


def legacy_algorithm(database: list, rng: np.random.Generator | None = None) -> float:
    """secret legacy algorithm"""
    if rng is None:
        rng = np.random.default_rng()
    value = database[-1]
    trimmed_value = min(5, max(4, value))
    noised_value = trimmed_value + rng.laplace(scale=1 / 3)
    return noised_value


def plot_outputs_histograms(algorithm: Callable, db0, db1, repetitions_number: int):
    results_0 = [algorithm(db0) for _ in range(repetitions_number)]
    results_1 = [algorithm(db1) for _ in range(repetitions_number)]
    plt.figure(figsize=(8, 6))
    plt.hist(
        results_0,
        bins=100,
        density=True,
        color='b',
        edgecolor='black',
        label=f'Mean: {np.mean(results_0)}\nStandard deviation: {np.std(results_0)}',
    )
    plt.hist(
        results_1,
        bins=100,
        density=True,
        color='y',
        edgecolor='yellow',
        label=f'Mean: {np.mean(results_1)}\nStandard deviation: {np.std(results_1)}',
    )

    plt.title("Histograms of results")
    plt.xlabel("result")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return


def epsilon_from_roc_point(fpr: float, tpr: float, delta: float) -> float:
    """One-sided plug-in lower bound from a single ROC point."""

    if tpr <= delta:
        return 0.0
    if fpr <= 0.0:
        return float("inf")
    if tpr - delta <= fpr:
        return 0.0
    return float(math.log((tpr - delta) / fpr))


def true_epsilon_gaussian_threshold(
    tau: float,
    delta: float,
    *,
    scale: float = 1.0,
    mu_neg: float = 0.0,
    mu_pos: float = 1.0,
) -> float:
    """Return the exact event-specific ``epsilon`` for shifted Gaussians and ``x > tau``."""

    from scipy.stats import norm

    dist_neg = norm(loc=mu_neg, scale=scale)
    dist_pos = norm(loc=mu_pos, scale=scale)
    fpr = float(FPR_from_threshold(dist_neg, tau))
    tpr = float(TPR_from_threshold(dist_pos, tau))
    return epsilon_from_roc_point(fpr, tpr, delta)


def run_repeated_gaussian_audits(
    tau: float,
    delta: float,
    *,
    scale: float = 1.0,
    mu_neg: float = 0.0,
    mu_pos: float = 1.0,
    n_audit: int = 500,
    n_repeats: int = 200,
    alpha_total: float = 0.05,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Return plug-in and safe epsilon values from repeated Gaussian audits."""

    rng = np.random.default_rng(seed)
    plug_values: list[float] = []
    safe_values: list[float] = []
    for _ in range(n_repeats):
        neg = rng.normal(mu_neg, scale, size=n_audit)
        pos = rng.normal(mu_pos, scale, size=n_audit)
        result = audit_point(neg, pos, tau, delta, alpha_total=alpha_total)
        plug_values.append(result.eps_plug)
        safe_values.append(result.eps_audit if result.eps_audit is not None else 0.0)
    return plug_values, safe_values


def empirical_roc(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return empirical ROC arrays from negative and positive samples."""

    samples_neg = np.asarray(samples_neg, dtype=float)
    samples_pos = np.asarray(samples_pos, dtype=float)
    labels = np.concatenate(
        [
            np.zeros(len(samples_neg), dtype=int),
            np.ones(len(samples_pos), dtype=int),
        ]
    )
    scores = np.concatenate([samples_neg, samples_pos])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def roc_point_at_threshold(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau: float,
) -> tuple[float, float]:
    """Return ``(FPR, TPR)`` for the strict event ``{x > tau}``."""

    n0 = len(samples_neg)
    n1 = len(samples_pos)
    if n0 == 0 or n1 == 0:
        raise ValueError("samples_neg and samples_pos must be non-empty")
    fpr = float(np.sum(samples_neg > tau) / n0)
    tpr = float(np.sum(samples_pos > tau) / n1)
    return fpr, tpr


def threshold_candidates(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
) -> np.ndarray:
    """Return candidate thresholds for the strict rule ``x > tau``."""

    values = np.sort(np.unique(np.concatenate([samples_neg, samples_pos])))
    if len(values) == 0:
        return np.array([0.0])
    candidates = [float(values[0] - 1.0)]
    if len(values) > 1:
        candidates.extend(0.5 * (values[:-1] + values[1:]))
    candidates.extend([float(values[-1]), float(values[-1] + 1.0)])
    return np.asarray(candidates, dtype=float)


def selected_threshold_from_empirical_roc(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    delta: float,
) -> tuple[float, tuple[float, float], float]:
    """Return ``(tau_star, (fpr, tpr), eps_plug)`` from one-sided empirical ROC points."""

    samples_neg = np.asarray(samples_neg, dtype=float)
    samples_pos = np.asarray(samples_pos, dtype=float)

    best_eps = -float("inf")
    best_tau = 0.0
    best_fpr = 0.0
    best_tpr = 0.0

    for tau in threshold_candidates(samples_neg, samples_pos):
        point_fpr, point_tpr = roc_point_at_threshold(samples_neg, samples_pos, float(tau))
        eps = epsilon_from_roc_point(point_fpr, point_tpr, delta)
        if eps > best_eps:
            best_eps = eps
            best_tau = float(tau)
            best_fpr = point_fpr
            best_tpr = point_tpr

    return best_tau, (best_fpr, best_tpr), best_eps


def one_sided_epsilon_from_roc_points(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, tuple[float, float] | None]:
    """Return the largest one-sided epsilon requirement among ROC points."""

    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")
    if len(fpr) == 0:
        raise ValueError("fpr and tpr must not be empty")

    best_eps = 0.0
    best_point: tuple[float, float] | None = None
    for point_fpr, point_tpr in zip(fpr, tpr):
        eps = epsilon_from_roc_point(float(point_fpr), float(point_tpr), delta)
        if eps > best_eps:
            best_eps = eps
            best_point = (float(point_fpr), float(point_tpr))
    return best_eps, best_point


def confusion_counts(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau: float,
) -> tuple[int, int, int, int]:
    """Return ``(FP, TN, TP, FN)`` for the event ``{x > tau}``."""

    samples_neg = np.asarray(samples_neg, dtype=float)
    samples_pos = np.asarray(samples_pos, dtype=float)
    fp = int(np.sum(samples_neg > tau))
    tn = int(len(samples_neg) - fp)
    tp = int(np.sum(samples_pos > tau))
    fn = int(len(samples_pos) - tp)
    return fp, tn, tp, fn


def safe_clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    return float(clopper_pearson_lower(successes, trials, alpha))


def safe_clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    return float(clopper_pearson_upper(successes, trials, alpha))


@dataclass(frozen=True)
class AuditResult:
    fp: int
    tn: int
    tp: int
    fn: int
    n0: int
    n1: int
    fpr_hat: float
    tpr_hat: float
    eps_plug: float
    tau_star: float
    delta: float
    alpha_total: float | None = None
    fpr_l: float | None = None
    fpr_u: float | None = None
    tpr_l: float | None = None
    tpr_u: float | None = None
    eps_audit: float | None = None
    no_positive_evidence: bool = False
    plug_in_infinity: bool = False


def audit_point(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau_star: float,
    delta: float,
    alpha_total: float | None = None,
) -> AuditResult:
    """Audit a fixed threshold event on fresh samples."""

    fp, tn, tp, fn = confusion_counts(samples_neg, samples_pos, tau_star)
    n0 = fp + tn
    n1 = tp + fn
    fpr_hat = fp / n0 if n0 else 0.0
    tpr_hat = tp / n1 if n1 else 0.0
    eps_plug = epsilon_from_roc_point(fpr_hat, tpr_hat, delta)
    no_positive_evidence = tpr_hat <= delta
    plug_in_infinity = fpr_hat <= 0.0 and tpr_hat > delta

    fpr_l = None
    fpr_u = None
    tpr_l = None
    tpr_u = None
    eps_audit = None
    if alpha_total is not None:
        alpha_each = alpha_total / 2.0
        fpr_l = safe_clopper_pearson_lower(fp, n0, alpha_each)
        fpr_u = safe_clopper_pearson_upper(fp, n0, alpha_each)
        tpr_l = safe_clopper_pearson_lower(tp, n1, alpha_each)
        tpr_u = safe_clopper_pearson_upper(tp, n1, alpha_each)
        eps_audit = epsilon_from_roc_point(fpr_u, tpr_l, delta)

    return AuditResult(
        fp=fp,
        tn=tn,
        tp=tp,
        fn=fn,
        n0=n0,
        n1=n1,
        fpr_hat=fpr_hat,
        tpr_hat=tpr_hat,
        eps_plug=eps_plug,
        tau_star=tau_star,
        delta=delta,
        alpha_total=alpha_total,
        fpr_l=fpr_l,
        fpr_u=fpr_u,
        tpr_l=tpr_l,
        tpr_u=tpr_u,
        eps_audit=eps_audit,
        no_positive_evidence=no_positive_evidence,
        plug_in_infinity=plug_in_infinity,
    )


def reference_epsilon_two_logistic(
    tau_star: float,
    delta: float,
    *,
    scale1: float = 1.0,
    scale2: float = 0.4,
    n_reference: int = 500_000,
    seed: int = 789,
) -> float:
    """High-precision event-specific reference from a large Monte Carlo sample."""

    _, _, epsilon = reference_point_two_logistic(
        tau_star,
        delta,
        scale1=scale1,
        scale2=scale2,
        n_reference=n_reference,
        seed=seed,
    )
    return epsilon


@lru_cache(maxsize=32)
def _reference_point_two_logistic_cached(
    tau_star: float,
    delta: float,
    scale1: float,
    scale2: float,
    n_reference: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    def sampler(size, rng=rng):
        return two_logistic_noise(size=size, scale1=scale1, scale2=scale2, rng=rng)

    samples_neg = sample_outputs(sampler, mu=0.0, n=n_reference, rng=rng)
    samples_pos = sample_outputs(sampler, mu=1.0, n=n_reference, rng=rng)
    fpr, tpr = roc_point_at_threshold(samples_neg, samples_pos, tau_star)
    return fpr, tpr, epsilon_from_roc_point(fpr, tpr, delta)


def reference_point_two_logistic(
    tau_star: float,
    delta: float,
    *,
    scale1: float = 1.0,
    scale2: float = 0.4,
    n_reference: int = 500_000,
    seed: int = 789,
) -> tuple[float, float, float]:
    """Return ``(fpr, tpr, epsilon)`` from a large Monte Carlo reference sample."""

    return _reference_point_two_logistic_cached(
        float(tau_star),
        float(delta),
        float(scale1),
        float(scale2),
        int(n_reference),
        int(seed),
    )


def plug_in_epsilon_from_empirical_roc(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    delta: float,
) -> float:
    """Return the one-sided plug-in epsilon from the full empirical ROC."""

    fpr, tpr, _ = empirical_roc(samples_neg, samples_pos)
    epsilon, _ = one_sided_epsilon_from_roc_points(fpr, tpr, delta)
    return epsilon


AuditFrameState = tuple[int, int, int, int, float, float]


def balanced_audit_sequence(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shuffled balanced ``(labels, values)`` for ``n_frames`` audit steps."""

    per_class = n_frames // 2
    labels = np.concatenate(
        [np.zeros(per_class, dtype=int), np.ones(per_class, dtype=int)]
    )
    values = np.concatenate([samples_neg[:per_class], samples_pos[:per_class]])
    order = rng.permutation(n_frames)
    return labels[order], values[order]


def fixed_threshold_audit_state_sequence(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau_star: float,
    *,
    n_frames: int = 200,
    seed: int = 0,
    labels: np.ndarray | None = None,
    values: np.ndarray | None = None,
) -> list[AuditFrameState]:
    """Return per-frame ``(TN, FN, FP, TP, FPR, TPR)`` for a fixed-threshold audit."""

    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if n_frames % 2 != 0:
        raise ValueError("n_frames must be even for balanced H0/H1 interleaving")

    samples_neg = np.asarray(samples_neg, dtype=float)
    samples_pos = np.asarray(samples_pos, dtype=float)
    per_class = n_frames // 2
    if len(samples_neg) < per_class or len(samples_pos) < per_class:
        raise ValueError("samples_neg and samples_pos must each contain at least n_frames // 2 values")

    if labels is None or values is None:
        labels, values = balanced_audit_sequence(
            samples_neg,
            samples_pos,
            n_frames,
            np.random.default_rng(seed),
        )

    tn = fn = fp = tp = 0
    states: list[AuditFrameState] = []
    for frame_index in range(n_frames):
        label = int(labels[frame_index])
        value = float(values[frame_index])
        positive_guess = value > tau_star
        if label == 0:
            if positive_guess:
                fp += 1
            else:
                tn += 1
        elif positive_guess:
            tp += 1
        else:
            fn += 1

        n_neg = tn + fp
        n_pos = fn + tp
        fpr = fp / n_neg if n_neg else 0.0
        tpr = tp / n_pos if n_pos else 0.0
        states.append((tn, fn, fp, tp, fpr, tpr))

    return states
