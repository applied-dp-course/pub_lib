"""Fixed audit sample scenes for website ``PrivateEstimationAuditROCVisualizer`` embeds."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from libdpy.assignment_specific.private_estimation.utils import (
    DEFAULT_K_SIGMA,
    DEFAULT_N_TRIALS_AUDIT,
    PUBLIC_INCOME_CANDIDATES,
    audit_panel,
    build_engineered_quantile_neighbor_pair,
    empirical_quantile_clipped_mean,
    extract_income,
    load_fulton,
    private_mu_sigma_clipped_mean,
    replace_one_row,
    split_base_and_pool,
)

_SCENE_NAMES = frozenset({"ms-repair-one", "ms-repair-three", "quantile-s3"})

_DEFAULT_SEED = 42
_DEFAULT_SAMPLE_SIZE = 1000
_DEFAULT_TARGET_EPS = 1.0
_DEFAULT_TARGET_DELTA = 1e-2
_DEFAULT_LOW_Q = 0.01
_DEFAULT_HIGH_Q = 0.99
_DEFAULT_K_SIGMA = DEFAULT_K_SIGMA
_PUBLIC_MS_LOWER = 0.0
_PUBLIC_MS_UPPER = float(PUBLIC_INCOME_CANDIDATES[-1])
_EPS_MS_LOCALIZE = 0.60
_EPS_MS_MEAN = 0.40
_MIN_PRIVATE_SIGMA = 2_500.0


def audit_embed_scene_names() -> frozenset[str]:
    """Return supported ``scene`` identifiers for static website embed discovery."""

    return _SCENE_NAMES


@lru_cache(maxsize=None)
def build_audit_embed_samples(
    scene: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(samples_neg, samples_pos, delta)`` for a fixed scene."""

    if scene not in _SCENE_NAMES:
        raise ValueError(f"unsupported audit embed scene: {scene!r}")

    income_all = extract_income(load_fulton())
    dataset, _ = split_base_and_pool(income_all, _DEFAULT_SAMPLE_SIZE, seed=_DEFAULT_SEED)

    if scene in {"ms-repair-one", "ms-repair-three"}:
        engineered_dataset = dataset.copy()
        out_idx = int(np.argmin(engineered_dataset))
        engineered_neighbor = replace_one_row(engineered_dataset, out_idx, 2_000_000.0)
        n_rounds = 1 if scene == "ms-repair-one" else 3
        seed_offset = 40 if scene == "ms-repair-one" else 41

        def mechanism(values: np.ndarray, rng: np.random.Generator) -> dict:
            return private_mu_sigma_clipped_mean(
                values,
                rng,
                eps_localize=_EPS_MS_LOCALIZE,
                eps_mean=_EPS_MS_MEAN,
                n_rounds=n_rounds,
                k=_DEFAULT_K_SIGMA,
                public_lower=_PUBLIC_MS_LOWER,
                public_upper=_PUBLIC_MS_UPPER,
                min_private_sigma=_MIN_PRIVATE_SIGMA,
            )

        panel = audit_panel(
            mechanism,
            engineered_dataset,
            engineered_neighbor,
            n_trials=DEFAULT_N_TRIALS_AUDIT,
            delta=_DEFAULT_TARGET_DELTA,
            seed=_DEFAULT_SEED + seed_offset,
            extractor=lambda result: result["estimate"],
            claimed_eps=_DEFAULT_TARGET_EPS,
            adversary_statistic="released private mu±4 sigma estimate",
        )
        return (
            panel.samples_neg,
            panel.samples_pos,
            _DEFAULT_TARGET_DELTA,
        )

    fabricated_dataset, fabricated_neighbor, _, _ = build_engineered_quantile_neighbor_pair(
        _DEFAULT_SAMPLE_SIZE
    )

    def quantile_mechanism(values: np.ndarray, rng: np.random.Generator) -> float:
        return empirical_quantile_clipped_mean(
            values,
            _DEFAULT_TARGET_EPS,
            _DEFAULT_LOW_Q,
            _DEFAULT_HIGH_Q,
            rng,
        )

    panel = audit_panel(
        quantile_mechanism,
        fabricated_dataset,
        fabricated_neighbor,
        n_trials=DEFAULT_N_TRIALS_AUDIT,
        delta=_DEFAULT_TARGET_DELTA,
        seed=_DEFAULT_SEED + 18,
        adversary_statistic="released empirical-quantile clipped mean",
    )
    return (
        panel.samples_neg,
        panel.samples_pos,
        _DEFAULT_TARGET_DELTA,
    )
