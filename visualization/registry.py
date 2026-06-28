"""Registry of site-exportable interactive plot constructors."""

from __future__ import annotations

from collections.abc import Callable

from libdpy.assignment_specific.exponential_mechanism.visualizations import (
    ExponentialMechanismInteractive,
)
from libdpy.assignment_specific.privacy_auditing.visualizations import (
    NaiveSafeEpsilonHistogram,
)
from libdpy.visualization.interactive import InteractiveSpec
from libdpy.visualization.privacy_plots import PrivacyPlot
from libdpy.visualization.roc_plots import empirical_roc_spec, theory_roc_spec
from libdpy.visualization.statistical_plots import LaplaceComparison

# Public names discovered by ``Ctor(...).embed()`` in website sources.
EMBED_CONSTRUCTOR_NAMES: tuple[str, ...] = (
    "PrivacyPlot",
    "TheoryROCVisualizer",
    "EmpiricalEpsilonFromDeltaVisualizer",
    "NaiveSafeEpsilonHistogram",
    "LaplaceComparison",
    "ExponentialMechanismInteractive",
)


def embed_spec_builders() -> dict[str, Callable[[dict], InteractiveSpec]]:
    """Return builders that construct an ``InteractiveSpec`` from literal embed kwargs."""

    return {
        "PrivacyPlot": lambda kwargs: PrivacyPlot(**kwargs).spec(),
        "TheoryROCVisualizer": lambda kwargs: theory_roc_spec(**kwargs),
        "EmpiricalEpsilonFromDeltaVisualizer": lambda kwargs: empirical_roc_spec(
            **{
                "compute_epsilon": False,
                "show_compute_epsilon_toggle": True,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key != "random_seed"
                },
                **(
                    {"sample_seed": kwargs["random_seed"]}
                    if "random_seed" in kwargs
                    else {}
                ),
            }
        ),
        "NaiveSafeEpsilonHistogram": lambda kwargs: NaiveSafeEpsilonHistogram(**kwargs).spec(),
        "LaplaceComparison": lambda kwargs: LaplaceComparison(**kwargs).spec(),
        "ExponentialMechanismInteractive": (
            lambda kwargs: ExponentialMechanismInteractive(**kwargs).spec()
        ),
    }
