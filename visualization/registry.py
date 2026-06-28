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
from libdpy.visualization.roc_plots import (
    EmpiricalEpsilonFromDeltaVisualizer,
    TheoryROCVisualizer,
)
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
        "TheoryROCVisualizer": (
            lambda kwargs: TheoryROCVisualizer(**{"auto_display": False, **kwargs}).spec()
        ),
        "EmpiricalEpsilonFromDeltaVisualizer": (
            lambda kwargs: EmpiricalEpsilonFromDeltaVisualizer(
                **{"auto_display": False, **kwargs}
            ).spec()
        ),
        "NaiveSafeEpsilonHistogram": lambda kwargs: NaiveSafeEpsilonHistogram(**kwargs).spec(),
        "LaplaceComparison": lambda kwargs: LaplaceComparison(**kwargs).spec(),
        "ExponentialMechanismInteractive": (
            lambda kwargs: ExponentialMechanismInteractive(**kwargs).spec()
        ),
    }
