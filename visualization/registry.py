"""Registry of site-exportable interactive plot constructors."""

from __future__ import annotations

from collections.abc import Callable

from libdpy.assignment_specific.exponential_mechanism.visualizations import (
    ExponentialMechanismInteractive,
)
from libdpy.assignment_specific.private_estimation.embed_interactives import (
    PrivateEstimationAuditROCVisualizer,
)
from libdpy.assignment_specific.reconstruction.reconstruction_3d_visualization import (
    Reconstruction3DSlabsPlot,
)
from libdpy.assignment_specific.reconstruction.reconstruction_lecture_visualization import (
    Reconstruction2DSlabPlot,
)
from libdpy.assignment_specific.privacy_auditing.visualizations import (
    NaiveSafeEpsilonHistogram,
)
from libdpy.visualization.roc_plots import (
    EmpiricalEpsilonFromDeltaVisualizer,
    TheoryROCVisualizer,
)
from libdpy.visualization.interactive import InteractiveSpec
from libdpy.visualization.privacy_plots import PrivacyPlot
from libdpy.visualization.statistical_plots import LaplaceComparison

# Public names discovered by ``Ctor(...).embed()`` in website sources.
EMBED_CONSTRUCTOR_NAMES: tuple[str, ...] = (
    "PrivacyPlot",
    "TheoryROCVisualizer",
    "EmpiricalEpsilonFromDeltaVisualizer",
    "PrivateEstimationAuditROCVisualizer",
    "NaiveSafeEpsilonHistogram",
    "LaplaceComparison",
    "ExponentialMechanismInteractive",
    "Reconstruction2DSlabPlot",
    "Reconstruction3DSlabsPlot",
)


def embed_spec_builders() -> dict[str, Callable[[dict], InteractiveSpec]]:
    """Return builders that construct an ``InteractiveSpec`` from literal embed kwargs."""

    return {
        "PrivacyPlot": lambda kwargs: PrivacyPlot(**kwargs).spec(),
        "TheoryROCVisualizer": lambda kwargs: TheoryROCVisualizer(**kwargs).spec(),
        "EmpiricalEpsilonFromDeltaVisualizer": (
            lambda kwargs: EmpiricalEpsilonFromDeltaVisualizer(**kwargs).spec()
        ),
        "PrivateEstimationAuditROCVisualizer": (
            lambda kwargs: PrivateEstimationAuditROCVisualizer(**kwargs).spec()
        ),
        "NaiveSafeEpsilonHistogram": lambda kwargs: NaiveSafeEpsilonHistogram(**kwargs).spec(),
        "LaplaceComparison": lambda kwargs: LaplaceComparison(**kwargs).spec(),
        "ExponentialMechanismInteractive": (
            lambda kwargs: ExponentialMechanismInteractive(**kwargs).spec()
        ),
        "Reconstruction2DSlabPlot": lambda kwargs: Reconstruction2DSlabPlot(**kwargs).spec(),
        "Reconstruction3DSlabsPlot": lambda kwargs: Reconstruction3DSlabsPlot(**kwargs).spec(),
    }
