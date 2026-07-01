"""Site-exportable interactives for the private-estimation lecture."""

from __future__ import annotations

from libdpy.assignment_specific.private_estimation.audit_embed_scenes import (
    audit_embed_scene_names,
    build_audit_embed_samples,
)
from libdpy.visualization.interactive import AbstractInteractivePlot, InteractiveSpec
from libdpy.visualization.roc_plots import empirical_roc_from_samples_spec


class PrivateEstimationAuditROCVisualizer(AbstractInteractivePlot):
    """Empirical ROC explorer for a fixed private-estimation audit scene."""

    def __init__(self, scene: str):
        if scene not in audit_embed_scene_names():
            supported = ", ".join(sorted(audit_embed_scene_names()))
            raise ValueError(f"scene must be one of: {supported}")
        self.scene = scene

    def spec(self) -> InteractiveSpec:
        samples_neg, samples_pos, delta = build_audit_embed_samples(
            self.scene
        )
        return empirical_roc_from_samples_spec(
            samples_neg,
            samples_pos,
            delta=delta,
            compute_epsilon=False,
            show_compute_epsilon_toggle=True,
        )
