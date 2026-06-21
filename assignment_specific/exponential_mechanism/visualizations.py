"""Reusable exponential-mechanism probability explorer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from IPython.display import display

from libdpy.visualization.interactive import ControlSpec, InteractiveSpec
from libdpy.visualization.interactive_widgets import render_ipywidgets


def exponential_mechanism_probabilities(
    epsilon: float,
    *,
    utilities: Sequence[float],
    sensitivity: float,
) -> np.ndarray:
    """Return numerically stable exact exponential-mechanism probabilities."""

    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    utility_array = np.asarray(utilities, dtype=float)
    if utility_array.ndim != 1 or len(utility_array) == 0:
        raise ValueError("utilities must be a nonempty one-dimensional sequence")
    logits = epsilon * utility_array / (2 * sensitivity)
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def make_exponential_mechanism_figure(
    epsilon: float,
    *,
    utilities: Sequence[float],
    sensitivity: float,
    labels: Sequence[str] | None = None,
) -> go.Figure:
    """Compare exact selection probabilities with normalized utility."""

    utility_array = np.asarray(utilities, dtype=float)
    probabilities = exponential_mechanism_probabilities(
        epsilon,
        utilities=utility_array,
        sensitivity=sensitivity,
    )
    utility_total = float(np.sum(utility_array))
    normalized_utilities = (
        utility_array / utility_total
        if utility_total > 0
        else np.full(len(utility_array), 1 / len(utility_array))
    )
    order = np.argsort(-utility_array)
    resolved_labels = (
        np.asarray(labels, dtype=object)
        if labels is not None
        else np.asarray([f"Pairing {index}" for index in range(len(utility_array))])
    )
    if len(resolved_labels) != len(utility_array):
        raise ValueError("labels and utilities must have the same length")

    figure = go.Figure(
        [
            go.Bar(
                x=resolved_labels[order],
                y=probabilities[order],
                name="Selection probability",
            ),
            go.Bar(
                x=resolved_labels[order],
                y=normalized_utilities[order],
                name="Normalized utility",
            ),
        ]
    )
    figure.update_layout(
        barmode="group",
        title=f"Exact Exponential-Mechanism Probabilities (ε={epsilon:.2f})",
        xaxis_title="Responses (sorted by utility)",
        yaxis_title="Normalized value",
        xaxis_tickangle=-45,
        height=520,
    )
    return figure


def exponential_mechanism_spec(
    utilities: Sequence[float],
    *,
    sensitivity: float,
    labels: Sequence[str] | None = None,
) -> InteractiveSpec:
    return InteractiveSpec(
        name="exponential_mechanism",
        artifact_name="exponential-mechanism",
        controls=(
            ControlSpec(
                name="epsilon",
                kind="slider",
                label="Epsilon",
                default=5.0,
                min=0.1,
                max=10.0,
                step=0.1,
            ),
        ),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_exponential_mechanism_figure,
        figure_factory=(
            "libdpy.assignment_specific.exponential_mechanism.visualizations:"
            "make_exponential_mechanism_figure"
        ),
        fixed_kwargs={
            "utilities": tuple(float(value) for value in utilities),
            "sensitivity": sensitivity,
            "labels": tuple(labels) if labels is not None else None,
        },
    )


def create_exponential_mechanism_interactive(
    utilities: Sequence[float],
    *,
    sensitivity: float,
    labels: Sequence[str] | None = None,
):
    rendered = render_ipywidgets(
        exponential_mechanism_spec(
            utilities,
            sensitivity=sensitivity,
            labels=labels,
        )
    )
    display(rendered.root)
    return rendered.root
