"""Precomputed, declarative ML weight-heatmap explorer."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from libdpy.visualization.interactive import (
    ControlSpec,
    InteractiveSpec,
    declarative_plotly_from_spec,
)


@dataclass(frozen=True)
class WeightHeatmapState:
    noise_factor: float
    image: np.ndarray
    loss: float
    accuracy: float


def precompute_weight_heatmap_states(
    noise_values: Sequence[float],
    *,
    evaluate: Callable,
    data,
    learning_params,
    clipping_methods,
) -> dict[float, WeightHeatmapState]:
    """Evaluate immutable states without mutating the supplied parameters."""

    states: dict[float, WeightHeatmapState] = {}
    for raw_noise in noise_values:
        noise = float(raw_noise)
        if noise < 0:
            raise ValueError("noise values must be nonnegative")
        parameters = copy.deepcopy(learning_params)
        parameters.noise_factor = noise
        weights, _predictions, loss, accuracy = evaluate(
            processed_data=data,
            learning_params=parameters,
            clipping_methods=clipping_methods,
            print_result=False,
        )
        image = np.asarray(weights[-1][:-1], dtype=float).reshape(28, 28).copy()
        image.setflags(write=False)
        states[noise] = WeightHeatmapState(
            noise_factor=noise,
            image=image,
            loss=float(loss),
            accuracy=float(accuracy),
        )
    return states


def make_weight_heatmap_figure(
    noise_factor: float,
    *,
    states_by_noise: Mapping[float, WeightHeatmapState],
) -> go.Figure:
    """Build one heatmap from an explicitly precomputed state."""

    if not states_by_noise:
        raise ValueError("states_by_noise must not be empty")
    key = min(states_by_noise, key=lambda value: abs(float(value) - noise_factor))
    state = states_by_noise[key]
    displayed_accuracy = 100 * state.accuracy if 0 <= state.accuracy <= 1 else state.accuracy
    figure = go.Figure(
        go.Heatmap(
            z=state.image,
            colorscale="gray",
            colorbar={"title": "Weight Values"},
            showscale=True,
        )
    )
    figure.update_layout(
        height=500,
        width=500,
        title=(
            f"Weight Visualization | Noise Factor: {state.noise_factor:.4f} | "
            f"Loss: {state.loss:.4f} | Accuracy: {displayed_accuracy:.2f}%"
        ),
    )
    return figure


def weight_heatmap_spec(
    states_by_noise: Mapping[float, WeightHeatmapState],
) -> InteractiveSpec:
    values = tuple(sorted(float(value) for value in states_by_noise))
    if not values:
        raise ValueError("states_by_noise must not be empty")
    step = min(np.diff(values)) if len(values) > 1 else 1.0
    return InteractiveSpec(
        name="ml_noise_weight_heatmap",
        artifact_name="ml-noise-weight-heatmap",
        controls=(
            ControlSpec(
                name="noise_factor",
                kind="slider",
                label="Noise Factor",
                default=values[0],
                min=values[0],
                max=values[-1],
                step=float(step),
                continuous=False,
            ),
        ),
        preferred_backend="plotly-declarative",
        allowed_backends=("plotly-declarative",),
        make_figure=make_weight_heatmap_figure,
        figure_factory=(
            "libdpy.assignment_specific.ml_and_overfitting.visualizations:"
            "make_weight_heatmap_figure"
        ),
        fixed_kwargs={"states_by_noise": states_by_noise},
    )


def create_weight_heatmap_slider(
    states_by_noise: Mapping[float, WeightHeatmapState],
    *,
    max_json_mb: float = 25.0,
) -> go.Figure:
    values = tuple(sorted(float(value) for value in states_by_noise))
    return declarative_plotly_from_spec(
        weight_heatmap_spec(states_by_noise),
        {"noise_factor": values},
        max_states=max(200, len(values)),
        max_json_mb=max_json_mb,
    )
