"""Backend-neutral specifications and renderers for interactive visualizations."""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import PurePosixPath
from typing import Any, Literal

import plotly.graph_objects as go

Backend = Literal["plotly-declarative", "wasm-marimo", "wasm-jupyterlite"]
ControlKind = Literal["slider", "select", "checkbox", "button_group"]

_VALID_BACKENDS = frozenset({"plotly-declarative", "wasm-marimo", "wasm-jupyterlite"})
_VALID_CONTROL_KINDS = frozenset({"slider", "select", "checkbox", "button_group"})
_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ControlSpec:
    """Describe one backend-neutral interactive control."""

    name: str
    kind: ControlKind
    label: str
    default: float | int | str | bool
    min: float | None = None
    max: float | None = None
    step: float | None = None
    values: Sequence[float | int | str | bool] | None = None
    continuous: bool = False

    def __post_init__(self) -> None:
        if not _PYTHON_IDENTIFIER.fullmatch(self.name):
            raise ValueError(f"control name must be a Python identifier: {self.name!r}")
        if self.kind not in _VALID_CONTROL_KINDS:
            raise ValueError(f"unsupported control kind: {self.kind!r}")
        if self.kind == "slider":
            if self.min is None or self.max is None or self.step is None:
                raise ValueError(f"slider {self.name!r} requires min, max, and step")
            if self.min > self.max:
                raise ValueError(f"slider {self.name!r} has min greater than max")
            if self.step <= 0:
                raise ValueError(f"slider {self.name!r} requires a positive step")
            if not self.min <= float(self.default) <= self.max:
                raise ValueError(f"slider {self.name!r} default is outside its range")
        elif self.values is None:
            raise ValueError(f"control {self.name!r} requires explicit values")


@dataclass(frozen=True)
class InteractiveSpec:
    """Own the computation and renderer metadata for an interactive."""

    name: str
    artifact_name: str
    controls: Sequence[ControlSpec]
    preferred_backend: Backend
    allowed_backends: Sequence[Backend]
    make_figure: Callable[..., object]
    figure_factory: str
    fixed_kwargs: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("interactive name must not be empty")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.artifact_name):
            raise ValueError(
                "artifact_name must contain lowercase letters, digits, and hyphens"
            )
        if self.preferred_backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unsupported preferred backend: {self.preferred_backend!r}"
            )
        if self.preferred_backend not in self.allowed_backends:
            raise ValueError("preferred_backend must be included in allowed_backends")
        if any(backend not in _VALID_BACKENDS for backend in self.allowed_backends):
            raise ValueError("allowed_backends contains an unsupported backend")
        names = [control.name for control in self.controls]
        if len(names) != len(set(names)):
            raise ValueError("control names must be unique")
        if ":" not in self.figure_factory:
            raise ValueError("figure_factory must use the 'module:function' format")

    def default_values(self) -> dict[str, float | int | str | bool]:
        return {control.name: control.default for control in self.controls}

    def default_figure(self) -> object:
        return self.make_figure(**self.default_values())


@dataclass(frozen=True)
class InteractiveEmbed:
    """HTML representation returned by ``InteractivePlot.embed()`` methods."""

    html: str

    def _repr_html_(self) -> str:
        return self.html

    def __str__(self) -> str:
        return self.html


# The iframe auto-loads (lazily, when scrolled near), and a loading overlay covers
# marimo's blank WASM boot screen until the widget actually renders. The overlay is
# hidden by this generic glue once the iframe's (same-origin) document contains a
# rendered <marimo-slider>, with a hard timeout fallback so it never sticks; a
# <noscript> rule hides the overlay when JavaScript is unavailable. No
# widget-specific computation lives here - it is purely presentational DOM glue.
_LOADING_WATCH = (
    "(function(f){"
    "var ov=f.parentNode.querySelector('.libdpy-interactive-loading');"
    "if(!ov){return;}"
    "var hide=function(){ov.style.display='none';};"
    "var ready=function(){"
    "var doc;try{doc=f.contentDocument;}catch(e){return true;}"
    "if(!doc){return false;}var found=false;"
    "(function walk(r){"
    "if(r.shadowRoot){walk(r.shadowRoot);}"
    "r.querySelectorAll('*').forEach(function(e){"
    "if(e.tagName&&e.tagName.toLowerCase()==='marimo-slider'){found=true;}"
    "if(e.shadowRoot){walk(e.shadowRoot);}});"
    "})(doc);return found;};"
    "var t=setInterval(function(){if(ready()){clearInterval(t);hide();}},400);"
    "setTimeout(function(){clearInterval(t);hide();},60000);"
    "})(this)"
)


def iframe_embed(
    spec: InteractiveSpec,
    *,
    src: str | None = None,
    height: int = 750,
    title: str | None = None,
) -> InteractiveEmbed:
    """Return an auto-loading iframe embed with a loading overlay."""

    if height <= 0:
        raise ValueError("height must be positive")
    iframe_src = escape(src or f"apps/{spec.artifact_name}/index.html", quote=True)
    iframe_title = escape(
        title or spec.description or spec.name.replace("_", " ").title(),
        quote=True,
    )
    name = escape(spec.name, quote=True)
    html = (
        f'<div class="libdpy-interactive" data-libdpy-interactive="{name}" '
        f'style="position:relative;min-height:{height}px;overflow:hidden;'
        'border:1px solid #d0d0d0;border-radius:6px;background:#fafafa;">'
        f'<iframe src="{iframe_src}" width="100%" height="{height}" loading="lazy" '
        f'title="{iframe_title}" data-libdpy-interactive="{name}" '
        f'style="border:0;width:100%;display:block;" '
        f'onload="{escape(_LOADING_WATCH, quote=True)}"></iframe>'
        '<div class="libdpy-interactive-loading" style="position:absolute;inset:0;'
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        'gap:0.6em;background:#fafafa;color:#555;pointer-events:none;">'
        '<div style="width:28px;height:28px;border:3px solid #ccc;'
        "border-top-color:#555;border-radius:50%;"
        'animation:libdpy-spin 0.9s linear infinite;"></div>'
        '<span style="font-size:0.85rem;">Loading interactive… '
        "(first load downloads a Python runtime)</span></div>"
        "<style>@keyframes libdpy-spin{to{transform:rotate(360deg)}}</style>"
        "<noscript><style>.libdpy-interactive-loading{display:none}</style></noscript>"
        "</div>"
    )
    return InteractiveEmbed(html)


def declarative_plotly_from_spec(
    spec: InteractiveSpec,
    grid: Mapping[str, Sequence[float | int | str | bool]],
    *,
    max_states: int = 200,
    max_json_mb: float = 5.0,
) -> go.Figure:
    """Precompute a one-control finite state space as a static Plotly figure.

    Multi-control state spaces deliberately fail closed. They require a renderer
    that can preserve independent controls, normally ``wasm-marimo``.
    """

    if "plotly-declarative" not in spec.allowed_backends:
        raise ValueError(f"{spec.name} does not allow the plotly-declarative backend")
    if len(grid) != 1:
        state_count = math.prod(len(values) for values in grid.values())
        raise ValueError(
            f"{spec.name} cannot be rendered with plotly-declarative: "
            f"{len(grid)} controls imply {state_count} precomputed states. "
            "Use wasm-marimo or provide a single explicit pedagogical discretization."
        )

    control_name, raw_values = next(iter(grid.items()))
    controls_by_name = {control.name: control for control in spec.controls}
    if control_name not in controls_by_name:
        raise ValueError(f"unknown control for {spec.name}: {control_name!r}")
    values = list(raw_values)
    if not values:
        raise ValueError(f"grid for {control_name!r} must not be empty")
    if len(values) > max_states:
        raise ValueError(
            f"{spec.name} cannot be rendered with plotly-declarative: "
            f"{len(values)} states exceed max_states={max_states}. Use wasm-marimo."
        )

    defaults = spec.default_values()
    figures: list[go.Figure] = []
    for value in values:
        parameters = {**defaults, control_name: value}
        figure = spec.make_figure(**parameters)
        if not isinstance(figure, go.Figure):
            raise TypeError(
                "plotly-declarative figure factories must return plotly.go.Figure"
            )
        figures.append(figure)

    trace_count = len(figures[0].data)
    if any(len(figure.data) != trace_count for figure in figures):
        raise ValueError(
            "figure factory must return the same trace count for every state"
        )

    frame_names = [f"{control_name}-{index}" for index in range(len(values))]
    result = go.Figure(
        data=figures[0].data,
        layout=figures[0].layout,
        frames=[
            go.Frame(name=frame_name, data=figure.data, traces=list(range(trace_count)))
            for frame_name, figure in zip(frame_names, figures)
        ],
    )
    result.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": f"{controls_by_name[control_name].label}: "},
                "pad": {"t": 45},
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [frame_name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": (
                            f"{value:g}"
                            if isinstance(value, (int, float))
                            else str(value)
                        ),
                    }
                    for frame_name, value in zip(frame_names, values)
                ],
            }
        ]
    )

    json_mb = len(result.to_json().encode("utf-8")) / (1024 * 1024)
    if json_mb > max_json_mb:
        raise ValueError(
            f"{spec.name} cannot be rendered with plotly-declarative: "
            f"generated Plotly JSON is {json_mb:.2f} MB, above max_json_mb={max_json_mb}. "
            "Use wasm-marimo."
        )
    return result


def marimo_app_source(
    spec: InteractiveSpec,
    *,
    wheel_filename: str,
    marimo_version: str = "0.23.9",
) -> str:
    """Generate an import-only marimo app for a WASM export."""

    if "wasm-marimo" not in spec.allowed_backends:
        raise ValueError(f"{spec.name} does not allow the wasm-marimo backend")
    if PurePosixPath(
        wheel_filename
    ).name != wheel_filename or not wheel_filename.endswith(".whl"):
        raise ValueError("wheel_filename must be a wheel basename")
    unsupported = [
        control.kind for control in spec.controls if control.kind != "slider"
    ]
    if unsupported:
        raise NotImplementedError(
            "the initial marimo renderer supports sliders only; "
            f"unsupported controls: {sorted(set(unsupported))}"
        )

    slider_lines = []
    control_names = []
    for control in spec.controls:
        control_names.append(control.name)
        slider_lines.append(
            f"    {control.name} = mo.ui.slider("
            f"start={control.min!r}, stop={control.max!r}, step={control.step!r}, "
            f"value={control.default!r}, label={control.label!r}, "
            "show_value=True, full_width=True)"
        )
    sliders = "\n".join(slider_lines)
    returns = ", ".join(control_names)
    if len(control_names) == 1:
        returns += ","
    figure_args = ", ".join(f"{name}={name}.value" for name in control_names)
    fixed_kwargs = json.dumps(dict(spec.fixed_kwargs), sort_keys=True)
    module_name, function_name = spec.figure_factory.split(":", maxsplit=1)

    return f"""# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo=={marimo_version}",
# ]
# ///

import marimo

__generated_with = "{marimo_version}"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _(mo):
    import sys

    if sys.platform == "emscripten":
        import micropip

        wheel_url = str(mo.notebook_location() / "public" / {wheel_filename!r})
        await micropip.install(wheel_url, deps=False)
    libdpy_ready = True
    return (libdpy_ready,)


@app.cell
def _(libdpy_ready):
    import importlib

    assert libdpy_ready
    make_figure = getattr(importlib.import_module({module_name!r}), {function_name!r})
    fixed_kwargs = {fixed_kwargs}
    return fixed_kwargs, make_figure


@app.cell
def _(mo):
{sliders}
    controls = mo.hstack([{", ".join(control_names)}], widths="equal", gap=2)
    return controls, {returns}


@app.cell
def _(controls):
    controls
    return


@app.cell
def _({", ".join(control_names)}, fixed_kwargs, make_figure):
    figure = make_figure({figure_args}, **fixed_kwargs)
    figure
    return


if __name__ == "__main__":
    app.run()
"""


def state_count(grid: Mapping[str, Sequence[Any]]) -> int:
    """Return the number of states in a finite declarative grid."""

    return math.prod(len(values) for values in grid.values())


def iter_states(
    grid: Mapping[str, Sequence[Any]],
) -> Sequence[dict[str, Any]]:
    """Materialize finite state dictionaries in deterministic key order."""

    names = list(grid)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(grid[name] for name in names))
    ]
