"""Backend-neutral specifications and renderers for interactive visualizations."""

from __future__ import annotations

import itertools
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import PurePosixPath
from typing import Any, Literal, MutableMapping

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

Backend = Literal["ipywidgets", "plotly-declarative", "wasm-marimo"]
ControlKind = Literal["slider", "select", "checkbox", "button_group", "toggle_button"]
SliderScale = Literal["linear", "log"]

_VALID_BACKENDS = frozenset({"ipywidgets", "plotly-declarative", "wasm-marimo"})
_MARIMO_SUPPORTED_CONTROL_KINDS = frozenset(
    {"slider", "select", "button_group", "checkbox", "toggle_button"}
)


def marimo_supported_control_kinds() -> frozenset[str]:
    """Return control kinds that the marimo WASM exporter can render today."""

    return _MARIMO_SUPPORTED_CONTROL_KINDS
_VALID_CONTROL_KINDS = frozenset(
    {"slider", "select", "checkbox", "button_group", "toggle_button"}
)
_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


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
    continuous: bool = True
    slider_scale: SliderScale = "linear"
    readout_format: str | None = None
    readout_formatter: Callable[[float], str] | None = None
    throttle_ms: int | None = None
    description: str = ""

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
            if self.slider_scale not in {"linear", "log"}:
                raise ValueError(
                    f"slider {self.name!r} has unsupported scale {self.slider_scale!r}"
                )
            if self.slider_scale == "log" and (
                self.min <= 0 or self.max <= 0 or float(self.default) <= 0
            ):
                raise ValueError(f"log slider {self.name!r} requires positive values")
        elif self.kind in {"select", "button_group"}:
            if self.values is None:
                raise ValueError(f"control {self.name!r} requires explicit values")
            if self.default not in self.values:
                raise ValueError(f"control {self.name!r} default is not one of its values")
        elif self.kind in {"checkbox", "toggle_button"} and not isinstance(self.default, bool):
            raise ValueError(f"{self.kind} {self.name!r} requires a boolean default")
        if self.throttle_ms is not None and self.throttle_ms < 0:
            raise ValueError(f"control {self.name!r} throttle_ms must be nonnegative")


@dataclass(frozen=True)
class ActionSpec:
    """Describe a state-changing action for an interactive notebook widget."""

    name: str
    label: str
    handler: Callable[
        [MutableMapping[str, Any], Mapping[str, float | int | str | bool]],
        Mapping[str, Any] | None,
    ]
    button_style: str = ""
    state_updates: Mapping[str, int] = field(default_factory=dict)
    """Declarative semantics for kernel-free renderers (e.g. ``wasm-marimo``).

    Maps a state key to the integer delta applied each time the action fires. The
    live ``ipywidgets`` renderer uses ``handler`` instead; the WASM renderer cannot
    serialize an arbitrary callable, so it relies on this declaration to model the
    action as a button accumulator. Both current actions just advance a seed by one.
    """

    def __post_init__(self) -> None:
        if not _PYTHON_IDENTIFIER.fullmatch(self.name):
            raise ValueError(f"action name must be a Python identifier: {self.name!r}")
        if not self.label:
            raise ValueError("action label must not be empty")
        for key, delta in self.state_updates.items():
            if not _PYTHON_IDENTIFIER.fullmatch(key):
                raise ValueError(f"action {self.name!r} state key must be an identifier: {key!r}")
            if not isinstance(delta, int) or isinstance(delta, bool):
                raise ValueError(f"action {self.name!r} state delta must be an int: {key!r}={delta!r}")


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
    actions: Sequence[ActionSpec] = ()
    initial_state: Mapping[str, Any] | Callable[[], Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("interactive name must not be empty")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.artifact_name):
            raise ValueError("artifact_name must contain lowercase letters, digits, and hyphens")
        if self.preferred_backend not in _VALID_BACKENDS:
            raise ValueError(f"unsupported preferred backend: {self.preferred_backend!r}")
        if self.preferred_backend not in self.allowed_backends:
            raise ValueError("preferred_backend must be included in allowed_backends")
        if any(backend not in _VALID_BACKENDS for backend in self.allowed_backends):
            raise ValueError("allowed_backends contains an unsupported backend")
        names = [control.name for control in self.controls]
        if len(names) != len(set(names)):
            raise ValueError("control names must be unique")
        action_names = [action.name for action in self.actions]
        if len(action_names) != len(set(action_names)):
            raise ValueError("action names must be unique")
        if set(names) & set(action_names):
            raise ValueError("control and action names must be distinct")
        if ":" not in self.figure_factory:
            raise ValueError("figure_factory must use the 'module:function' format")

    def default_values(self) -> dict[str, float | int | str | bool]:
        return {control.name: control.default for control in self.controls}

    def new_state(self) -> dict[str, Any]:
        initial = self.initial_state() if callable(self.initial_state) else self.initial_state
        return dict(initial)

    def figure_kwargs(
        self,
        control_values: Mapping[str, float | int | str | bool] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            **dict(self.fixed_kwargs),
            **(self.default_values() if control_values is None else dict(control_values)),
            **(self.new_state() if state is None else dict(state)),
        }

    def default_figure(self) -> object:
        return self.make_figure(**self.figure_kwargs())


class AbstractInteractivePlot(ABC):
    """Base class for plot-specific wrappers backed by an ``InteractiveSpec``."""

    @abstractmethod
    def spec(self) -> InteractiveSpec:
        """Return the backend-neutral specification for this plot."""

    def build_spec(self) -> InteractiveSpec:
        """Alias used by the refactor plan and newer plot wrappers."""

        return self.spec()

    def widget(self, **renderer_options):
        """Build the live notebook widget through the shared renderer."""

        from .interactive_widgets import render_ipywidgets

        return render_ipywidgets(self.build_spec(), **renderer_options)

    def figure(self, **control_values) -> go.Figure:
        """Build a concrete figure without creating or displaying widgets."""

        spec = self.build_spec()
        values = {**spec.default_values(), **control_values}
        figure = spec.make_figure(**spec.figure_kwargs(values))
        if not isinstance(figure, go.Figure):
            raise TypeError("Plotly interactive figure factories must return plotly.go.Figure")
        return figure

    def show(self, **renderer_options):
        """Display and return the live notebook widget."""

        from IPython.display import display

        rendered = self.widget(**renderer_options)
        display(rendered.root)
        return rendered.root

    def embed(
        self,
        *,
        height: int | None = None,
        mode: str = "page",
        src: str | None = None,
    ) -> "InteractiveEmbed | Any":
        """Return a lazy iframe for the generated WASM artifact of this plot.

        Used by the static site build: the website discovers ``Plot(...).embed()``
        calls and exports each spec to a self-contained marimo WASM app, so the
        interactive keeps working with no live kernel. In Colab, ``embed`` falls
        back to :meth:`show` so the same uploaded notebook runs with live widgets.
        """

        if mode not in {"page", "deck"}:
            raise ValueError("mode must be 'page' or 'deck'")
        if _running_in_colab():
            return self.show()
        resolved_height = height if height is not None else (620 if mode == "deck" else 750)
        return iframe_embed(self.spec(), src=src, height=resolved_height)


class SpecInteractivePlot(AbstractInteractivePlot):
    """Concrete adapter for callers that already have an ``InteractiveSpec``."""

    def __init__(self, spec: InteractiveSpec):
        self._spec = spec

    def spec(self) -> InteractiveSpec:
        return self._spec


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
    assume_constant_data: bool = False,
) -> go.Figure:
    """Precompute a one-control finite state space as a static Plotly figure.

    Multi-control state spaces deliberately fail closed. They require a renderer
    that can preserve independent controls, normally ``wasm-marimo``.
    """

    if "plotly-declarative" not in spec.allowed_backends:
        raise ValueError(f"{spec.name} does not allow the plotly-declarative backend")
    if spec.actions or spec.new_state():
        raise ValueError(
            f"{spec.name} cannot be rendered with plotly-declarative: "
            "stateful actions require a live renderer"
        )
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
        figure = spec.make_figure(**spec.figure_kwargs(parameters, {}))
        if not isinstance(figure, go.Figure):
            raise TypeError("plotly-declarative figure factories must return plotly.go.Figure")
        figures.append(figure)

    trace_count = len(figures[0].data)
    if any(len(figure.data) != trace_count for figure in figures):
        raise ValueError("figure factory must return the same trace count for every state")

    frame_names = [f"{control_name}-{index}" for index in range(len(values))]
    default_value = controls_by_name[control_name].default
    active_index = min(
        range(len(values)),
        key=lambda index: (
            abs(float(values[index]) - float(default_value))
            if isinstance(values[index], (int, float)) and isinstance(default_value, (int, float))
            else 0 if values[index] == default_value else 1
        ),
    )
    if assume_constant_data:
        data_is_constant = True
    else:
        serialized_data = [go.Figure(data=figure.data).to_json() for figure in figures]
        data_is_constant = len(set(serialized_data)) == 1
    layout_json = [figure.layout.to_plotly_json() for figure in figures]
    layout_keys = set().union(*(layout.keys() for layout in layout_json))
    dynamic_layout_keys = {
        key
        for key in layout_keys
        if len(
            {
                json.dumps(
                    layout.get(key),
                    sort_keys=True,
                    cls=PlotlyJSONEncoder,
                )
                for layout in layout_json
            }
        )
        > 1
    }
    frame_layouts = [
        {key: layout.get(key) for key in dynamic_layout_keys} for layout in layout_json
    ]
    result = go.Figure(
        data=figures[active_index].data,
        layout=figures[active_index].layout,
        frames=[
            go.Frame(
                name=frame_name,
                data=[] if data_is_constant else figure.data,
                layout=frame_layout,
                traces=[] if data_is_constant else list(range(trace_count)),
            )
            for frame_name, figure, frame_layout in zip(
                frame_names,
                figures,
                frame_layouts,
            )
        ],
    )
    result.update_layout(
        sliders=[
            {
                "active": active_index,
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
                        "label": (f"{value:g}" if isinstance(value, (int, float)) else str(value)),
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
    if PurePosixPath(wheel_filename).name != wheel_filename or not wheel_filename.endswith(".whl"):
        raise ValueError("wheel_filename must be a wheel basename")
    supported_kinds = marimo_supported_control_kinds()
    unsupported = [control.kind for control in spec.controls if control.kind not in supported_kinds]
    if unsupported:
        raise NotImplementedError(
            "the marimo renderer supports sliders, selects, button groups, checkboxes, "
            f"and toggle buttons; unsupported controls: {sorted(set(unsupported))}"
        )

    # Each action is modelled as a button accumulator that advances one state key.
    state = spec.new_state()
    state_source: dict[str, str] = {}
    for action in spec.actions:
        if len(action.state_updates) != 1:
            raise NotImplementedError(
                f"the marimo renderer needs action {action.name!r} to update exactly one state key"
            )
        (key, delta), = action.state_updates.items()
        if key not in state:
            raise ValueError(f"action {action.name!r} updates unknown state key {key!r}")
        if key in state_source:
            raise NotImplementedError(f"state key {key!r} is driven by more than one action")
        state_source[key] = f"{action.name}.value"

    control_lines = []
    control_names = []
    for control in spec.controls:
        control_names.append(control.name)
        if control.kind == "slider":
            control_lines.append(
                f"    {control.name} = mo.ui.slider("
                f"start={control.min!r}, stop={control.max!r}, step={control.step!r}, "
                f"value={control.default!r}, label={control.label!r}, "
                "show_value=True, full_width=True)"
            )
        elif control.kind == "select":
            options = json.dumps(list(control.values), sort_keys=True)
            control_lines.append(
                f"    {control.name} = mo.ui.dropdown("
                f"options={options}, value={control.default!r}, label={control.label!r})"
            )
        elif control.kind == "button_group":
            options = json.dumps(list(control.values), sort_keys=True)
            control_lines.append(
                f"    {control.name} = mo.ui.radio("
                f"options={options}, value={control.default!r}, label={control.label!r})"
            )
        elif control.kind == "checkbox":
            control_lines.append(
                f"    {control.name} = mo.ui.checkbox("
                f"value={bool(control.default)!r}, label={control.label!r})"
            )
        else:  # toggle_button
            control_lines.append(
                f"    {control.name} = mo.ui.button("
                f"value={bool(control.default)!r}, "
                "on_click=lambda value: not value, "
                f"label={control.label!r})"
            )

    action_lines = []
    action_names = []
    for action in spec.actions:
        (key, delta), = action.state_updates.items()
        action_names.append(action.name)
        action_lines.append(
            f"    {action.name} = mo.ui.button("
            f"value={state[key]!r}, on_click=lambda value: value + {delta!r}, "
            f"label={action.label!r})"
        )

    widget_names = control_names + action_names
    if action_names:
        layout = (
            f"mo.vstack([mo.hstack([{', '.join(control_names)}], widths='equal', gap=2), "
            f"mo.hstack([{', '.join(action_names)}], gap=1)])"
        )
    else:
        layout = f"mo.hstack([{', '.join(control_names)}], widths='equal', gap=2)"

    returns = ", ".join(widget_names)
    if len(widget_names) == 1:
        returns += ","

    figure_arg_parts = [f"{name}={name}.value" for name in control_names]
    for key in state:
        figure_arg_parts.append(
            f"{key}={state_source[key]}" if key in state_source else f"{key}={state[key]!r}"
        )
    figure_args = ", ".join(figure_arg_parts)
    figure_deps = ", ".join(widget_names)
    fixed_kwargs = json.dumps(dict(spec.fixed_kwargs), sort_keys=True)
    module_name, function_name = spec.figure_factory.split(":", maxsplit=1)
    pyodide_packages = json.dumps(["numpy", "scipy", "scikit-learn"])

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

        await micropip.install({pyodide_packages})
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
{chr(10).join(control_lines + action_lines)}
    controls = {layout}
    return controls, {returns}


@app.cell
def _({figure_deps}, fixed_kwargs, make_figure):
    figure = make_figure({figure_args}, **fixed_kwargs)
    figure
    return


@app.cell
def _(controls):
    controls
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
        dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))
    ]
