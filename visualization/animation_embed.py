"""Helpers for embedding build-generated animation artifacts in website content."""

from __future__ import annotations

_VALID_OUTPUTS = frozenset({"gif", "player", "mp4"})
_EXTENSIONS = {"gif": ".gif", "player": ".html", "mp4": ".mp4"}


def animation_artifact_path(
    collection: str,
    item: str,
    name: str,
    *,
    output: str = "gif",
    depth: int = 3,
) -> str:
    """Return a relative URL from a rendered page to a generated animation artifact.

    ``collection`` and ``item`` mirror the content path
    ``content/<collection>/<item>/animations/<name>.py``. ``depth`` is the number of
    ``..`` segments from the rendered page back to ``website/`` (default ``3`` for
    pages under ``content/<collection>/<item>/``).
    """

    if output not in _VALID_OUTPUTS:
        raise ValueError(f"output must be one of {sorted(_VALID_OUTPUTS)}")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    prefix = "/".join([".."] * depth)
    extension = _EXTENSIONS[output]
    return f"{prefix}/_generated/animations/{collection}/{item}/{name}{extension}"


def animation_markdown_image(
    collection: str,
    item: str,
    name: str,
    *,
    alt: str,
    output: str = "gif",
    depth: int = 3,
    width: str | None = None,
    fig_align: str | None = None,
) -> str:
    """Return Quarto/Markdown image syntax for a flat animation artifact."""

    path = animation_artifact_path(collection, item, name, output=output, depth=depth)
    attributes: list[str] = []
    if width is not None:
        attributes.append(f'width="{width}"')
    if fig_align is not None:
        attributes.append(f"fig-align=\"{fig_align}\"")
    suffix = f"{{{','.join(attributes)}}}" if attributes else ""
    return f"![{alt}]({path}){suffix}"


def animation_player_iframe(
    collection: str,
    item: str,
    name: str,
    *,
    depth: int = 3,
    height: int = 430,
) -> str:
    """Return an iframe embed for an HTML animation player artifact."""

    path = animation_artifact_path(collection, item, name, output="player", depth=depth)
    return (
        f'<iframe src="{path}" width="100%" height="{height}" style="border:0" '
        'onload="this.style.height=this.contentWindow.document.documentElement.scrollHeight+\'px\'">'
        "</iframe>"
    )
