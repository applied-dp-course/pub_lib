"""Helpers for embedding hand-authored external browser apps in website content."""

from __future__ import annotations


def external_app_path(
    app_path: str,
    *,
    depth: int = 3,
    index: str = "index.html",
) -> str:
    """Return a relative URL from a rendered page to an external app entrypoint.

    ``app_path`` mirrors manifest paths such as
    ``apps/lecture-presentations/reconstruction-attacks/reconstruction-2d-slab``.
    ``depth`` is the number of ``..`` segments from the rendered page back to
    ``website/`` (default ``3`` for pages under ``content/<collection>/<item>/``).
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")
    normalized = app_path.strip("/").replace("\\", "/")
    prefix = "/".join([".."] * depth)
    return f"{prefix}/{normalized}/{index}"


def external_app_iframe(
    app_path: str,
    *,
    depth: int = 3,
    height: int = 640,
    title: str = "",
    loading: str = "lazy",
) -> str:
    """Return an iframe embed for a hand-authored external app."""

    path = external_app_path(app_path, depth=depth)
    title_attr = f' title="{title}"' if title else ""
    return (
        f'<iframe src="{path}" width="100%" height="{height}" style="border:0" '
        f'loading="{loading}"{title_attr}></iframe>'
    )
