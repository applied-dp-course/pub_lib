"""Repository inventory checks for the plotting policy (Phase 0, warning mode)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from libdpy.visualization.interactive import InteractiveSpec, marimo_supported_control_kinds
from libdpy.visualization.registry import EMBED_CONSTRUCTOR_NAMES, embed_spec_builders

_GENERATED_BUNDLE_SKIP = frozenset(
    {
        "lecture-presentations/reconstruction-attacks/reconstruction-2d-slab",
        "lecture-presentations/reconstruction-attacks/reconstruction-3d-slabs",
    }
)

_EXTERNAL_APP_EMBED_MARKERS = (
    "external_app_iframe",
    "external_app_embed",
)

_ANIMATION_IFRAME_PREFIXES = (
    "_generated/animations/",
)

_RAW_IFRAME_SCAN_ROOTS = (
    "website/content",
    "website/pages",
)

_RAW_IFRAME_EXCLUDE_ROOTS = (
    "website/authoring",
)

_LIBDPY_SCAN_ROOT = Path("code_base_dev/libdpy")

# Rendered pages that should run ``smoke_full_page_wasm`` once ``_site`` exists.
FULL_PAGE_WASM_SMOKE_ROUTES: tuple[str, ...] = (
    "pages/index.html",
    "content/blog-posts/hypothesis-testing/post.html",
    "content/blog-posts/privacy-auditing/post.html",
    "content/lecture-presentations/hypothesis-testing/presentation.html",
    "content/lecture-presentations/privacy-auditing/presentation.html",
    "content/site-posts/gaussian-privacy-tradeoff/index.html",
    "content/tools/privacy-tradeoff-explorer/index.html",
)

# Categories that fail in strict mode (pre-render, no built ``_site`` required).
INVENTORY_STRICT_CATEGORIES: frozenset[str] = frozenset(
    {
        "raw_iframe",
        "legacy_generated_app",
        "plt_show",
        "figure_show",
        "unregistered_embed",
        "marimo_controls",
        "orphan_generated_app",
        "missing_generated_app",
    }
)

# Additional categories checked after ``quarto render`` when ``_site`` exists.
INVENTORY_POST_RENDER_STRICT_CATEGORIES: frozenset[str] = frozenset(
    {
        "doubled_defer",
        "full_page_smoke",
        "missing_animation_artifact",
    }
)

# Local names whose ``.show()`` calls are treated as figure-display side effects.
_FIGURE_SHOW_RECEIVER_NAMES = frozenset({"fig", "figure"})

# ``plot_*`` compatibility wrappers may call ``display(fig)`` but not ``fig.show()``.
_FIGURE_SHOW_ALLOWLIST_FUNCTIONS: frozenset[str] = frozenset()

# Routes intentionally excluded from the full-page WASM smoke suite.
FULL_PAGE_WASM_SMOKE_EXEMPT: dict[str, str] = {}


@dataclass(frozen=True)
class PlotInventoryFinding:
    category: str
    message: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.message}"


def workspace_root(start: Path | None = None) -> Path:
    """Return the monorepo root that contains ``code_base_dev`` and ``website``."""

    path = (start or Path(__file__)).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "code_base_dev" / "libdpy").is_dir() and (
            candidate / "website" / "scripts"
        ).is_dir():
            return candidate
    raise FileNotFoundError("could not locate workspace root from plot_inventory.py")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_source_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    return files


def find_raw_iframe_embeds(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    iframe_pattern = re.compile(r"<iframe\b", re.IGNORECASE)
    for scan_root in _RAW_IFRAME_SCAN_ROOTS:
        directory = root / scan_root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".qmd", ".ipynb", ".md", ".html"}:
                continue
            if any(_relative(path, root).startswith(prefix) for prefix in _RAW_IFRAME_EXCLUDE_ROOTS):
                continue
            text = path.read_text(encoding="utf-8")
            if not iframe_pattern.search(text):
                continue
            relative = _relative(path, root)
            for match in iframe_pattern.finditer(text):
                snippet_start = max(0, match.start() - 20)
                snippet_end = min(len(text), match.start() + 180)
                snippet = " ".join(text[snippet_start:snippet_end].split())
                if any(marker in snippet for marker in _EXTERNAL_APP_EMBED_MARKERS):
                    continue
                if any(prefix in snippet for prefix in _ANIMATION_IFRAME_PREFIXES):
                    continue
                findings.append(
                    PlotInventoryFinding(
                        "raw_iframe",
                        f"{relative}: raw <iframe> embed outside approved helpers ({snippet[:120]}...)",
                    )
                )
                break
    return findings


def find_legacy_generated_app_bundles(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    website = root / "website"

    legacy_top_level = website / "apps" / "privacy-plot-norm-6197737a49"
    if legacy_top_level.is_dir():
        findings.append(
            PlotInventoryFinding(
                "legacy_generated_app",
                "website/apps/privacy-plot-norm-6197737a49/ should move to _generated/apps/",
            )
        )

    for path in sorted((website / "content").glob("**/apps/*")):
        if not path.is_dir():
            continue
        relative = _relative(path, root)
        if any(part in relative for part in _GENERATED_BUNDLE_SKIP):
            continue
        findings.append(
            PlotInventoryFinding(
                "legacy_generated_app",
                f"{relative}/ belongs under website/_generated/apps/, not content/**/apps/",
            )
        )
    return findings


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _enclosing_function_name(node: ast.AST) -> str | None:
    current = getattr(node, "parent", None)
    while current is not None:
        if isinstance(current, ast.FunctionDef):
            return current.name
        current = getattr(current, "parent", None)
    return None


def find_plt_show_in_library(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    lib_root = root / _LIBDPY_SCAN_ROOT
    if not lib_root.is_dir():
        return findings
    for path in sorted(lib_root.rglob("*.py")):
        if "__pycache__" in path.parts or "build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "plt.show()" not in text and ".show()" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        _attach_parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "show"):
                continue
            if not isinstance(func.value, ast.Name):
                continue
            receiver = func.value.id
            if receiver == "plt":
                findings.append(
                    PlotInventoryFinding(
                        "plt_show",
                        f"{_relative(path, root)}:{node.lineno}: library helper calls plt.show()",
                    )
                )
                continue
            if receiver not in _FIGURE_SHOW_RECEIVER_NAMES:
                continue
            enclosing = _enclosing_function_name(node)
            if enclosing in _FIGURE_SHOW_ALLOWLIST_FUNCTIONS:
                continue
            findings.append(
                PlotInventoryFinding(
                    "figure_show",
                    f"{_relative(path, root)}:{node.lineno}: library helper calls {receiver}.show()",
                )
            )
    return findings


def _load_build_interactives_module(root: Path):
    try:
        from website.scripts import build_interactives  # type: ignore[import-not-found]

        return build_interactives
    except ImportError:
        import importlib.util
        import sys

        script = root / "website" / "scripts" / "build_interactives.py"
        if not script.is_file():
            return None
        spec = importlib.util.spec_from_file_location("build_interactives", script)
        if spec is None or spec.loader is None:
            return None
        build_interactives = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = build_interactives
        spec.loader.exec_module(build_interactives)
        return build_interactives


def find_unregistered_embed_constructors(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    build_interactives = _load_build_interactives_module(root)
    if build_interactives is None:
        return findings

    for warning in build_interactives.discover_unsupported_embeds(root / "website"):
        findings.append(PlotInventoryFinding("unregistered_embed", warning))
    return findings


def _example_embed_kwargs(name: str) -> dict:
    from scipy import stats

    if name == "PrivacyPlot":
        return {
            "distribution_types": [stats.norm],
            "sensitivity": 1.0,
            "std": 1.5,
        }
    if name == "TheoryROCVisualizer":
        return {"distribution": "Laplace", "scale": 1.0}
    if name == "EmpiricalEpsilonFromDeltaVisualizer":
        return {"n_samples": 100, "distribution": "Laplace", "scale": 1.0}
    if name == "NaiveSafeEpsilonHistogram":
        return {}
    if name == "LaplaceComparison":
        return {"loc1": 55, "loc2": 56}
    if name == "ExponentialMechanismInteractive":
        return {"utilities": [1.0, 2.0, 3.0], "sensitivity": 1.0}
    raise KeyError(name)


def find_unsupported_marimo_controls(root: Path) -> list[PlotInventoryFinding]:
    del root
    findings: list[PlotInventoryFinding] = []
    builders = embed_spec_builders()
    for name in EMBED_CONSTRUCTOR_NAMES:
        try:
            spec = builders[name](_example_embed_kwargs(name))
        except Exception as error:  # noqa: BLE001 - inventory only
            findings.append(
                PlotInventoryFinding(
                    "marimo_controls",
                    f"{name}: could not build spec for marimo inventory ({error})",
                )
            )
            continue
        findings.extend(_unsupported_marimo_findings(spec, context=name))
    return findings


def _unsupported_marimo_findings(
    spec: InteractiveSpec,
    *,
    context: str,
) -> list[PlotInventoryFinding]:
    supported = marimo_supported_control_kinds()
    unsupported = sorted({control.kind for control in spec.controls if control.kind not in supported})
    if not unsupported:
        return []
    return [
        PlotInventoryFinding(
            "marimo_controls",
            f"{context}: site-exportable spec uses controls unsupported by marimo WASM: "
            f"{unsupported}",
        )
    ]


def find_doubled_deferred_attributes(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    site = root / "website" / "_site"
    if not site.is_dir():
        return findings
    for path in sorted(site.rglob("*.html")):
        text = path.read_text(encoding="utf-8")
        if "data-libdpy-data-libdpy-src" in text:
            findings.append(
                PlotInventoryFinding(
                    "doubled_defer",
                    f"{_relative(path, root)}: contains doubled data-libdpy-data-libdpy-src attribute",
                )
            )
    return findings


def _literal_string(node: ast.expr, *, context: str) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise ValueError(f"{context} requires a literal string argument")


def _animation_embed_calls_from_block(block: str) -> list[tuple[str, str, str, str, int]]:
    """Return ``(collection, item, name, output, lineno)`` for animation helper calls."""

    calls: list[tuple[str, str, str, str, int]] = []
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = None
        if isinstance(function, ast.Name):
            name = function.id
        elif isinstance(function, ast.Attribute):
            name = function.attr
        if name not in {"animation_player_iframe", "animation_markdown_image"}:
            continue
        if len(node.args) < 3:
            continue
        try:
            collection = _literal_string(node.args[0], context=name)
            item = _literal_string(node.args[1], context=name)
            artifact_name = _literal_string(node.args[2], context=name)
        except ValueError:
            continue
        output = "player" if name == "animation_player_iframe" else "gif"
        for keyword in node.keywords:
            if keyword.arg == "output" and keyword.value is not None:
                try:
                    output = _literal_string(keyword.value, context=f"{name} output")
                except ValueError:
                    continue
        calls.append((collection, item, artifact_name, output, node.lineno))
    return calls


def _discover_animation_embed_calls(root: Path) -> list[tuple[Path, str, str, str, str, int]]:
    build_interactives = _load_build_interactives_module(root)
    if build_interactives is None:
        return []

    discovered: list[tuple[Path, str, str, str, str, int]] = []
    website = root / "website"
    for path in build_interactives._discovery_paths(website):
        for block in build_interactives._python_blocks(path):
            for collection, item, artifact_name, output, lineno in _animation_embed_calls_from_block(
                block
            ):
                discovered.append((path, collection, item, artifact_name, output, lineno))
    return discovered


def _rendered_page_for_source(source: Path, *, root: Path) -> Path | None:
    website = root / "website"
    site = root / "website" / "_site"
    try:
        relative = source.relative_to(website)
    except ValueError:
        return None
    if relative.suffix == ".qmd":
        return site / relative.with_suffix(".html")
    if relative.suffix == ".ipynb":
        return site / relative.with_suffix(".html")
    return None


def _animation_artifact_extension(output: str) -> str:
    return {"gif": ".gif", "player": ".html", "mp4": ".mp4"}[output]


def find_missing_animation_artifacts(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    site = root / "website" / "_site"
    if not site.is_dir():
        return findings

    for source, collection, item, artifact_name, output, lineno in _discover_animation_embed_calls(
        root
    ):
        extension = _animation_artifact_extension(output)
        artifact_relative = (
            f"website/_site/_generated/animations/{collection}/{item}/{artifact_name}{extension}"
        )
        artifact_path = root / artifact_relative
        source_relative = _relative(source, root)
        if not artifact_path.is_file():
            findings.append(
                PlotInventoryFinding(
                    "missing_animation_artifact",
                    f"{source_relative}:{lineno}: expected rendered artifact missing: "
                    f"{artifact_relative}",
                )
            )
            continue
        rendered_page = _rendered_page_for_source(source, root=root)
        if rendered_page is None or not rendered_page.is_file():
            findings.append(
                PlotInventoryFinding(
                    "missing_animation_artifact",
                    f"{source_relative}:{lineno}: rendered page missing for animation embed "
                    f"({artifact_name}{extension})",
                )
            )
            continue
        rendered_text = rendered_page.read_text(encoding="utf-8")
        if f"{artifact_name}{extension}" not in rendered_text:
            findings.append(
                PlotInventoryFinding(
                    "missing_animation_artifact",
                    f"{_relative(rendered_page, root)}: source {source_relative}:{lineno} "
                    f"embeds {artifact_name}{extension} but the rendered page does not reference it",
                )
            )
    return findings


def find_orphan_generated_apps(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    build_interactives = _load_build_interactives_module(root)
    if build_interactives is None:
        return findings

    website = root / "website"
    apps_root = website / "_generated" / "apps"
    if not apps_root.is_dir():
        return findings

    expected = {
        build_interactives.output_directory_for(use, website)
        for use in build_interactives.discover_interactives(website)
    }
    for marker in sorted(apps_root.rglob(".libdpy-interactive")):
        app_directory = marker.parent
        if app_directory not in expected:
            findings.append(
                PlotInventoryFinding(
                    "orphan_generated_app",
                    f"{_relative(app_directory, root)}/ is not referenced by any discovered "
                    ".embed() call",
                )
            )
    return findings


def find_missing_generated_apps(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    build_interactives = _load_build_interactives_module(root)
    if build_interactives is None:
        return findings

    website = root / "website"
    expected = {
        build_interactives.output_directory_for(use, website)
        for use in build_interactives.discover_interactives(website)
    }
    for app_directory in sorted(expected):
        marker = app_directory / ".libdpy-interactive"
        if not marker.is_file():
            findings.append(
                PlotInventoryFinding(
                    "missing_generated_app",
                    f"{_relative(app_directory, root)}/ is expected from a discovered "
                    ".embed() call but was not built",
                )
            )
    return findings


def find_missing_full_page_wasm_smoke_coverage(root: Path) -> list[PlotInventoryFinding]:
    findings: list[PlotInventoryFinding] = []
    site = root / "website" / "_site"
    if not site.is_dir():
        return [
            PlotInventoryFinding(
                "full_page_smoke",
                "website/_site missing; run render before verifying deferred WASM parent pages",
            )
        ]

    covered = set(FULL_PAGE_WASM_SMOKE_ROUTES) | set(FULL_PAGE_WASM_SMOKE_EXEMPT)
    for path in sorted(site.rglob("*.html")):
        text = path.read_text(encoding="utf-8")
        if "data-libdpy-src" not in text and "libdpy-interactive" not in text:
            continue
        route = path.relative_to(site).as_posix()
        if route in covered:
            continue
        findings.append(
            PlotInventoryFinding(
                "full_page_smoke",
                f"{route}: deferred WASM parent page is not listed in "
                "FULL_PAGE_WASM_SMOKE_ROUTES or FULL_PAGE_WASM_SMOKE_EXEMPT",
            )
        )
    return findings


def collect_plot_inventory_findings(
    root: Path | None = None,
    *,
    include_post_render: bool = True,
) -> list[PlotInventoryFinding]:
    """Run all Phase 0 inventory checks and return findings (warning mode)."""

    workspace = workspace_root(root)
    findings: list[PlotInventoryFinding] = []
    findings.extend(find_raw_iframe_embeds(workspace))
    findings.extend(find_legacy_generated_app_bundles(workspace))
    findings.extend(find_plt_show_in_library(workspace))
    findings.extend(find_unregistered_embed_constructors(workspace))
    findings.extend(find_unsupported_marimo_controls(workspace))
    findings.extend(find_orphan_generated_apps(workspace))
    findings.extend(find_missing_generated_apps(workspace))
    if include_post_render:
        findings.extend(find_doubled_deferred_attributes(workspace))
        findings.extend(find_missing_full_page_wasm_smoke_coverage(workspace))
        findings.extend(find_missing_animation_artifacts(workspace))
    return findings


def strict_inventory_findings(
    findings: list[PlotInventoryFinding],
    *,
    post_render: bool = False,
) -> list[PlotInventoryFinding]:
    """Return findings that should fail CI in strict mode."""

    allowed = INVENTORY_STRICT_CATEGORIES
    if post_render:
        allowed = allowed | INVENTORY_POST_RENDER_STRICT_CATEGORIES
    return [finding for finding in findings if finding.category in allowed]
