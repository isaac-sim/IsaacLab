# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that every test file's Kit markers agree with what the file actually does.

Kit-dependence is a property of *importing* a test module: a module that constructs
:class:`~isaaclab.app.AppLauncher` at module scope boots Isaac Sim during pytest collection,
before any fixture runs. The ``kit`` / ``kit_cameras`` / ``kitless`` markers make that
property declarative so the runner can group files that share a launch configuration into a
single process instead of paying Kit startup once per file.

A marker is only useful if it cannot drift from reality, which is what this test enforces:

* ``kit`` / ``kit_cameras`` -- the file calls :func:`~isaaclab.test.launch.launch_kit` at
  module scope with the matching ``cameras`` argument, and never constructs ``AppLauncher``
  or ``SimulationApp`` itself. Direct construction would boot a second, unshared app.
* ``kitless`` -- the file never launches Kit and does not import a Kit runtime package at
  module scope, so it can run in a process where Kit was never started.
* ``unit`` -- same requirement as ``kitless``, which turns the marker's registered
  description ("does not launch the simulator") into a checked invariant.
* At most one module-scope ``pytestmark`` assignment, since a second assignment silently
  rebinds the name and discards the markers from the first.

The checks are AST-based rather than text-based because a source-text search cannot tell an
``AppLauncher`` reference in a docstring from a real call -- several kit-free files mention
``AppLauncher`` only to document that they do not use it.

Files outside :data:`_ENFORCED_ROOTS` are not yet *required* to carry a marker; the
consistency rules above still apply to them whenever they do. Extend that tuple as each
package is migrated.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCAN_ROOTS = ("source", "scripts")

_EXCLUDED_PARTS = frozenset(
    {
        # Own pytest.ini / rootdir; deliberately excluded from the main collector too.
        "install_ci",
        # Vendored copies of the source tree produced by the wheel builder.
        "build",
        # Virtual environments and the Isaac Sim symlink.
        ".venv",
        "env_isaaclab",
        "_isaac_sim",
    }
)

# Packages that only exist inside a running Kit application. ``pxr`` is deliberately absent:
# OpenUSD is importable kit-less through the ``usd-core`` wheel, so importing it says nothing
# about whether Kit is running.
_KIT_RUNTIME_PREFIXES = ("omni", "carb", "isaacsim")

# Directories where a test file is required to declare `kit`, `kit_cameras`, or `kitless`.
# Grows one package at a time as files are migrated off module-scope ``AppLauncher``.
_ENFORCED_ROOTS: tuple[str, ...] = ()

_PROFILE_MARKERS = ("kit", "kit_cameras", "kitless")


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _module_scope_nodes(tree: ast.Module):
    """Yield every node that executes at module import, without entering callables.

    Descends through module-level control flow (``if`` / ``try`` / ``with``) because those
    bodies still run at import, but stops at function, class, and lambda boundaries because
    those bodies only run when called.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        for child in ast.iter_child_nodes(node):
            stack.append(child)


def _call_name(node: ast.AST) -> str | None:
    """Return the called function's bare name, for ``f()`` and ``mod.f()`` alike."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _marker_names(node: ast.AST) -> list[str]:
    """Return the marker names in a ``pytest.mark.<name>`` expression or a list of them."""
    if isinstance(node, ast.List | ast.Tuple):
        return [name for element in node.elts for name in _marker_names(element)]
    if isinstance(node, ast.Call):
        return _marker_names(node.func)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        # pytest.mark.<name>
        if node.value.attr == "mark":
            return [node.attr]
    return []


class _FileFacts:
    """What a single test file declares and what it actually does at module scope."""

    def __init__(self, path: Path, tree: ast.Module):
        self.path = path
        self.pytestmark_assignments: list[int] = []
        self.markers: set[str] = set()
        self.launch_kit_cameras: bool | None = None
        self.module_scope_launcher: list[tuple[str, int]] = []
        self.launch_kit_anywhere = False
        self.kit_runtime_imports: list[tuple[str, int]] = []

        module_scope = set()
        for node in _module_scope_nodes(tree):
            module_scope.add(id(node))

            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
            ):
                self.pytestmark_assignments.append(node.lineno)
                self.markers.update(_marker_names(node.value))

            name = _call_name(node)
            if name in ("AppLauncher", "SimulationApp"):
                self.module_scope_launcher.append((name, node.lineno))
            elif name == "launch_kit":
                self.launch_kit_cameras = any(
                    keyword.arg == "cameras" and isinstance(keyword.value, ast.Constant) and keyword.value.value
                    for keyword in node.keywords
                )

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _KIT_RUNTIME_PREFIXES:
                        self.kit_runtime_imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.split(".")[0] in _KIT_RUNTIME_PREFIXES:
                    self.kit_runtime_imports.append((node.module, node.lineno))

        # Decorator markers (e.g. a per-test `@pytest.mark.unit`) count toward the file's
        # marker set, and AppLauncher use anywhere -- not just module scope -- disqualifies
        # a file from claiming `kitless`.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for decorator in node.decorator_list:
                    self.markers.update(_marker_names(decorator))
            name = _call_name(node)
            if name == "launch_kit":
                self.launch_kit_anywhere = True
            elif name in ("AppLauncher", "SimulationApp") and id(node) not in module_scope:
                self.module_scope_launcher.append((f"{name} (deferred)", node.lineno))

    @property
    def rel(self) -> str:
        return self.path.relative_to(_REPO_ROOT).as_posix()

    @property
    def profile_markers(self) -> list[str]:
        return [marker for marker in _PROFILE_MARKERS if marker in self.markers]

    @property
    def launches_kit_directly(self) -> list[tuple[str, int]]:
        return self.module_scope_launcher


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _iter_test_files():
    for root in _SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("test_*.py")):
            if _EXCLUDED_PARTS.isdisjoint(path.parts):
                yield path


@pytest.fixture(scope="module")
def facts() -> list[_FileFacts]:
    """Parse every test file once and return the extracted facts."""
    collected = []
    for path in _iter_test_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError as exc:
            pytest.fail(f"{path.relative_to(_REPO_ROOT).as_posix()} failed to parse: {exc}")
        collected.append(_FileFacts(path, tree))
    assert collected, f"no test files discovered under {_SCAN_ROOTS} -- the scan roots are wrong"
    return collected


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_pytestmark_is_assigned_at_most_once(facts: list[_FileFacts]):
    """A second module-scope ``pytestmark`` rebinds the name and drops the first one's markers."""
    offenders = [
        f"{f.rel}: lines {sorted(f.pytestmark_assignments)}" for f in facts if len(f.pytestmark_assignments) > 1
    ]
    assert not offenders, (
        "These files assign `pytestmark` more than once at module scope. The later assignment"
        " replaces the earlier one, so the markers declared first are silently lost:\n  "
        + "\n  ".join(offenders)
        + "\n\nFix: merge them into a single list, e.g. `pytestmark = [pytest.mark.a, pytest.mark.b]`."
    )


def test_profile_markers_are_mutually_exclusive(facts: list[_FileFacts]):
    """A file runs in exactly one of the launch configurations, so it declares only one."""
    offenders = [f"{f.rel}: {', '.join(f.profile_markers)}" for f in facts if len(f.profile_markers) > 1]
    assert not offenders, "These files declare more than one of `kit`, `kit_cameras`, `kitless`:\n  " + "\n  ".join(
        offenders
    )


def test_kit_marked_files_use_launch_kit(facts: list[_FileFacts]):
    """`kit` / `kit_cameras` files share the process app; they must not build their own."""
    offenders = []
    for f in facts:
        markers = f.profile_markers
        if not markers or markers[0] == "kitless":
            continue
        if f.launches_kit_directly:
            where = ", ".join(f"{name} at line {line}" for name, line in f.launches_kit_directly)
            offenders.append(f"{f.rel}: declares `{markers[0]}` but constructs {where}")
            continue
        if f.launch_kit_cameras is None:
            offenders.append(f"{f.rel}: declares `{markers[0]}` but never calls launch_kit() at module scope")
            continue
        wants_cameras = markers[0] == "kit_cameras"
        if f.launch_kit_cameras != wants_cameras:
            expected = "launch_kit(cameras=True)" if wants_cameras else "launch_kit()"
            offenders.append(f"{f.rel}: declares `{markers[0]}` but does not call {expected}")

    assert not offenders, (
        "These files' Kit markers disagree with how they launch Kit:\n  "
        + "\n  ".join(offenders)
        + "\n\nFix: call `launch_kit()` (or `launch_kit(cameras=True)`) from"
        " `isaaclab.test.launch` at module scope instead of constructing AppLauncher, and make"
        " the marker match the `cameras` argument."
    )


@pytest.mark.parametrize("marker", ["kitless", "unit"])
def test_kit_free_files_do_not_touch_kit(marker: str, facts: list[_FileFacts]):
    """`kitless` and `unit` files must run in a process where Kit was never started."""
    offenders = []
    for f in facts:
        if marker not in f.markers:
            continue
        if f.launches_kit_directly:
            where = ", ".join(f"{name} at line {line}" for name, line in f.launches_kit_directly)
            offenders.append(f"{f.rel}: constructs {where}")
        if f.launch_kit_anywhere:
            offenders.append(f"{f.rel}: calls launch_kit()")
        if f.kit_runtime_imports:
            where = ", ".join(f"`{name}` at line {line}" for name, line in f.kit_runtime_imports)
            offenders.append(f"{f.rel}: imports {where} at module scope")

    assert not offenders, (
        f"These files are marked `{marker}` but depend on a running Kit:\n  "
        + "\n  ".join(offenders)
        + f"\n\nKit runtime packages: {_KIT_RUNTIME_PREFIXES}."
        f"\nFix: drop the `{marker}` marker and declare `kit`, or move the Kit import inside the"
        " test function so it is not paid at collection."
    )


def test_migrated_packages_declare_a_marker(facts: list[_FileFacts]):
    """Within a migrated package, every test file states its launch configuration."""
    if not _ENFORCED_ROOTS:
        pytest.skip("no packages are enforced yet; extend _ENFORCED_ROOTS as files are migrated")

    offenders = [f.rel for f in facts if f.rel.startswith(_ENFORCED_ROOTS) and not f.profile_markers]
    assert not offenders, (
        "These files are in a migrated package but declare none of `kit`, `kit_cameras`,"
        " `kitless`:\n  " + "\n  ".join(offenders)
    )


def test_shareable_file_list_is_derived_from_the_markers():
    """``tools/kit_test_files.py`` must agree with the markers, and order cameras first.

    CI batches test files by asking that script which ones can share a Kit app, instead of
    carrying a hand-written list that goes stale as files are added or reclassified. These are
    the invariants a caller relies on.
    """
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from kit_test_files import shareable_test_files  # noqa: PLC0415
    from test_settings import TESTS_TO_SKIP  # noqa: PLC0415

    directory = _REPO_ROOT / "source" / "isaaclab" / "test" / "sim"
    selected = shareable_test_files(directory)
    names = [path.name for path in selected]
    assert names, f"no shareable files found in {directory}"
    assert len(names) == len(set(names)), f"duplicate entries: {names}"

    sources = {path.name: path.read_text(encoding="utf-8") for path in directory.glob("test_*.py")}

    def marks(name: str, marker: str) -> bool:
        return f"pytest.mark.{marker}" in sources[name]

    expected = {
        name
        for name, source in sources.items()
        if name not in TESTS_TO_SKIP and "pytest.mark.kit" in source and "pytest.mark.kit_solo" not in source
    }
    assert set(names) == expected, (
        "the derived list disagrees with the markers:"
        f"\n  only in list:     {sorted(set(names) - expected)}"
        f"\n  only in markers:  {sorted(expected - set(names))}"
    )

    # A camera-enabled app can serve tests that do not need cameras, but cameras cannot be
    # enabled after startup, so every kit_cameras file must precede every plain kit file.
    is_camera = [marks(name, "kit_cameras") for name in names]
    assert is_camera == sorted(is_camera, reverse=True), (
        "kit_cameras files must come first, otherwise a plain `kit` file boots the app without"
        f" cameras and the later launch_kit(cameras=True) raises. Got: {names}"
    )


def test_kitless_files_import_without_kit(facts: list[_FileFacts]):
    """Importing every `kitless` module must not pull in Kit through a helper module.

    The AST rules only see each file's own imports. A shared test utility that imports Kit
    would slip past them, so this imports the real modules in one subprocess and checks that
    ``omni.kit.app`` never appears in :data:`sys.modules`.
    """
    modules = sorted(f.rel for f in facts if "kitless" in f.markers)
    if not modules:
        pytest.skip("no files are marked `kitless` yet")

    script = textwrap.dedent(f"""
        import importlib.util, json, os, sys

        offenders = []
        for rel in {modules!r}:
            # pytest puts a test file's own directory on sys.path (rootdir/conftest handling),
            # which is how these modules reach their sibling helpers. Mirror that here.
            directory = os.path.dirname(rel)
            if directory not in sys.path:
                sys.path.insert(0, directory)

            name = "_kitless_probe_" + rel.replace("/", "_")[:-3]
            spec = importlib.util.spec_from_file_location(name, rel)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                offenders.append(f"{{rel}}: import failed: {{type(exc).__name__}}: {{exc}}")
                continue
            if "omni.kit.app" in sys.modules:
                offenders.append(f"{{rel}}: importing it started Kit")
                break
        print("__RESULTS__" + json.dumps(offenders))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=600)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("__RESULTS__")), None)
    assert line is not None, (
        f"kitless import probe did not report results\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    offenders = json.loads(line[len("__RESULTS__") :])
    assert not offenders, "These `kitless` files pull in Kit transitively:\n  " + "\n  ".join(offenders)
