# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package-boundary guard for core ``isaaclab`` test code.

The core package (``source/isaaclab``) declares no interpackage dependencies in
its ``pyproject.toml``, so its test suite *and* the shared test fixtures shipped
inside the package (``isaaclab.test.*``, e.g. ``isaaclab.test.env_cfgs``) must
stay within the core dependency boundary. Borrowing a higher-level sibling
package inverts the dependency direction and hides genuine cross-package
integration tests among the unit tests, working against the migration toward
simulator-free unit tests built on core-only fixtures.

The guard is data-driven so it can be extended without touching any logic:

* :data:`_BANNED_PACKAGES` — the undeclared first-party packages core test code
  must not import. Only ``isaaclab_tasks`` is enforced today (OMPE-100996). To
  extend the boundary to another package (for example ``isaaclab_assets``), add
  it here together with any temporary allowlist below — no code change needed.
* :data:`_TEMPORARY_HANDOFF_ALLOWLIST` — genuine cross-package integration tests
  still living in the core tree, keyed by banned package. The test-separation
  follow-up relocates these to ``tests/integration/<pkg>/``; once a file moves
  out, remove its entry (``test_handoff_allowlist_entries_are_current`` fails on
  stale entries).
* :data:`_EXEMPT_SEGMENTS` — path segments that are out of scope for this guard:
  ``install_ci`` (installs the full stack on purpose) and ``benchmark`` (owned by
  separate benchmark work; not modified by OMPE-100996).

This test is purely static (AST-based) and requires no simulator.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# This file lives at ``source/isaaclab/test/dependencies/<file>.py``.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Roots holding core test code: the test suite and the fixtures shipped in the
# package (importable as ``isaaclab.test.*``). Both must respect the boundary.
_SCAN_ROOTS = (
    _REPO_ROOT / "source" / "isaaclab" / "test",
    _REPO_ROOT / "source" / "isaaclab" / "isaaclab" / "test",
)

# Undeclared first-party packages core test code must not import. Enforced today:
# only ``isaaclab_tasks`` (OMPE-100996). Add packages here to widen the boundary.
_BANNED_PACKAGES: tuple[str, ...] = ("isaaclab_tasks",)

# Path segments excluded from the guard, with the reason each is out of scope.
_EXEMPT_SEGMENTS: dict[str, str] = {
    "install_ci": "installs and imports the full stack to validate documented install paths",
    "benchmark": "benchmark suite/support is owned by separate benchmark work, out of scope for OMPE-100996",
}

# Temporary per-package allowlist (repo-root-relative POSIX paths): genuine
# cross-package integration tests awaiting relocation to ``tests/integration/<pkg>/``.
# Remove an entry once its file leaves the core tree.
_TEMPORARY_HANDOFF_ALLOWLIST: dict[str, frozenset[str]] = {
    "isaaclab_tasks": frozenset({
        "source/isaaclab/test/controllers/test_pink_ik.py",
        "source/isaaclab/test/sensors/test_outdated_sensor.py",
        "source/isaaclab/test/sensors/test_tiled_camera_env.py",
    }),
}


def _match_banned_package(module: str, banned: tuple[str, ...]) -> str | None:
    """Return the banned package *module* belongs to, or ``None``.

    Matches ``<pkg>`` and ``<pkg>.*`` while deliberately excluding sibling
    top-level packages that merely share a prefix (e.g. ``isaaclab_tasks`` must
    not match ``isaaclab_tasks_experimental``).
    """
    for package in banned:
        if module == package or module.startswith(f"{package}."):
            return package
    return None


def _iter_banned_imports(source: str, filename: str, banned: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return ``(package, statement)`` for each import of a *banned* package (AST-based).

    Only real ``import`` / ``from ... import`` statements are reported; string
    literals that merely mention a package name (package-name lists, ``python -c``
    snippets) are ignored. Relative imports cannot reference a top-level package
    and are skipped.
    """
    offenders: list[tuple[str, str]] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package = _match_banned_package(alias.name, banned)
                if package is not None:
                    offenders.append((package, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                package = _match_banned_package(node.module, banned)
                if package is not None:
                    names = ", ".join(alias.name for alias in node.names)
                    offenders.append((package, f"from {node.module} import {names}"))
    return offenders


def _repo_rel(path: Path) -> str:
    """POSIX path of *path* relative to the repository root."""
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_exempt(path: Path) -> bool:
    """Whether *path* lives under an out-of-scope segment (see :data:`_EXEMPT_SEGMENTS`)."""
    parts = set(path.relative_to(_REPO_ROOT).parts)
    return any(segment in parts for segment in _EXEMPT_SEGMENTS)


def _iter_core_test_files() -> list[Path]:
    """All Python files across the scan roots, excluding bytecode caches."""
    files: set[Path] = set()
    for root in _SCAN_ROOTS:
        files.update(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


# Files subject to the boundary: everything discovered except out-of-scope segments.
_GUARDED_FILES = [p for p in _iter_core_test_files() if not _is_exempt(p)]
_GUARDED_IDS = [_repo_rel(p) for p in _GUARDED_FILES]


@pytest.mark.parametrize("path", _GUARDED_FILES, ids=_GUARDED_IDS or ["no-files-discovered"])
def test_core_test_respects_package_boundary(path: Path):
    """Core test code must not import a banned first-party package."""
    rel = _repo_rel(path)
    offenders = [
        (package, statement)
        for package, statement in _iter_banned_imports(path.read_text(encoding="utf-8"), str(path), _BANNED_PACKAGES)
        if rel not in _TEMPORARY_HANDOFF_ALLOWLIST.get(package, frozenset())
    ]
    if offenders:
        listing = "\n".join(f"  [{package}] {statement}" for package, statement in offenders)
        pytest.fail(
            f"{rel} crosses the core package boundary:\n{listing}\n\n"
            "Core (isaaclab) test code declares no dependency on these packages. Rebuild the needed "
            "fixture with core APIs (see isaaclab.test.env_cfgs), or — if this is a genuine cross-package "
            "integration test — coordinate its move to tests/integration/<pkg>/ with the test-separation "
            "follow-up before adding it to _TEMPORARY_HANDOFF_ALLOWLIST."
        )


def test_boundary_guard_covers_shared_fixtures_and_tests():
    """Canary: the guard scanned the test suite and the shared in-package fixtures."""
    scanned = set(_iter_core_test_files())
    assert Path(__file__).resolve() in scanned, "Discovery did not include the test-suite scan root."

    env_cfgs = _REPO_ROOT / "source" / "isaaclab" / "isaaclab" / "test" / "env_cfgs.py"
    assert env_cfgs in scanned, "Shared fixture env_cfgs.py is not covered by the boundary guard."
    assert env_cfgs in set(_GUARDED_FILES), "Shared fixture env_cfgs.py must not be exempt from the guard."


def test_import_detector_flags_representative_task_imports():
    """Positive control: the detector flags the representative pre-refactor task imports.

    These mirror the borrowed imports the refactor removed from the core tests
    (e.g. ``from isaaclab_tasks.utils.parse_cfg import parse_env_cfg``), proving
    the guard catches real regressions rather than passing vacuously.
    """
    source = (
        "import isaaclab_tasks\n"
        "from isaaclab_tasks.utils.parse_cfg import parse_env_cfg\n"
        "import isaaclab_tasks.manager_based.classic.cartpole as cartpole\n"
    )
    offenders = _iter_banned_imports(source, "<representative>", _BANNED_PACKAGES)
    assert len(offenders) == 3
    assert set(offenders) == {
        ("isaaclab_tasks", "import isaaclab_tasks"),
        ("isaaclab_tasks", "from isaaclab_tasks.utils.parse_cfg import parse_env_cfg"),
        ("isaaclab_tasks", "import isaaclab_tasks.manager_based.classic.cartpole"),
    }


def test_import_detector_ignores_core_and_sibling_packages():
    """The detector must not flag core imports, prefix-sharing siblings, or string mentions."""
    source = (
        "import isaaclab\n"
        "from isaaclab.envs import ManagerBasedEnv\n"
        "import isaaclab_tasks_experimental\n"
        "from isaaclab_tasks_experimental.foo import bar\n"
        'PKGS = ["isaaclab_tasks", "isaaclab_tasks_experimental"]\n'
        'subprocess.run(["python", "-c", "import isaaclab_tasks"])\n'
    )
    assert _iter_banned_imports(source, "<siblings>", _BANNED_PACKAGES) == []


def test_boundary_generalizes_to_additional_packages_via_config_only():
    """The mechanism is package-agnostic: widening the boundary is a config change.

    The same detector that enforces ``isaaclab_tasks`` today flags any other
    first-party package once it is added to the banned set — no logic change.
    """
    source = "import isaaclab_assets\nfrom isaaclab_assets.robots import FRANKA_PANDA_CFG\n"
    assert _iter_banned_imports(source, "<future>", _BANNED_PACKAGES) == []
    offenders = _iter_banned_imports(source, "<future>", ("isaaclab_assets",))
    assert {package for package, _ in offenders} == {"isaaclab_assets"}


_ALLOWLIST_ITEMS = sorted(
    (package, rel_path) for package, files in _TEMPORARY_HANDOFF_ALLOWLIST.items() for rel_path in files
)


@pytest.mark.parametrize(
    ("package", "rel_path"),
    _ALLOWLIST_ITEMS,
    ids=[f"{package}:{rel_path}" for package, rel_path in _ALLOWLIST_ITEMS] or ["empty-allowlist"],
)
def test_handoff_allowlist_entries_are_current(package: str, rel_path: str):
    """Each temporary allowlist entry must exist and still import its banned package.

    When the follow-up relocates one of these integration tests out of the core
    tree, its entry goes stale and this test fails, prompting removal from
    :data:`_TEMPORARY_HANDOFF_ALLOWLIST`.
    """
    path = _REPO_ROOT / rel_path
    assert path.is_file(), f"Stale handoff allowlist entry (file moved or removed): {rel_path}"
    offenders = _iter_banned_imports(path.read_text(encoding="utf-8"), str(path), (package,))
    assert offenders, f"Allowlisted file {rel_path} no longer imports {package}; remove it from the allowlist."
