# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard core test code against direct ``isaaclab_tasks`` imports.

The core package is a lower-level dependency of task packages, so its test
suite and the shared fixtures shipped in ``isaaclab.test`` must not borrow task
implementations. Genuine cross-package integration tests still awaiting
relocation are listed in a temporary handoff allowlist.

This increment checks only static ``import`` and ``from ... import`` statements
for ``isaaclab_tasks``, including statements under ``TYPE_CHECKING``. Dynamic
imports and imports from other sibling packages remain follow-up work.

The guard is data-driven:

* :data:`_BANNED_PACKAGES` lists the package boundary enforced by this
  increment.
* :data:`_TEMPORARY_HANDOFF_ALLOWLIST` lists genuine cross-package integration
  tests that still live in the core tree. Stale entries fail
  :func:`test_handoff_allowlist_entries_are_current`.
* :data:`_EXEMPT_SEGMENTS` lists test areas that intentionally remain out of
  scope: ``install_ci`` validates full-stack installations, while ``benchmark``
  is owned by separate benchmark work.

This test is purely static and requires no simulator.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# This file lives at ``source/isaaclab/test/dependencies/<file>.py``.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Scan both the core test suite and the shared fixtures importable from
# ``isaaclab.test``.
_SCAN_ROOTS = (
    _REPO_ROOT / "source" / "isaaclab" / "test",
    _REPO_ROOT / "source" / "isaaclab" / "isaaclab" / "test",
)

# This increment enforces only the direct ``isaaclab_tasks`` boundary.
_BANNED_PACKAGES: tuple[str, ...] = ("isaaclab_tasks",)

# Path segments excluded from the guard, with the reason each is out of scope.
_EXEMPT_SEGMENTS: dict[str, str] = {
    "install_ci": "installs and imports the full stack to validate documented install paths",
    "benchmark": "benchmark suite and support code are owned by separate benchmark work",
}

# Genuine cross-package integration tests awaiting relocation out of the core
# tree. The first three are pre-existing handoffs; the final two preserve
# task-backed coverage split from the core-only environment tests.
_TEMPORARY_HANDOFF_ALLOWLIST: dict[str, frozenset[str]] = {
    "isaaclab_tasks": frozenset(
        {
            "source/isaaclab/test/controllers/test_pink_ik.py",
            "source/isaaclab/test/envs/test_action_state_recorder_term_task_integration.py",
            "source/isaaclab/test/envs/test_manager_based_rl_env_obs_spaces_task_integration.py",
            "source/isaaclab/test/sensors/test_outdated_sensor.py",
            "source/isaaclab/test/sensors/test_tiled_camera_env.py",
        }
    ),
}


def _match_banned_package(module: str, banned: tuple[str, ...]) -> str | None:
    """Return the banned package that contains a module, if any.

    Prefix-sharing top-level packages do not match. For example,
    ``isaaclab_tasks_experimental`` is not part of ``isaaclab_tasks``.
    """
    for package in banned:
        if module == package or module.startswith(f"{package}."):
            return package
    return None


def _iter_banned_imports(source: str, filename: str, banned: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return the banned package and statement for each direct import.

    The complete AST is scanned, so imports under runtime conditionals or
    ``TYPE_CHECKING`` are source dependencies and are reported. Dynamic import
    calls and string literals are intentionally ignored.
    """
    offenders: list[tuple[str, str]] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package = _match_banned_package(alias.name, banned)
                if package is not None:
                    offenders.append((package, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            package = _match_banned_package(node.module, banned)
            if package is not None:
                names = ", ".join(alias.name for alias in node.names)
                offenders.append((package, f"from {node.module} import {names}"))
    return offenders


def _repo_rel(path: Path) -> str:
    """Return a repository-relative POSIX path."""
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_exempt(path: Path) -> bool:
    """Return whether a path belongs to an out-of-scope test area."""
    parts = set(path.relative_to(_REPO_ROOT).parts)
    return any(segment in parts for segment in _EXEMPT_SEGMENTS)


def _iter_core_test_files() -> list[Path]:
    """Return all Python files in the core test roots."""
    files: set[Path] = set()
    for root in _SCAN_ROOTS:
        files.update(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(files)


_GUARDED_FILES = [path for path in _iter_core_test_files() if not _is_exempt(path)]
_GUARDED_IDS = [_repo_rel(path) for path in _GUARDED_FILES]


@pytest.mark.parametrize("path", _GUARDED_FILES, ids=_GUARDED_IDS or ["no-files-discovered"])
def test_core_test_respects_package_boundary(path: Path):
    """Core test code must not directly import a banned package."""
    rel_path = _repo_rel(path)
    offenders = [
        (package, statement)
        for package, statement in _iter_banned_imports(
            path.read_text(encoding="utf-8"), str(path), _BANNED_PACKAGES
        )
        if rel_path not in _TEMPORARY_HANDOFF_ALLOWLIST.get(package, frozenset())
    ]
    if offenders:
        listing = "\n".join(f"  [{package}] {statement}" for package, statement in offenders)
        pytest.fail(
            f"{rel_path} crosses the core package boundary:\n{listing}\n\n"
            "Core test code must not depend on isaaclab_tasks. Rebuild the fixture with core APIs or, "
            "for a genuine cross-package integration test, coordinate its relocation before adding a "
            "temporary handoff allowlist entry."
        )


def test_boundary_guard_covers_shared_fixtures_and_tests():
    """The guard must scan both the test suite and shared in-package fixtures."""
    scanned = set(_iter_core_test_files())
    assert Path(__file__).resolve() in scanned, "Discovery did not include the test-suite scan root."

    env_cfgs = _REPO_ROOT / "source" / "isaaclab" / "isaaclab" / "test" / "env_cfgs.py"
    assert env_cfgs in scanned, "Shared fixture env_cfgs.py is not covered by the boundary guard."
    assert env_cfgs in set(_GUARDED_FILES), "Shared fixture env_cfgs.py must not be exempt from the guard."


def test_import_detector_flags_representative_task_imports():
    """The detector must flag direct task imports, including type-only imports."""
    source = (
        "from typing import TYPE_CHECKING\n"
        "import isaaclab_tasks\n"
        "from isaaclab_tasks.utils.parse_cfg import parse_env_cfg\n"
        "import isaaclab_tasks.core.cartpole as cartpole\n"
        "if TYPE_CHECKING:\n"
        "    from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg\n"
    )
    offenders = _iter_banned_imports(source, "<representative>", _BANNED_PACKAGES)
    assert len(offenders) == 4
    assert set(offenders) == {
        ("isaaclab_tasks", "import isaaclab_tasks"),
        ("isaaclab_tasks", "from isaaclab_tasks.utils.parse_cfg import parse_env_cfg"),
        ("isaaclab_tasks", "import isaaclab_tasks.core.cartpole"),
        (
            "isaaclab_tasks",
            "from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg",
        ),
    }


def test_import_detector_ignores_out_of_scope_references():
    """The detector must ignore core, sibling, string, and dynamic references."""
    source = (
        "import importlib\n"
        "import isaaclab\n"
        "from isaaclab.envs import ManagerBasedEnv\n"
        "import isaaclab_tasks_experimental\n"
        "from isaaclab_tasks_experimental.foo import bar\n"
        'PACKAGES = ["isaaclab_tasks", "isaaclab_tasks_experimental"]\n'
        'importlib.import_module("isaaclab_tasks")\n'
    )
    assert _iter_banned_imports(source, "<out-of-scope>", _BANNED_PACKAGES) == []


def test_boundary_generalizes_to_additional_packages_via_config_only():
    """The detector can enforce another package through configuration alone."""
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
    """Each handoff entry must exist and still directly import its package."""
    path = _REPO_ROOT / rel_path
    assert path.is_file(), f"Stale handoff allowlist entry (file moved or removed): {rel_path}"
    offenders = _iter_banned_imports(path.read_text(encoding="utf-8"), str(path), (package,))
    assert offenders, f"Allowlisted file {rel_path} no longer imports {package}; remove it from the allowlist."
