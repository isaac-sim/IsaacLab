# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for the public Newton-backend solver preset names.

These tests prevent reintroduction of the legacy preset names ``newton`` and
``kamino``, which were renamed to ``newton_mjwarp`` and ``newton_kamino`` to
disambiguate from the Newton backend label, package, and visualizer (which all
retain the bare ``newton`` spelling).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_TEXT_ROOTS = (
    _REPO_ROOT / "docs" / "source",
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "source" / "isaaclab_tasks",
    _REPO_ROOT / "source" / "isaaclab",
    _REPO_ROOT / "source" / "isaaclab_visualizers",
)
_TEXT_SUFFIXES = {".py", ".rst", ".md"}

# Files that intentionally reference the deprecated names (deprecation tests,
# the alias map, and this scanner's own pattern strings).
_SCAN_EXCLUDE_FILES = frozenset(
    {
        "test_hydra.py",
        "test_newton_solver_preset_names.py",
        "hydra.py",
    }
)

# Legacy preset names that must not appear as public preset references in
# user-facing surfaces. Word-boundary anchors (``\b``) keep ``newton_mjwarp``,
# ``newton_renderer``, ``newton_kamino`` and similar prefixed names from
# matching, since ``_`` is a Python word character.
_LEGACY_TOKEN = r"(?:newton|kamino)"
_LEGACY_PRESET_PATTERNS = (
    # CLI: ``presets=...,newton`` or ``presets=newton,...``
    re.compile(rf"presets=[^\s`]*\b{_LEGACY_TOKEN}\b"),
    # CLI: ``env.<path>=newton`` or ``env.<path>=kamino``
    re.compile(rf"\benv\.[\w.]*={_LEGACY_TOKEN}\b"),
    # Dict-literal preset entry, e.g. ``"newton": NewtonCfg(...)``
    re.compile(rf"""[\"']{_LEGACY_TOKEN}[\"']\s*:\s*\w*Cfg\b"""),
    # ``PresetCfg`` field declaration, e.g. ``    newton: NewtonCfg = ...``
    re.compile(rf"^\s+{_LEGACY_TOKEN}\s*:\s*[A-Za-z_][\w.]*Cfg\b", re.MULTILINE),
    # ``PresetCfg`` field assignment, e.g. ``    kamino = NewtonCfg(...)``
    re.compile(rf"^\s+{_LEGACY_TOKEN}\s*=\s*[A-Za-z_][\w.]*Cfg\(", re.MULTILINE),
)

# RST inline literals are caught only for files known to enumerate per-task
# preset choices, where bare ``newton``/``kamino`` literals must refer to the
# (renamed) physics preset rather than the Newton backend, package, or
# visualizer that share the spelling.
_PRESET_TABLE_FILES = ("docs/source/overview/environments.rst",)
_LEGACY_RST_LITERAL_PATTERN = re.compile(rf"``{_LEGACY_TOKEN}``")

# Field declaration scoped to physics-related Cfg classes only (used by the
# task-level scanner to keep its message focused on env config files).
_LEGACY_PHYSICS_FIELD_PATTERN = re.compile(
    rf"^\s+{_LEGACY_TOKEN}\s*(?::\s*(?:NewtonCfg|SimulationCfg)\b|=\s*(?:NewtonCfg|SimulationCfg)\()",
    re.MULTILINE,
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in _TEXT_ROOTS:
        files.extend(path for path in root.rglob("*") if path.suffix in _TEXT_SUFFIXES)
    return sorted(files)


def test_public_examples_use_renamed_preset_names() -> None:
    """Public examples should use ``newton_mjwarp``/``newton_kamino``, not the legacy names."""
    files = _iter_text_files()
    assert files, f"no text files scanned -- _TEXT_ROOTS likely stale: {_TEXT_ROOTS!r}"

    offenders: list[str] = []
    for path in files:
        if "CHANGELOG" in path.name or "changelog.d" in path.parts:
            continue
        if path.name in _SCAN_EXCLUDE_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _LEGACY_PRESET_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {match.group(0)}")
        if path.relative_to(_REPO_ROOT).as_posix() in _PRESET_TABLE_FILES:
            for match in _LEGACY_RST_LITERAL_PATTERN.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {match.group(0)}")

    assert not offenders, "Legacy Newton-backend solver preset references found:\n" + "\n".join(offenders)


def test_rendering_test_utils_maps_newton_label_to_newton_mjwarp() -> None:
    """Golden-image fixtures keep the ``"newton"`` backend label; the helper must
    map it to the ``"newton_mjwarp"`` preset so Hydra resolves the right config.
    """
    import sys

    import pytest

    pytest.importorskip("torch")  # rendering_test_utils imports torch eagerly

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from rendering_test_utils import _physics_preset_name
    finally:
        sys.path.pop(0)

    assert _physics_preset_name("newton") == "newton_mjwarp"
    assert _physics_preset_name("physx") == "physx"
    assert _physics_preset_name("ovphysx") == "ovphysx"


def test_task_physics_presets_use_renamed_field_names() -> None:
    """Task ``PresetCfg`` field names should use ``newton_mjwarp`` / ``newton_kamino``."""
    tasks_root = _REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks"
    files = sorted(tasks_root.rglob("*.py"))
    assert files, f"no task config files scanned at {tasks_root}"

    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in _LEGACY_PHYSICS_FIELD_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {match.group(0).strip()}")

    assert not offenders, "Legacy Newton-backend solver field declarations found:\n" + "\n".join(offenders)
