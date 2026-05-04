# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for the public MJWarp preset name."""

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
_LEGACY_PRESET_PATTERNS = (
    re.compile(r"presets=[^\s`]*\bnewton\b"),
    re.compile(r"``newton(?!_renderer)(?=,|\.\.\.)"),
    re.compile(r"\benv\.[\w.]*=newton\b"),
    re.compile(r"[\"']newton[\"']\s*:\s*NewtonCfg\b"),
    re.compile(r"^\s*newton\s*:\s*[A-Za-z_][\w.]*Cfg\b", re.MULTILINE),
)
_LEGACY_PHYSICS_FIELD_PATTERN = re.compile(
    r"^\s*newton\s*(?::\s*(?:NewtonCfg|SimulationCfg)\b|=\s*(?:NewtonCfg|SimulationCfg)\()",
    re.MULTILINE,
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in _TEXT_ROOTS:
        files.extend(path for path in root.rglob("*") if path.suffix in _TEXT_SUFFIXES)
    return sorted(files)


def test_public_examples_use_mjwarp_preset_name():
    """Public examples should use ``mjwarp`` for the Newton MJWarp solver preset."""
    offenders: list[str] = []
    for path in _iter_text_files():
        if "CHANGELOG" in path.name or "changelog.d" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _LEGACY_PRESET_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {match.group(0)}")
        if path.relative_to(_REPO_ROOT).as_posix() == "docs/source/overview/environments.rst":
            for match in re.finditer(r"``newton``", text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {match.group(0)}")

    assert not offenders, "Legacy Newton preset references found:\n" + "\n".join(offenders)


def test_task_physics_presets_use_mjwarp_field_name():
    """Task physics presets should expose MJWarp as ``mjwarp``, not ``newton``."""
    offenders: list[str] = []
    tasks_root = _REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks"
    for path in sorted(tasks_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _LEGACY_PHYSICS_FIELD_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {match.group(0).strip()}")

    assert not offenders, "Legacy Newton physics preset fields found:\n" + "\n".join(offenders)
