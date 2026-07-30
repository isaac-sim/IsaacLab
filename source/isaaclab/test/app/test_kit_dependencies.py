# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""App-free checks for Isaac Lab Kit experience dependencies."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_APPS_DIR = Path(__file__).resolve().parents[4] / "apps"
_BASE_KITS = (
    "isaaclab.python.kit",
    "isaaclab.python.headless.kit",
)
_REPLICATOR_BUNDLED_DEPENDENCIES = (
    "omni.kit.pip_archive",
    "omni.warp.core",
)


def _read_kit(name: str) -> str:
    """Return the text of kit ``name`` under ``apps/``."""
    return (_APPS_DIR / name).read_text()


def _direct_dependencies(content: str) -> set[str]:
    """Return extension names declared in all ``[dependencies]`` sections."""
    dependencies = set()
    in_dependencies = False
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("["):
            in_dependencies = stripped_line == "[dependencies]"
        elif in_dependencies:
            match = re.match(r'^"([^"]+)"\s*=', stripped_line)
            if match is not None:
                dependencies.add(match.group(1))
    return dependencies


def _excluded_extensions(content: str) -> set[str]:
    """Return extension names listed in an ``excluded`` setting."""
    match = re.search(r"(?:app\.extensions\.)?excluded\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if match is None:
        return set()
    return set(re.findall(r'"([^"]+)"', match.group(1)))


@pytest.mark.parametrize("kit_name", _BASE_KITS)
def test_base_kits_enable_replicator_for_isaac_rtx_renderer(kit_name):
    """Base Kit experiences explicitly enable Replicator for the Isaac RTX renderer."""
    dependencies = _direct_dependencies(_read_kit(kit_name))
    assert "omni.replicator.core" in dependencies


@pytest.mark.parametrize("kit_name", _BASE_KITS)
def test_base_kits_exclude_replicator_bundled_python_dependencies(kit_name):
    """Base Kit experiences keep Replicator's bundled Python dependencies disabled."""
    excluded_extensions = _excluded_extensions(_read_kit(kit_name))
    assert set(_REPLICATOR_BUNDLED_DEPENDENCIES) <= excluded_extensions
