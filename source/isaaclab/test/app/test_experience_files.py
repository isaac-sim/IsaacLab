# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shipped experience files. These inspect the files and do not launch the simulator."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

APPS_DIR = Path(__file__).resolve().parents[4] / "apps"

# ``.kit`` files repeat section headers, so they cannot be parsed as TOML.
_SECTION_RE = re.compile(r"^\s*\[+(?P<name>[^\[\]]+)\]+\s*$")
_DEPENDENCY_RE = re.compile(r'^\s*"(?P<name>[^"]+)"\s*=')


def _kit_dependencies(experience: str) -> set[str]:
    """Collect the extension names declared in an experience file's dependency sections."""
    dependencies: set[str] = set()
    in_dependencies = False

    for line in (APPS_DIR / experience).read_text(encoding="utf-8").splitlines():
        section = _SECTION_RE.match(line)
        if section is not None:
            in_dependencies = section.group("name").strip() == "dependencies"
            continue
        if in_dependencies:
            dependency = _DEPENDENCY_RE.match(line)
            if dependency is not None:
                dependencies.add(dependency.group("name"))

    return dependencies


@pytest.mark.parametrize("experience", ["isaaclab.python.kit", "isaaclab.python.headless.kit"])
def test_base_experiences_enable_native_storage(experience: str):
    """Test the base experiences load the extension that applies ``ISAACSIM_ASSET_ROOT``."""
    assert "isaacsim.storage.native" in _kit_dependencies(experience)


@pytest.mark.parametrize(
    "experience, base",
    [
        ("isaaclab.python.rendering.kit", "isaaclab.python"),
        ("isaaclab.python.xr.openxr.kit", "isaaclab.python"),
        ("isaaclab.python.headless.rendering.kit", "isaaclab.python.headless"),
        ("isaaclab.python.xr.openxr.headless.kit", "isaaclab.python.xr.openxr"),
    ],
)
def test_derived_experiences_inherit_native_storage(experience: str, base: str):
    """Test the derived experiences inherit native storage from a base experience."""
    assert base in _kit_dependencies(experience)
