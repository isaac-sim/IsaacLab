# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency and coverage contracts for the direct rendering suite."""

import ast
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from rendering_cases import KIT_RENDERING_CASES, KITLESS_RENDERING_CASES

_SUITE = Path(__file__).parent
_SOURCE = _SUITE.parents[2]


def test_rendering_is_downstream_and_task_independent() -> None:
    """Runtime packages cannot depend on this test project; its scenes cannot depend on tasks."""
    project = tomllib.loads((_SUITE.parents[1] / "pyproject.toml").read_text())
    assert project["tool"]["setuptools"] == {"packages": [], "py-modules": []}
    for manifest in _SOURCE.glob("isaaclab*/pyproject.toml"):
        runtime = tomllib.loads(manifest.read_text())["project"]
        dependencies = list(runtime.get("dependencies", []))
        for extra in runtime.get("optional-dependencies", {}).values():
            dependencies.extend(extra)
        assert project["project"]["name"] not in {Requirement(dep).name for dep in dependencies}
    for file in _SUITE.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            assert not any(name.startswith(("isaaclab_tasks", "gymnasium")) for name in imports), file


def test_case_and_golden_inventory() -> None:
    """Each case is unique; both OVRTX readers share one complete reference inventory."""
    kit_ids = [case.id for case in KIT_RENDERING_CASES]
    kitless_ids = [f"{stage}-{case.id}" for stage, case in KITLESS_RENDERING_CASES]
    assert len(kit_ids) == len(set(kit_ids))
    assert len(kitless_ids) == len(set(kitless_ids))
    expected = {
        Path(case.scene) / case.golden_filename(aov, "kit") for case in KIT_RENDERING_CASES for aov in case.golden_aovs
    } | {
        Path(case.scene) / case.golden_filename(aov) for _, case in KITLESS_RENDERING_CASES for aov in case.golden_aovs
    }
    root = _SUITE / "golden_images/renderers"
    assert {path.relative_to(root) for path in root.rglob("*.png")} == expected
    subsets = tomllib.loads((_SOURCE.parent / ".github/test-subsets/postmerge-rendering.toml").read_text())
    nodes = {
        f"{_SUITE.relative_to(_SOURCE.parent)}/test_rendering_kit.py::test_rendering[{case}]" for case in kit_ids
    } | {
        f"{_SUITE.relative_to(_SOURCE.parent)}/test_rendering_kitless.py::test_rendering[{case}]"
        for case in kitless_ids
    }
    for subset in subsets.values():
        for node in subset:
            assert (_SOURCE.parent / node.split("::")[0]).is_file(), node
            if "_isaaclab_testing" in node:
                assert node in nodes, node
