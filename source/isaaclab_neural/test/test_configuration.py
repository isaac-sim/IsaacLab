# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration import isolation and the neural library dependency boundary."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_configuration_imports_without_tasks_or_native_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class Missing:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'isaaclab_tasks', 'newton_neural'}:
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Missing())
from isaaclab_neural import NeRDCfg, NeRDTrainingCfg
from isaaclab_neural.physics import NeRDSolverCfg, MJWarpNeRDSolverCfg
from isaaclab_neural.record.recording_cfg import NeRDRecorderCfg
for cfg in (NeRDCfg, NeRDTrainingCfg, NeRDSolverCfg, MJWarpNeRDSolverCfg, NeRDRecorderCfg):
    cfg()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_library_has_no_task_composition():
    package = Path(__file__).resolve().parents[1]
    assert not (package / "test/test_nerd_presets.py").exists()
    library = package / "isaaclab_neural"
    assert not list((library / "entrypoints").glob("*.py"))
    forbidden = {
        "collect_presets",
        "enumerate_task_presets",
        "resolve_presets",
        "resolve_task_config",
        "setup_preset_cli",
    }
    for path in (*library.rglob("*.py"), *library.rglob("*.pyi")):
        assert "preset" not in path.name, path
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("isaaclab_tasks"), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("isaaclab_tasks") for alias in node.names), path
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden, path
