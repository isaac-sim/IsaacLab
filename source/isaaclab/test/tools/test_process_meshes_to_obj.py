# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


def _load_process_meshes_module():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "tools" / "process_meshes_to_obj.py"
    spec = importlib.util.spec_from_file_location("test_process_meshes_to_obj_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blender_command_preserves_paths_with_spaces(monkeypatch, tmp_path):
    module = _load_process_meshes_module()
    blender_path = str(tmp_path / "Blender App" / "blender")
    input_path = str(tmp_path / "input meshes" / "robot part.stl")
    output_path = str(tmp_path / "output meshes" / "robot part.obj")
    run_mock = Mock()
    monkeypatch.setattr(module, "BLENDER_EXE_PATH", blender_path)
    monkeypatch.setattr(module.subprocess, "run", run_mock)

    module.run_blender_convert2obj(input_path, output_path)

    script_path = str(Path(module.__file__).resolve().parent / "blender_obj.py")
    run_mock.assert_called_once_with(
        [blender_path, "--background", "--python", script_path, "--", "-i", input_path, "-o", output_path],
        check=True,
    )


def test_blender_conversion_requires_executable(monkeypatch):
    module = _load_process_meshes_module()
    monkeypatch.setattr(module, "BLENDER_EXE_PATH", None)

    with pytest.raises(FileNotFoundError, match="Blender executable"):
        module.run_blender_convert2obj("input.stl", "output.obj")
