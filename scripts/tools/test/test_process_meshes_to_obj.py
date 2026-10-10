# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.tools import process_meshes_to_obj

pytestmark = pytest.mark.unit


def test_blender_command_preserves_paths_with_spaces(monkeypatch):
    blender_path = "/path to/blender"
    input_path = "/path to/input mesh.stl"
    output_path = "/path to/output mesh.obj"
    run_mock = Mock()
    monkeypatch.setattr(process_meshes_to_obj, "BLENDER_EXE_PATH", blender_path)
    monkeypatch.setattr(process_meshes_to_obj.subprocess, "run", run_mock)

    process_meshes_to_obj.run_blender_convert2obj(input_path, output_path)

    script_path = str(Path(process_meshes_to_obj.__file__).with_name("blender_obj.py"))
    run_mock.assert_called_once_with(
        [blender_path, "--background", "--python", script_path, "--", "-i", input_path, "-o", output_path],
        check=True,
    )


def test_blender_conversion_requires_executable(monkeypatch):
    monkeypatch.setattr(process_meshes_to_obj, "BLENDER_EXE_PATH", None)

    with pytest.raises(FileNotFoundError, match="Blender executable"):
        process_meshes_to_obj.run_blender_convert2obj("input.stl", "output.obj")
