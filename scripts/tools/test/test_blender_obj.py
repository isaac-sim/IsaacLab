# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import sys
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def blender_obj_module(monkeypatch):
    fake_bpy = Mock()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("scripts.tools.blender_obj", None)
    return importlib.import_module("scripts.tools.blender_obj"), fake_bpy


def test_convert_to_obj_supports_current_directory_output(blender_obj_module, tmp_path):
    blender_obj, fake_bpy = blender_obj_module
    input_path = tmp_path / "input.stl"
    input_path.touch()

    blender_obj.convert_to_obj(str(input_path), "robot.obj")

    fake_bpy.ops.export_scene.obj.assert_called_once_with(
        filepath="robot.obj",
        check_existing=False,
        axis_forward="Y",
        axis_up="Z",
        global_scale=1,
        path_mode="RELATIVE",
    )


def test_save_usd_replaces_only_output_extension(blender_obj_module, tmp_path):
    blender_obj, fake_bpy = blender_obj_module
    input_path = tmp_path / "input.stl"
    input_path.touch()
    output_path = tmp_path / "objects" / "robot.obj"

    blender_obj.convert_to_obj(str(input_path), str(output_path), save_usd=True)

    fake_bpy.ops.wm.usd_export.assert_called_once_with(
        filepath=str(output_path.with_suffix(".usd")), check_existing=False
    )
