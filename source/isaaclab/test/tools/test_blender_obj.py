# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


def _load_blender_obj_module(monkeypatch):
    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(
            wm=SimpleNamespace(read_factory_settings=Mock(), collada_import=Mock(), usd_export=Mock()),
            import_mesh=SimpleNamespace(stl=Mock()),
            export_scene=SimpleNamespace(obj=Mock()),
        )
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)

    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "tools" / "blender_obj.py"
    spec = importlib.util.spec_from_file_location("test_blender_obj_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_bpy


def test_convert_to_obj_supports_current_directory_output(monkeypatch, tmp_path):
    module, fake_bpy = _load_blender_obj_module(monkeypatch)
    input_path = tmp_path / "input.stl"
    input_path.touch()
    monkeypatch.chdir(tmp_path)

    module.convert_to_obj(str(input_path), "robot.obj")

    fake_bpy.ops.export_scene.obj.assert_called_once_with(
        filepath="robot.obj",
        check_existing=False,
        axis_forward="Y",
        axis_up="Z",
        global_scale=1,
        path_mode="RELATIVE",
    )


def test_save_usd_replaces_only_output_extension(monkeypatch, tmp_path):
    module, fake_bpy = _load_blender_obj_module(monkeypatch)
    input_path = tmp_path / "input.stl"
    input_path.touch()
    output_dir = tmp_path / "objects"
    output_path = output_dir / "robot.obj"

    module.convert_to_obj(str(input_path), str(output_path), save_usd=True)

    fake_bpy.ops.wm.usd_export.assert_called_once_with(
        filepath=str(output_dir / "robot.usd"), check_existing=False
    )
