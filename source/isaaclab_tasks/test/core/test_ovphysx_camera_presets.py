# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX camera-task preset resolution."""

import sys

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_ovphysx.physics import OvPhysxCfg

from isaaclab.app import scan
from isaaclab.sim import CuboidCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize(
    "task_name,presets",
    [
        ("Isaac-Reorient-Cube-Shadow-Camera-Direct", "ovphysx,ovrtx,rgb"),
        ("Isaac-Reorient-KukaAllegro-Camera", "ovphysx,ovrtx,rgb64,single_camera,cube"),
        ("Isaac-Lift-KukaAllegro-Camera", "ovphysx,ovrtx,rgb64,single_camera,cube"),
    ],
)
def test_ovphysx_camera_preset_resolves_kitless(task_name: str, presets: str):
    """OVPhysX camera presets should select kitless physics and rendering backends."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], f"presets={presets}"]
        env_cfg, _ = resolve_task_config(task_name, "rsl_rl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    camera_cfg = env_cfg.tiled_camera if hasattr(env_cfg, "tiled_camera") else env_cfg.scene.base_camera

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(camera_cfg.renderer_cfg, OVRTXRendererCfg)
    assert scan(env_cfg).needs_kit is False


@pytest.mark.parametrize(
    "task_name",
    [
        "Isaac-Reorient-KukaAllegro-Camera",
        "Isaac-Lift-KukaAllegro-Camera",
    ],
)
def test_kuka_ovphysx_preset_uses_supported_task_configuration(task_name: str):
    """Kuka OVPhysX presets should select supported object, event, and curriculum configs."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], "presets=ovphysx,ovrtx,rgb64,single_camera"]
        env_cfg, _ = resolve_task_config(task_name, "rsl_rl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    robot_asset_cfg = env_cfg.events.robot_physics_material.params["asset_cfg"]
    object_asset_cfg = env_cfg.events.object_physics_material.params["asset_cfg"]

    assert isinstance(env_cfg.scene.object.spawn, CuboidCfg)
    assert robot_asset_cfg.body_names == ".*"
    assert object_asset_cfg.body_names == ".*"
    assert env_cfg.events.variable_gravity is None
    assert env_cfg.curriculum.gravity_adr is None
