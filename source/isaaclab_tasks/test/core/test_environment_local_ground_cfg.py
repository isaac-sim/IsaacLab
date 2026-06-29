# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative task scene configuration normalization."""

import copy

import gymnasium as gym
import pytest

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import resolve_cfg_presets

from isaaclab_tasks.contrib.cabinet.config.franka.ik_abs_env_cfg import FrankaCabinetEnvCfg as FrankaCabinetIKAbsEnvCfg
from isaaclab_tasks.contrib.cabinet.config.franka.ik_rel_env_cfg import FrankaCabinetEnvCfg as FrankaCabinetIKRelEnvCfg
from isaaclab_tasks.contrib.lift.config.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg
from isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg
from isaaclab_tasks.core.dexsuite.config.kuka_allegro.camera_cfg import RAYCASTER_CAMERA_MESH_PRIM_PATHS
from isaaclab_tasks.core.reach.config.ur_10.joint_pos_env_cfg import UR10ReachEnvCfg
from isaaclab_tasks.core.reach.reach_env_cfg import TableCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_reach_and_lift_tables_share_one_definition():
    """Resolved Reach and Lift tables should differ only by their root binding."""
    reach_table_presets = TableCfg()
    ur10_table = resolve_cfg_presets(UR10ReachEnvCfg()).scene.table
    lift_table = OpenArmCubeLiftEnvCfg().scene.table

    assert ur10_table.prim_path == "{ENV_REGEX_NS}/Table"
    assert reach_table_presets.newton_mjwarp.prim_path == "{ENV_REGEX_NS}/Table"

    ur10_definition = copy.deepcopy(ur10_table)
    lift_definition = copy.deepcopy(lift_table)
    ur10_definition.prim_path = lift_definition.prim_path = "{SCENE_SLOT}"
    assert ur10_definition == lift_definition


def test_task_scenes_keep_native_environment_local_grounds():
    """Each representative task should keep its native floor height in every environment."""
    openarm_scene = OpenArmCubeLiftEnvCfg().scene
    franka_scene = FrankaCabinetEnvCfg().scene
    ur10_scene = resolve_cfg_presets(UR10ReachEnvCfg()).scene

    grounds = (openarm_scene.plane, franka_scene.plane, ur10_scene.ground)
    scenes = (openarm_scene, franka_scene, ur10_scene)
    assert all(ground.prim_path == "{ENV_REGEX_NS}/GroundPlane" for ground in grounds)
    assert all(ground.spawn.size == (2.0, 2.0) for ground in grounds)
    assert all(ground.spawn.size[0] <= scene.env_spacing for scene, ground in zip(scenes, grounds, strict=True))
    assert all(ground.collision_group == 0 for ground in grounds)
    assert all(scene.filter_collisions for scene in scenes)
    assert [ground.init_state.pos[2] for ground in grounds] == [-1.05, 0.0, -1.05]
    assert franka_scene.robot.init_state.pos[2] == 0.0
    assert franka_scene.cabinet.init_state.pos[2] == 0.4


def test_franka_cabinet_ik_scenes_are_declarative():
    """IK variants should replace the robot in their scene type without post-init mutation."""
    joint_scene = FrankaCabinetEnvCfg().scene
    ik_abs_scene = FrankaCabinetIKAbsEnvCfg().scene
    ik_rel_scene = FrankaCabinetIKRelEnvCfg().scene

    assert ik_abs_scene.robot == ik_rel_scene.robot
    assert ik_abs_scene.robot != joint_scene.robot
    assert ik_abs_scene.ee_frame == ik_rel_scene.ee_frame == joint_scene.ee_frame


def test_dexsuite_raycaster_targets_environment_local_ground():
    """Dexsuite ray-caster presets should follow the per-environment ground path."""
    target_paths = [target.prim_expr for target in RAYCASTER_CAMERA_MESH_PRIM_PATHS]

    assert "{ENV_REGEX_NS}/GroundPlane" in target_paths
    assert "/World/GroundPlane" not in target_paths


def _registered_task_ids() -> list[str]:
    """Return task IDs owned by the isaaclab_tasks package."""
    task_ids = []
    for task_spec in gym.registry.values():
        entry_point = task_spec.kwargs.get("env_cfg_entry_point")
        if isinstance(entry_point, str):
            module_name = entry_point.split(":", maxsplit=1)[0]
        else:
            module_name = getattr(entry_point, "__module__", type(entry_point).__module__)
        if module_name.startswith("isaaclab_tasks."):
            task_ids.append(task_spec.id)
    return sorted(task_ids)


def _scene_fields(scene_cfg: InteractiveSceneCfg) -> dict[str, object]:
    """Return user-declared scene fields."""
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    return {name: value for name, value in vars(scene_cfg).items() if name not in base_fields and value is not None}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_registered_flat_task_grounds_are_environment_local():
    """Every declarative flat task should use bounded, environment-local ground assets."""
    checked_ids = []
    flat_terrain_ids = []

    for task_id in _registered_task_ids():
        scene_cfg = parse_env_cfg(task_id, device="cpu").scene
        fields = _scene_fields(scene_cfg)
        terrains = [value for value in fields.values() if isinstance(value, TerrainImporterCfg)]
        if any(terrain.terrain_type != "plane" for terrain in terrains):
            continue

        grounds = [
            value
            for value in fields.values()
            if isinstance(value, AssetBaseCfg) and isinstance(value.spawn, sim_utils.GroundPlaneCfg)
        ]
        if not grounds and not terrains:
            continue

        assert scene_cfg.filter_collisions, task_id
        for ground in grounds:
            assert ground.prim_path == "{ENV_REGEX_NS}/GroundPlane", task_id
            assert ground.collision_group == 0, task_id
            assert ground.spawn.size[0] <= scene_cfg.env_spacing, task_id
            assert ground.spawn.size[1] <= scene_cfg.env_spacing, task_id

        if terrains:
            flat_terrain_ids.append(task_id)
        checked_ids.append(task_id)

    assert checked_ids
    assert flat_terrain_ids
