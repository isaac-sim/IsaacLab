# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Franka deformable lift camera benchmark task configs."""

from __future__ import annotations

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.benchmark.franka_soft.franka_cloth_camera_env_cfg import FrankaClothCameraEnvCfg
from isaaclab_tasks.benchmark.franka_soft.franka_soft_camera_env_cfg import (
    FrankaSoftCameraEnvCfg,
    FrankaSoftCameraSceneCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import PhysicsCfg as FrankaClothPhysicsCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    FrankaSoftSceneCfg as FrankaSoftCoreSceneCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def test_tasks_registered():
    assert "Isaac-Lift-Cloth-Franka-Camera-Benchmark" in gym.registry
    assert "Isaac-Lift-Soft-Franka-Camera-Benchmark" in gym.registry


def test_load_env_cfg_from_registry():
    cloth_cfg = load_cfg_from_registry("Isaac-Lift-Cloth-Franka-Camera-Benchmark", "env_cfg_entry_point")
    soft_cfg = load_cfg_from_registry("Isaac-Lift-Soft-Franka-Camera-Benchmark", "env_cfg_entry_point")
    assert isinstance(cloth_cfg, FrankaClothCameraEnvCfg)
    assert isinstance(soft_cfg, FrankaSoftCameraEnvCfg)


def test_data_type_preset_cascades_to_camera_and_observation():
    cloth_cfg = resolve_presets(FrankaClothCameraEnvCfg(), {"depth"})
    soft_cfg = resolve_presets(FrankaSoftCameraEnvCfg(), {"depth"})
    assert cloth_cfg is not None
    assert soft_cfg is not None
    assert cloth_cfg.scene.tiled_camera.data_types == ["depth"]
    assert cloth_cfg.observations.policy.image.params["data_type"] == "depth"
    assert soft_cfg.scene.tiled_camera.data_types == ["depth"]
    assert soft_cfg.observations.policy.image.params["data_type"] == "depth"


def test_camera_configs_match_core_physics_presets():
    cloth_core_presets = set(collect_presets(FrankaClothPhysicsCfg())[""]) - {"default"}
    cloth_camera_presets = set(collect_presets(FrankaClothCameraEnvCfg())["sim.physics"]) - {"default"}
    assert cloth_camera_presets == cloth_core_presets

    soft_core_presets = set(collect_presets(FrankaSoftCoreSceneCfg())[""]) - {"default"}
    soft_camera_presets = set(collect_presets(FrankaSoftCameraSceneCfg())[""]) - {"default"}
    assert soft_camera_presets == soft_core_presets


def test_soft_scene_preserves_physx_replication_setting():
    scene_cfg = resolve_presets(FrankaSoftCameraSceneCfg(), {"physx"})
    assert scene_cfg is not None
    assert scene_cfg.replicate_physics is False


def test_soft_scene_preserves_newton_replication_setting():
    scene_cfg = resolve_presets(FrankaSoftCameraSceneCfg(), {"newton_mjwarp_vbd"})
    assert scene_cfg is not None
    assert scene_cfg.replicate_physics is True
