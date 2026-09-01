# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral regressions for the Deploy GearAssembly configuration."""

import math
from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from pxr import Usd

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.deploy.gear_assembly.config.rizon_4s.ik_newton_env_cfg import (
    Rizon4sGearAssemblyIKNewtonEnvCfg,
)
from isaaclab_tasks.contrib.deploy.gear_assembly.config.rizon_4s.joint_pos_env_cfg import (
    Rizon4sGearAssemblyEnvCfg,
)
from isaaclab_tasks.contrib.deploy.gear_assembly.gear_assembly_env_cfg import NEWTON_GEAR_ASSETS_DIR
from isaaclab_tasks.contrib.deploy.mdp.events import randomize_gears_and_base_pose
from isaaclab_tasks.contrib.deploy.mdp.rewards import keypoint_ee_grasp_error
from isaaclab_tasks.contrib.deploy.mdp.terminations import reset_when_gear_dropped
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_RIZON_NEWTON_IK_TASK = "IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-Newton-IK"
_GEAR_ASSET_NAMES = ("factory_gear_small", "factory_gear_medium", "factory_gear_large")
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


class _TensorData:
    def __init__(self, value: torch.Tensor):
        self.torch = value


class _RecordingAsset:
    def __init__(
        self,
        default_root_pose: torch.Tensor,
        root_link_pos_w: torch.Tensor | None = None,
        root_link_quat_w: torch.Tensor | None = None,
    ):
        num_envs = default_root_pose.shape[0]
        self.data = SimpleNamespace(
            default_root_pose=_TensorData(default_root_pose),
            default_root_vel=_TensorData(torch.zeros((num_envs, 6))),
            root_link_pos_w=_TensorData(
                default_root_pose[:, :3].clone() if root_link_pos_w is None else root_link_pos_w
            ),
            root_link_quat_w=_TensorData(
                default_root_pose[:, 3:7].clone() if root_link_quat_w is None else root_link_quat_w
            ),
        )
        self.pose_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.velocity_writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_root_pose_to_sim_index(self, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.pose_writes.append((root_pose.clone(), env_ids.clone()))

    def write_root_velocity_to_sim_index(self, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.velocity_writes.append((root_velocity.clone(), env_ids.clone()))


class _RecordingRobot(_RecordingAsset):
    _BODY_INDICES = {"flange": 0, "left_finger_tip": 1, "right_finger_tip": 2}

    def __init__(self, body_positions: torch.Tensor):
        num_envs = body_positions.shape[0]
        default_pose = torch.tensor([(*([0.0] * 3), *_IDENTITY_QUAT)]).repeat(num_envs, 1)
        super().__init__(default_pose)
        self.data.body_link_pos_w = _TensorData(body_positions)
        self.data.body_link_quat_w = _TensorData(
            torch.tensor(_IDENTITY_QUAT).repeat(num_envs, body_positions.shape[1], 1)
        )

    def find_bodies(self, body_names: list[str]) -> tuple[list[int], list[str]]:
        return [self._BODY_INDICES[name] for name in body_names], body_names


class _FakeScene(dict):
    def __init__(self, assets: dict[str, _RecordingAsset], num_envs: int):
        super().__init__(assets)
        self.env_origins = torch.zeros((num_envs, 3))


class _GearTypeManager:
    def __init__(self, selected_indices: torch.Tensor):
        self.selected_indices = selected_indices

    def get_all_gear_type_indices(self) -> torch.Tensor:
        return self.selected_indices


def _pose(position: tuple[float, float, float], num_envs: int = 1) -> torch.Tensor:
    return torch.tensor([(*position, *_IDENTITY_QUAT)], dtype=torch.float32).repeat(num_envs, 1)


def _gear_assets(num_envs: int, root_positions: torch.Tensor | None = None) -> dict[str, _RecordingAsset]:
    assets = {}
    for gear_index, name in enumerate(_GEAR_ASSET_NAMES):
        default_pose = _pose((0.0, 0.0, 0.0), num_envs)
        position = None if root_positions is None else root_positions + torch.tensor([0.0, 0.0, gear_index])
        assets[name] = _RecordingAsset(default_pose, root_link_pos_w=position)
    return assets


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-Deploy-GearAssembly-UR10e-2F140",
        "IsaacContrib-Deploy-GearAssembly-UR10e-2F85",
    ],
)
def test_ur10e_gear_assembly_default_num_envs(task_name: str):
    """UR10e GearAssembly training configs should fit on 16 GB GPUs by default."""
    env_cfg = parse_env_cfg(task_name)

    assert env_cfg.scene.num_envs == 2048


def test_rizon_newton_ik_registered_default_is_usable():
    """The registered Newton-IK task should select a supported Newton shard by default."""
    env_cfg = parse_env_cfg(_RIZON_NEWTON_IK_TASK)

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert isinstance(env_cfg.actions.arm_action, NewtonInverseKinematicsActionCfg)
    assert env_cfg.scene.num_envs == 256
    env_cfg.validate()


@pytest.mark.parametrize("preset_name", ["newton_mjwarp", "newton_sdf", "newton_hydroelastic"])
def test_rizon_newton_presets_use_supported_shard(preset_name: str):
    """Every Newton solver preset should resolve to the supported per-rank shard size."""
    env_cfg = resolve_presets(Rizon4sGearAssemblyIKNewtonEnvCfg(), {preset_name})

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.scene.num_envs == 256
    env_cfg.validate()


def test_rizon_newton_ik_rejects_physx():
    """Newton IK must not pass validation with a PhysX backend."""
    env_cfg = resolve_presets(Rizon4sGearAssemblyIKNewtonEnvCfg(), {"physx"})

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    with pytest.raises(ValueError, match="Newton inverse-kinematics actions require a Newton physics preset"):
        env_cfg.validate()


def test_rizon_newton_collision_capacity_tracks_environment_count():
    """Newton collision presets should reject shards larger than their configured capacity."""
    from isaaclab_tasks.contrib.deploy.gear_assembly.gear_assembly_env_cfg import (
        _GEAR_TRIANGLE_PAIRS_PER_ENV,
    )

    env_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_sdf"})
    env_cfg.scene.num_envs += 1

    with pytest.raises(ValueError, match="triangle pairs per environment"):
        env_cfg.validate()

    env_cfg.sim.physics.collision_cfg.max_triangle_pairs = env_cfg.scene.num_envs * _GEAR_TRIANGLE_PAIRS_PER_ENV
    env_cfg.validate()


def test_reset_composes_shaft_offsets_in_randomized_base_frame():
    """Reset placement should rotate and translate each shaft offset with the base."""
    base = _RecordingAsset(_pose((2.0, 3.0, 0.0)))
    assets = {"factory_gear_base": base, **_gear_assets(1)}
    scene = _FakeScene(assets, 1)
    scene.env_origins[:] = torch.tensor([[10.0, -2.0, 0.0]])
    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        scene=scene,
        _gear_type_manager=_GearTypeManager(torch.tensor([0])),
    )
    term = randomize_gears_and_base_pose(SimpleNamespace(), env)
    gear_offsets = {
        "gear_small": [1.0, 0.0, 0.0],
        "gear_medium": [0.0, 2.0, 0.0],
        "gear_large": [-1.0, 0.0, 0.0],
    }

    term(
        env,
        torch.tensor([0]),
        pose_range={"yaw": (math.pi / 2, math.pi / 2)},
        gear_pos_range={"z": (0.2, 0.2)},
        gear_offsets=gear_offsets,
        seated_gear_z_offset=0.05,
    )

    base_world_position = torch.tensor([12.0, 1.0, 0.0])
    expected_positions = (
        base_world_position + torch.tensor([0.0, 1.0, 0.2]),
        base_world_position + torch.tensor([-2.0, 0.0, 0.05]),
        base_world_position + torch.tensor([0.0, -1.0, 0.05]),
    )
    for asset_name, expected_position in zip(_GEAR_ASSET_NAMES, expected_positions, strict=True):
        written_pose, written_env_ids = assets[asset_name].pose_writes[-1]
        torch.testing.assert_close(written_pose[0, :3], expected_position, atol=1.0e-5, rtol=0.0)
        torch.testing.assert_close(written_env_ids, torch.tensor([0]))


def test_pin_term_writes_only_unselected_gears_including_empty_masks():
    """The per-step pin term should preserve selected gears and accept empty masked writes."""
    from isaaclab_tasks.contrib.deploy.gear_assembly.config.rizon_4s.events import (
        pin_unselected_gears_to_shafts,
    )

    num_envs = 4
    base_positions = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    base = _RecordingAsset(_pose((0.0, 0.0, 0.0), num_envs), root_link_pos_w=base_positions)
    assets = {"factory_gear_base": base, **_gear_assets(num_envs)}
    manager = _GearTypeManager(torch.zeros(num_envs, dtype=torch.long))
    env = SimpleNamespace(device="cpu", num_envs=num_envs, scene=_FakeScene(assets, num_envs))
    env._gear_type_manager = manager
    gear_offsets = {
        "gear_small": [1.0, 0.0, 0.0],
        "gear_medium": [0.0, 2.0, 0.0],
        "gear_large": [-1.0, 0.0, 0.0],
    }
    cfg = SimpleNamespace(params={"gear_offsets": gear_offsets, "seated_gear_z_offset": 0.05})
    term = pin_unselected_gears_to_shafts(cfg, env)

    term(env, None, gear_offsets, 0.05)
    empty_pose, empty_env_ids = assets["factory_gear_small"].pose_writes[-1]
    assert empty_pose.shape == (0, 7)
    assert empty_env_ids.shape == (0,)

    for gear in assets.values():
        gear.pose_writes.clear()
    manager.selected_indices = torch.tensor([0, 1, 2, 0])
    term(env, None, gear_offsets, 0.05)

    expected_written_envs = {
        "factory_gear_small": torch.tensor([1, 2]),
        "factory_gear_medium": torch.tensor([0, 2, 3]),
        "factory_gear_large": torch.tensor([0, 1, 3]),
    }
    for asset_name, expected_env_ids in expected_written_envs.items():
        _, written_env_ids = assets[asset_name].pose_writes[-1]
        torch.testing.assert_close(written_env_ids, expected_env_ids)


def test_reward_and_drop_distance_use_fingertip_midpoint():
    """Grasp reward and drop detection should use the midpoint of explicit fingertip bodies."""
    body_positions = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    robot = _RecordingRobot(body_positions)
    gear_assets = _gear_assets(1, root_positions=torch.tensor([[1.0, 0.0, 0.0]]))
    env = SimpleNamespace(device="cpu", num_envs=1, scene=_FakeScene({"robot": robot, **gear_assets}, 1))
    env._gear_type_manager = _GearTypeManager(torch.tensor([0]))

    reward_term = object.__new__(keypoint_ee_grasp_error)
    reward_term.robot_asset = robot
    reward_term.eef_idx = 0
    reward_term.grasp_center_body_indices = [1, 2]
    reward_term.grasp_rot_offset_tensor = torch.tensor([_IDENTITY_QUAT])
    reward_term.gear_grasp_offsets_stacked = torch.zeros((3, 3))
    reward_term.gear_type_indices = torch.tensor([0])
    reward_term._get_selected_gear_poses = lambda _env: (
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([_IDENTITY_QUAT]),
    )

    eef_position, _, target_position, _ = reward_term._get_grasp_corrected_target(env)
    torch.testing.assert_close(eef_position, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(eef_position, target_position)

    drop_cfg = SimpleNamespace(
        params={
            "robot_asset_cfg": SimpleNamespace(name="robot"),
            "end_effector_body_name": "flange",
            "grasp_rot_offset": _IDENTITY_QUAT,
            "gear_offsets_grasp": {
                "gear_small": [0.0, 0.0, 0.0],
                "gear_medium": [0.0, 0.0, 0.0],
                "gear_large": [0.0, 0.0, 0.0],
            },
            "grasp_center_body_names": ("left_finger_tip", "right_finger_tip"),
        }
    )
    drop_term = reset_when_gear_dropped(drop_cfg, env)
    assert not drop_term(env, distance_threshold=0.1).item()

    robot.data.body_link_pos_w.torch[:, 1:3] += 3.0
    assert drop_term(env, distance_threshold=0.1).item()


def test_rizon_gear_assets_author_required_sdf_metadata():
    """Package-local gear colliders should expose the schemas required by both SDF backends."""
    point_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_sdf"})
    hydro_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_hydroelastic"})
    for asset_name in ("factory_gear_base", *_GEAR_ASSET_NAMES):
        point_spawn = getattr(point_cfg.scene, asset_name).spawn
        hydro_spawn = getattr(hydro_cfg.scene, asset_name).spawn
        assert point_spawn.collision_props.hydroelastic_enabled is False
        assert hydro_spawn.collision_props.hydroelastic_enabled is True

        usd_path = f"{NEWTON_GEAR_ASSETS_DIR}/{asset_name}/{asset_name}.usda"
        stage = Usd.Stage.Open(usd_path)
        sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]
        assert len(sdf_prims) == 1

        sdf_prim = sdf_prims[0]
        authored_schemas = set(sdf_prim.GetPrimStack()[0].GetInfo("apiSchemas").prependedItems)
        assert {
            "PhysicsCollisionAPI",
            "PhysicsMeshCollisionAPI",
            "PhysxSDFMeshCollisionAPI",
            "NewtonSDFCollisionAPI",
        } <= authored_schemas
        assert sdf_prim.GetAttribute("physics:approximation").Get() == "sdf"
