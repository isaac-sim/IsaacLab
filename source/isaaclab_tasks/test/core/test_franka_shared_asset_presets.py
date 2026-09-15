# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared Franka Menagerie asset contract across core tasks."""

import re

import pytest

from isaaclab.sim import CuboidCfg, MultiAssetSpawnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetSceneCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_ENV_ROOT = "/World/envs/env_0"
_MENAGERIE_ARM_PATH = (
    f"{_ENV_ROOT}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
    "panda_link4/panda_link5/panda_link6/panda_link7"
)


def _matches_environment_path(pattern: str, path: str) -> bool:
    return re.fullmatch(pattern.replace("{ENV_REGEX_NS}", _ENV_ROOT), path) is not None


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-Reach-Franka",
        "Isaac-Reach-Franka-OSC",
        "Isaac-Lift-Franka",
        "Isaac-Reorient-Franka",
        "Isaac-Open-Drawer-Franka",
        "Isaac-Open-Drawer-Franka-Direct",
    ],
)
@pytest.mark.parametrize(
    ("physics_preset", "expected_variants"),
    [
        ("isaacsim_physx", {"Physics": "physx", "Colliders": "gripper_only"}),
        ("newton_mjwarp", {"Physics": "mujoco", "Colliders": "gripper_only"}),
    ],
)
def test_rigid_franka_tasks_use_validated_collider_lods(
    task: str, physics_preset: str, expected_variants: dict[str, str]
) -> None:
    """Rigid tasks use one Menagerie root and the validated collider preset for each backend."""
    cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=(physics_preset,))

    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/franka_panda.usda")
    assert cfg.scene.robot.spawn.variants == expected_variants
    assert all(actuator.viscous_friction == 0.0 for actuator in cfg.scene.robot.actuators.values())


def test_lift_auto_physx_uses_ovphysx_compatible_object_setup() -> None:
    """Automatic PhysX keeps Lift compatible with the kitless OvPhysX fast path."""
    auto_cfg = resolve_presets(load_cfg_from_registry("Isaac-Lift-Franka", "env_cfg_entry_point"), selected=("physx",))
    isaacsim_cfg = resolve_presets(
        load_cfg_from_registry("Isaac-Lift-Franka", "env_cfg_entry_point"), selected=("isaacsim_physx",)
    )

    assert isinstance(auto_cfg.scene.object.spawn, CuboidCfg)
    assert isinstance(isaacsim_cfg.scene.object.spawn, MultiAssetSpawnerCfg)


@pytest.mark.parametrize(
    ("task", "selected_presets"),
    [
        ("Isaac-Lift-Soft-Franka", ("isaacsim_physx",)),
        ("Isaac-Lift-Soft-Franka", ("newton_mjwarp_vbd_proxy",)),
        ("Isaac-Lift-Cloth-Franka", ("isaacsim_physx",)),
        ("Isaac-Lift-Cloth-Franka", ("newton_mjwarp_vbd_proxy",)),
        ("Isaac-Lift-Cable-Franka", ("newton_mjwarp_vbd_proxy",)),
        ("Isaac-Lift-Soft-Franka-Camera", ("isaacsim_physx", "isaacsim_rtx")),
        ("Isaac-Lift-Soft-Franka-Camera", ("newton_mjwarp_vbd_proxy", "newton_renderer")),
        ("Isaac-Lift-Cloth-Franka-Camera", ("isaacsim_physx", "isaacsim_rtx")),
        ("Isaac-Lift-Cloth-Franka-Camera", ("newton_mjwarp_vbd_proxy", "newton_renderer")),
        ("Isaac-Lift-Cable-Franka-Camera", ("newton_mjwarp_vbd_proxy", "newton_renderer")),
    ],
)
def test_deformable_franka_tasks_use_gripper_only_colliders(task: str, selected_presets: tuple[str, ...]) -> None:
    """Deformable state and camera tasks avoid unrelated full-arm contact geometry."""
    cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=selected_presets)

    expected_physics = "physx" if selected_presets[0] == "isaacsim_physx" else "mujoco"
    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/franka_panda.usda")
    assert cfg.scene.robot.spawn.variants == {"Physics": expected_physics, "Colliders": "gripper_only"}
    assert all(actuator.viscous_friction == 0.0 for actuator in cfg.scene.robot.actuators.values())


def test_franka_cabinet_frame_paths_match_menagerie_hierarchy() -> None:
    """Cabinet frame selectors resolve every required Menagerie body."""
    cfg = FrankaCabinetSceneCfg(num_envs=1, env_spacing=2.0)
    targets = {target.name: target.prim_path for target in cfg.ee_frame.target_frames}
    paths = {
        cfg.ee_frame.prim_path: f"{_ENV_ROOT}/Robot/Geometry/panda_link0",
        targets["ee_tcp"]: f"{_MENAGERIE_ARM_PATH}/panda_hand",
        targets["tool_leftfinger"]: f"{_MENAGERIE_ARM_PATH}/panda_hand/panda_leftfinger",
        targets["tool_rightfinger"]: f"{_MENAGERIE_ARM_PATH}/panda_hand/panda_rightfinger",
    }

    assert all(_matches_environment_path(pattern, path) for pattern, path in paths.items())
