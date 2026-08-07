# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-Reach-UR10"


def _load_env_cfg(*presets: str):
    cfg = load_cfg_from_registry(_TASK, "env_cfg_entry_point")
    return resolve_presets(cfg, selected=presets)


def _without_physics(cfg):
    cfg_dict = cfg.to_dict()
    cfg_dict["sim"].pop("physics")
    return cfg_dict


@pytest.mark.parametrize(
    ("presets", "physics_type"),
    [
        ((), "NewtonCfg"),
        (("isaacsim_physx",), "PhysxCfg"),
        (("newton_mjwarp",), "NewtonCfg"),
    ],
)
def test_reach_ur10_physics_presets_resolve(presets, physics_type):
    cfg = _load_env_cfg(*presets)

    cfg.validate()
    assert type(cfg.actions.arm_action).__name__ == "JointPositionActionCfg"
    assert type(cfg.sim.physics).__name__ == physics_type
    assert type(cfg.scene.table).__name__ == "AssetBaseCfg"
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.decimation == 4
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1.0 / 30.0)


def test_reach_ur10_physics_presets_change_only_physics():
    physx = _load_env_cfg("isaacsim_physx")
    newton = _load_env_cfg("newton_mjwarp")

    assert _without_physics(physx) == _without_physics(newton)


def test_reach_ur10_uses_shared_mdp_and_arm_motion_penalty():
    cfg = _load_env_cfg("isaacsim_physx")

    assert cfg.actions.arm_action.joint_names == [".*"]
    assert cfg.actions.arm_action.scale == pytest.approx(0.5)
    assert cfg.actions.arm_action.clip is None
    assert cfg.rewards.joint_vel.params["asset_cfg"].joint_names == [".*"]
    assert cfg.rewards.action_rate.weight == pytest.approx(-0.0001)
    assert cfg.rewards.action_magnitude.weight == pytest.approx(-0.005)
    assert cfg.rewards.joint_vel.weight == pytest.approx(-0.0001)
    assert cfg.terminations.success.params == {
        "command_name": "ee_pose",
    }
    assert cfg.commands.ee_pose.position_success_threshold == pytest.approx(0.05)
    assert cfg.commands.ee_pose.orientation_success_threshold == pytest.approx(0.2)
