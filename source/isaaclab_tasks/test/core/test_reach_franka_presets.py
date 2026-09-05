# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from gymnasium.envs.registration import registry

import isaaclab.envs.mdp as mdp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-Reach-Franka"
_CONTRIB_DIFFIK_ABS_TASK = "IsaacContrib-Reach-Franka-IK-Abs"


def _load_env_cfg(*presets: str):
    return _load_reach_env_cfg(_TASK, *presets)


def _load_reach_env_cfg(task: str, *presets: str):
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    return resolve_presets(cfg, selected=presets)


def _without_actions(cfg):
    cfg_dict = cfg.to_dict()
    cfg_dict.pop("actions")
    return cfg_dict


def test_reach_diffik_abs_legacy_task_is_a_deprecated_alias():
    spec = registry[_CONTRIB_DIFFIK_ABS_TASK]

    assert spec.kwargs["deprecated"] == {"alias": "--task Isaac-Reach-Franka physics=isaacsim_physx presets=diffik_abs"}
    with pytest.warns(FutureWarning, match="presets=diffik_abs"):
        legacy_cfg = load_cfg_from_registry(_CONTRIB_DIFFIK_ABS_TASK, "env_cfg_entry_point")

    canonical_cfg = _load_env_cfg("diffik_abs", "isaacsim_physx")
    legacy_cfg = resolve_presets(legacy_cfg)
    assert legacy_cfg.to_dict() == canonical_cfg.to_dict()


_REACH_PRESET_CASES = [
    (_TASK, (), "JointPositionActionCfg", "NewtonCfg"),
    (_TASK, ("isaacsim_physx",), "JointPositionActionCfg", "PhysxCfg"),
    (_TASK, ("newton_mjwarp",), "JointPositionActionCfg", "NewtonCfg"),
    (_TASK, ("ovphysx",), "JointPositionActionCfg", "OvPhysxCfg"),
    (_TASK, ("diffik",), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
    (_TASK, ("diffik", "isaacsim_physx"), "DifferentialInverseKinematicsActionCfg", "PhysxCfg"),
    (_TASK, ("diffik", "newton_mjwarp"), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
    (_TASK, ("diffik_abs", "isaacsim_physx"), "DifferentialInverseKinematicsActionCfg", "PhysxCfg"),
    (_TASK, ("diffik_abs", "newton_mjwarp"), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
    (_TASK, ("diffik_abs", "ovphysx"), "DifferentialInverseKinematicsActionCfg", "OvPhysxCfg"),
    (_TASK, ("newton_ik", "newton_mjwarp"), "NewtonInverseKinematicsActionCfg", "NewtonCfg"),
    ("Isaac-Reach-UR10", (), "JointPositionActionCfg", "NewtonCfg"),
    ("Isaac-Reach-UR10", ("isaacsim_physx",), "JointPositionActionCfg", "PhysxCfg"),
    ("Isaac-Reach-UR10", ("newton_mjwarp",), "JointPositionActionCfg", "NewtonCfg"),
]


@pytest.mark.parametrize(
    ("task", "presets", "action_type", "physics_type"),
    [pytest.param(*case, id=f"{case[0]}-{'-'.join(case[1]) or 'default'}") for case in _REACH_PRESET_CASES],
)
def test_reach_presets_resolve_supported_combinations(task, presets, action_type, physics_type):
    cfg = _load_reach_env_cfg(task, *presets)

    cfg.validate()
    assert type(cfg.actions.arm_action).__name__ == action_type
    assert type(cfg.sim.physics).__name__ == physics_type


def test_reach_ur10_physics_presets_change_only_physics():
    """UR10 backend selections must preserve the task configuration."""
    physx = _load_reach_env_cfg("Isaac-Reach-UR10", "isaacsim_physx")
    newton = _load_reach_env_cfg("Isaac-Reach-UR10", "newton_mjwarp")

    physx_cfg = physx.to_dict()
    newton_cfg = newton.to_dict()
    physx_cfg["sim"].pop("physics")
    newton_cfg["sim"].pop("physics")
    assert physx_cfg == newton_cfg


def test_reach_action_presets_change_only_the_action_configuration():
    joint_pos_physx = _load_env_cfg("joint_pos", "isaacsim_physx")
    diffik_physx = _load_env_cfg("diffik", "isaacsim_physx")
    joint_pos_newton = _load_env_cfg("joint_pos", "newton_mjwarp")
    diffik_newton = _load_env_cfg("diffik", "newton_mjwarp")
    newton_ik = _load_env_cfg("newton_ik", "newton_mjwarp")

    assert _load_env_cfg().actions.arm_action.to_dict() == joint_pos_newton.actions.arm_action.to_dict()
    assert _without_actions(joint_pos_physx) == _without_actions(diffik_physx)
    assert _without_actions(joint_pos_newton) == _without_actions(diffik_newton)
    assert _without_actions(joint_pos_newton) == _without_actions(newton_ik)


def test_reach_success_requires_position_and_orientation():
    cfg = _load_env_cfg()
    success = cfg.terminations.success

    assert cfg.commands.ee_pose.position_success_threshold == pytest.approx(0.05)
    assert cfg.commands.ee_pose.orientation_success_threshold == pytest.approx(0.2)
    assert success.func is mdp.pose_command_success
    assert success.params == {"command_name": "ee_pose"}
    assert cfg.rewards.success.func.__name__ == "is_terminated_term"
    assert cfg.rewards.success.weight == pytest.approx(10.0)
    assert cfg.rewards.success.params == {"term_keys": ["success"]}

    angles = torch.tensor([0.19, 0.19, 0.21])
    body_quaternions = torch.zeros(3, 1, 4)
    body_quaternions[:, 0, 2] = torch.sin(angles / 2)
    body_quaternions[:, 0, 3] = torch.cos(angles / 2)
    command_values = torch.zeros(3, 7)
    command_values[:, 6] = 1.0
    robot_data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.zeros(3, 3)),
        root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)),
        body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.04, 0.0, 0.0]], [[0.06, 0.0, 0.0]], [[0.04, 0.0, 0.0]]])),
        body_quat_w=SimpleNamespace(torch=body_quaternions),
    )
    command = object.__new__(mdp.UniformPoseCommand)
    command.robot = SimpleNamespace(data=robot_data)
    command.body_idx = 0
    command.pose_command_b = command_values
    command.pose_command_w = torch.zeros_like(command_values)
    command.cfg = cfg.commands.ee_pose
    command._env = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
    command._track_success = True
    command._succeeded = torch.zeros(3, dtype=torch.bool)

    class CommandManager:
        def get_term(self, name):
            assert name == "ee_pose"
            return command

    env = SimpleNamespace(command_manager=CommandManager())
    succeeded = mdp.pose_command_success(env, **success.params)

    assert torch.equal(succeeded, torch.tensor([True, False, False]))
    assert torch.equal(command._succeeded, succeeded)

    command.cfg = command.cfg.replace(orientation_success_threshold=None)
    command._succeeded.zero_()
    position_only_succeeded = mdp.pose_command_success(env, **success.params)

    assert torch.equal(position_only_succeeded, torch.tensor([True, False, True]))


def test_reach_newton_ik_rejects_physx():
    cfg = _load_env_cfg("newton_ik", "isaacsim_physx")

    with pytest.raises(ValueError, match="requires a Newton physics preset"):
        cfg.validate()
