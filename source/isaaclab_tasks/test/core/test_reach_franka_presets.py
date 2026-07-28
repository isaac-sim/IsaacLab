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
_CONTRIB_DIFFIK_TASK = "IsaacContrib-Reach-Franka-IK-Rel"
_NEWTON_IK_TASK = "Isaac-Reach-Franka-Newton-IK-Rel"
_NEWTON_IK_V0_TASK = "Isaac-Reach-Franka-Newton-IK-Rel-v0"
_OSC_TASK = "Isaac-Reach-Franka-OSC"


def _load_env_cfg(*presets: str):
    cfg = load_cfg_from_registry(_TASK, "env_cfg_entry_point")
    return resolve_presets(cfg, selected=presets)


def _without_actions(cfg):
    cfg_dict = cfg.to_dict()
    cfg_dict.pop("actions")
    return cfg_dict


def test_reach_uses_controller_neutral_franka_config_module():
    entry_point = registry[_TASK].kwargs["env_cfg_entry_point"]

    assert entry_point.endswith(".franka_reach_env_cfg:FrankaReachEnvCfg")
    assert _CONTRIB_DIFFIK_TASK not in registry
    assert _NEWTON_IK_TASK not in registry
    assert _NEWTON_IK_V0_TASK not in registry


@pytest.mark.parametrize(
    ("presets", "action_type", "physics_type"),
    [
        ((), "JointPositionActionCfg", "NewtonCfg"),
        (("isaacsim_physx",), "JointPositionActionCfg", "PhysxCfg"),
        (("newton_mjwarp",), "JointPositionActionCfg", "NewtonCfg"),
        (("ovphysx",), "JointPositionActionCfg", "OvPhysxCfg"),
        (("diffik",), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
        (("diffik", "isaacsim_physx"), "DifferentialInverseKinematicsActionCfg", "PhysxCfg"),
        (("diffik", "newton_mjwarp"), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
        (("newton_ik", "newton_mjwarp"), "NewtonInverseKinematicsActionCfg", "NewtonCfg"),
    ],
)
def test_reach_action_and_physics_presets_resolve_supported_combinations(presets, action_type, physics_type):
    cfg = _load_env_cfg(*presets)

    cfg.validate()
    assert type(cfg.actions.arm_action).__name__ == action_type
    assert type(cfg.sim.physics).__name__ == physics_type
    assert type(cfg.scene.table).__name__ == "AssetBaseCfg"
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.decimation == 4
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1.0 / 30.0)


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


def test_reach_penalizes_action_magnitude_rate_and_physical_joint_motion():
    rewards = _load_env_cfg().rewards
    action_rate = rewards.action_rate
    action_magnitude = rewards.action_magnitude
    joint_vel = rewards.joint_vel

    assert action_rate.func.__name__ == "action_rate_l2"
    assert action_rate.weight == pytest.approx(-0.0001)
    assert action_magnitude.func.__name__ == "action_l2"
    assert action_magnitude.weight == pytest.approx(-0.005)
    assert joint_vel.func.__name__ == "joint_vel_l2"
    assert joint_vel.weight == pytest.approx(-0.0001)
    assert joint_vel.params["asset_cfg"].joint_names == ["panda_joint.*"]
    assert _load_env_cfg().curriculum.joint_vel.params == {
        "term_name": "joint_vel",
        "weight": -0.001,
        "num_steps": 4500,
    }
    assert _load_env_cfg().curriculum.action_rate.params == {
        "term_name": "action_rate",
        "weight": -0.005,
        "num_steps": 4500,
    }
    assert "end_effector_position_tracking_fine_grained" not in rewards.to_dict()


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


def test_reach_uses_menagerie_franka():
    robot = _load_env_cfg().scene.robot

    assert robot.spawn.usd_path.endswith("/Robots/FrankaEmika/franka_panda.usda")
    assert list(robot.actuators) == ["panda_arm", "panda_hand"]
    assert robot.actuators["panda_arm"].velocity_limit_sim == {
        "panda_joint[1-4]": 20.0,
        "panda_joint[5-7]": 25.0,
    }
    assert robot.actuators["panda_arm"].stiffness == {
        "panda_joint[1-2]": 1000.0,
        "panda_joint[3-4]": 750.0,
        "panda_joint[5-7]": 300.0,
    }
    assert robot.actuators["panda_arm"].damping == {
        "panda_joint[1-2]": 20.0,
        "panda_joint[3-4]": 4.0,
        "panda_joint[5-7]": 2.0,
    }
    assert robot.actuators["panda_arm"].armature is None
    assert robot.actuators["panda_hand"].stiffness is None
    assert robot.actuators["panda_hand"].damping is None


def test_reach_osc_uses_menagerie_franka_effort_actuator():
    cfg = load_cfg_from_registry(_OSC_TASK, "env_cfg_entry_point")

    assert cfg.scene.robot.spawn.usd_path.endswith("/Robots/FrankaEmika/franka_panda.usda")
    assert cfg.scene.robot.actuators["panda_arm"].stiffness == 0.0
    assert cfg.scene.robot.actuators["panda_arm"].damping == 0.0


def test_reach_newton_uses_high_frequency_solver_timing():
    cfg = _load_env_cfg("newton_mjwarp")

    assert cfg.sim.physics.use_cuda_graph is True
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.sim.physics.solver_cfg.update_data_interval == 2
    assert cfg.sim.dt / cfg.sim.physics.num_substeps == pytest.approx(1.0 / 240.0)
    assert cfg.decimation * cfg.sim.physics.num_substeps == 8


def test_reach_joint_position_action_configuration():
    action = _load_env_cfg().actions.arm_action

    assert action.asset_name == "robot"
    assert action.joint_names == ["panda_joint.*"]
    assert action.scale == 0.5
    assert action.use_default_offset is True
    assert action.clip is None


def test_reach_diffik_action_configuration():
    action = _load_env_cfg("diffik").actions.arm_action

    assert action.asset_name == "robot"
    assert action.joint_names == ["panda_joint.*"]
    assert action.body_name == "panda_hand"
    assert action.controller.command_type == "pose"
    assert action.controller.use_relative_mode is True
    assert action.controller.ik_method == "dls"
    assert action.controller.ik_params == {"lambda_val": 0.01}
    assert action.scale == (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    assert action.body_offset.pos == [0.0, 0.0, 0.107]
    assert action.clip is None


def test_reach_diffik_action_configuration_is_identical_across_backends():
    physx_action = _load_env_cfg("diffik", "isaacsim_physx").actions.arm_action
    newton_action = _load_env_cfg("diffik", "newton_mjwarp").actions.arm_action

    assert physx_action.to_dict() == newton_action.to_dict()
    assert newton_action.controller.ik_params == {"lambda_val": 0.01}
    assert newton_action.scale == (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    assert newton_action.clip is None


def test_reach_newton_ik_action_configuration():
    cfg = _load_env_cfg("newton_ik", "newton_mjwarp")
    action = cfg.actions.arm_action
    diffik_action = _load_env_cfg("diffik", "newton_mjwarp").actions.arm_action

    assert action.asset_name == "robot"
    assert action.joint_names == ["panda_joint.*"]
    assert action.controller.optimizer == "lm"
    assert action.controller.jacobian_mode == "analytic"
    assert action.controller.iterations == 4
    assert len(action.objectives) == 2
    assert action.objectives[0].body_name == "panda_hand"
    assert action.objectives[0].body_offset_pos == (0.0, 0.0, 0.107)
    assert action.objectives[0].command_type == "pose"
    assert action.objectives[0].use_relative_mode is True
    assert action.objectives[0].scale == diffik_action.scale == (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    assert action.objectives[0].rotation_weight == 2.0
    assert action.objectives[1].weight == 0.1
    assert action.clip is None


def test_reach_newton_ik_rejects_physx():
    cfg = _load_env_cfg("newton_ik", "isaacsim_physx")

    with pytest.raises(ValueError, match="requires a Newton physics preset"):
        cfg.validate()
