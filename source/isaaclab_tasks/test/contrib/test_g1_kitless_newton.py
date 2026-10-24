# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for the G1 kitless Newton task boundary."""

import math
import re
from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.physics import NewtonCfg

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)
from isaaclab_tasks.contrib.locomanip_pick_place.g1_kitless_newton import (
    ACTUATOR_PROFILE,
    JOINT_LIMIT_MARGIN_RAD,
    JOINT_TARGET_MAX_STEP_RAD,
    POSITION_MAX_STEP_M,
    ROTATION_MAX_STEP_RAD,
    G1NewtonBoundedPinkInverseKinematicsAction,
    bound_pink_joint_targets,
    condition_kitless_newton_g1_action,
    teleop_action_target_issue,
)

RECORDED_FIRST_ACTION = torch.tensor(
    [
        -0.11201994,
        0.43588385,
        0.89166349,
        -0.12487421,
        -0.11215594,
        0.69946498,
        0.69467705,
        0.14433567,
        0.41181007,
        0.91755670,
        0.00914924,
        -0.05320583,
        0.65801126,
        0.75107032,
        *([0.0] * 14),
    ],
    dtype=torch.float32,
)


def _quat_angle(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, dim=0)
    b = torch.nn.functional.normalize(b, dim=0)
    return float(2.0 * torch.acos(torch.clamp(torch.abs(torch.sum(a * b)), 0.0, 1.0)))


def _fake_env():
    left = torch.tensor([0.23491086, 0.21602505, 0.63518274, -0.13474128, 0.79248637, -0.02669795, -0.59422028])
    right = torch.tensor([-0.30430639, 0.52459145, 0.78784978, 0.19782960, 0.52823287, 0.41972736, 0.71109974])
    poses = torch.stack((left, right)).unsqueeze(0)
    robot = SimpleNamespace(
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        data=SimpleNamespace(body_link_pose_w=poses),
    )
    cfg = SimpleNamespace(
        actions=SimpleNamespace(
            upper_body_ik=SimpleNamespace(
                target_eef_link_names={
                    "left_wrist": "left_wrist_yaw_link",
                    "right_wrist": "right_wrist_yaw_link",
                }
            )
        )
    )
    return SimpleNamespace(scene={"robot": robot}, cfg=cfg), left, right


def test_g1_newton_contract_is_backend_scoped_and_complete() -> None:
    default_cfg = FixedBaseUpperBodyIKG1EnvCfg()
    newton_cfg = FixedBaseUpperBodyIKG1EnvCfg()
    newton_cfg.sim.physics = NewtonCfg()
    newton_cfg.stabilize_for_kitless_newton_mjwarp()

    assert newton_cfg.sim.physics.solver_cfg.nconmax == 4096
    assert newton_cfg.sim.physics.solver_cfg.njmax == 1536
    for name, selected in ACTUATOR_PROFILE.items():
        actuator = newton_cfg.scene.robot.actuators[name]
        assert {key: getattr(actuator, key) for key in selected} == selected
    assert newton_cfg.actions.upper_body_ik.class_type is G1NewtonBoundedPinkInverseKinematicsAction
    assert newton_cfg.isaac_teleop.action_conditioner is condition_kitless_newton_g1_action
    assert newton_cfg.isaac_teleop.action_validator is teleop_action_target_issue
    assert newton_cfg.scene.packing_table.spawn._packing_table_collision_prepared is True
    assert newton_cfg.scene.object.spawn._kitless_newton_wheel_collision is True

    assert default_cfg.isaac_teleop.action_conditioner is None
    assert default_cfg.isaac_teleop.action_validator is None
    assert not hasattr(default_cfg.scene.packing_table.spawn, "_packing_table_collision_prepared")
    assert not hasattr(default_cfg.scene.object.spawn, "_kitless_newton_wheel_collision")


@pytest.mark.parametrize("task", ["fixed_base", "locomanipulation"])
def test_g1_contact_mappings_are_bilateral_and_nested(task) -> None:
    from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg

    cfg = FixedBaseUpperBodyIKG1EnvCfg() if task == "fixed_base" else LocomanipulationG1EnvCfg()

    assert cfg.scene.left_hand_contact.prim_path == "{ENV_REGEX_NS}/Robot/left_hand/left_hand_[^/]*_link"
    assert cfg.scene.right_hand_contact.prim_path == "{ENV_REGEX_NS}/Robot/right_hand/right_hand_[^/]*_link"
    assert cfg.haptic_feedback.left_sensor_name == "left_hand_contact"
    assert cfg.haptic_feedback.right_sensor_name == "right_hand_contact"

    compiled_labels = (
        "/World/envs/env_0/Robot/left_hand/left_hand_index_0_link",
        "/World/envs/env_0/Robot/right_hand/right_hand_thumb_2_link",
        "/World/envs/env_0/Robot/torso_link",
    )
    left_pattern = cfg.scene.left_hand_contact.prim_path.replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+")
    right_pattern = cfg.scene.right_hand_contact.prim_path.replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+")
    left = [label for label in compiled_labels if re.fullmatch(left_pattern, label)]
    right = [label for label in compiled_labels if re.fullmatch(right_pattern, label)]
    assert left == [compiled_labels[0]]
    assert right == [compiled_labels[1]]
    assert set(left).isdisjoint(right)


def test_locomanipulation_preserves_agile_policy_order_and_action_contract() -> None:
    from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg

    cfg = LocomanipulationG1EnvCfg()
    controller = cfg.actions.lower_body_joint_pos.copy()
    upper_body_class = cfg.actions.upper_body_ik.class_type
    cfg.sim.physics = NewtonCfg()
    cfg.stabilize_for_kitless_newton_mjwarp()

    assert cfg.scene.robot.joint_ordering == "physx"
    assert cfg.actions.lower_body_joint_pos == controller
    assert cfg.actions.upper_body_ik.class_type == upper_body_class
    assert cfg.isaac_teleop.action_conditioner is None
    assert cfg.sim.physics.solver_cfg.integrator == "implicitfast"
    assert cfg.sim.physics.num_substeps == 2


def test_recorded_first_action_is_ramped_and_finite() -> None:
    env, left, right = _fake_env()
    conditioned = condition_kitless_newton_g1_action(env, RECORDED_FIRST_ACTION, {})

    assert teleop_action_target_issue(conditioned) is None
    assert torch.isfinite(conditioned).all()
    assert float(torch.linalg.norm(conditioned[:3] - left[:3])) <= POSITION_MAX_STEP_M + 1.0e-6
    assert float(torch.linalg.norm(conditioned[7:10] - right[:3])) <= POSITION_MAX_STEP_M + 1.0e-6
    assert _quat_angle(conditioned[3:7], left[3:7]) <= ROTATION_MAX_STEP_RAD + 1.0e-5
    assert _quat_angle(conditioned[10:14], right[3:7]) <= ROTATION_MAX_STEP_RAD + 1.0e-5
    assert torch.equal(conditioned[14:], RECORDED_FIRST_ACTION[14:])


@pytest.mark.parametrize(
    "angle, axis, sign", [(0.9, (1, 0, 0), 1), (1.2, (0, 1, 1), -1), (0.1, (0, 0, 1), 1), (0, (1, 0, 0), -1)]
)
def test_conditioned_wrist_rotation_obeys_angular_cap(angle, axis, sign):
    """The public action conditioner follows the short rotation without exceeding its cap."""
    env, _, _ = _fake_env()
    pose = env.scene["robot"].data.body_link_pose_w
    pose[0, :, 3:7] = torch.tensor([1.0, 0, 0, 0])
    direction = torch.nn.functional.normalize(torch.tensor(axis, dtype=torch.float32), dim=0)
    target = sign * torch.cat((torch.tensor([math.cos(angle / 2)]), direction * math.sin(angle / 2)))
    action = RECORDED_FIRST_ACTION.clone()
    action[3:7] = action[10:14] = target
    conditioned = condition_kitless_newton_g1_action(env, action, {})
    identity = torch.tensor([1.0, 0, 0, 0])
    for result in (conditioned[3:7], conditioned[10:14]):
        assert torch.isfinite(result).all()
        assert torch.linalg.norm(result).item() == pytest.approx(1, abs=1.0e-6)
        assert _quat_angle(identity, result) <= min(angle, ROTATION_MAX_STEP_RAD) + 1.0e-5
        assert _quat_angle(result, target) == pytest.approx(max(0, angle - ROTATION_MAX_STEP_RAD), abs=1.0e-5)


def test_invalid_actions_fail_closed() -> None:
    nonfinite = RECORDED_FIRST_ACTION.clone()
    nonfinite[0] = float("nan")
    assert teleop_action_target_issue(nonfinite) == "non-finite action target"

    oversized = RECORDED_FIRST_ACTION.clone()
    oversized[14] = 7.0
    assert teleop_action_target_issue(oversized) == "unreasonable hand joint target magnitude"
    assert "unexpected G1 controller action size" in teleop_action_target_issue(torch.zeros(27))

    missing_wrist_env = SimpleNamespace(scene={}, cfg=SimpleNamespace())
    with pytest.raises(RuntimeError, match="could not resolve the task-selected G1 wrist poses"):
        condition_kitless_newton_g1_action(missing_wrist_env, RECORDED_FIRST_ACTION, {})


def test_pink_targets_are_slew_limited_and_clamped_to_common_limits() -> None:
    current = torch.tensor([0.25, 0.0, -0.5])
    proposed = torch.tensor([0.5215, -4.0, 4.0])
    lower = torch.tensor([-0.2617994, -2.0, -1.0])
    upper = torch.tensor([0.2617994, 2.0, 1.0])

    bounded = bound_pink_joint_targets(current, proposed, lower, upper)

    assert torch.all(bounded >= lower + JOINT_LIMIT_MARGIN_RAD)
    assert torch.all(bounded <= upper - JOINT_LIMIT_MARGIN_RAD)
    assert float(bounded[0]) == pytest.approx(0.2617994 - JOINT_LIMIT_MARGIN_RAD)
    assert float(bounded[1]) == pytest.approx(-JOINT_TARGET_MAX_STEP_RAD)
    assert float(bounded[2]) == pytest.approx(-0.5 + JOINT_TARGET_MAX_STEP_RAD)


def test_nonfinite_pink_target_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="non-finite Pink joint target boundary"):
        bound_pink_joint_targets(torch.zeros(1), torch.tensor([float("nan")]), -torch.ones(1), torch.ones(1))


def test_pink_limit_cache_keeps_live_articulation_limits(monkeypatch):
    from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

    proposed = torch.tensor([[0.4]])
    monkeypatch.setattr(PinkInverseKinematicsAction, "_compute_ik_solutions", lambda self: proposed)
    action = G1NewtonBoundedPinkInverseKinematicsAction.__new__(G1NewtonBoundedPinkInverseKinematicsAction)
    action._env = SimpleNamespace(device="cpu")
    limits = torch.tensor([[[-1.0, 1.0]]])
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.tensor([[0.0]])),
            joint_pos_limits=SimpleNamespace(torch=limits),
        )
    )
    action._isaaclab_controlled_joint_ids = [0]
    action._ik_controllers = [
        SimpleNamespace(
            pink_configuration=SimpleNamespace(
                controlled_model=SimpleNamespace(
                    lowerPositionLimit=[-0.5],
                    upperPositionLimit=[0.5],
                )
            ),
            pink_to_isaac_lab_controlled_ordering=[0],
        )
    ]
    action._pink_joint_limits = None
    torch.testing.assert_close(action._compute_ik_solutions(), torch.tensor([[0.1]]))
    cached = action._pink_joint_limits
    limits[0, 0, 1] = 0.05
    torch.testing.assert_close(action._compute_ik_solutions(), torch.tensor([[0.049]]))
    assert action._pink_joint_limits is cached
