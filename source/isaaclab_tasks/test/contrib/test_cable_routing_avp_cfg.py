# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-free tests for the YAM cable-routing AVP teleoperation configuration."""

import gymnasium as gym
import numpy as np
import pytest
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
)

import isaaclab.envs.mdp as env_mdp

import isaaclab_tasks.contrib.cable_routing  # noqa: F401
from isaaclab_tasks.contrib.cable_routing import cable_routing_avp_env_cfg as avp_cfg_module
from isaaclab_tasks.contrib.cable_routing.avp_action_adapter import (
    CableRoutingAVPActionAdapter,
    CableRoutingAVPActionAdapterCfg,
)
from isaaclab_tasks.contrib.cable_routing.cable_routing_avp_env_cfg import (
    AVP_TELEOP_ACTION_DIM,
    AVP_TELEOP_ACTION_LAYOUT,
    CableRoutingAVPTeleopEnvCfg,
    _build_yam_cable_routing_avp_pipeline,
)
from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import CableRoutingEnvCfg
from isaaclab_tasks.utils import load_cfg_from_registry


def test_cable_routing_avp_registry_resolves_single_scene_config() -> None:
    """The AVP task resolves independently from the RSL-RL training registrations."""
    task_id = "IsaacContrib-CableRouting-YAM-AVP-Teleop"
    spec = gym.spec(task_id)

    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.disable_env_checker
    assert spec.kwargs == {
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.cable_routing.cable_routing_avp_env_cfg:CableRoutingAVPTeleopEnvCfg"
        )
    }

    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert isinstance(cfg, CableRoutingAVPTeleopEnvCfg)
    assert cfg.scene.num_envs == 1
    assert not cfg.commands.route.debug_vis
    assert cfg.episode_length_s == 300.0
    assert cfg.xr.anchor_pos == (0.0, -0.65, 0.0)
    assert cfg.xr.near_plane == 0.05
    assert avp_cfg_module._TELEOP_AVAILABLE
    assert isinstance(cfg.teleop_action_adapter, CableRoutingAVPActionAdapterCfg)
    assert cfg.teleop_action_adapter.class_type is CableRoutingAVPActionAdapter
    assert not cfg.commands.route.reset_replay.enabled

    # Teleoperation intentionally replaces only the manager action terms. The
    # learning tasks retain their original 14-D relative-joint action space.
    assert isinstance(cfg.actions.left_arm, env_mdp.JointPositionActionCfg)
    assert cfg.actions.left_arm.asset_name == "yam_left"
    assert cfg.actions.left_arm.joint_names == ["joint[1-6]"]
    assert cfg.actions.left_arm.scale == 1.0
    assert not cfg.actions.left_arm.use_default_offset

    assert isinstance(cfg.actions.right_arm, NewtonInverseKinematicsActionCfg)
    assert cfg.actions.right_arm.asset_name == "yam_right"
    assert cfg.actions.right_arm.joint_names == ["joint[1-6]"]
    assert cfg.actions.right_arm.isolate_articulation_model
    assert cfg.actions.right_arm.use_cuda_graph
    assert cfg.actions.right_arm.controller.optimizer == "lm"
    assert cfg.actions.right_arm.controller.jacobian_mode == "analytic"
    assert cfg.actions.right_arm.controller.iterations == 12
    assert len(cfg.actions.right_arm.objectives) == 2
    pose_objective, limit_objective = cfg.actions.right_arm.objectives
    assert isinstance(pose_objective, NewtonIKPoseObjectiveCfg)
    assert pose_objective.body_name == "link_6"
    assert pose_objective.body_offset_pos == (0.0, -0.044, 0.1297)
    assert pose_objective.body_offset_rot == (0.0, 0.0, -0.7071067812, 0.7071067812)
    assert pose_objective.command_type == "pose"
    assert not pose_objective.use_relative_mode
    assert pose_objective.scale == 1.0
    assert isinstance(limit_objective, NewtonIKJointLimitObjectiveCfg)
    assert limit_objective.weight > 0.0

    # One hand target per 1/120 s simulation step minimizes teleoperation latency.
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.decimation == 1
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.sim.use_newton_actuators

    cfg.play_mode()
    assert cfg.scene.num_envs == 1
    assert not cfg.commands.route.debug_vis
    assert not cfg.commands.route.reset_replay.enabled


def test_cable_routing_avp_config_builds_pipeline_lazily_without_tuning_ui(monkeypatch) -> None:
    """Config construction neither imports/builds the graph nor requests the optional UI."""

    def unexpected_pipeline_build():
        raise AssertionError("the IsaacTeleop graph must be built only when its session starts")

    monkeypatch.setattr(avp_cfg_module, "_build_yam_cable_routing_avp_pipeline", unexpected_pipeline_build)
    cfg = CableRoutingAVPTeleopEnvCfg()

    assert cfg.isaac_teleop.pipeline_builder is unexpected_pipeline_build
    assert cfg.isaac_teleop.retargeters_to_tune is None
    assert cfg.isaac_teleop.target_frame_prim_path is None
    assert cfg.isaac_teleop.retargeting_execution.mode.value == "sync"
    assert not cfg.isaac_teleop.teleoperation_active_default


def test_cable_routing_avp_action_override_does_not_change_training_contract() -> None:
    """Newton IK and 120 Hz nominal control remain isolated to the interactive AVP configuration."""
    training_cfg = CableRoutingEnvCfg()
    teleop_cfg = CableRoutingAVPTeleopEnvCfg()

    assert isinstance(training_cfg.actions.left_arm, env_mdp.RelativeJointPositionActionCfg)
    assert isinstance(training_cfg.actions.right_arm, env_mdp.RelativeJointPositionActionCfg)
    assert training_cfg.decimation == 4
    assert training_cfg.commands.route.reset_replay.enabled
    assert isinstance(teleop_cfg.actions.left_arm, env_mdp.JointPositionActionCfg)
    assert isinstance(teleop_cfg.actions.right_arm, NewtonInverseKinematicsActionCfg)
    assert teleop_cfg.decimation == 1
    assert not teleop_cfg.commands.route.reset_replay.enabled

    # Direct Cartesian teleoperation needs a responsive right-arm drive, but it
    # must not silently change the dynamics used by the RL environments or the
    # left arm that is held stationary during AVP control.
    assert set(training_cfg.scene.yam_right.actuators) == {"arm", "gripper_drive", "gripper_passive"}
    assert set(teleop_cfg.scene.yam_left.actuators) == {"arm", "gripper_drive", "gripper_passive"}
    assert set(teleop_cfg.scene.yam_right.actuators) == {
        "arm_proximal",
        "arm_joint4",
        "arm_distal",
        "gripper_drive",
        "gripper_passive",
    }
    proximal = teleop_cfg.scene.yam_right.actuators["arm_proximal"]
    joint4 = teleop_cfg.scene.yam_right.actuators["arm_joint4"]
    distal = teleop_cfg.scene.yam_right.actuators["arm_distal"]
    assert (
        proximal.joint_names_expr,
        proximal.stiffness,
        proximal.damping,
        proximal.effort_limit_sim,
        proximal.velocity_limit_sim,
        proximal.armature,
    ) == (["joint[1-3]"], 160.0, 12.0, 40.0, 2.0, 0.032)
    assert (
        joint4.joint_names_expr,
        joint4.stiffness,
        joint4.damping,
        joint4.effort_limit_sim,
        joint4.velocity_limit_sim,
        joint4.armature,
    ) == (["joint4"], 80.0, 4.0, 15.0, 2.0, 0.0018)
    assert (
        distal.joint_names_expr,
        distal.stiffness,
        distal.damping,
        distal.effort_limit_sim,
        distal.velocity_limit_sim,
        distal.armature,
    ) == (["joint[5-6]"], 60.0, 5.0, 15.0, 2.0, 0.0018)


def test_cable_routing_avp_pipeline_has_complete_8d_right_hand_only_layout() -> None:
    """The graph emits only the right pinch pose and gripper in adapter order."""
    pytest.importorskip("isaacteleop")

    pipeline = _build_yam_cable_routing_avp_pipeline()
    action_type = pipeline.output_types()["action"].types[0]
    assert AVP_TELEOP_ACTION_DIM == 8
    assert action_type.shape == (AVP_TELEOP_ACTION_DIM,)
    assert AVP_TELEOP_ACTION_LAYOUT == (
        "right_pos_x",
        "right_pos_y",
        "right_pos_z",
        "right_quat_x",
        "right_quat_y",
        "right_quat_z",
        "right_quat_w",
        "right_gripper",
    )

    # TensorReorderer silently leaves unknown labels at zero, so guard both its
    # requested order and the complete resolved source-to-destination mapping.
    try:
        reorderer = pipeline.output_mapping["action"].module._target_module
        output_order = tuple(reorderer._output_order)
        resolved_destination_indices = sorted(mapping[2] for mapping in reorderer._mapping)
    except AttributeError:
        pytest.skip("relies on isaacteleop graph internals; public layout accessor unavailable")
    assert output_order == AVP_TELEOP_ACTION_LAYOUT
    assert resolved_destination_indices == list(range(AVP_TELEOP_ACTION_DIM))

    leaves = {(type(node).__name__, node.name) for node in pipeline.get_leaf_nodes()}
    assert leaves == {("HandsSource", "hands"), ("ValueInput", "world_T_anchor")}


def test_cable_routing_avp_pipeline_builds_camera_invariant_right_hand_tool_basis() -> None:
    """Map the hand plane to the YAM pad plane without trusting XR wrist axes."""
    pytest.importorskip("isaacteleop")
    from isaacteleop.retargeting_engine.interface import ComputeContext, OptionalTensorGroup, TensorGroup
    from isaacteleop.retargeting_engine.tensor_types import HandInputIndex, HandJointIndex
    from scipy.spatial.transform import Rotation

    pipeline = _build_yam_cable_routing_avp_pipeline()
    side = "right"
    hand_key = "hand_right"
    try:
        reorderer_subgraph = pipeline.output_mapping["action"].module
        pose_subgraph = reorderer_subgraph._input_connections[f"{side}_pose"].module
        pose_retargeter = pose_subgraph._target_module
    except AttributeError:
        pytest.skip("relies on isaacteleop graph internals; public retargeter accessor unavailable")

    pose_cfg = pose_retargeter._config
    assert not pose_cfg.use_wrist_position
    assert pose_cfg.use_wrist_rotation
    assert pose_cfg.target_offset_roll == 0.0
    assert pose_cfg.target_offset_pitch == 0.0
    assert pose_cfg.target_offset_yaw == 0.0

    def compute_pose(hand):
        output = TensorGroup(pose_retargeter.output_spec()["ee_pose"])
        pose_retargeter._compute_fn(
            {hand_key: hand},
            {"ee_pose": output},
            ComputeContext(),
        )
        return np.asarray(output[0])

    hand = OptionalTensorGroup(pose_retargeter.input_spec()[hand_key])
    assert np.isnan(compute_pose(hand)).all()

    joint_positions = np.zeros((26, 3), dtype=np.float32)
    joint_positions[HandJointIndex.WRIST] = (0.0, 0.0, 0.0)
    # Right-hand fingers point along world +X; little-to-index spans world +Y.
    joint_positions[HandJointIndex.INDEX_PROXIMAL] = (1.0, 0.05, 0.0)
    joint_positions[HandJointIndex.LITTLE_PROXIMAL] = (1.0, -0.05, 0.0)
    joint_positions[HandJointIndex.THUMB_TIP] = (0.20, -0.10, 1.00)
    joint_positions[HandJointIndex.INDEX_TIP] = (0.40, 0.10, 1.20)
    hand[HandInputIndex.JOINT_POSITIONS] = joint_positions
    joint_orientations = np.zeros((26, 4), dtype=np.float32)
    joint_orientations[:, 3] = 1.0
    # The runtime wrist basis is deliberately unrelated. Semantic targeting
    # must be reconstructed from joint positions instead of copying this pose.
    joint_orientations[HandJointIndex.WRIST] = Rotation.from_euler("XYZ", (63.0, -27.0, 41.0), degrees=True).as_quat()
    hand[HandInputIndex.JOINT_ORIENTATIONS] = joint_orientations
    hand[HandInputIndex.JOINT_RADII] = np.zeros(26, dtype=np.float32)
    required_joints = (
        HandJointIndex.WRIST,
        HandJointIndex.INDEX_PROXIMAL,
        HandJointIndex.LITTLE_PROXIMAL,
        HandJointIndex.THUMB_TIP,
        HandJointIndex.INDEX_TIP,
    )
    for missing_joint in required_joints:
        joint_valid = np.zeros(26, dtype=np.uint8)
        joint_valid[list(required_joints)] = 1
        joint_valid[missing_joint] = 0
        hand[HandInputIndex.JOINT_VALID] = joint_valid
        assert np.isnan(compute_pose(hand)).all(), missing_joint

    joint_valid = np.zeros(26, dtype=np.uint8)
    joint_valid[list(required_joints)] = 1
    hand[HandInputIndex.JOINT_VALID] = joint_valid
    # Columns of the output rotation are the physical YAM grasp-frame axes.
    # The planar pad spans contact +X/+Z, so it must span the human hand's
    # across-palm/finger-forward plane. For a right hand, contact +X follows
    # index-to-little (world -Y here), selecting the -90-degree roll about the
    # already-correct forward axis. Contact +Y is the pad normal/jaw axis.
    expected_pose = np.array([0.30, 0.00, 1.10, -0.5, 0.5, -0.5, 0.5], dtype=np.float32)
    np.testing.assert_allclose(compute_pose(hand), expected_pose, atol=1.0e-6, rtol=1.0e-6)
    expected_axes = np.eye(3, dtype=np.float32)[[1, 2, 0]]
    expected_axes[0:2] *= -1.0
    actual_axes = Rotation.from_quat(compute_pose(hand)[3:7]).apply(np.eye(3, dtype=np.float32))
    np.testing.assert_allclose(actual_axes, expected_axes, atol=1.0e-6, rtol=1.0e-6)

    # Rotating/translating the entire tracked hand (for example through a side
    # camera's XR anchor) must transform the semantic tool pose exactly once.
    anchor_rot = Rotation.from_euler("ZX", (35.0, -20.0), degrees=True)
    anchor_pos = np.array((0.4, -0.3, 0.2), dtype=np.float32)
    transformed_positions = anchor_rot.apply(joint_positions) + anchor_pos
    hand[HandInputIndex.JOINT_POSITIONS] = transformed_positions.astype(np.float32)
    transformed_pose = compute_pose(hand)
    expected_position = anchor_rot.apply(expected_pose[:3]) + anchor_pos
    expected_rotation = (anchor_rot * Rotation.from_quat(expected_pose[3:7])).as_quat()
    np.testing.assert_allclose(transformed_pose[:3], expected_position, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(transformed_pose[3:7], expected_rotation, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(
        Rotation.from_quat(transformed_pose[3:7]).apply((0.0, 0.0, 1.0)),
        anchor_rot.apply((1.0, 0.0, 0.0)),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # Coincident knuckles cannot define an across-palm pad tangent. Report lost
    # tracking instead of emitting an invalid quaternion into Newton IK.
    degenerate_positions = transformed_positions.copy()
    degenerate_positions[HandJointIndex.LITTLE_PROXIMAL] = degenerate_positions[HandJointIndex.INDEX_PROXIMAL]
    hand[HandInputIndex.JOINT_POSITIONS] = degenerate_positions.astype(np.float32)
    assert np.isnan(compute_pose(hand)).all()

    input_names = set(reorderer_subgraph._input_connections)
    assert input_names == {"right_pose", "right_gripper"}
    connected_node_names = {
        selector.module._target_module.name for selector in reorderer_subgraph._input_connections.values()
    }
    assert connected_node_names == {"right_ee_pose", "right_gripper"}
