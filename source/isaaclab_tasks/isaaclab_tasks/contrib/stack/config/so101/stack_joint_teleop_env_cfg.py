# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_teleop import IsaacTeleopCfg

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp

from . import stack_joint_pos_env_cfg

# ``isaaclab_teleop`` imports cleanly without the optional ``isaacteleop`` package (the heavy
# import is deferred to session start), so no import guard is needed here. This lightweight
# marker lets the env test suite categorize this as a teleop env (see
# ``isaaclab_tasks/test/env_test_utils.py::_is_teleop_env``).
_TELEOP_AVAILABLE = True

# Canonical SO-101 DOF names, in the order the leader arm streams them (matches
# ``so101_new_calib.urdf`` and the ``so101_leader`` device schema). The first five are the arm;
# ``gripper`` is the jaw DOF. This order is the positional contract with the flattened action
# tensor and the ``TensorReorderer`` ``output_order`` in :func:`_build_so101_joint_teleop_pipeline`.
_SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_SO101_JOINTS = _SO101_ARM_JOINTS + ["gripper"]

# Tensor collection id the ``so101_leader`` device pushes over the OpenXR tensor transport. The
# ``JointStateSource`` below subscribes to this id; it must match the ``collection_id`` the leader
# plugin (or any pusher emitting the same schema) advertises.
_SO101_LEADER_COLLECTION_ID = "so101_leader"


def _build_so101_joint_teleop_pipeline():
    """Build an IsaacTeleop joint-space pipeline driven by the SO-101 leader arm.

    Unlike the XR-controller IK pipeline (see :mod:`.stack_ik_abs_env_cfg`), this path mirrors the
    physical leader arm's joint encoders straight onto the follower robot's joint targets -- no XR
    anchor, controller source, or base-frame rebasing is involved. The
    :class:`~isaacteleop.retargeting_engine.deviceio_source_nodes.JointStateSource` polls the
    ``so101_leader`` joint-state tracker, the
    :class:`~isaacteleop.retargeters.JointStateRetargeter` (``mode="joint"``) remaps device DOF
    names to robot joint targets, and a
    :class:`~isaacteleop.retargeters.TensorReorderer` flattens them into a single action tensor.

    Because the leader and follower are the same SO-101 morphology, joint angles map 1:1 (identity
    ``device_joints -> target_joints``). If the two units are miscalibrated, tune per-joint
    ``scale`` / ``offset`` / ``sign`` on :class:`JointStateRetargeterConfig` rather than here.

    Returns:
        OutputCombiner with a single ``"action"`` output containing the flattened 6-D joint
        action tensor: ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
        gripper]``.
    """
    from isaacteleop.retargeters import (
        JointStateRetargeter,
        JointStateRetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import JointStateSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    # Live leader joint-state source. ``joint_names`` fixes the static output layout and must match
    # the retargeter's ``device_joints`` order below (the graph type-checks this at ``connect``).
    source = JointStateSource(
        name="leader",
        collection_id=_SO101_LEADER_COLLECTION_ID,
        joint_names=_SO101_JOINTS,
    )

    # Identity joint mirror: device DOF -> robot joint target, same names and order.
    retargeter = JointStateRetargeter(
        name="leader",
        mode="joint",
        config=JointStateRetargeterConfig(
            device_joints=_SO101_JOINTS,
            target_joints=_SO101_JOINTS,
        ),
    )
    connected_retargeter = retargeter.connect({JointStateRetargeter.JOINTS: source.output(JointStateSource.JOINTS)})

    # Flatten the name-keyed joint targets into a single ordered action vector. ``output_order``
    # is the positional contract with the action terms declared in :class:`SO101JointTeleopActionsCfg`
    # (arm joints first, then gripper).
    reorderer = TensorReorderer(
        input_config={"joint_targets": _SO101_JOINTS},
        output_order=_SO101_JOINTS,
        name="action_reorderer",
        input_types={"joint_targets": "scalar"},
    )
    connected_reorderer = reorderer.connect({"joint_targets": connected_retargeter.output("joint_targets")})

    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class SO101JointTeleopActionsCfg:
    """Action terms for SO-101 joint-space teleop, ordered to match the pipeline ``output_order``.

    This is a fresh (non-inheriting) action container declared in field order
    ``[arm_action, gripper_action]`` so the flattened action concatenates as
    ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`` -- the positional
    contract with the teleop pipeline's ``output_order`` in
    :func:`_build_so101_joint_teleop_pipeline`. Both terms are absolute
    :class:`~isaaclab_tasks.contrib.stack.mdp.JointPositionActionCfg` (``scale=1.0``,
    ``use_default_offset=False``) so the leader's raw joint angles [rad] pass through unmodified.
    Inheriting from the base ``ActionsCfg`` would keep its delta-scaled arm term and binary gripper.
    """

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class SO101CubeStackEnvCfg(stack_joint_pos_env_cfg.SO101CubeStackEnvCfg):
    """SO-101 cube-stack environment teleoperated by the SO-101 leader arm (joint-space control).

    Reuses the seated robot, cube workspace, and end-effector frames from the joint-position base
    env and swaps in an absolute joint-mirror action plus an IsaacTeleop joint-space pipeline. The
    leader arm's encoders drive the follower's five arm joints and gripper DOF directly, so the
    follower reproduces the operator's pose without any inverse kinematics or XR tracking.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Replace the actions container with an absolute joint mirror. The base env's arm term is
        # delta-scaled (``scale=0.5``, ``use_default_offset=True``) with a binary gripper, which is
        # wrong for streaming absolute leader-arm angles; here both terms pass the raw joint targets
        # [rad] straight through. The 2-field order is the positional contract with the pipeline
        # ``output_order`` ([arm joints, gripper]).
        self.actions = SO101JointTeleopActionsCfg(
            arm_action=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=_SO101_ARM_JOINTS,
                scale=1.0,
                use_default_offset=False,
            ),
            gripper_action=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper"],
                scale=1.0,
                use_default_offset=False,
            ),
        )

        # IsaacTeleop joint-space pipeline. Unlike the IK env this needs no ``target_frame_prim_path``
        # (there is no pose to rebase into the base frame) and does not read the XR anchor -- the
        # leader plugin streams joint state over the OpenXR tensor transport, which the session still
        # provides. ``xr_cfg`` is forwarded only to satisfy the session's XR bootstrap. Launch the
        # ``so101_leader`` device (collection id ``so101_leader``) alongside the sim; it is not
        # spawned here as a plugin so operators can point at real or synthetic hardware.
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_so101_joint_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
