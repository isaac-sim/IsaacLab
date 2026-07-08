# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import numpy as np
from isaaclab_teleop import IsaacTeleopCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp

from . import stack_joint_pos_env_cfg
from .pose_ik_action import SO101PoseIKActionCfg
from .pose_ik_controller import SO101PoseIKControllerCfg
from .stack_joint_pos_env_cfg import SO101_GRIPPER_CLOSE, SO101_GRIPPER_OPEN

##
# Pre-defined configs
##
from isaaclab_assets.robots.so101 import SO101_HIGH_PD_CFG  # isort: skip

# ``isaaclab_teleop`` imports cleanly without the optional ``isaacteleop`` package (the heavy
# import is deferred to session start), so no import guard is needed here. This lightweight
# marker lets the env test suite categorize this as a teleop env (see
# ``isaaclab_tasks/test/env_test_utils.py::_is_teleop_env``).
_TELEOP_AVAILABLE = True

# Analog-gripper action affine: ``joint = offset + scale * closedness`` maps the retargeter's
# closedness ``c in [0, 1]`` to a ``gripper`` joint target [rad]. ``offset == OPEN`` (so ``c=0``
# is fully open) and ``offset + scale == CLOSE`` (so ``c=1`` is fully closed). Tied to the
# single source of truth in ``stack_joint_pos_env_cfg`` so the endpoints cannot drift.
_SO101_GRIPPER_OFFSET = SO101_GRIPPER_OPEN  # [rad] closedness c=0 -> open
_SO101_GRIPPER_SCALE = SO101_GRIPPER_CLOSE - SO101_GRIPPER_OPEN  # [rad] c=1 -> close

# Clutch EE reset-origin home: the ``base_T_ee`` transform [m] for the gripper's pose in the
# robot base frame at the seated reset pose. The clutch seeds its home from this on reset / first
# engage (no live end-effector is read; the base rebase is handled upstream by
# ``target_frame_prim_path`` and the EE state forward is the clutch's own running home). The robot
# ``init_state`` MUST forward-kinematics to this value, or engaging snaps the arm -- this is the
# forward kinematics of the seated ``_SO101_STACK_INIT_JOINT_POS`` measured in sim, so re-measure
# (read ``ee_frame.data.target_pos_source`` at reset) if those init joints change. Only the
# translation drives the IK position command; the orientation command is the live controller grip
# orientation composed with the clutch's fixed calibration offset, not this transform's rotation.
_BASE_T_GRIPPER_HOME = np.eye(4, dtype=np.float32)
_BASE_T_GRIPPER_HOME[:3, 3] = (0.01918, -0.18852, 0.18887)

# Fixed controller-grip -> gripper_link orientation calibration offset [x, y, z, w] (xyzw),
# passed to the clutch as a body-frame right multiply. ``None`` uses the clutch default (Rz(pi)).
# Derivation: with ``q_grip`` the controller grip orientation and ``q_G0`` the gripper_link
# orientation in the base frame (both xyzw) at the reset pose, the offset is
# ``quat_inv(q_grip) (x) q_G0`` (xyzw); tuning it here needs no Teleop rebuild.
# Set to RPY (roll, pitch, yaw) = (-90, 0, 60) degrees (intrinsic XYZ), i.e.
# ``Rotation.from_euler("XYZ", [-90, 0, 60], degrees=True).as_quat()``.
_SO101_ORIENTATION_OFFSET_XYZW: tuple[float, float, float, float] | None = (
    -0.61237244,
    0.35355339,
    0.35355339,
    0.61237244,
)


def _build_so101_stack_pipeline():
    """Build an IsaacTeleop retargeting pipeline for SO-101 cube stacking.

    Creates a SO101ClutchRetargeter for right-hand clutch-rebased full-pose tracking and a
    SO101GripperRetargeter for right-hand analog gripper control, flattened into a single action
    tensor via TensorReorderer.

    The SO-101 is 5-DOF and driven by a full-SE3-pose IK over all 5 arm joints (shoulder_pan,
    shoulder_lift, elbow_flex, wrist_flex, wrist_roll): the clutch emits the controller pose
    (position + grip orientation composed with a fixed calibration offset) and the IK tracks
    position exactly while best-effort tracking orientation (soft-weighted). The gripper jaw
    tracks the trigger proportionally (analog), and the EE position is clutch-rebased around a
    captured origin so engaging teleop does not teleport the arm.

    Returns:
        OutputCombiner with a single "action" output containing the flattened
        8D action tensor: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper].
    """
    from isaacteleop.retargeters import (
        SO101ClutchRetargeter,
        SO101GripperRetargeter,
        TensorReorderer,
    )
    from isaacteleop.retargeters.SO101.gripper_retargeter import GRIPPER_COMMAND_KEY, GRIPPER_ELEMENT_LABEL
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")

    # External input: the anchor transform 4x4 provided by IsaacTeleopDevice. With
    # ``target_frame_prim_path`` set to the robot base (see the cfg below) the device feeds
    # ``base_T_world @ world_T_anchor``, so the transformed controller stream below arrives
    # already in the robot base frame -- the IK command's root frame.
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Rebase controller poses into the robot base frame so the downstream clutch retargeter
    # receives base-frame data matching the absolute-pose IK command frame.
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # Clutch (relative-origin) EE-pose retargeter (right hand). Emits the same absolute 7D
    # "ee_pose" contract as Se3AbsRetargeter (node name + output key "ee_pose"), but rebases
    # controller motion around an origin captured on engage. It seeds the home from the static
    # ``home_base_T_ee`` reset-origin on reset / first engage and keeps its own running home
    # thereafter (so a mid-task re-clutch resumes from the last commanded pose). The robot
    # ``init_state`` must reset the arm to ``_BASE_T_GRIPPER_HOME`` so the first engage does not
    # jump. The full pose (position + orientation) drives the 5-joint SE3 IK below.
    clutch = SO101ClutchRetargeter(
        name="ee_pose",
        input_device=ControllersSource.RIGHT,
        home_base_T_ee=_BASE_T_GRIPPER_HOME,
        orientation_offset=(
            np.array(_SO101_ORIENTATION_OFFSET_XYZW, dtype=np.float64)
            if _SO101_ORIENTATION_OFFSET_XYZW is not None
            else None
        ),
    )
    connected_clutch = clutch.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    # Analog Gripper Retargeter (right hand). Emits a proportional jaw closedness in [0, 1]
    # from the controller trigger; the node name "gripper" matches its gripper_command/
    # gripper_value channel and the gripper_action term.
    gripper = SO101GripperRetargeter(name="gripper", input_device=ControllersSource.RIGHT)
    connected_gripper = gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    # TensorReorderer to flatten into a single action vector
    # SO101ClutchRetargeter outputs a 7D NDArray (pos xyz + quat xyzw)
    # SO101GripperRetargeter outputs a single float (gripper closedness)
    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    gripper_elements = [GRIPPER_ELEMENT_LABEL]

    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_pose_elements,
            GRIPPER_COMMAND_KEY: gripper_elements,
        },
        # Full-pose SE3 IK: pass the whole 7D pose (xyz + quat xyzw) through, then the gripper.
        # Output order: [pos_xyz, quat_xyzw, gripper] — the positional contract with the
        # SO101PoseIKAction (7D arm: pos + quat) + gripper.
        output_order=ee_pose_elements + gripper_elements,
        name="action_reorderer",
        input_types={
            "ee_pose": "array",
            GRIPPER_COMMAND_KEY: "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_clutch.output("ee_pose"),
            GRIPPER_COMMAND_KEY: connected_gripper.output(GRIPPER_COMMAND_KEY),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class SO101IkActionsCfg:
    """Action terms for SO-101 IK-Abs teleop, ordered to match the pipeline ``output_order``.

    This is a fresh (non-inheriting) action container declared in field order
    ``[arm_action, gripper_action]`` so the flattened action concatenates as
    ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]`` — the positional contract
    with the teleop pipeline's ``output_order`` in :func:`_build_so101_stack_pipeline`. The arm
    action covers 7 dimensions (pos xyz + quat xyzw); gripper covers 1. Inheriting from the base
    ``ActionsCfg`` would append new fields after the base's and mis-order the concat.
    """

    arm_action: SO101PoseIKActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class SO101CubeStackEnvCfg(stack_joint_pos_env_cfg.SO101CubeStackEnvCfg):
    """SO-101 cube-stack environment with absolute task-space (IK) control.

    The SO-101 has only 5 actuated arm DOF, so a full 6-DOF pose target is over-determined by
    one DOF. We command an absolute end-effector **SE3 pose** via a full-pose differential IK
    over all 5 arm joints (``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``,
    ``wrist_roll``): the 3 linear task rows track position exactly (weight 1) and the
    orientation rows are soft-weighted (``orientation_weight``) so orientation is
    best-effort and never leaks error into position. Orientation is further restricted to the wrist
    (``orientation_joint_names=("wrist_flex", "wrist_roll")``): ``wrist_roll`` takes the gripper
    spin about the (vertical) approach axis and ``wrist_flex`` the tilt, while ``shoulder_pan``
    serves position only -- so the base never swings to satisfy a commanded orientation. The
    resulting action is
    ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]`` (orientation xyzw). The IK
    is kept well-conditioned near singularities via the soft orientation weight,
    manipulability-aware damped least squares (adaptive lambda keyed off the full weighted task
    Jacobian, so it is coupled to the orientation weight), and null-space joint-limit avoidance.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch to a stiffer PD controller so the arm tracks IK targets well. SO101_HIGH_PD_CFG
        # carries only the so101.py asset default init_state (zeros, no root pos), so ``replace``
        # would DISCARD the parent's seated + stack-posed init_state. Preserve it explicitly so
        # the IK-Abs task is also seated on the table and starts in the stack joint pose.
        #
        # Also soften the gripper to mimic the Franka panda_hand's grasp feel. The SO-101 jaw asset
        # default (effort 10 N·m, velocity 10 rad/s) snaps shut fast and hard, pushing through the
        # cube and ejecting it. The Franka panda_hand (effort 200 N, stiffness 2e3) is a *prismatic*
        # finger pair, so its gains do not transfer to this *revolute* jaw [N·m, rad]; instead we
        # behavior-match by capping the closing speed and grip torque so it closes gently and holds
        # without penetrating. ``effort_limit_sim`` is the "strength" knob and ``velocity_limit_sim``
        # the "speed" knob; stiffness/damping keep the asset defaults. Tune in-sim (lower effort if
        # it still pushes through; raise it if the cube drops).
        self.scene.robot = SO101_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=self.scene.robot.init_state,
            actuators={
                "arm": SO101_HIGH_PD_CFG.actuators["arm"],
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    effort_limit_sim=1.0,  # was 10.0 -- cap grip torque so it can't push through
                    velocity_limit_sim=2.0,  # was 10.0 -- close gently instead of snapping shut
                    stiffness=17.8,
                    damping=0.60,
                ),
            },
        )

        # Replace the actions container. The 2-field order here is the positional contract
        # with the pipeline ``output_order`` ([arm pos+quat, gripper]); a true action_dim check
        # needs the loaded articulation, so the sim-free test is the real guard.
        self.actions = SO101IkActionsCfg(
            # Full-pose SE3 IK over all 5 arm joints (including wrist_roll). Solves a 6-row
            # pose task [pos_xyz, ori_xyz]; the 3 orientation rows are soft-weighted.
            arm_action=SO101PoseIKActionCfg(
                asset_name="robot",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                scale=1.0,
                controller=SO101PoseIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    # Manipulability-aware damped least squares keeps the 5-DOF arm well-conditioned
                    # near singularities (adaptive lambda keyed off the smallest task-Jacobian
                    # singular value).
                    ik_method="adaptive_dls",
                    ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                    # Track all 3 orientation axes (including spin about the vertical approach
                    # axis), but restrict orientation to the wrist via ``orientation_joint_names``
                    # below so the base never serves it: ``wrist_roll`` takes the spin (controller
                    # twist about vertical / stage Z) and ``wrist_flex`` the tilt, while
                    # ``shoulder_pan`` stays position-only (heading to the target).
                    orientation_weight=0.5,
                    # Orientation is a wrist-only task (see above). Position still uses all 5 joints.
                    orientation_joint_names=("wrist_flex", "wrist_roll"),
                    # Null-space joint-limit avoidance (0 disables it).
                    joint_limit_avoidance_gain=0.5,
                    joint_limit_avoidance_margin=0.3,
                ),
            ),
            # Analog gripper: continuous JointPositionActionCfg mapping the retargeter's
            # closedness c in [0, 1] to a ``gripper`` joint target [rad] via the affine
            # ``joint = offset + scale * c`` (offset == OPEN, offset + scale == CLOSE). This
            # replaces the joint-pos task's BinaryJointPositionActionCfg with proportional
            # control. ``use_default_offset=False`` so the offset is the absolute open angle.
            gripper_action=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper"],
                scale=_SO101_GRIPPER_SCALE,
                offset=_SO101_GRIPPER_OFFSET,
                use_default_offset=False,
            ),
        )

        # IsaacTeleop-based teleoperation pipeline
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_so101_stack_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            # Rebase all teleop output poses into the robot base frame: the device reads this
            # prim's world transform each frame and left-multiplies its inverse onto the XR
            # anchor (``base_T_world @ world_T_anchor``), so the clutch retargeter works
            # in the base frame -- the IK command's root frame (the FrameTransformer source
            # also uses ``Robot/base``). The clutch's reset-origin home (_BASE_T_GRIPPER_HOME)
            # provides the engage seed, so no live end-effector feed is needed. Teleop runs a
            # single env (env_0).
            target_frame_prim_path="/World/envs/env_0/Robot/base",
        )
