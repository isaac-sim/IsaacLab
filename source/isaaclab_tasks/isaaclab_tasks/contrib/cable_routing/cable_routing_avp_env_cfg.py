# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apple Vision Pro right-hand teleoperation for YAM cable routing."""

from __future__ import annotations

from isaaclab_newton.envs.mdp import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg, NewtonIKSolverCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCfg
from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

import isaaclab.envs.mdp as env_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass

from .avp_action_adapter import CableRoutingAVPActionAdapterCfg
from .cable_routing_env_cfg import YAM_GRIPPER_CLOSED_POS, YAM_GRIPPER_OPEN_POS, CableRoutingEnvCfg
from .yam_frames import YAM_CONTACT_FRAME_OFFSET_POS, YAM_CONTACT_FRAME_OFFSET_QUAT

# Marker consumed by ``env_test_utils._is_teleop_env`` to bucket optional
# IsaacTeleop environments in the test suite.
_TELEOP_AVAILABLE = True

AVP_TELEOP_ACTION_LAYOUT = (
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_quat_x",
    "right_quat_y",
    "right_quat_z",
    "right_quat_w",
    "right_gripper",
)
"""Semantic layout of the raw 8-D IsaacTeleop tensor."""

AVP_TELEOP_ACTION_DIM = len(AVP_TELEOP_ACTION_LAYOUT)
"""Size of the raw AVP pose-and-gripper command tensor."""


def _build_yam_cable_routing_avp_pipeline():
    """Build the right-hand tracking pipeline used by the AVP task.

    The pipeline returns the right thumb-index pinch midpoint in the simulation
    world frame. The runtime adapter rebases that pose into the right YAM's root
    frame for the native Newton IK action term.

    Returns:
        The pipeline output combiner. Its ``"action"`` output is ordered as
        ``[right_pose_xyzw(7), right_grip]``.
    """
    import numpy as np
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import HandInputIndex, HandJointIndex, TransformMatrix
    from scipy.spatial.transform import Rotation

    class _YAMSemanticHandPoseRetargeter(Se3AbsRetargeter):
        """Build the physical YAM grasp-frame orientation from right-hand anatomy.

        OpenXR hand-joint orientations do not carry the ``grip_surface`` axis
        contract. Copying the wrist quaternion therefore makes a robot-specific
        gripper appear rolled or pointed sideways on different runtimes. Joint
        positions are unambiguous: the wrist-to-knuckle direction is the YAM
        contact frame's finger-length ``+Z`` axis, and index-to-little across the
        palm is the pad's transverse-tangent ``+X`` axis. ``+Y = +Z cross +X`` is
        the pad normal/jaw axis. Consequently the full YAM ``+X/+Z`` pad plane,
        rather than only its forward tangent, follows the human hand plane.
        """

        _REQUIRED_JOINTS = (
            HandJointIndex.WRIST,
            HandJointIndex.INDEX_PROXIMAL,
            HandJointIndex.LITTLE_PROXIMAL,
            HandJointIndex.THUMB_TIP,
            HandJointIndex.INDEX_TIP,
        )
        _MIN_AXIS_NORM = 1.0e-5

        @staticmethod
        def _invalid_pose(outputs) -> None:
            outputs["ee_pose"][0] = np.full(7, np.nan, dtype=np.float32)

        def _compute_fn(self, inputs, outputs, context) -> None:
            if context.execution_events.reset:
                self._last_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            hand = inputs[self._config.input_device]
            if hand.is_none:
                self._invalid_pose(outputs)
                return

            joint_valid = np.from_dlpack(hand[HandInputIndex.JOINT_VALID])
            if not all(bool(joint_valid[joint]) for joint in self._REQUIRED_JOINTS):
                self._invalid_pose(outputs)
                return

            joint_positions = np.from_dlpack(hand[HandInputIndex.JOINT_POSITIONS])
            wrist = joint_positions[HandJointIndex.WRIST]
            index_knuckle = joint_positions[HandJointIndex.INDEX_PROXIMAL]
            little_knuckle = joint_positions[HandJointIndex.LITTLE_PROXIMAL]

            forward = 0.5 * (index_knuckle + little_knuckle) - wrist
            forward_norm = np.linalg.norm(forward)
            if not np.isfinite(forward_norm) or forward_norm < self._MIN_AXIS_NORM:
                self._invalid_pose(outputs)
                return
            forward = forward / forward_norm

            # The signed index-to-little direction selects the negative
            # 90-degree roll about finger-forward. Using the reverse direction
            # would preserve the same pad plane but turn the pad the wrong way.
            # Remove any perspective/skew component along the fingers before
            # using it as the pad's transverse tangent.
            pad_tangent = little_knuckle - index_knuckle
            pad_tangent = pad_tangent - np.dot(pad_tangent, forward) * forward
            pad_tangent_norm = np.linalg.norm(pad_tangent)
            if not np.isfinite(pad_tangent_norm) or pad_tangent_norm < self._MIN_AXIS_NORM:
                self._invalid_pose(outputs)
                return
            pad_tangent = pad_tangent / pad_tangent_norm

            pad_normal = np.cross(forward, pad_tangent)
            pad_normal_norm = np.linalg.norm(pad_normal)
            if not np.isfinite(pad_normal_norm) or pad_normal_norm < self._MIN_AXIS_NORM:
                self._invalid_pose(outputs)
                return
            pad_normal = pad_normal / pad_normal_norm
            # Re-orthogonalize the transverse tangent so numerical noise cannot
            # reach Newton IK. Columns are the authored YAM contact axes.
            pad_tangent = np.cross(pad_normal, forward)

            tool_rotation = np.column_stack((pad_tangent, pad_normal, forward))
            rotation = Rotation.from_matrix(tool_rotation).as_quat()
            if np.dot(rotation, self._last_pose[3:7]) < 0.0:
                rotation = -rotation

            thumb_tip = joint_positions[HandJointIndex.THUMB_TIP]
            index_tip = joint_positions[HandJointIndex.INDEX_TIP]
            position = 0.5 * (thumb_tip + index_tip)
            pose = np.concatenate((position, rotation)).astype(np.float32)
            if not np.isfinite(pose).all():
                self._invalid_pose(outputs)
                return
            self._last_pose = pose
            outputs["ee_pose"][0] = pose

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # Report the right thumb-index pinch center and an anatomical orientation
    # whose axes already describe the physical YAM grasp frame. The adapter
    # clutches translation only; orientation stays absolute so finger-forward
    # can never inherit an arbitrary offset from the robot's engagement pose.
    pose_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=False,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    pose_retargeter = _YAMSemanticHandPoseRetargeter(pose_cfg, name="right_ee_pose")
    connected_pose = pose_retargeter.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    # Only hand tracking is connected. GripperRetargeter therefore uses
    # thumb-index pinch distance rather than a controller trigger.
    gripper_retargeter = GripperRetargeter(
        GripperRetargeterConfig(hand_side="right"),
        name="right_gripper",
    )
    connected_gripper = gripper_retargeter.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    right_pose_elements = list(AVP_TELEOP_ACTION_LAYOUT[0:7])
    right_gripper_elements = [AVP_TELEOP_ACTION_LAYOUT[7]]

    reorderer = TensorReorderer(
        input_config={
            "right_pose": right_pose_elements,
            "right_gripper": right_gripper_elements,
        },
        output_order=list(AVP_TELEOP_ACTION_LAYOUT),
        name="right_hand_action_reorderer",
        input_types={
            "right_pose": "array",
            "right_gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "right_pose": connected_pose.output("ee_pose"),
            "right_gripper": connected_gripper.output("gripper_command"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline


@configclass
class CableRoutingAVPTeleopActionsCfg:
    """Hold the left YAM and drive the right YAM with native Newton IK."""

    left_arm = env_mdp.JointPositionActionCfg(
        asset_name="yam_left",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    left_gripper = env_mdp.BinaryJointPositionActionCfg(
        asset_name="yam_left",
        joint_names=["left_finger"],
        open_command_expr={"left_finger": YAM_GRIPPER_OPEN_POS},
        close_command_expr={"left_finger": YAM_GRIPPER_CLOSED_POS},
    )
    right_arm = NewtonInverseKinematicsActionCfg(
        asset_name="yam_right",
        joint_names=["joint[1-6]"],
        isolate_articulation_model=True,
        use_cuda_graph=True,
        controller=NewtonIKSolverCfg(
            optimizer="lm",
            jacobian_mode="analytic",
            sampler="none",
            n_seeds=1,
            iterations=12,
            lambda_initial=0.05,
        ),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_pinch",
                body_name="link_6",
                body_offset_pos=YAM_CONTACT_FRAME_OFFSET_POS,
                body_offset_rot=YAM_CONTACT_FRAME_OFFSET_QUAT,
                command_type="pose",
                use_relative_mode=False,
                scale=1.0,
                position_weight=1.0,
                rotation_weight=2.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )
    right_gripper = env_mdp.BinaryJointPositionActionCfg(
        asset_name="yam_right",
        joint_names=["left_finger"],
        open_command_expr={"left_finger": YAM_GRIPPER_OPEN_POS},
        close_command_expr={"left_finger": YAM_GRIPPER_CLOSED_POS},
    )


@configclass
class CableRoutingAVPTeleopEnvCfg(CableRoutingEnvCfg):
    """Single-scene right-hand AVP variant using native Newton IK."""

    actions: CableRoutingAVPTeleopActionsCfg = CableRoutingAVPTeleopActionsCfg()
    teleop_action_adapter: CableRoutingAVPActionAdapterCfg = CableRoutingAVPActionAdapterCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The training drive is deliberately conservative, while the authored
        # Menagerie gains are underdamped for direct Cartesian pose steps. Split
        # only the teleoperated right arm into its physical proximal/distal
        # groups and use the measured middle ground: it preserves the asset's
        # armatures and 2 rad/s limit while avoiding both large overshoot and the
        # old multi-second overdamped response. Collision geometry is unchanged.
        right_gripper_drive = self.scene.yam_right.actuators["gripper_drive"]
        right_gripper_passive = self.scene.yam_right.actuators["gripper_passive"]
        self.scene.yam_right.actuators = {
            "arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=2.0,
                stiffness=160.0,
                damping=12.0,
                armature=0.032,
            ),
            "arm_joint4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=15.0,
                velocity_limit_sim=2.0,
                stiffness=80.0,
                damping=4.0,
                armature=0.0018,
            ),
            "arm_distal": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-6]"],
                effort_limit_sim=15.0,
                velocity_limit_sim=2.0,
                stiffness=60.0,
                damping=5.0,
                armature=0.0018,
            ),
            "gripper_drive": right_gripper_drive,
            "gripper_passive": right_gripper_passive,
        }

        # Teleoperation is intentionally single-scene: AVP engagement state and
        # the operator's rendered viewpoint both refer to environment zero.
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.5
        # Replay banks are a training curriculum. Building the default 4096
        # snapshots in this single interactive scene would add substantial
        # startup time without improving direct hand control.
        self.commands.route.reset_replay.enabled = False
        # Teleoperation accepts a new hand target every 120 Hz physics step. The
        # same ten Newton substeps, contacts, cable solver, and collision geometry
        # are retained; only the policy/control decimation is reduced.
        self.decimation = 1
        self.sim.render_interval = self.decimation
        # Keep actuator evaluation inside Newton's captured physics loop instead
        # of crossing the Torch/Warp boundary for every environment step.
        self.sim.use_newton_actuators = True
        self.commands.route.debug_vis = False
        self.episode_length_s = 300.0

        # Place the tracking origin in front of the table. With the OpenXR-to-USD
        # basis conversion, an operator looking forward sees the centered board
        # roughly 0.65 m away at its physical 0.77 m tabletop height.
        self.xr = XrCfg(
            anchor_pos=(0.0, -0.65, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
            near_plane=0.05,
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_yam_cable_routing_avp_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            # This graph is lightweight; synchronous execution removes the
            # default pipelined mode's one-completed-frame control latency.
            retargeting_execution=RetargetingExecutionConfig(mode="sync"),
            # The action adapter performs the world-to-right-base transform.
            target_frame_prim_path=None,
            teleoperation_active_default=False,
            app_name="IsaacLab YAM Cable Routing AVP",
        )

    def play_mode(self) -> None:
        """Keep the AVP task single-scene without controller-frame clutter."""
        super().play_mode()
        self.scene.num_envs = 1
        self.commands.route.debug_vis = False
