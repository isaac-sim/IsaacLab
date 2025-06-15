# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import binary_joint_actions, joint_actions, joint_actions_to_limits, non_holonomic_actions, task_space_actions

##
# Joint actions.
##


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions). Defaults to None."""


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for the relative joint position action term.

    See :class:`RelativeJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.RelativeJointPositionAction

    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to zero.
    """


@configclass
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.JointVelocityAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """


@configclass
class JointEffortActionCfg(JointActionCfg):
    """Configuration for the joint effort action term.

    See :class:`JointEffortAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.JointEffortAction


##
# Joint actions rescaled to limits.
##


@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.JointPositionToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    rescale_to_limits: bool = True
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """

    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions). Defaults to None."""


@configclass
class EMAJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
    """Configuration for the exponential moving average (EMA) joint position action term.

    See :class:`EMAJointPositionToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.EMAJointPositionToLimitsAction

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving average window.
    """


##
# Gripper actions.
##


@configclass
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""

    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointPositionAction


@configclass
class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint velocity action term.

    See :class:`BinaryJointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointVelocityAction


##
# Non-holonomic actions.
##


@configclass
class NonHolonomicActionCfg(ActionTermCfg):
    """Configuration for the non-holonomic action term with dummy joints at the base.

    See :class:`NonHolonomicAction` for more details.
    """

    class_type: type[ActionTerm] = non_holonomic_actions.NonHolonomicAction

    body_name: str = MISSING
    """Name of the body which has the dummy mechanism connected to."""

    x_joint_name: str = MISSING
    """The dummy joint name in the x direction."""

    y_joint_name: str = MISSING
    """The dummy joint name in the y direction."""

    yaw_joint_name: str = MISSING
    """The dummy joint name in the yaw direction."""

    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for the action. Defaults to (1.0, 1.0)."""

    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factor for the action. Defaults to (0.0, 0.0)."""

    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions).

    The expected keys are "v", and "yaw". Defaults to None for no clipping.
    """


##
# Task-space Actions.
##


@configclass
class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = task_space_actions.DifferentialInverseKinematicsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""

    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""

    # TODO: Should this be simplified to a list of tuples? More compact, less readable?
    # TODO: Do we want to have an homogeneous behavior for the clip range? I think we do
    # or we'd need unique clip names so that it's not confusing to the user.

    clip: dict[str, tuple] | None = None
    """Clip range of the controller's command in the world frame (dict of regex expressions).

    The expected keys are "position", "orientation", and "wrench". Defaults to None for no clipping.
    For "position" we expect a tuple of (min, max) for each dimension. (x, y, z) in this order.
    For "orientation" we expect a tuple of (min, max) for each dimension. (roll, pitch, yaw) in this order.

    Example:
    ..code-block:: python
        {
            "position": ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), # (x, y, z)
            "orientation": ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), # (roll, pitch, yaw)
        }

    ..note::
        This means that regardless of the :attr:`controller.use_relative_mode` setting, the clip range is always
        applied in the world frame. This is done so that the meaning of the clip range is consistent across
        different modes.

    ..note::
        If the :attr:`controller.command_type` is set to "pose", then both the position and orientation clip ranges
        must be provided. To clip either one or the other, one can set large values to the clip range.


    """


@configclass
class OperationalSpaceControllerActionCfg(ActionTermCfg):
    """Configuration for operational space controller action term.

    See :class:`OperationalSpaceControllerAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = task_space_actions.OperationalSpaceControllerAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    body_name: str = MISSING
    """Name of the body or frame for which motion/force control is performed."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    task_frame_rel_path: str = None
    """The path of a ``RigidObject``, relative to the sub-environment, representing task frame. Defaults to None."""

    controller_cfg: OperationalSpaceControllerCfg = MISSING
    """The configuration for the operational space controller."""

    position_scale: float = 1.0
    """Scale factor for the position targets. Defaults to 1.0."""

    orientation_scale: float = 1.0
    """Scale factor for the orientation (quad for ``pose_abs`` or axis-angle for ``pose_rel``). Defaults to 1.0."""

    wrench_scale: float = 1.0
    """Scale factor for the wrench targets. Defaults to 1.0."""

    stiffness_scale: float = 1.0
    """Scale factor for the stiffness commands. Defaults to 1.0."""

    damping_ratio_scale: float = 1.0
    """Scale factor for the damping ratio commands. Defaults to 1.0."""

    nullspace_joint_pos_target: str = "none"
    """The joint targets for the null-space control: ``"none"``, ``"zero"``, ``"default"``, ``"center"``.

    Note: Functional only when ``nullspace_control`` is set to ``"position"`` within the
        ``OperationalSpaceControllerCfg``.
    """

    # TODO: Here the clip effects are not homogeneous, but they have unique names that relate
    # to specific control modes so it's fine to me.

    clip_pose_abs: list[tuple[float, float]] | None = None
    """Clip range for the absolute pose targets. Defaults to None for no clipping.

    The expected format is a list of tuples, each containing two values. This effectively bounds
    the reachable range of the end-effector in the world frame.

    Example:
    ..code-block:: python
        clip_pose_abs = [
            (min_x, max_x),
            (min_y, max_y),
            (min_z, max_z),
            (min_roll, max_roll),
            (min_pitch, max_pitch),
            (min_yaw, max_yaw),
        ]
    """
    clip_pose_rel: list[tuple[float, float]] | None = None
    """Clip range for the relative pose targets. Defaults to None for no clipping.

    The expected format is a list of tuples, each containing two values. This effectively limits
    the end-effector's velocity in the task frame.

    Example:
    ..code-block:: python
        clip_pose_rel = [
            (min_x, max_x),
            (min_y, max_y),
            (min_z, max_z),
            (min_roll, max_roll),
            (min_pitch, max_pitch),
            (min_yaw, max_yaw),
        ]
    """
    clip_wrench_abs: list[tuple[float, float]] | None = None
    """Clip range for the absolute wrench targets. Defaults to None for no clipping.

    The expected format is a list of tuples, each containing two values. This effectively limits
    the maximum force and torque that can be commanded in the task frame.

    Example:
    ..code-block:: python
        clip_wrench_abs = [
            (min_fx, max_fx),
            (min_fy, max_fy),
            (min_fz, max_fz),
            (min_tx, max_tx),
            (min_ty, max_ty),
            (min_tz, max_tz),
        ]
    """
