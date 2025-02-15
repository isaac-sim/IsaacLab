# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .articulation import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    class_type: type = Articulation

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of DOF position limits (parsed from the asset) to use. Defaults to 1.0.

    The joint position limits are scaled by this factor to allow for a limited range of motion.
    This is accessible in the articulation data through :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    actuated_joint_names: list[str] | str = ".*"
    """List of joint names or regular expression to specify the actuated joints. Defaults to '.*' which means all
    joints are actuated."""

    mimic_joints_info: dict[str, dict[str, float | str]] = {}
    """Mimic joints configuration for the articulation. Defaults to an empty dictionary.

    The key indicates the name of the joint that is being mimicked. The value is a dictionary with the following keys:

    * ``"parent"``: The name of the parent joint.
    * ``"multiplier"``: The multiplier for the mimic joint.
    * ``"offset"``: The offset for the mimic joint. Defaults to 0.0.

    For example, the following configuration mimics the joint ``"joint_1"`` with the parent joint ``"joint_0"``
    with a multiplier of ``2.0`` and an offset of ``1.0``:

    .. code-block:: python

        mimic_joints_info = {
            "joint_1": {"parent": "joint_0", "multiplier": 2.0, "offset": 1.0}
        }

    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""
