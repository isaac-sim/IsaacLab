# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.actuators import ActuatorBaseCfg, actuator_pd
from isaaclab.sim import JointDrivePropertiesCfg
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

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim under the :attr:`prim_path`. Defaults to None, in which case the class
    will search for a prim with the USD ArticulationRootAPI on it.

    This path should be relative to the :attr:`prim_path` of the asset. If the asset is loaded from a USD file,
    this path should be relative to the root of the USD stage. For instance, if the loaded USD file at :attr:`prim_path`
    contains two articulations, one at `/robot1` and another at `/robot2`, and you want to use `robot2`,
    then you should set this to `/robot2`.

    The path must start with a slash (`/`).
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of joint position limits (parsed from the asset) to use. Defaults to 1.0.

    The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    joint position limits. This isn't used by the simulation, but is useful for learning agents to prevent the joint
    positions from violating the limits, such as for termination conditions.

    The soft joint position limits are accessible through the :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""

    actuator_value_resolution_debug_print = False
    """Print the resolution of actuator final value when input cfg is different from USD value, Defaults to False
    """

    def __post_init__(self):
        """This Post init is meant to determine if we need to apply the new drive model api to the joints for this articulation.
        If the drive model is not explicitly enabled by the USD, but an actuator configuration attempts to configure the new
        api, it will silent default to legacy behavior. If we see an attempt to parametrize the drive model, we configure
        the primitive spawner to enable the API so that when the actuators are configured, after the primitive is spawned
        the necessary attributes exist and are available to be configured by the usual configuration parser logic.

        .. Note: Importantly, configuring the drive model on any actuator in the articulation will enable the api on _all_
           actuators in the articulation (with some default values). It is not currently possible to enable the api on
           specific joint indices. This may lead to unexpected impacts on drive behavior or frame rate."""

        if self.actuators is not None:
            if len(self.actuators.keys()) > 0:
                for act_cfg in self.actuators.values():
                    if act_cfg.class_type == actuator_pd.ImplicitActuator:
                        if act_cfg.drive_model is not None:
                            if hasattr(self.spawn, "joint_drive_props"):
                                if self.spawn.joint_drive_props is not None:
                                    self.spawn.joint_drive_props.enable_physx_perf_env = True
                                else:
                                    self.spawn.joint_drive_props = JointDrivePropertiesCfg(enable_physx_perf_env=True)
