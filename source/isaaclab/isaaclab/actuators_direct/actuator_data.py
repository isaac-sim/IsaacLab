# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


class ActuatorData:
    """Data container for all actuators in an articulation.

    This class contains the data for an actuator. The data includes the computed effort, applied effort,
    effort limit, velocity limit, control mode, stiffness, damping, armature and friction.
    """
    effort: wp.array
    """The effort for the actuator group. Shape is (num_envs, num_joints)."""

    computed_effort: wp.array
    """The computed effort for the actuator group. Shape is (num_envs, num_joints)."""

    applied_effort: wp.array
    """The applied effort for the actuator group. Shape is (num_envs, num_joints).

    This is the effort obtained after clipping the :attr:`computed_effort` based on the
    actuator characteristics.
    """

    effort_limit: wp.array
    """The effort limit for the actuator group. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`effort_limit` and :attr:`effort_limit_sim` are the same.
    """

    effort_limit_sim: wp.array
    """The effort limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`effort_limit` and :attr:`effort_limit_sim` are the same.
    """

    velocity_limit: wp.array
    """The velocity limit for the actuator group. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`velocity_limit` and :attr:`velocity_limit_sim` are the same.
    """

    velocity_limit_sim: wp.array
    """The velocity limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`velocity_limit` and :attr:`velocity_limit_sim` are the same.
    """

    control_mode: wp.array(dtype=wp.int32)
    """The control mode of the actuator. Shape is (num_envs, num_joints).
    
    * 0: No control
    * 1: Position control
    * 2: Velocity control
    """

    stiffness: wp.array
    """The stiffness (P gain) of the PD controller. Shape is (num_envs, num_joints)."""

    damping: wp.array
    """The damping (D gain) of the PD controller. Shape is (num_envs, num_joints)."""

    armature: wp.array
    """The armature of the actuator joints. Shape is (num_envs, num_joints)."""

    friction: wp.array
    """The joint friction of the actuator joints. Shape is (num_envs, num_joints)."""

    dynamic_friction: wp.array
    """The joint dynamic friction of the actuator joints. Shape is (num_envs, num_joints)."""

    viscous_friction: wp.array
    """The joint viscous friction of the actuator joints. Shape is (num_envs, num_joints)."""

    all_joint_mask: wp.array
    """The mask of all joints in the articulation. Shape is (num_envs, num_joints)."""

    all_env_mask: wp.array
    """The mask of all environments in the articulation. Shape is (num_envs, num_joints)."""

    def add_data(self, name: str, data: wp.array):
        """Add data to the actuator data.

        Args:
            name: The name of the data.
            data: The data to add.
        """

        # Check that the data is a wp.array
        if not isinstance(data, wp.array):
            raise ValueError(f"Data must be a wp.array. Got {type(data)}")

        # Check that the shape of the data is (num_envs, num_joints)
        if data.shape != (self.all_env_mask.shape[0], self.all_joint_mask.shape[0]):
            raise ValueError(f"Data must have the shape (num_envs, num_joints). Got {data.shape} but expected ({self.all_env_mask.shape[0]}, {self.all_joint_mask.shape[0]})")

        setattr(self, name, data)
