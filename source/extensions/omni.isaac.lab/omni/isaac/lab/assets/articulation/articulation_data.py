# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from ..rigid_object import RigidObjectData


@dataclass
class ArticulationData(RigidObjectData):
    """Data container for an articulation."""

    ##
    # Properties.
    ##

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    ##
    # Default states.
    ##

    default_joint_pos: torch.Tensor = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints)."""

    default_joint_vel: torch.Tensor = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints)."""

    ##
    # Joint states <- From simulation.
    ##

    joint_pos: torch.Tensor = None
    """Joint positions of all joints. Shape is (num_instances, num_joints)."""

    joint_vel: torch.Tensor = None
    """Joint velocities of all joints. Shape is (num_instances, num_joints)."""

    joint_acc: torch.Tensor = None
    """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""

    ##
    # Joint commands -- Set into simulation.
    ##

    joint_pos_target: torch.Tensor = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_vel_target: torch.Tensor = None
    """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_effort_target: torch.Tensor = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    ##
    # Joint properties.
    ##

    joint_stiffness: torch.Tensor = None
    """Joint stiffness provided to simulation. Shape is (num_instances, num_joints)."""

    joint_damping: torch.Tensor = None
    """Joint damping provided to simulation. Shape is (num_instances, num_joints)."""

    joint_armature: torch.Tensor = None
    """Joint armature provided to simulation. Shape is (num_instances, num_joints)."""

    joint_friction: torch.Tensor = None
    """Joint friction provided to simulation. Shape is (num_instances, num_joints)."""

    joint_limits: torch.Tensor = None
    """Joint limits provided to simulation. Shape is (num_instances, num_joints, 2)."""

    ##
    # Default joint properties
    ##

    default_joint_stiffness: torch.Tensor = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints)."""

    default_joint_damping: torch.Tensor = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints)."""

    default_joint_armature: torch.Tensor = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints)."""

    default_joint_friction: torch.Tensor = None
    """Default joint friction of all joints. Shape is (num_instances, num_joints)."""

    default_joint_limits: torch.Tensor = None
    """Default joint limits of all joints. Shape is (num_instances, num_joints, 2)."""

    ##
    # Joint commands -- Explicit actuators.
    ##

    computed_torque: torch.Tensor = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.

    Note: The torques are zero for implicit actuator models.
    """

    applied_torque: torch.Tensor = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.

    Note: The torques are zero for implicit actuator models.
    """

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: torch.Tensor = None
    """Fixed tendon stiffness provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_damping: torch.Tensor = None
    """Fixed tendon damping provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit_stiffness: torch.Tensor = None
    """Fixed tendon limit stiffness provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_rest_length: torch.Tensor = None
    """Fixed tendon rest length provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_offset: torch.Tensor = None
    """Fixed tendon offset provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit: torch.Tensor = None
    """Fixed tendon limits provided to simulation. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Default fixed tendon properties
    ##

    default_fixed_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_damping: torch.Tensor = None
    """Default tendon damping of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_rest_length: torch.Tensor = None
    """Default tendon rest length of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_offset: torch.Tensor = None
    """Default tendon offset of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_limit: torch.Tensor = None
    """Default tendon limits of all tendons. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Other Data.
    ##

    soft_joint_pos_limits: torch.Tensor = None
    """Joint positions limits for all joints. Shape is (num_instances, num_joints, 2)."""

    soft_joint_vel_limits: torch.Tensor = None
    """Joint velocity limits for all joints. Shape is (num_instances, num_joints)."""

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
