# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import isaacsim.core.utils.stage as stage_utils
import omni.log
import omni.physics.tensors.impl.api as physx
from pxr import PhysxSchema, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg, ImplicitActuator
from isaaclab.utils.types import ArticulationActions

from ..asset_base import AssetBase
from .articulation_data import ArticulationData

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


class Articulation(AssetBase):
    """An articulation asset class.

    An articulation is a collection of rigid bodies connected by joints. The joints can be either
    fixed or actuated. The joints can be of different types, such as revolute, prismatic, D-6, etc.
    However, the articulation class has currently been tested with revolute and prismatic joints.
    The class supports both floating-base and fixed-base articulations. The type of articulation
    is determined based on the root joint of the articulation. If the root joint is fixed, then
    the articulation is considered a fixed-base system. Otherwise, it is considered a floating-base
    system. This can be checked using the :attr:`Articulation.is_fixed_base` attribute.

    For an asset to be considered an articulation, the root prim of the asset must have the
    `USD ArticulationRootAPI`_. This API is used to define the sub-tree of the articulation using
    the reduced coordinate formulation. On playing the simulation, the physics engine parses the
    articulation root prim and creates the corresponding articulation in the physics engine. The
    articulation root prim can be specified using the :attr:`AssetBaseCfg.prim_path` attribute.

    The articulation class also provides the functionality to augment the simulation of an articulated
    system with custom actuator models. These models can either be explicit or implicit, as detailed in
    the :mod:`isaaclab.actuators` module. The actuator models are specified using the
    :attr:`ArticulationCfg.actuators` attribute. These are then parsed and used to initialize the
    corresponding actuator models, when the simulation is played.

    During the simulation step, the articulation class first applies the actuator models to compute
    the joint commands based on the user-specified targets. These joint commands are then applied
    into the simulation. The joint commands can be either position, velocity, or effort commands.
    As an example, the following snippet shows how this can be used for position commands:

    .. code-block:: python

        # an example instance of the articulation class
        my_articulation = Articulation(cfg)

        # set joint position targets
        my_articulation.set_joint_position_target(position)
        # propagate the actuator models and apply the computed commands into the simulation
        my_articulation.write_data_to_sim()

        # step the simulation using the simulation context
        sim_context.step()

        # update the articulation state, where dt is the simulation time step
        my_articulation.update(dt)

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html

    """

    cfg: ArticulationCfg
    """Configuration instance for the articulations."""

    actuators: dict[str, ActuatorBase]
    """Dictionary of actuator instances for the articulation.

    The keys are the actuator names and the values are the actuator instances. The actuator instances
    are initialized based on the actuator configurations specified in the :attr:`ArticulationCfg.actuators`
    attribute. They are used to compute the joint commands during the :meth:`write_data_to_sim` function.
    """

    def __init__(self, cfg: ArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> ArticulationData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self.root_physx_view.shared_metatype.fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self.root_physx_view.shared_metatype.dof_count

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        return self.root_physx_view.max_fixed_tendons

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self.root_physx_view.shared_metatype.link_count

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self.root_physx_view.shared_metatype.dof_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in articulation."""
        return self._fixed_tendon_names

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        return self.root_physx_view.shared_metatype.link_names

    @property
    def root_physx_view(self) -> physx.ArticulationView:
        """Articulation view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = slice(None)
        # reset actuators
        for actuator in self.actuators.values():
            actuator.reset(env_ids)
        # reset external wrench
        self._external_force_b[env_ids] = 0.0
        self._external_torque_b[env_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=False,
            )

        # apply actuator models
        self._apply_actuator_model()
        # write actions into simulation
        self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES)
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:
            self.root_physx_view.set_dof_position_targets(self._joint_pos_target_sim, self._ALL_INDICES)
            self.root_physx_view.set_dof_velocity_targets(self._joint_vel_target_sim, self._ALL_INDICES)

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_fixed_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find fixed tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint
                names with fixed tendons.
            tendon_subsets: A subset of joints with fixed tendons to search for. Defaults to None, which means
                all joints in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        if tendon_subsets is None:
            # tendons follow the joint names they are attached to
            tendon_subsets = self.fixed_tendon_names
        # find tendons
        return string_utils.resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    """
    Operations - Writers.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """

        # set into simulation
        self.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # set into simulation
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # set into simulation
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_physx_view.set_root_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_link_state_w[env_ids, :7] = root_pose.clone()
        self._data.root_state_w[env_ids, :7] = self._data.root_link_state_w[env_ids, :7]
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_link_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_physx_view.set_root_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        com_pos = self.data.com_pos_b[env_ids, 0, :]
        com_quat = self.data.com_quat_b[env_ids, 0, :]

        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_rotate(math_utils.quat_inv(com_quat), -com_pos),
            math_utils.quat_inv(com_quat),
        )

        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
        self.write_root_link_pose_to_sim(root_pose=root_link_pose, env_ids=physx_env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, 7:] = root_velocity.clone()
        self._data.body_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_root_velocities(self._data.root_state_w[:, 7:], indices=physx_env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """

        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_com_state_w[env_ids, 7:] = root_velocity.clone()
        self._data.root_state_w[env_ids, 7:] = self._data.root_com_state_w[env_ids, 7:]
        self._data.body_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_root_velocities(self._data.root_com_state_w[:, 7:], indices=physx_env_ids)

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        root_com_velocity = root_velocity.clone()
        quat = self.data.root_link_state_w[env_ids, 3:7]
        com_pos_b = self.data.com_pos_b[env_ids, 0, :]
        # transform given velocity to center of mass
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_rotate(quat, com_pos_b), dim=-1
        )
        # write center of mass velocity to sim
        self.write_root_com_velocity_to_sim(root_velocity=root_com_velocity, env_ids=physx_env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_pos[env_ids, joint_ids] = position
        self._data.joint_vel[env_ids, joint_ids] = velocity
        self._data._previous_joint_vel[env_ids, joint_ids] = velocity
        self._data.joint_acc[env_ids, joint_ids] = 0.0
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_physx_view.set_dof_positions(self._data.joint_pos, indices=physx_env_ids)
        self.root_physx_view.set_dof_velocities(self._data.joint_vel, indices=physx_env_ids)

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_stiffness[env_ids, joint_ids] = stiffness
        # set into simulation
        self.root_physx_view.set_dof_stiffnesses(self._data.joint_stiffness.cpu(), indices=physx_env_ids.cpu())

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the damping for.
                Defaults to None (all joints).
            env_ids: The environment indices to set the damping for.
                Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_damping[env_ids, joint_ids] = damping
        # set into simulation
        self.root_physx_view.set_dof_dampings(self._data.joint_damping.cpu(), indices=physx_env_ids.cpu())

    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint max velocity to the simulation.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the max velocity for. Defaults to None (all joints).
            env_ids: The environment indices to set the max velocity for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # move tensor to cpu if needed
        if isinstance(limits, torch.Tensor):
            limits = limits.to(self.device)

        # set into internal buffers
        self._data.joint_velocity_limits = self.root_physx_view.get_dof_max_velocities().to(self.device)
        self._data.joint_velocity_limits[env_ids, joint_ids] = limits
        # set into simulation
        self.root_physx_view.set_dof_max_velocities(self._data.joint_velocity_limits.cpu(), indices=physx_env_ids.cpu())

    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint effort limits into the simulation.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # move tensor to cpu if needed
        if isinstance(limits, torch.Tensor):
            limits = limits.cpu()
        # set into internal buffers
        torque_limit_all = self.root_physx_view.get_dof_max_forces()
        torque_limit_all[env_ids, joint_ids] = limits
        # set into simulation
        self.root_physx_view.set_dof_max_forces(torque_limit_all.cpu(), indices=physx_env_ids.cpu())

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint armature into the simulation.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_armature[env_ids, joint_ids] = armature
        # set into simulation
        self.root_physx_view.set_dof_armatures(self._data.joint_armature.cpu(), indices=physx_env_ids.cpu())

    def write_joint_friction_to_sim(
        self,
        joint_friction: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint friction into the simulation.

        Args:
            joint_friction: Joint friction. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_friction[env_ids, joint_ids] = joint_friction
        # set into simulation
        self.root_physx_view.set_dof_friction_coefficients(self._data.joint_friction.cpu(), indices=physx_env_ids.cpu())

    def write_joint_limits_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint limits into the simulation.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2).
            joint_ids: The joint indices to set the limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions exceed the new limits. Defaults to True.
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_limits[env_ids, joint_ids] = limits
        # update default joint pos to stay within the new limits
        if torch.any(
            (self._data.default_joint_pos[env_ids, joint_ids] < limits[..., 0])
            | (self._data.default_joint_pos[env_ids, joint_ids] > limits[..., 1])
        ):
            self._data.default_joint_pos[env_ids, joint_ids] = torch.clamp(
                self._data.default_joint_pos[env_ids, joint_ids], limits[..., 0], limits[..., 1]
            )
            violation_message = (
                "Some default joint positions are outside of the range of the new joint limits. Default joint positions"
                " will be clamped to be within the new joint limits."
            )
            if warn_limit_violation:
                # warn level will show in console
                omni.log.warn(violation_message)
            else:
                # info level is only written to log file
                omni.log.info(violation_message)
        # set into simulation
        self.root_physx_view.set_dof_limits(self._data.joint_limits.cpu(), indices=physx_env_ids.cpu())

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
        """
        if forces.any() or torques.any():
            self.has_external_wrench = True
        else:
            self.has_external_wrench = False

        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        # -- body_ids
        if body_ids is None:
            body_ids = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
        elif isinstance(body_ids, slice):
            body_ids = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)[body_ids]
        elif not isinstance(body_ids, torch.Tensor):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # note: we need to do this complicated indexing since torch doesn't support multi-indexing
        # create global body indices from env_ids and env_body_ids
        # (env_id * total_bodies_per_env) + body_id
        indices = body_ids.repeat(len(env_ids), 1) + env_ids.unsqueeze(1) * self.num_bodies
        indices = indices.view(-1)
        # set into internal buffers
        # note: these are applied in the write_to_sim function
        self._external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
        self._external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint position targets into internal buffers.

        .. note::
            This function does not apply the joint targets to the simulation. It only fills the buffers with
            the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.joint_pos_target[env_ids, joint_ids] = target

    def set_joint_velocity_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint velocity targets into internal buffers.

        .. note::
            This function does not apply the joint targets to the simulation. It only fills the buffers with
            the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.joint_vel_target[env_ids, joint_ids] = target

    def set_joint_effort_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint efforts into internal buffers.

        .. note::
            This function does not apply the joint targets to the simulation. It only fills the buffers with
            the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.joint_effort_target[env_ids, joint_ids] = target

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon stiffness into internal buffers.

        .. note::
            This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
            the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set stiffness
        self._data.fixed_tendon_stiffness[env_ids, fixed_tendon_ids] = stiffness

    def set_fixed_tendon_damping(
        self,
        damping: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon damping into internal buffers.

        .. note::
            This function does not apply the tendon damping to the simulation. It only fills the buffers with
            the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set damping
        self._data.fixed_tendon_damping[env_ids, fixed_tendon_ids] = damping

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon limit stiffness efforts into internal buffers.

        .. note::
            This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
            the desired values. To apply the tendon limit stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set limit_stiffness
        self._data.fixed_tendon_limit_stiffness[env_ids, fixed_tendon_ids] = limit_stiffness

    def set_fixed_tendon_limit(
        self,
        limit: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon limit efforts into internal buffers.

        .. note::
             This function does not apply the tendon limit to the simulation. It only fills the buffers with
             the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
             limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)).
             fixed_tendon_ids: The tendon indices to set the limit for. Defaults to None (all fixed tendons).
             env_ids: The environment indices to set the limit for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set limit
        self._data.fixed_tendon_limit[env_ids, fixed_tendon_ids] = limit

    def set_fixed_tendon_rest_length(
        self,
        rest_length: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon rest length efforts into internal buffers.

        .. note::
            This function does not apply the tendon rest length to the simulation. It only fills the buffers with
            the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set rest_length
        self._data.fixed_tendon_rest_length[env_ids, fixed_tendon_ids] = rest_length

    def set_fixed_tendon_offset(
        self,
        offset: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon offset efforts into internal buffers.

        .. note::
            This function does not apply the tendon offset to the simulation. It only fills the buffers with
            the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set offset
        self._data.fixed_tendon_offset[env_ids, fixed_tendon_ids] = offset

    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write fixed tendon properties into the simulation.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the limits for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            physx_env_ids = self._ALL_INDICES
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)

        # set into simulation
        self.root_physx_view.set_fixed_tendon_properties(
            self._data.fixed_tendon_stiffness,
            self._data.fixed_tendon_damping,
            self._data.fixed_tendon_limit_stiffness,
            self._data.fixed_tendon_limit,
            self._data.fixed_tendon_rest_length,
            self._data.fixed_tendon_offset,
            indices=physx_env_ids,
        )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find articulation root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find an articulation when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD ArticulationRootAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single articulation when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one articulation in the prim path tree."
            )

        # resolve articulation root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- articulation
        self._root_physx_view = self._physics_sim_view.create_articulation_view(root_prim_path_expr.replace(".*", "*"))

        # check if the articulation was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create articulation at: {self.cfg.prim_path}. Please check PhysX logs.")

        # log information about the articulation
        omni.log.info(f"Articulation initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        omni.log.info(f"Is fixed root: {self.is_fixed_base}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        omni.log.info(f"Body names: {self.body_names}")
        omni.log.info(f"Number of joints: {self.num_joints}")
        omni.log.info(f"Joint names: {self.joint_names}")
        omni.log.info(f"Number of fixed tendons: {self.num_fixed_tendons}")

        # container for data access
        self._data = ArticulationData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        self._process_actuators_cfg()
        self._process_fixed_tendons()
        # validate configuration
        self._validate_cfg()
        # update the robot data
        self.update(0.0)
        # log joint information
        self._log_articulation_joint_info()

    def _create_buffers(self):
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # external forces and torques
        self.has_external_wrench = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)

        # asset data
        # -- properties
        self._data.joint_names = self.joint_names
        self._data.body_names = self.body_names

        # -- bodies
        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()

        # -- default joint state
        self._data.default_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_vel = torch.zeros_like(self._data.default_joint_pos)

        # -- joint commands
        self._data.joint_pos_target = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_vel_target = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_effort_target = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_stiffness = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_damping = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_armature = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_friction = torch.zeros_like(self._data.default_joint_pos)
        self._data.joint_limits = torch.zeros(self.num_instances, self.num_joints, 2, device=self.device)

        # -- joint commands (explicit)
        self._data.computed_torque = torch.zeros_like(self._data.default_joint_pos)
        self._data.applied_torque = torch.zeros_like(self._data.default_joint_pos)

        # -- tendons
        if self.num_fixed_tendons > 0:
            self._data.fixed_tendon_stiffness = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.fixed_tendon_damping = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.fixed_tendon_limit_stiffness = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.fixed_tendon_limit = torch.zeros(
                self.num_instances, self.num_fixed_tendons, 2, device=self.device
            )
            self._data.fixed_tendon_rest_length = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.fixed_tendon_offset = torch.zeros(self.num_instances, self.num_fixed_tendons, device=self.device)

        # -- other data
        self._data.soft_joint_pos_limits = torch.zeros(self.num_instances, self.num_joints, 2, device=self.device)
        self._data.soft_joint_vel_limits = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.gear_ratio = torch.ones(self.num_instances, self.num_joints, device=self.device)

        # -- initialize default buffers related to joint properties
        self._data.default_joint_stiffness = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_damping = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_armature = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_friction = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_limits = torch.zeros(self.num_instances, self.num_joints, 2, device=self.device)

        # -- initialize default buffers related to fixed tendon properties
        if self.num_fixed_tendons > 0:
            self._data.default_fixed_tendon_stiffness = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.default_fixed_tendon_damping = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.default_fixed_tendon_limit_stiffness = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.default_fixed_tendon_limit = torch.zeros(
                self.num_instances, self.num_fixed_tendons, 2, device=self.device
            )
            self._data.default_fixed_tendon_rest_length = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )
            self._data.default_fixed_tendon_offset = torch.zeros(
                self.num_instances, self.num_fixed_tendons, device=self.device
            )

        # soft joint position limits (recommended not to be too close to limits).
        joint_pos_limits = self.root_physx_view.get_dof_limits()
        joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
        joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor
        # add to data
        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

        # create buffers to store processed actions from actuator models
        self._joint_pos_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_vel_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_effort_target_sim = torch.zeros_like(self._data.joint_pos_target)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)
        # -- joint state
        # joint pos
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_pos, self.joint_names
        )
        self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)
        # joint vel
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_vel, self.joint_names
        )
        self._data.default_joint_vel[:, indices_list] = torch.tensor(values_list, device=self.device)

        # -- joint limits
        self._data.default_joint_limits = self.root_physx_view.get_dof_limits().to(device=self.device).clone()
        self._data.joint_limits = self._data.default_joint_limits.clone()

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None

    """
    Internal helpers -- Actuators.
    """

    def _process_actuators_cfg(self):
        """Process and apply articulation joint properties."""
        # create actuators
        self.actuators = dict()
        # flag for implicit actuators
        # if this is false, we by-pass certain checks when doing actuator-related operations
        self._has_implicit_actuators = False

        # cache the values coming from the usd
        self._data.default_joint_stiffness = self.root_physx_view.get_dof_stiffnesses().to(self.device).clone()
        self._data.default_joint_damping = self.root_physx_view.get_dof_dampings().to(self.device).clone()
        self._data.default_joint_armature = self.root_physx_view.get_dof_armatures().to(self.device).clone()
        self._data.default_joint_friction = self.root_physx_view.get_dof_friction_coefficients().to(self.device).clone()

        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # type annotation for type checkers
            actuator_cfg: ActuatorBaseCfg
            # create actuator group
            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)
            # check if any joints are found
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            # create actuator collection
            # note: for efficiency avoid indexing when over all indices
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=(
                    slice(None) if len(joint_names) == self.num_joints else torch.tensor(joint_ids, device=self.device)
                ),
                num_envs=self.num_instances,
                device=self.device,
                stiffness=self._data.default_joint_stiffness[:, joint_ids],
                damping=self._data.default_joint_damping[:, joint_ids],
                armature=self._data.default_joint_armature[:, joint_ids],
                friction=self._data.default_joint_friction[:, joint_ids],
                effort_limit=self.root_physx_view.get_dof_max_forces().to(self.device).clone()[:, joint_ids],
                velocity_limit=self.root_physx_view.get_dof_max_velocities().to(self.device).clone()[:, joint_ids],
            )
            # log information on actuator groups
            omni.log.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}' and"
                f" joint names: {joint_names} [{joint_ids}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # set the passed gains and limits into the simulation
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(actuator.damping, joint_ids=actuator.joint_indices)
                self.write_joint_effort_limit_to_sim(actuator.effort_limit, joint_ids=actuator.joint_indices)
                self.write_joint_velocity_limit_to_sim(actuator.velocity_limit, joint_ids=actuator.joint_indices)
                self.write_joint_armature_to_sim(actuator.armature, joint_ids=actuator.joint_indices)
                self.write_joint_friction_to_sim(actuator.friction, joint_ids=actuator.joint_indices)
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.write_joint_stiffness_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_effort_limit_to_sim(1.0e9, joint_ids=actuator.joint_indices)
                self.write_joint_velocity_limit_to_sim(actuator.velocity_limit, joint_ids=actuator.joint_indices)
                self.write_joint_armature_to_sim(actuator.armature, joint_ids=actuator.joint_indices)
                self.write_joint_friction_to_sim(actuator.friction, joint_ids=actuator.joint_indices)
            # Store the actual default stiffness and damping values for explicit and implicit actuators (not written the sim)
            self._data.default_joint_stiffness[:, actuator.joint_indices] = actuator.stiffness
            self._data.default_joint_damping[:, actuator.joint_indices] = actuator.damping

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != (self.num_joints - self.num_fixed_tendons):
            omni.log.warn(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints - self.num_fixed_tendons}."
            )

    def _process_fixed_tendons(self):
        """Process fixed tendons."""
        # create a list to store the fixed tendon names
        self._fixed_tendon_names = list()

        # parse fixed tendons properties if they exist
        if self.num_fixed_tendons > 0:
            stage = stage_utils.get_current_stage()
            joint_paths = self.root_physx_view.dof_paths[0]

            # iterate over all joints to find tendons attached to them
            for j in range(self.num_joints):
                usd_joint_path = joint_paths[j]
                # check whether joint has tendons - tendon name follows the joint name it is attached to
                joint = UsdPhysics.Joint.Get(stage, usd_joint_path)
                if joint.GetPrim().HasAPI(PhysxSchema.PhysxTendonAxisRootAPI):
                    joint_name = usd_joint_path.split("/")[-1]
                    self._fixed_tendon_names.append(joint_name)

            self._data.fixed_tendon_names = self._fixed_tendon_names
            self._data.default_fixed_tendon_stiffness = self.root_physx_view.get_fixed_tendon_stiffnesses().clone()
            self._data.default_fixed_tendon_damping = self.root_physx_view.get_fixed_tendon_dampings().clone()
            self._data.default_fixed_tendon_limit_stiffness = (
                self.root_physx_view.get_fixed_tendon_limit_stiffnesses().clone()
            )
            self._data.default_fixed_tendon_limit = self.root_physx_view.get_fixed_tendon_limits().clone()
            self._data.default_fixed_tendon_rest_length = self.root_physx_view.get_fixed_tendon_rest_lengths().clone()
            self._data.default_fixed_tendon_offset = self.root_physx_view.get_fixed_tendon_offsets().clone()

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        # process actions per group
        for actuator in self.actuators.values():
            # prepare input for actuator model based on cached data
            # TODO : A tensor dict would be nice to do the indexing of all tensors together
            control_action = ArticulationActions(
                joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
                joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
                joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            # compute joint command from the actuator model
            control_action = actuator.compute(
                control_action,
                joint_pos=self._data.joint_pos[:, actuator.joint_indices],
                joint_vel=self._data.joint_vel[:, actuator.joint_indices],
            )
            # update targets (these are set into the simulation)
            if control_action.joint_positions is not None:
                self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts
            # update state of the actuator model
            # -- torques
            self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
            self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort
            # -- actuator data
            self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
            # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
            if hasattr(actuator, "gear_ratio"):
                self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio

    """
    Internal helpers -- Debugging.
    """

    def _validate_cfg(self):
        """Validate the configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        # check that the default values are within the limits
        joint_pos_limits = self.root_physx_view.get_dof_limits()[0].to(self.device)
        out_of_range = self._data.default_joint_pos[0] < joint_pos_limits[:, 0]
        out_of_range |= self._data.default_joint_pos[0] > joint_pos_limits[:, 1]
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        # throw error if any of the default joint positions are out of the limits
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default positions out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limits = joint_pos_limits[idx]
                joint_pos = self.data.default_joint_pos[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limits[0]:.3f}, {joint_limits[1]:.3f}]\n"
            raise ValueError(msg)

        # check that the default joint velocities are within the limits
        joint_max_vel = self.root_physx_view.get_dof_max_velocities()[0].to(self.device)
        out_of_range = torch.abs(self._data.default_joint_vel[0]) > joint_max_vel
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default velocities out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limits = [-joint_max_vel[idx], joint_max_vel[idx]]
                joint_vel = self.data.default_joint_vel[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limits[0]:.3f}, {joint_limits[1]:.3f}]\n"
            raise ValueError(msg)

    def _log_articulation_joint_info(self):
        """Log information about the articulation's simulated joints."""
        # read out all joint parameters from simulation
        # -- gains
        stiffnesses = self.root_physx_view.get_dof_stiffnesses()[0].tolist()
        dampings = self.root_physx_view.get_dof_dampings()[0].tolist()
        # -- properties
        armatures = self.root_physx_view.get_dof_armatures()[0].tolist()
        frictions = self.root_physx_view.get_dof_friction_coefficients()[0].tolist()
        # -- limits
        position_limits = self.root_physx_view.get_dof_limits()[0].tolist()
        velocity_limits = self.root_physx_view.get_dof_max_velocities()[0].tolist()
        effort_limits = self.root_physx_view.get_dof_max_forces()[0].tolist()
        # create table for term information
        table = PrettyTable(float_format=".3f")
        table.title = f"Simulation Joint Information (Prim path: {self.cfg.prim_path})"
        table.field_names = [
            "Index",
            "Name",
            "Stiffness",
            "Damping",
            "Armature",
            "Friction",
            "Position Limits",
            "Velocity Limits",
            "Effort Limits",
        ]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self.joint_names):
            table.add_row([
                index,
                name,
                stiffnesses[index],
                dampings[index],
                armatures[index],
                frictions[index],
                position_limits[index],
                velocity_limits[index],
                effort_limits[index],
            ])
        # convert table to string
        omni.log.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + table.get_string())

        # read out all tendon parameters from simulation
        if self.num_fixed_tendons > 0:
            # -- gains
            ft_stiffnesses = self.root_physx_view.get_fixed_tendon_stiffnesses()[0].tolist()
            ft_dampings = self.root_physx_view.get_fixed_tendon_dampings()[0].tolist()
            # -- limits
            ft_limit_stiffnesses = self.root_physx_view.get_fixed_tendon_limit_stiffnesses()[0].tolist()
            ft_limits = self.root_physx_view.get_fixed_tendon_limits()[0].tolist()
            ft_rest_lengths = self.root_physx_view.get_fixed_tendon_rest_lengths()[0].tolist()
            ft_offsets = self.root_physx_view.get_fixed_tendon_offsets()[0].tolist()
            # create table for term information
            tendon_table = PrettyTable(float_format=".3f")
            tendon_table.title = f"Simulation Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Limit",
                "Rest Length",
                "Offset",
            ]
            # add info on each term
            for index in range(self.num_fixed_tendons):
                tendon_table.add_row([
                    index,
                    ft_stiffnesses[index],
                    ft_dampings[index],
                    ft_limit_stiffnesses[index],
                    ft_limits[index],
                    ft_rest_lengths[index],
                    ft_offsets[index],
                ])
            # convert table to string
            omni.log.info(f"Simulation parameters for tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string())
