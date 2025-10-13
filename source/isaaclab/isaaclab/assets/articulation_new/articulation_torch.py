# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import omni.log
import warp as wp
from isaacsim.core.simulation_manager import SimulationManager
from newton import JointType, Model
from newton.selection import ArticulationView as NewtonArticulationView
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators_warp import ActuatorBaseWarp, ActuatorBaseWarpCfg, ImplicitActuatorWarp
from isaaclab.sim._impl.newton_manager import NewtonManager

from ..asset_base import AssetBase
from .articulation_data_torch import ArticulationDataTorch
from isaaclab.assets.core.joint_properties.joint import Joint
from isaaclab.assets.core.body_properties.body import Body
from isaaclab.assets.core.root_properties.root import Root
from isaaclab.assets.core.kernels import (
    generate_mask_from_ids,
    populate_empty_array,
    update_batched_array_with_array_masked,
    update_array_with_value,
)

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg

class ArticulationTorch(AssetBase):
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

    actuators: dict[str, ActuatorBaseWarp]
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
    def data(self) -> ArticulationDataTorch:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._root_newton_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self._root_newton_view.is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self._joint.num_joints

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        return 0

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in articulation."""
        return 0

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self._body.num_bodies

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self._joint.joint_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in articulation."""
        # TODO: check if the articulation has fixed tendons
        return []

    @property
    def spatial_tendon_names(self) -> list[str]:
        """Ordered names of spatial tendons in articulation."""
        # TODO: check if the articulation has spatial tendons
        return []

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        return self._body.body_names

    @property
    def root_newton_view(self) -> NewtonArticulationView:
        """Articulation view for the asset (Newton).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_newton_view

    @property
    def root_newton_model(self) -> Model:
        """Newton model for the asset."""
        return self._root_newton_view.model

    """
    Operations.
    """

    def reset(self, mask: wp.array):
        # use ellipses object to skip initial indices.
        self._root.reset(mask)
        self._joint.reset(mask)
        self._body.reset(mask)

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # Wrenches are automatically applied by set_external_force_and_torque.
        # apply actuator models
        self._apply_actuator_model()

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names, and indices.
        """
        mask, names, indices = self._body.find_bodies(name_keys, preserve_order)
        return wp.to_torch(mask), names, indices

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint mask, names, and indices.
        """
        mask, names, indices = self._joint.find_joints(name_keys, joint_subset, preserve_order)
        return wp.to_torch(mask), names, indices

    def find_fixed_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
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
            A tuple of lists containing the tendon mask, names, and indices.
        """
        raise NotImplementedError("Fixed tendons are not supported in Newton.")

    def find_spatial_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find spatial tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the tendon names.
            tendon_subsets: A subset of tendons to search for. Defaults to None, which means all tendons
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon mask, names, and indices.
        """
        raise NotImplementedError("Spatial tendons are not supported in Newton.")
    """
    Operations - State Writers.
    """

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_state: Root state in simulation frame. Shape is (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")

        self._root.write_root_state_to_sim(root_state, env_mask)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_state: Root state in simulation frame. Shape is (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_com_state_to_sim(root_state, env_mask)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_state: Root state in simulation frame. Shape is (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_link_state_to_sim(root_state, env_mask)

    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_link_pose_to_sim(root_pose, env_mask=env_mask)

    def write_root_link_pose_to_sim(
        self,
        pose: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose ``wp.transformf`` comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_link_pose_to_sim(pose, env_mask)

    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_com_pose_to_sim(root_pose, env_mask)

    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.
        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.
        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_com_velocity_to_sim(root_velocity, env_mask)

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.
        NOTE: If both env_mask and env_ids are provided, then env_mask will be used.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        self._root.write_root_link_velocity_to_sim(root_velocity, env_mask)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint positions and velocities to the simulation.

        Note: If both joint_mask and joint_ids are provided, then joint_mask will be used. If both env_mask and env_ids
        are provided, then env_mask will be used.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_state_to_sim(position, velocity, joint_mask, env_mask)

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint positions to the simulation.

        Note: If both joint_mask and joint_ids are provided, then joint_mask will be used. If both env_mask and env_ids
        are provided, then env_mask will be used.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_position_to_sim(position, joint_mask, env_mask)

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint velocities to the simulation.

        Args:
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_velocity_to_sim(velocity, joint_mask, env_mask)

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_control_mode_to_sim(
        self,
        control_mode: torch.Tensor | int,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ):
        """Write joint control mode into the simulation.

        Args:
            control_mode: Joint control mode. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).

        Raises:
            ValueError: If the control mode is invalid.
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_control_mode_to_sim(control_mode, joint_mask, env_mask)

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_stiffness_to_sim(stiffness, joint_mask, env_mask)

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_damping_to_sim(damping, joint_mask, env_mask)

    def write_joint_position_limit_to_sim(
        self,
        upper_limits: torch.Tensor | float,
        lower_limits: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint position limits into the simulation.

        Args:
            upper_limits: Joint upper limits. Shape is (num_instances, num_joints).
            lower_limits: Joint lower limits. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_position_limit_to_sim(upper_limits, lower_limits, joint_mask, env_mask)

    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. warn:: This function is ignored when using the Mujoco solver.

        Args:
            limits: Joint max velocity. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_velocity_limit_to_sim(limits, joint_mask, env_mask)

    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        Args:
            limits: Joint torque limits. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_effort_limit_to_sim(limits, joint_mask, env_mask)

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_armature_to_sim(armature, joint_mask, env_mask)

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | float,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        r"""Write joint friction coefficients into the simulation.

        The joint friction is a unitless quantity. It relates the magnitude of the spatial force transmitted
        from the parent body to the child body to the maximal friction force that may be applied by the solver
        to resist the joint motion.

        Mathematically, this means that: :math:`F_{resist} \leq \mu F_{spatial}`, where :math:`F_{resist}`
        is the resisting force applied by the solver and :math:`F_{spatial}` is the spatial force
        transmitted from the parent body to the child body. The simulated friction effect is therefore
        similar to static and Coulomb friction.

        Args:
            joint_friction_coeff: Joint friction coefficients. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_mask: The environment mask. Shape is (num_instances,).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.write_joint_friction_coefficient_to_sim(joint_friction_coeff, joint_mask, env_mask)

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_mask: torch.Tensor | None = None,
        body_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=wp.zeros(0, 3), torques=wp.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (num_instances, num_bodies, 3).
            torques: External torques in bodies' local frame. Shape is (num_instances, num_bodies, 3).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if body_mask is None:
            if body_ids is not None:
                body_mask = torch.zeros(self.num_bodies, dtype=torch.bool, device=self.device)
                body_mask[body_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if body_ids is not None:
                omni.log.warn("body_ids is not None, but body_mask is provided. Ignoring body_ids.")
        self._body.set_external_force_and_torque(forces, torques, body_mask, env_mask)

    def set_joint_position_target(
        self,
        target: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.set_joint_position_target(target, joint_mask, env_mask)

    def set_joint_velocity_target(
        self,
        target: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.set_joint_velocity_target(target, joint_mask, env_mask)

    def set_joint_effort_target(
        self,
        target: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        joint_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            if env_ids is not None:
                env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
                env_mask[env_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if env_ids is not None:
                omni.log.warn("env_ids is not None, but env_mask is provided. Ignoring env_ids.")
        if joint_mask is None:
            if joint_ids is not None:
                joint_mask = torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
                joint_mask[joint_ids] = True
                omni.log.warn("To optimize performances avoid passing ids and use mask instead. Ids may be removed in the future.")
        else:
            if joint_ids is not None:
                omni.log.warn("joint_ids is not None, but joint_mask is provided. Ignoring joint_ids.")
        self._joint.set_joint_effort_target(target, joint_mask, env_mask)

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        fixed_tendon_mask: torch.Tensor | None = None,
        fixed_tendon_ids: torch.Tensor | Sequence[int] | None = None,
        env_mask: torch.Tensor | None = None,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon stiffness is not supported in Newton.")

    def set_fixed_tendon_damping(
        self,
        damping: wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            damping: Fixed tendon damping. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon damping is not supported in Newton.")

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness efforts into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon limit stiffness is not supported in Newton.")

    def set_fixed_tendon_position_limit(
        self,
        limit: wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
            limit: Fixed tendon limit. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon position limit is not supported in Newton.")

    def set_fixed_tendon_rest_length(
        self,
        rest_length: wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length efforts into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            rest_length: Fixed tendon rest length. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon rest length is not supported in Newton.")

    def set_fixed_tendon_offset(
        self,
        offset: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon offset is not supported in Newton.")

    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation.

        Args:
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon properties are not supported in Newton.")

    def set_spatial_tendon_stiffness(
        self,
        stiffness: wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon stiffness is not supported in Newton.")

    def set_spatial_tendon_damping(
        self,
        damping: wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            damping: Spatial tendon damping. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon damping is not supported in Newton.")

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon limit stiffness is not supported in Newton.")

    def set_spatial_tendon_offset(
        self,
        offset: wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            offset: Spatial tendon offset. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon offset is not supported in Newton.")

    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation.

        Args:
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon properties are not supported in Newton.")

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        if self.cfg.articulation_root_prim_path is not None:
            # The articulation root prim path is specified explicitly, so we can just use this.
            root_prim_path_expr = self.cfg.prim_path + self.cfg.articulation_root_prim_path
        else:
            # No articulation root prim path was specified, so we need to search
            # for it. We search for this in the first environment and then
            # create a regex that matches all environments.
            first_env_matching_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
            if first_env_matching_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
            first_env_matching_prim_path = first_env_matching_prim.GetPath().pathString

            # Find all articulation root prims in the first environment.
            first_env_root_prims = sim_utils.get_all_matching_child_prims(
                first_env_matching_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            )
            if len(first_env_root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find an articulation when resolving '{first_env_matching_prim_path}'."
                    " Please ensure that the prim has 'USD ArticulationRootAPI' applied."
                )
            if len(first_env_root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single articulation when resolving '{first_env_matching_prim_path}'."
                    f" Found multiple '{first_env_root_prims}' under '{first_env_matching_prim_path}'."
                    " Please ensure that there is only one articulation in the prim path tree."
                )

            # resolve articulation root prim back into regex expression
            first_env_root_prim_path = first_env_root_prims[0].GetPath().pathString
            root_prim_path_relative_to_prim_path = first_env_root_prim_path[len(first_env_matching_prim_path) :]
            root_prim_path_expr = self.cfg.prim_path + root_prim_path_relative_to_prim_path

        prim_path = root_prim_path_expr.replace(".*", "*")

        # Perf implication when filtering fixed joints. --> Removing the joints from the middle.
        # May need to copy stuff. --> DoFs? Careful with joint properties.... 
        self._root_newton_view = NewtonArticulationView(
            NewtonManager.get_model(), prim_path, verbose=True, exclude_joint_types=[JointType.FREE, JointType.FIXED]
        )

        # container for data access
        self._data = ArticulationDataWarp(self._root_newton_view, self.device)

        # create backend setters for the data containers
        self._joint = Joint(self._root_newton_view, self._data.joint_data, self.cfg.soft_joint_pos_limit_factor, self.device)
        self._body = Body(self._root_newton_view, self._data.body_data, self.device)
        self._root = Root(self._root_newton_view, self._data.root_data, self.device)

        # process configuration
        self._process_cfg()
        self._process_actuators_cfg()
        self._process_tendons()
        # validate configuration
        self._validate_cfg()
        # update the robot data
        self.update(0.0)
        # log joint information
        self._log_articulation_info()

        # Offsets the spawned pose by the default root pose prior to initializing the solver. This ensures that the
        # solver is initialized at the correct pose, avoiding potential miscalculations in the maximum number of
        # constraints or contact required to run the simulation.
        # TODO: Do this is warp directly?
        generated_pose = wp.to_torch(self._data.default_root_pose).clone()
        generated_pose[:, :2] += wp.to_torch(self._root_newton_view.get_root_transforms(NewtonManager.get_model()))[
            :, :2
        ]
        self._root_newton_view.set_root_transforms(NewtonManager.get_state_0(), generated_pose)
        self._root_newton_view.set_root_transforms(NewtonManager.get_model(), generated_pose)


    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default pose with quaternion given as (w, x, y, z) --> (x, y, z, w)
        default_root_pose = tuple(self.cfg.init_state.pos) + (
            self.cfg.init_state.rot[1],
            self.cfg.init_state.rot[2],
            self.cfg.init_state.rot[3],
            self.cfg.init_state.rot[0],
        )
        # update the default root pose
        wp.launch(
            update_array_with_value,
            dim=(self.num_instances,),
            inputs=[
                wp.transformf(*default_root_pose),
                self._data.default_root_pose,
                self._root._ALL_ENV_MASK,
            ],
        )
        # default velocity
        default_root_velocity = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        wp.launch(
            update_array_with_value,
            dim=(self.num_instances,),
            inputs=[
                wp.spatial_vectorf(*default_root_velocity),
                self._data.default_root_vel,
                self._root._ALL_ENV_MASK,
            ],
        )
        # -- joint pos
        # joint pos
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_pos, self.joint_names
        )
        # Compute the mask once and use it for all joint operations
        self._joint._JOINT_MASK.fill_(False)
        wp.launch(
            generate_mask_from_ids,
            dim=(self.num_joints,),
            inputs=[
                self._joint._JOINT_MASK,
                wp.array(indices_list, dtype=wp.int32, device=self.device),
            ],
        )
        tmp_joint_data = wp.zeros((self.num_joints,), dtype=wp.float32, device=self.device)
        wp.launch(
            populate_empty_array,
            dim=(self.num_joints,),
            inputs=[
                wp.array(values_list, dtype=wp.float32, device=self.device),
                tmp_joint_data,
                wp.array(indices_list, dtype=wp.int32, device=self.device),
            ],
        )
        wp.launch(
            update_batched_array_with_array_masked,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                tmp_joint_data,
                self._data.default_joint_pos,
                self._joint._ALL_ENV_MASK,
                self._joint._JOINT_MASK,
            ],
        )
        # joint vel
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_vel, self.joint_names
        )
        wp.launch(
            populate_empty_array,
            dim=(self.num_joints,),
            inputs=[
                wp.array(values_list, dtype=wp.float32, device=self.device),
                tmp_joint_data,
                wp.array(indices_list, dtype=wp.int32, device=self.device),
            ],
        )
        wp.launch(
            update_batched_array_with_array_masked,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                tmp_joint_data,
                self._data.default_joint_vel,
                self._joint._ALL_ENV_MASK,
                self._joint._JOINT_MASK,
            ],
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)

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

        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # type annotation for type checkers
            actuator_cfg: ActuatorBaseWarpCfg
            # create actuator group
            joint_mask, joint_names, joint_indices = self.find_joints(actuator_cfg.joint_names_expr)
            # check if any joints are found
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            # create actuator collection
            # note: for efficiency avoid indexing when over all indices
            actuator: ActuatorBaseWarp = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_mask=joint_mask,
                env_mask=self._joint._ALL_ENV_MASK,
                articulation_data=self._data.joint_data,
                device=self.device,
            )
            # log information on actuator groups
            model_type = "implicit" if actuator.is_implicit_model else "explicit"
            omni.log.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (type: {model_type}) and joint names: {joint_names} [{joint_mask}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # set the passed gains and limits into the simulation
            # TODO: write out all joint parameters from simulation
            if isinstance(actuator, ImplicitActuatorWarp):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self.write_joint_stiffness_to_sim(self.data.joint_stiffness, joint_mask=actuator._joint_mask)
                self.write_joint_damping_to_sim(self.data.joint_damping, joint_mask=actuator._joint_mask)
                # Sets the control mode for the implicit actuators
                self.write_joint_control_mode_to_sim(self.data.joint_control_mode, joint_mask=actuator._joint_mask)

                # When using implicit actuators, we bind the commands sent from the user to the simulation.
                # We only run the actuator model to compute the estimated joint efforts.
                self.data.joint_target = self.data.sim_bind_joint_target
                self.data.joint_effort_target = self.data.sim_bind_joint_effort
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.write_joint_stiffness_to_sim(0.0, joint_mask=actuator._joint_mask)
                self.write_joint_damping_to_sim(0.0, joint_mask=actuator._joint_mask)
                # Set the control mode to None when using explicit actuators
                self.write_joint_control_mode_to_sim(0, joint_mask=actuator._joint_mask)
                # Bind the applied effort to the simulation effort
                self.data.applied_effort = self.data.sim_bind_joint_effort

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != (self.num_joints - self.num_fixed_tendons):
            omni.log.warn(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints - self.num_fixed_tendons}."
            )

    def _process_tendons(self):
        """Process fixed and spatialtendons."""
        # create a list to store the fixed tendon names
        self._fixed_tendon_names = list()
        self._spatial_tendon_names = list()
        # parse fixed tendons properties if they exist
        if self.num_fixed_tendons > 0:
            raise NotImplementedError("Tendons are not implemented yet")

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        # process actions per group
        for actuator in self.actuators.values():
            actuator.compute()
            # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
            # if hasattr(actuator, "gear_ratio"):
            #    self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio

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
        joint_pos_limits = torch.stack(
            (
                wp.to_torch(self._root_newton_view.get_attribute("joint_limit_lower", NewtonManager.get_model())),
                wp.to_torch(self._root_newton_view.get_attribute("joint_limit_upper", NewtonManager.get_model())),
            ),
            dim=2,
        )[0].to(self.device)
        out_of_range = wp.to_torch(self._data.default_joint_pos)[0] < joint_pos_limits[:, 0]
        out_of_range |= wp.to_torch(self._data.default_joint_pos)[0] > joint_pos_limits[:, 1]
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        # throw error if any of the default joint positions are out of the limits
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default positions out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = joint_pos_limits[idx]
                joint_pos = self.data.default_joint_pos[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

    def _log_articulation_info(self):
        """Log information about the articulation.

        Note: We purposefully read the values from the simulator to ensure that the values are configured as expected.
        """
        # TODO: read out all joint parameters from simulation
        # read out all joint parameters from simulation
        # -- gains
        stiffnesses = torch.zeros([self.num_joints], dtype=torch.float32, device=self.device).tolist()
        dampings = torch.zeros([self.num_joints], dtype=torch.float32, device=self.device).tolist()
        # -- properties
        armatures = torch.zeros([self.num_joints], dtype=torch.float32, device=self.device).tolist()
        frictions = torch.zeros([self.num_joints], dtype=torch.float32, device=self.device).tolist()
        # -- limits
        position_limits = torch.zeros([self.num_joints, 2], dtype=torch.float32, device=self.device).tolist()
        velocity_limits = torch.zeros([self.num_joints], dtype=torch.float32, device=self.device).tolist()
        effort_limits = torch.zeros([self.num_joints], dtype=torch.float32, device=self.device).tolist()
        # create table for term information
        joint_table = PrettyTable()
        joint_table.title = f"Simulation Joint Information (Prim path: {self.cfg.prim_path})"
        joint_table.field_names = [
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
        joint_table.float_format = ".3"
        joint_table.custom_format["Position Limits"] = lambda f, v: f"[{v[0]:.3f}, {v[1]:.3f}]"
        # set alignment of table columns
        joint_table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self.joint_names):
            joint_table.add_row([
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
        omni.log.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + joint_table.get_string())

        # read out all fixed tendon parameters from simulation
        if self.num_fixed_tendons > 0:
            raise NotImplementedError("Tendons are not implemented yet")

        if self.num_spatial_tendons > 0:
            raise NotImplementedError("Tendons are not implemented yet")
