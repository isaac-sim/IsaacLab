# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from pxr import UsdPhysics

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg, ImplicitActuator
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.sim.utils.queries import find_first_matching_prim, get_all_matching_child_prims
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.assets.articulation import kernels as articulation_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

from .articulation_data import ArticulationData

if TYPE_CHECKING:
    import omni.physics.tensors.impl.api as physx

    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

# import logger
logger = logging.getLogger(__name__)


class Articulation(BaseArticulation):
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

    __backend_name__: str = "physx"
    """The name of the backend for the articulation."""

    actuators: dict
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
        return self.root_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self.root_view.shared_metatype.fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self.root_view.shared_metatype.dof_count

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        return self.root_view.max_fixed_tendons

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in articulation."""
        return self.root_view.max_spatial_tendons

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self.root_view.shared_metatype.link_count

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self.root_view.shared_metatype.dof_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in articulation."""
        return self._fixed_tendon_names

    @property
    def spatial_tendon_names(self) -> list[str]:
        """Ordered names of spatial tendons in articulation."""
        return self._spatial_tendon_names

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        return self.root_view.shared_metatype.link_names

    @property
    def root_view(self) -> physx.ArticulationView:
        """Root view for the asset.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_view

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer for the articulation."""
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer for the articulation."""
        return self._permanent_wrench_composer

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the articulation.

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # use ellipses object to skip initial indices.
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = slice(None)
        # reset actuators
        for actuator in self.actuators.values():
            actuator.reset(env_ids)
        # reset external wrenches.
        self._instantaneous_wrench_composer.reset(env_ids, env_mask)
        self._permanent_wrench_composer.reset(env_ids, env_mask)

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench
                self._instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES,
                    env_ids=self._ALL_INDICES,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self._instantaneous_wrench_composer.composed_force.flatten().view(wp.float32),
                    torque_data=self._instantaneous_wrench_composer.composed_torque.flatten().view(wp.float32),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
            else:
                # Apply permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self._permanent_wrench_composer.composed_force.flatten().view(wp.float32),
                    torque_data=self._permanent_wrench_composer.composed_torque.flatten().view(wp.float32),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
        self._instantaneous_wrench_composer.reset()

        # apply actuator models
        self._apply_actuator_model()
        # write actions into simulation
        self.root_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES)
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:
            self.root_view.set_dof_position_targets(self._joint_pos_target_sim, self._ALL_INDICES)
            self.root_view.set_dof_velocity_targets(self._joint_vel_target_sim, self._ALL_INDICES)

    def update(self, dt: float):
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        self.data.update(dt)

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
        return resolve_matching_names(name_keys, self.body_names, preserve_order)

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
        return resolve_matching_names(name_keys, joint_subset, preserve_order)

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
        return resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    def find_spatial_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find spatial tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the tendon names.
            tendon_subsets: A subset of tendons to search for. Defaults to None, which means all tendons
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        if tendon_subsets is None:
            tendon_subsets = self.spatial_tendon_names
        # find tendons
        return resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    """
    Operations - State Writers.
    """

    def write_root_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expect partial data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim_index(root_pose, env_ids=env_ids)

    def write_root_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim_mask(root_pose, env_mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_pose,
                env_ids,
                full_data,
            ],
            outputs=[
                self.data._root_link_pose_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_link_pose_w.timestamp = self.data._sim_timestamp
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        # Need to invalidate the buffer to trigger the update with the new state.
        self.data._body_link_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_root_transforms(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_root_link_pose_to_sim_index(root_pose, env_ids=env_ids, full_data=True)

    def write_root_com_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        # Note: we are doing a single launch for faster performance. Prior versions would call
        # write_root_link_pose_to_sim after this.
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_pose,
                self.data.body_com_pose_b,
                env_ids,
                full_data,
            ],
            outputs=[
                self.data._root_com_pose_w.data,
                self.data._root_link_pose_w.data,
                None,  # self.data._root_com_state_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_com_pose_w.timestamp = self.data._sim_timestamp
        self.data._root_link_pose_w.timestamp = self.data._sim_timestamp
        self.data._root_com_state_w.timestamp = -1.0
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        # Need to invalidate the buffer to trigger the update with the new state.
        self.data._body_link_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_root_transforms(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_root_com_pose_to_sim_index(root_pose, env_ids=env_ids, full_data=True)

    def write_root_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        .. note:: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expect partial data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        .. note:: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim_mask(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_com_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        .. note:: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or
                (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_velocity,
                env_ids,
                self.data._num_bodies,
                full_data,
            ],
            outputs=[
                self.data._root_com_vel_w.data,
                self.data._body_com_acc_w.data,
                None,  # self.data._root_state_w.data,
                None,  # self.data._root_com_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_acc_w.timestamp = self.data._sim_timestamp
        self.data._root_state_w.timestamp = -1.0
        self.data._root_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_root_velocities(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        .. note:: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES

        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_root_com_velocity_to_sim_index(root_velocity, env_ids=env_ids, full_data=True)

    def write_root_link_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        .. note:: This sets the velocity of the root's frame rather than the roots center of mass.

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6) or
                (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        # Note: we are doing a single launch for faster performance. Prior versions would do multiple launches.
        wp.launch(
            shared_kernels.set_root_link_velocity_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_velocity,
                self.data.body_com_pose_b,
                self.data.root_link_pose_w,
                env_ids,
                self.data._num_bodies,
                full_data,
            ],
            outputs=[
                self.data._root_link_vel_w.data,
                self.data._root_com_vel_w.data,
                self.data._body_com_acc_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
                None,  # self.data._root_com_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_link_vel_w.timestamp = self.data._sim_timestamp
        self.data._root_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_acc_w.timestamp = self.data._sim_timestamp
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        self.data._root_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_root_velocities(self.data._root_link_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        .. note:: This sets the velocity of the root's frame rather than the roots center of mass.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_root_link_velocity_to_sim_index(root_velocity, env_ids=env_ids, full_data=True)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ):
        """Deprecated, same as :meth:`write_joint_position_to_sim_index` and
        :meth:`write_joint_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_position_to_sim_index' and 'write_joint_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # set into simulation
        self.write_joint_position_to_sim_index(position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim_index(velocity, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_state_to_sim_mask(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        joint_mask: wp.array | None = None,
    ):
        """Write joint positions and velocities over selected environment mask into the simulation.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            env_mask: Environment mask. If None, then all indices are used.
            joint_mask: Joint mask. If None, then all joints are used.
        """
        # set into simulation
        self.write_joint_position_to_sim_mask(position, env_mask=env_mask, joint_mask=joint_mask)
        self.write_joint_velocity_to_sim_mask(velocity, env_mask=env_mask, joint_mask=joint_mask)

    def write_joint_position_to_sim_index(
        self,
        position: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint positions over selected environment indices into the simulation.

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                position,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_pos.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._joint_pos.timestamp = self.data._sim_timestamp
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self.data._body_com_vel_w.timestamp = -1.0
        self.data._body_link_vel_w.timestamp = -1.0
        self.data._body_com_pose_b.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_link_pose_w.timestamp = -1.0

        self.data._body_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_dof_positions(self.data._joint_pos.data, indices=env_ids)

    def write_joint_position_to_sim_mask(
        self,
        position: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint positions over selected environment mask into the simulation.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_joint_position_to_sim_index(position, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def write_joint_velocity_to_sim_index(
        self,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint velocities to the simulation.

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            articulation_kernels.write_joint_vel_data,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                velocity,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_vel.data,
                self.data._previous_joint_vel,
                self.data._joint_acc.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._joint_vel.timestamp = self.data._sim_timestamp
        self.data._joint_acc.timestamp = self.data._sim_timestamp
        # set into simulation
        self.root_view.set_dof_velocities(self.data._joint_vel.data, indices=env_ids)

    def write_joint_velocity_to_sim_mask(
        self,
        velocity: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint velocities over selected environment mask into the simulation.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_joint_velocity_to_sim_index(velocity, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_stiffness_to_sim_index(
        self,
        stiffness: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint stiffness over selected environment indices into the simulation.

        .. note::
            This method expect partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        if isinstance(stiffness, float):
            wp.launch(
                articulation_kernels.float_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    stiffness,
                    env_ids,
                    joint_ids,
                ],
                outputs=[
                    self.data._joint_stiffness,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    stiffness,
                    env_ids,
                    joint_ids,
                    full_data,
                ],
                outputs=[
                    self.data._joint_stiffness,
                ],
                device=self.device,
            )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_stiffnesses(wp.clone(self.data._joint_stiffness, device="cpu"), indices=cpu_env_ids)

    def write_joint_stiffness_to_sim_mask(
        self,
        stiffness: torch.Tensor | wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint stiffness over selected environment mask into the simulation.

        .. note::
            This method expect full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            stiffness: Joint stiffness. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the the right code path is taken inside the kernel.
        self.write_joint_stiffness_to_sim_index(stiffness, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def write_joint_damping_to_sim_index(
        self,
        damping: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint damping over selected environment indices into the simulation.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # Note This function isn't setting the values for actuator models. (#128)
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        if isinstance(damping, float):
            wp.launch(
                articulation_kernels.float_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    damping,
                    env_ids,
                    joint_ids,
                ],
                outputs=[
                    self.data._joint_damping,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    damping,
                    env_ids,
                    joint_ids,
                    full_data,
                ],
                outputs=[
                    self.data._joint_damping,
                ],
                device=self.device,
            )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_dampings(wp.clone(self.data._joint_damping, device="cpu"), indices=cpu_env_ids)

    def write_joint_damping_to_sim_mask(
        self,
        damping: torch.Tensor | wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint damping over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            damping: Joint damping. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_damping_to_sim_index(damping, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def write_joint_position_limit_to_sim_index(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
        warn_limit_violation: bool = True,
    ):
        """Write joint position limits over selected environment indices into the simulation.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2) or (num_instances, num_joints, 2).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        # Note This function isn't setting the values for actuator models. (#128)
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        clamped_defaults = wp.zeros(1, dtype=wp.int32, device=self.device)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        # Note: we are doing a single launch for faster performance. Prior versions would do this in multiple launches.
        if isinstance(limits, float):
            raise ValueError("Joint position limits must be a tensor or array, not a float.")
        wp.launch(
            articulation_kernels.write_joint_limit_data_to_buffer,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                limits,
                self.cfg.soft_joint_pos_limit_factor,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_pos_limits,
                self.data._soft_joint_pos_limits,
                self.data._default_joint_pos,
                clamped_defaults,
            ],
            device=self.device,
        )
        # Log a warning if the default joint positions are outside of the new limits.
        if clamped_defaults.numpy()[0] > 0:
            violation_message = (
                "Some default joint positions are outside of the range of the new joint limits. Default joint positions"
                " will be clamped to be within the new joint limits."
            )
            if warn_limit_violation:
                logger.warning(violation_message)
            else:
                logger.info(violation_message)
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_limits(wp.clone(self.data._joint_pos_limits, device="cpu"), indices=cpu_env_ids)

    def write_joint_position_limit_to_sim_mask(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint position limits over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limits: Joint limits. Shape is (num_instances, num_joints, 2).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_position_limit_to_sim_index(
            limits, joint_ids=joint_ids, env_ids=env_ids, full_data=True, warn_limit_violation=warn_limit_violation
        )

    def write_joint_velocity_limit_to_sim_index(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint max velocity over selected environment indices into the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        if isinstance(limits, float):
            wp.launch(
                articulation_kernels.float_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                ],
                outputs=[
                    self.data._joint_vel_limits,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                    full_data,
                ],
                outputs=[
                    self.data._joint_vel_limits,
                ],
                device=self.device,
            )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_max_velocities(wp.clone(self.data._joint_vel_limits, device="cpu"), indices=cpu_env_ids)

    def write_joint_velocity_limit_to_sim_mask(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint max velocity over selected environment mask into the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limits: Joint max velocity. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_velocity_limit_to_sim_index(limits, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def write_joint_effort_limit_to_sim_index(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint effort limits over selected environment indices into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # Note This function isn't setting the values for actuator models. (#128)
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        if isinstance(limits, float):
            wp.launch(
                articulation_kernels.float_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                ],
                outputs=[
                    self.data._joint_effort_limits,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                    full_data,
                ],
                outputs=[
                    self.data._joint_effort_limits,
                ],
                device=self.device,
            )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_max_forces(wp.clone(self.data._joint_effort_limits, device="cpu"), indices=cpu_env_ids)

    def write_joint_effort_limit_to_sim_mask(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint effort limits over selected environment mask into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limits: Joint torque limits. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_effort_limit_to_sim_index(limits, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def write_joint_armature_to_sim_index(
        self,
        armature: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Write joint armature over selected environment indices into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        if isinstance(armature, float):
            wp.launch(
                articulation_kernels.float_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    armature,
                    env_ids,
                    joint_ids,
                ],
                outputs=[
                    self.data._joint_armature,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(env_ids.shape[0], joint_ids.shape[0]),
                inputs=[
                    armature,
                    env_ids,
                    joint_ids,
                    full_data,
                ],
                outputs=[
                    self.data._joint_armature,
                ],
                device=self.device,
            )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_armatures(wp.clone(self.data._joint_armature, device="cpu"), indices=cpu_env_ids)

    def write_joint_armature_to_sim_mask(
        self,
        armature: torch.Tensor | wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint armature over selected environment mask into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            armature: Joint armature. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_armature_to_sim_index(armature, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def write_joint_friction_coefficient_to_sim_index(
        self,
        joint_friction_coeff: torch.Tensor | wp.array | float,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        r"""Write joint friction coefficients over selected environment indices into the simulation.

        For Isaac Sim versions below 5.0, only the static friction coefficient is set.
        This limits the resisting force or torque up to a maximum proportional to the transmitted
        spatial force: :math:`\|F_{resist}\| \leq \mu_s \, \|F_{spatial}\|`.

        For Isaac Sim versions 5.0 and above, the static, dynamic, and viscous friction coefficients
        are set. The model combines Coulomb (static & dynamic) friction with a viscous term:

        - Static friction :math:`\mu_s` defines the maximum effort that prevents motion at rest.
        - Dynamic friction :math:`\mu_d` applies once motion begins and remains constant during motion.
        - Viscous friction :math:`c_v` is a velocity-proportional resistive term.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            joint_friction_coeff: Static friction coefficient :math:`\mu_s`.
                Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient :math:`\mu_d`.
                Same shape as above. If None, the dynamic coefficient is not updated.
            joint_viscous_friction_coeff: Viscous friction coefficient :math:`c_v`.
                Same shape as above. If None, the viscous coefficient is not updated.
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Get the friction properties from the simulation.
        friction_props = wp.clone(self.root_view.get_dof_friction_properties(), device=self.device)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        # Note: we are doing a single launch for faster performance. Prior versions would do this in multiple launches.
        wp.launch(
            articulation_kernels.write_joint_friction_data_to_buffer,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                joint_friction_coeff,
                joint_dynamic_friction_coeff,
                joint_viscous_friction_coeff,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_friction_coeff,
                self.data._joint_dynamic_friction_coeff,
                self.data._joint_viscous_friction_coeff,
                friction_props,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_friction_properties(wp.clone(friction_props, device="cpu"), indices=cpu_env_ids)

    def write_joint_friction_coefficient_to_sim_mask(
        self,
        joint_friction_coeff: torch.Tensor | wp.array,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | None = None,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        r"""Write joint friction coefficients over selected environment mask into the simulation.

        For Isaac Sim versions below 5.0, only the static friction coefficient is set.
        This limits the resisting force or torque up to a maximum proportional to the transmitted
        spatial force: :math:`\|F_{resist}\| \leq \mu_s \, \|F_{spatial}\|`.

        For Isaac Sim versions 5.0 and above, the static, dynamic, and viscous friction coefficients
        are set. The model combines Coulomb (static & dynamic) friction with a viscous term:

        - Static friction :math:`\mu_s` defines the maximum effort that prevents motion at rest.
        - Dynamic friction :math:`\mu_d` applies once motion begins and remains constant during motion.
        - Viscous friction :math:`c_v` is a velocity-proportional resistive term.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            joint_friction_coeff: Static friction coefficient :math:`\mu_s`.
                Shape is (num_instances, num_joints).
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient :math:`\mu_d`.
                Same shape as above. If None, the dynamic coefficient is not updated.
            joint_viscous_friction_coeff: Viscous friction coefficient :math:`c_v`.
                Same shape as above. If None, the viscous coefficient is not updated.
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff,
            joint_dynamic_friction_coeff=joint_dynamic_friction_coeff,
            joint_viscous_friction_coeff=joint_viscous_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
            full_data=True,
        )

    def write_joint_dynamic_friction_coefficient_to_sim_index(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Write joint dynamic friction coefficient over selected environment indices into the simulation.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            joint_dynamic_friction_coeff: Joint dynamic friction coefficient. Shape is (len(env_ids), len(joint_ids))
                or (num_instances, num_joints) if full_data.
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Get the friction properties from the simulation.
        friction_props = wp.clone(self.root_view.get_dof_friction_properties(), device=self.device)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        # Note: we are doing a single launch for faster performance. Prior versions would do this in multiple launches.
        wp.launch(
            articulation_kernels.write_joint_friction_param_to_buffer,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                joint_dynamic_friction_coeff,
                env_ids,
                joint_ids,
                1,
                full_data,
            ],
            outputs=[
                self.data._joint_dynamic_friction_coeff,
                friction_props,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_friction_properties(wp.clone(friction_props, device="cpu"), indices=cpu_env_ids)

    def write_joint_dynamic_friction_coefficient_to_sim_mask(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint dynamic friction coefficient over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            joint_dynamic_friction_coeff: Joint dynamic friction coefficient. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_dynamic_friction_coefficient_to_sim_index(
            joint_dynamic_friction_coeff, joint_ids=joint_ids, env_ids=env_ids, full_data=True
        )

    def write_joint_viscous_friction_coefficient_to_sim_index(
        self,
        joint_viscous_friction_coeff: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Write joint viscous friction coefficient over selected environment indices into the simulation.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            joint_viscous_friction_coeff: Joint viscous friction coefficient. Shape is (len(env_ids), len(joint_ids))
                or (num_instances, num_joints) if full_data.
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        if get_isaac_sim_version().major < 5:
            logger.warning("Setting joint viscous friction coefficients are not supported in Isaac Sim < 5.0")
            return
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Get the friction properties from the simulation.
        friction_props = wp.clone(self.root_view.get_dof_friction_properties(), device=self.device)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        # Note: we are doing a single launch for faster performance. Prior versions would do this in multiple launches.
        wp.launch(
            articulation_kernels.write_joint_friction_param_to_buffer,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                joint_viscous_friction_coeff,
                env_ids,
                joint_ids,
                2,
                full_data,
            ],
            outputs=[
                self.data._joint_viscous_friction_coeff,
                friction_props,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_dof_friction_properties(wp.clone(friction_props, device="cpu"), indices=cpu_env_ids)

    def write_joint_viscous_friction_coefficient_to_sim_mask(
        self,
        joint_viscous_friction_coeff: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint viscous friction coefficient over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            joint_viscous_friction_coeff: Joint viscous friction coefficient. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve masks
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff, joint_ids=joint_ids, env_ids=env_ids, full_data=True
        )

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set masses of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)) or (num_instances, num_bodies)
                if full_data.
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                masses,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_mass,
            ],
            device=self.device,
        )

        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_masses(wp.clone(self.data._body_mass, device="cpu"), indices=cpu_env_ids)

    def set_masses_mask(
        self,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_masses_index(masses, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_coms_index(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set center of mass pose of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids), 7) or
                (num_instances, num_bodies, 7) if full_data.
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                coms,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_com_pose_b.data,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        # Convert from wp.transformf to flat (N, M, 7) array for PhysX
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        body_com_flat = (
            wp.clone(self.data._body_com_pose_b.data, device="cpu")
            .view(wp.float32)
            .reshape((self.num_instances, self.num_bodies, 7))
        )
        self.root_view.set_coms(body_com_flat, indices=cpu_env_ids)

    def set_coms_mask(
        self,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (num_instances, num_bodies, 7).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_coms_index(coms, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_inertias_index(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set inertias of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9) or
                (num_instances, num_bodies, 9) if full_data.
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                inertias,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_inertia,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        cpu_env_ids = wp.clone(env_ids, device="cpu")
        self.root_view.set_inertias(wp.clone(self.data._body_inertia, device="cpu"), indices=cpu_env_ids)

    def set_inertias_mask(
        self,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_inertias_index(inertias, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_joint_position_target_index(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set joint position targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints)
                if full_data.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                target,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_pos_target,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the targets to the simulation.

    def set_joint_position_target_mask(
        self,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            target: Joint position targets. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_joint_position_target_index(target, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def set_joint_velocity_target_index(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set joint velocity targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints)
                if full_data.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                target,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_vel_target,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the targets to the simulation.

    def set_joint_velocity_target_mask(
        self,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers using masks.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            target: Joint velocity targets. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_joint_velocity_target_index(target, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    def set_joint_effort_target_index(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set joint efforts into internal buffers using indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints)
                if full_data.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                target,
                env_ids,
                joint_ids,
                full_data,
            ],
            outputs=[
                self.data._joint_effort_target,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the targets to the simulation.

    def set_joint_effort_target_mask(
        self,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            target: Joint effort targets. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if joint_mask is not None:
            joint_ids = wp.nonzero(joint_mask)
        else:
            joint_ids = self._ALL_JOINT_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_joint_effort_target_index(target, joint_ids=joint_ids, env_ids=env_ids, full_data=True)

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness_index(
        self,
        stiffness: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers using indices.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)) or
                (num_instances, num_fixed_tendons) if full_data.
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        fixed_tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[
                stiffness,
                env_ids,
                fixed_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._fixed_tendon_stiffness,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the stiffness to the simulation.

    def set_fixed_tendon_stiffness_mask(
        self,
        stiffness: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers using masks.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if fixed_tendon_mask is not None:
            fixed_tendon_ids = wp.nonzero(fixed_tendon_mask)
        else:
            fixed_tendon_ids = self._ALL_FIXED_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_fixed_tendon_stiffness_index(
            stiffness, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_fixed_tendon_damping_index(
        self,
        damping: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set fixed tendon damping into internal buffers using indices.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)) or
                (num_instances, num_fixed_tendons) if full_data.
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        fixed_tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[
                damping,
                env_ids,
                fixed_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._fixed_tendon_damping,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the damping to the simulation.

    def set_fixed_tendon_damping_mask(
        self,
        damping: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers using masks.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            damping: Fixed tendon damping. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if fixed_tendon_mask is not None:
            fixed_tendon_ids = wp.nonzero(fixed_tendon_mask)
        else:
            fixed_tendon_ids = self._ALL_FIXED_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_fixed_tendon_damping_index(damping, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids, full_data=True)

    def set_fixed_tendon_limit_stiffness_index(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set fixed tendon limit stiffness into internal buffers using indices.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)) or
                (num_instances, num_fixed_tendons) if full_data.
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        fixed_tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[
                limit_stiffness,
                env_ids,
                fixed_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._fixed_tendon_limit_stiffness,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the limit stiffness to the simulation.

    def set_fixed_tendon_limit_stiffness_mask(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness into internal buffers using masks.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if fixed_tendon_mask is not None:
            fixed_tendon_ids = wp.nonzero(fixed_tendon_mask)
        else:
            fixed_tendon_ids = self._ALL_FIXED_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_fixed_tendon_limit_stiffness_index(
            limit_stiffness, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_fixed_tendon_position_limit_index(
        self,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set fixed tendon position limit into internal buffers using indices.

        This function does not apply the tendon position limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon position limit, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limit: Fixed tendon position limit. Shape is (len(env_ids), len(fixed_tendon_ids)) or
                (num_instances, num_fixed_tendons) if full_data.
            fixed_tendon_ids: The tendon indices to set the position limit for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        fixed_tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[
                limit,
                env_ids,
                fixed_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._fixed_tendon_pos_limits,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the position limit to the simulation.

    def set_fixed_tendon_position_limit_mask(
        self,
        limit: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon position limit into internal buffers using masks.

        This function does not apply the tendon position limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon position limit, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limit: Fixed tendon position limit. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if fixed_tendon_mask is not None:
            fixed_tendon_ids = wp.nonzero(fixed_tendon_mask)
        else:
            fixed_tendon_ids = self._ALL_FIXED_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_fixed_tendon_position_limit_index(
            limit, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_fixed_tendon_rest_length_index(
        self,
        rest_length: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set fixed tendon rest length into internal buffers using indices.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)) or
                (num_instances, num_fixed_tendons) if full_data.
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        fixed_tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[
                rest_length,
                env_ids,
                fixed_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._fixed_tendon_rest_length,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the rest length to the simulation.

    def set_fixed_tendon_rest_length_mask(
        self,
        rest_length: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length into internal buffers using masks.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            rest_length: Fixed tendon rest length. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if fixed_tendon_mask is not None:
            fixed_tendon_ids = wp.nonzero(fixed_tendon_mask)
        else:
            fixed_tendon_ids = self._ALL_FIXED_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_fixed_tendon_rest_length_index(
            rest_length, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_fixed_tendon_offset_index(
        self,
        offset: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set fixed tendon offset into internal buffers using indices.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)) or
                (num_instances, num_fixed_tendons) if full_data.
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        fixed_tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[
                offset,
                env_ids,
                fixed_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._fixed_tendon_offset,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the offset to the simulation.

    def set_fixed_tendon_offset_mask(
        self,
        offset: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon offset into internal buffers using masks.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            offset: Fixed tendon offset. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if fixed_tendon_mask is not None:
            fixed_tendon_ids = wp.nonzero(fixed_tendon_mask)
        else:
            fixed_tendon_ids = self._ALL_FIXED_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_fixed_tendon_offset_index(offset, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids, full_data=True)

    def write_fixed_tendon_properties_to_sim_index(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation using indices.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            fixed_tendon_ids: The fixed tendon indices to write the properties for. Defaults to None
                (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        # Write fixed tendon properties to the simulation.
        self.root_view.set_fixed_tendon_properties(
            self.data.fixed_tendon_stiffness,
            self.data.fixed_tendon_damping,
            self.data.fixed_tendon_limit_stiffness,
            self.data.fixed_tendon_pos_limits,
            self.data.fixed_tendon_rest_length,
            self.data.fixed_tendon_offset,
            indices=env_ids,
        )

    def write_fixed_tendon_properties_to_sim_mask(
        self,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation using masks.

        .. tip::
            For maximum performance we recommend using the mask method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_fixed_tendon_properties_to_sim_index(env_ids=env_ids)

    def set_spatial_tendon_stiffness_index(
        self,
        stiffness: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers using indices.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)) or
                (num_instances, num_spatial_tendons) if full_data.
            spatial_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        spatial_tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], spatial_tendon_ids.shape[0]),
            inputs=[
                stiffness,
                env_ids,
                spatial_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._spatial_tendon_stiffness,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the stiffness to the simulation.

    def set_spatial_tendon_stiffness_mask(
        self,
        stiffness: torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers using masks.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if spatial_tendon_mask is not None:
            spatial_tendon_ids = wp.nonzero(spatial_tendon_mask)
        else:
            spatial_tendon_ids = self._ALL_SPATIAL_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_spatial_tendon_stiffness_index(
            stiffness, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_spatial_tendon_damping_index(
        self,
        damping: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set spatial tendon damping into internal buffers using indices.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the
        :meth:`write_spatial_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)) or
                (num_instances, num_spatial_tendons) if full_data.
            spatial_tendon_ids: The tendon indices to set the damping for. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        spatial_tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], spatial_tendon_ids.shape[0]),
            inputs=[
                damping,
                env_ids,
                spatial_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._spatial_tendon_damping,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the damping to the simulation.

    def set_spatial_tendon_damping_mask(
        self,
        damping: torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers using masks.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the
        :meth:`write_spatial_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            damping: Spatial tendon damping. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if spatial_tendon_mask is not None:
            spatial_tendon_ids = wp.nonzero(spatial_tendon_mask)
        else:
            spatial_tendon_ids = self._ALL_SPATIAL_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_spatial_tendon_damping_index(
            damping, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_spatial_tendon_limit_stiffness_index(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers using indices.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)) or
                (num_instances, num_spatial_tendons) if full_data.
            spatial_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None
                (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        spatial_tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], spatial_tendon_ids.shape[0]),
            inputs=[
                limit_stiffness,
                env_ids,
                spatial_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._spatial_tendon_limit_stiffness,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the limit stiffness to the simulation.

    def set_spatial_tendon_limit_stiffness_mask(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers using masks.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if spatial_tendon_mask is not None:
            spatial_tendon_ids = wp.nonzero(spatial_tendon_mask)
        else:
            spatial_tendon_ids = self._ALL_SPATIAL_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_spatial_tendon_limit_stiffness_index(
            limit_stiffness, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids, full_data=True
        )

    def set_spatial_tendon_offset_index(
        self,
        offset: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set spatial tendon offset into internal buffers using indices.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the
        :meth:`write_spatial_tendon_properties_to_sim_index` method.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)) or
                (num_instances, num_spatial_tendons) if full_data.
            spatial_tendon_ids: The tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        spatial_tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], spatial_tendon_ids.shape[0]),
            inputs=[
                offset,
                env_ids,
                spatial_tendon_ids,
                full_data,
            ],
            outputs=[
                self.data._spatial_tendon_offset,
            ],
            device=self.device,
        )
        # Only updates internal buffers, does not apply the offset to the simulation.

    def set_spatial_tendon_offset_mask(
        self,
        offset: torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset into internal buffers using masks.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the
        :meth:`write_spatial_tendon_properties_to_sim_mask` method.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            offset: Spatial tendon offset. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if spatial_tendon_mask is not None:
            spatial_tendon_ids = wp.nonzero(spatial_tendon_mask)
        else:
            spatial_tendon_ids = self._ALL_SPATIAL_TENDON_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_spatial_tendon_offset_index(
            offset, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids, full_data=True
        )

    def write_spatial_tendon_properties_to_sim_index(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation using indices.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve indices
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = self._ALL_INDICES
        elif isinstance(env_ids, list):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        # Write spatial tendon properties to the simulation.
        self.root_view.set_spatial_tendon_properties(
            self.data.spatial_tendon_stiffness,
            self.data.spatial_tendon_damping,
            self.data.spatial_tendon_limit_stiffness,
            self.data.spatial_tendon_offset,
            indices=env_ids,
        )

    def write_spatial_tendon_properties_to_sim_mask(
        self,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation using masks.

        .. tip::
            For maximum performance we recommend using the mask method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            spatial_tendon_mask: Spatial tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        # Write spatial tendon properties to the simulation.
        self.write_spatial_tendon_properties_to_sim_index(env_ids=env_ids)

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
            first_env_matching_prim = find_first_matching_prim(self.cfg.prim_path)
            if first_env_matching_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
            first_env_matching_prim_path = first_env_matching_prim.GetPath().pathString

            # Find all articulation root prims in the first environment.
            first_env_root_prims = get_all_matching_child_prims(
                first_env_matching_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
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

            # Now we convert the found articulation root from the first
            # environment back into a regex that matches all environments.
            first_env_root_prim_path = first_env_root_prims[0].GetPath().pathString
            root_prim_path_relative_to_prim_path = first_env_root_prim_path[len(first_env_matching_prim_path) :]
            root_prim_path_expr = self.cfg.prim_path + root_prim_path_relative_to_prim_path

        # -- articulation
        self._root_view = self._physics_sim_view.create_articulation_view(root_prim_path_expr.replace(".*", "*"))

        # check if the articulation was created
        if self.root_view._backend is None:
            raise RuntimeError(f"Failed to create articulation at: {root_prim_path_expr}. Please check PhysX logs.")

        # log information about the articulation
        logger.info(f"Articulation initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        logger.info(f"Is fixed root: {self.is_fixed_base}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        logger.info(f"Body names: {self.body_names}")
        logger.info(f"Number of joints: {self.num_joints}")
        logger.info(f"Joint names: {self.joint_names}")
        logger.info(f"Number of fixed tendons: {self.num_fixed_tendons}")

        # container for data access
        self._data = ArticulationData(self.root_view, self.device)

        # create buffers
        self._create_buffers()
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
        # Let the articulation data know that it is fully instantiated and ready to use.
        self.data.is_primed = True

    def _create_buffers(self):
        self._ALL_INDICES = wp.array(np.arange(self.num_instances, dtype=np.int32), device=self.device)
        self._ALL_JOINT_INDICES = wp.array(np.arange(self.num_joints, dtype=np.int32), device=self.device)
        self._ALL_BODY_INDICES = wp.array(np.arange(self.num_bodies, dtype=np.int32), device=self.device)
        self._ALL_FIXED_TENDON_INDICES = wp.array(np.arange(self.num_fixed_tendons, dtype=np.int32), device=self.device)
        self._ALL_SPATIAL_TENDON_INDICES = wp.array(
            np.arange(self.num_spatial_tendons, dtype=np.int32), device=self.device
        )

        # external wrench composer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # asset named data
        self.data.joint_names = self.joint_names
        self.data.body_names = self.body_names
        # tendon names are set in _process_tendons function

        # -- joint commands (sent to the simulation after actuator processing)
        self._joint_pos_target_sim = wp.zeros_like(self.data.joint_pos_target, device=self.device)
        self._joint_vel_target_sim = wp.zeros_like(self.data.joint_pos_target, device=self.device)
        self._joint_effort_target_sim = wp.zeros_like(self.data.joint_pos_target, device=self.device)

        # soft joint position limits (recommended not to be too close to limits).
        wp.launch(
            articulation_kernels.update_soft_joint_pos_limits,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                self.data.joint_pos_limits,
                self.cfg.soft_joint_pos_limit_factor,
            ],
            outputs=[
                self.data.soft_joint_pos_limits,
            ],
            device=self.device,
        )

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # Note we cast to tuple to avoid torch/numpy type mismatch.
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        default_root_vel = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        default_root_pose = np.tile(np.array(default_root_pose, dtype=np.float32), (self.num_instances, 1))
        default_root_vel = np.tile(np.array(default_root_vel, dtype=np.float32), (self.num_instances, 1))
        self.data.default_root_pose = wp.array(default_root_pose, dtype=wp.transformf, device=self.device)
        self.data.default_root_vel = wp.array(default_root_vel, dtype=wp.spatial_vectorf, device=self.device)

        # -- joint state
        pos_idx_list, _, pos_val_list = resolve_matching_names_values(self.cfg.init_state.joint_pos, self.joint_names)
        vel_idx_list, _, vel_val_list = resolve_matching_names_values(self.cfg.init_state.joint_vel, self.joint_names)
        wp.launch(
            articulation_kernels.update_default_joint_values,
            dim=(self.num_instances, len(pos_idx_list)),
            inputs=[
                wp.array(pos_val_list, dtype=wp.float32, device=self.device),
                wp.array(pos_idx_list, dtype=wp.int32, device=self.device),
            ],
            outputs=[
                self.data.default_joint_pos,
            ],
            device=self.device,
        )
        wp.launch(
            articulation_kernels.update_default_joint_values,
            dim=(self.num_instances, len(vel_idx_list)),
            inputs=[
                wp.array(vel_val_list, dtype=wp.float32, device=self.device),
                wp.array(vel_idx_list, dtype=wp.int32, device=self.device),
            ],
            outputs=[
                self.data.default_joint_vel,
            ],
            device=self.device,
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        self._root_view = None

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
            actuator_cfg: ActuatorBaseCfg
            # create actuator group
            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)
            # check if any joints are found
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            # resolve joint indices
            # we pass a slice if all joints are selected to avoid indexing overhead
            if len(joint_names) == self.num_joints:
                joint_ids = slice(None)
            else:
                joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.int32)
            # create actuator collection
            # note: for efficiency avoid indexing when over all indices
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=wp.to_torch(self._data.joint_stiffness)[:, joint_ids],
                damping=wp.to_torch(self._data.joint_damping)[:, joint_ids],
                armature=wp.to_torch(self._data.joint_armature)[:, joint_ids],
                friction=wp.to_torch(self._data.joint_friction_coeff)[:, joint_ids],
                dynamic_friction=wp.to_torch(self._data.joint_dynamic_friction_coeff)[:, joint_ids],
                viscous_friction=wp.to_torch(self._data.joint_viscous_friction_coeff)[:, joint_ids],
                effort_limit=wp.to_torch(self._data.joint_effort_limits)[:, joint_ids].clone(),
                velocity_limit=wp.to_torch(self._data.joint_vel_limits)[:, joint_ids],
            )
            # log information on actuator groups
            model_type = "implicit" if actuator.is_implicit_model else "explicit"
            logger.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (type: {model_type}) and joint names: {joint_names} [{joint_ids}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # Store the configured values from the actuator model
            # note: this is the value configured in the actuator model (for implicit and explicit actuators)
            joint_ids = actuator.joint_indices
            if joint_ids == slice(None):
                joint_ids = self._ALL_JOINT_INDICES
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(self.num_instances, joint_ids.shape[0]),
                inputs=[
                    actuator.stiffness,
                    self._ALL_INDICES,
                    joint_ids,
                    False,
                ],
                outputs=[
                    self.data._joint_stiffness,
                ],
                device=self.device,
            )
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(self.num_instances, joint_ids.shape[0]),
                inputs=[
                    actuator.damping,
                    self._ALL_INDICES,
                    joint_ids,
                    False,
                ],
                outputs=[
                    self.data._joint_damping,
                ],
                device=self.device,
            )
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(self.num_instances, joint_ids.shape[0]),
                inputs=[
                    actuator.armature,
                    self._ALL_INDICES,
                    joint_ids,
                    False,
                ],
                outputs=[
                    self.data._joint_armature,
                ],
                device=self.device,
            )
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(self.num_instances, joint_ids.shape[0]),
                inputs=[
                    actuator.friction,
                    self._ALL_INDICES,
                    joint_ids,
                    False,
                ],
                outputs=[
                    self.data._joint_friction_coeff,
                ],
                device=self.device,
            )
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(self.num_instances, joint_ids.shape[0]),
                inputs=[
                    actuator.dynamic_friction,
                    self._ALL_INDICES,
                    joint_ids,
                    False,
                ],
                outputs=[
                    self.data._joint_dynamic_friction_coeff,
                ],
                device=self.device,
            )
            wp.launch(
                shared_kernels.write_2d_data_to_buffer_with_indices,
                dim=(self.num_instances, joint_ids.shape[0]),
                inputs=[
                    actuator.viscous_friction,
                    self._ALL_INDICES,
                    joint_ids,
                    False,
                ],
                outputs=[
                    self.data._joint_viscous_friction_coeff,
                ],
                device=self.device,
            )
            # set the passed gains and limits into the simulation
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self.write_joint_stiffness_to_sim_index(actuator.stiffness, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim_index(actuator.damping, joint_ids=actuator.joint_indices)
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.write_joint_stiffness_to_sim_index(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim_index(0.0, joint_ids=actuator.joint_indices)

            # Set common properties into the simulation
            self.write_joint_effort_limit_to_sim_index(actuator.effort_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_velocity_limit_to_sim_index(actuator.velocity_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_armature_to_sim_index(actuator.armature, joint_ids=actuator.joint_indices)
            self.write_joint_friction_coefficient_to_sim_index(actuator.friction, joint_ids=actuator.joint_indices)
            self.write_joint_dynamic_friction_coefficient_to_sim_index(
                actuator.dynamic_friction, joint_ids=actuator.joint_indices
            )
            self.write_joint_viscous_friction_coefficient_to_sim_index(
                actuator.viscous_friction, joint_ids=actuator.joint_indices
            )

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != (self.num_joints - self.num_fixed_tendons):
            logger.warning(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints - self.num_fixed_tendons}."
            )

        if self.cfg.actuator_value_resolution_debug_print:
            t = PrettyTable(["Group", "Property", "Name", "ID", "USD Value", "ActutatorCfg Value", "Applied"])
            for actuator_group, actuator in self.actuators.items():
                group_count = 0
                for property, resolution_details in actuator.joint_property_resolution_table.items():
                    for prop_idx, resolution_detail in enumerate(resolution_details):
                        actuator_group_str = actuator_group if group_count == 0 else ""
                        property_str = property if prop_idx == 0 else ""
                        fmt = [f"{v:.2e}" if isinstance(v, float) else str(v) for v in resolution_detail]
                        t.add_row([actuator_group_str, property_str, *fmt])
                        group_count += 1
            logger.warning(f"\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n{t}")

    def _process_tendons(self):
        """Process fixed and spatial tendons."""
        # create a list to store the fixed tendon names
        self._fixed_tendon_names = list()
        self._spatial_tendon_names = list()
        # parse fixed tendons properties if they exist
        if self.num_fixed_tendons > 0 or self.num_spatial_tendons > 0:
            joint_paths = self.root_view.dof_paths[0]

            # iterate over all joints to find tendons attached to them
            for j in range(self.num_joints):
                usd_joint_path = joint_paths[j]
                # check whether joint has tendons - tendon name follows the joint name it is attached to
                joint = UsdPhysics.Joint.Get(self.stage, usd_joint_path)
                joint_applied_str = str(joint.GetPrim().GetAppliedSchemas())
                if "PhysxTendonAxisRootAPI" in joint_applied_str:
                    self._fixed_tendon_names.append(usd_joint_path.split("/")[-1])
                elif (
                    "PhysxTendonAttachmentRootAPI" in joint_applied_str
                    or "PhysxTendonAttachmentLeafAPI" in joint_applied_str
                ):
                    self._spatial_tendon_names.append(usd_joint_path.split("/")[-1])

            # store the fixed tendon names
            self._data.fixed_tendon_names = self._fixed_tendon_names
            self._data.spatial_tendon_names = self._spatial_tendon_names

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
                joint_positions=wp.to_torch(self._data.joint_pos_target)[:, actuator.joint_indices],
                joint_velocities=wp.to_torch(self._data.joint_vel_target)[:, actuator.joint_indices],
                joint_efforts=wp.to_torch(self._data.joint_effort_target)[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            # compute joint command from the actuator model
            control_action = actuator.compute(
                control_action,
                joint_pos=wp.to_torch(self._data.joint_pos)[:, actuator.joint_indices],
                joint_vel=wp.to_torch(self._data.joint_vel)[:, actuator.joint_indices],
            )
            # update targets (these are set into the simulation)
            joint_indices = actuator.joint_indices
            if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
                joint_indices = self._ALL_JOINT_INDICES
            if hasattr(actuator, "gear_ratio"):
                gear_ratio = actuator.gear_ratio
            else:
                gear_ratio = None
            wp.launch(
                articulation_kernels.update_targets,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=[
                    control_action.joint_positions,
                    control_action.joint_velocities,
                    control_action.joint_efforts,
                    joint_indices,
                ],
                outputs=[
                    self._joint_pos_target_sim,
                    self._joint_vel_target_sim,
                    self._joint_effort_target_sim,
                ],
                device=self.device,
            )
            # update state of the actuator model
            wp.launch(
                articulation_kernels.update_actuator_state_model,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=[
                    actuator.computed_effort,
                    actuator.applied_effort,
                    gear_ratio,
                    actuator.velocity_limit,
                    joint_indices,
                ],
                outputs=[
                    self._data.computed_torque,
                    self._data.applied_torque,
                    self._data.gear_ratio,
                    self._data.soft_joint_vel_limits,
                ],
                device=self.device,
            )

    """
    Internal helpers -- Debugging.
    """

    def _validate_cfg(self):
        """Validate the configuration after processing.

        .. note::
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        # check that the default values are within the limits
        joint_pos_limits = wp.to_torch(wp.clone(self.root_view.get_dof_limits(), device=self.device))[0]
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
                joint_pos = wp.to_torch(self._data.default_joint_pos)[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

        # check that the default joint velocities are within the limits
        joint_max_vel = wp.to_torch(wp.clone(self.root_view.get_dof_max_velocities(), device=self.device))[0]
        out_of_range = torch.abs(wp.to_torch(self._data.default_joint_vel)[0]) > joint_max_vel
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default velocities out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = [-joint_max_vel[idx], joint_max_vel[idx]]
                joint_vel = wp.to_torch(self._data.default_joint_vel)[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

    def _log_articulation_info(self):
        """Log information about the articulation.

        .. note:: We purposefully read the values from the simulator to ensure that the values are configured as
            expected.
        """

        # define custom formatters for large numbers and limit ranges
        def format_large_number(_, v: float) -> str:
            """Format large numbers using scientific notation."""
            if abs(v) >= 1e3:
                return f"{v:.1e}"
            else:
                return f"{v:.3f}"

        def format_limits(_, v: tuple[float, float]) -> str:
            """Format limit ranges using scientific notation."""
            if abs(v[0]) >= 1e3 or abs(v[1]) >= 1e3:
                return f"[{v[0]:.1e}, {v[1]:.1e}]"
            else:
                return f"[{v[0]:.3f}, {v[1]:.3f}]"

        # read out all joint parameters from simulation
        # -- gains
        # Use data properties which have already been cloned and stored during initialization
        # This avoids issues with indexedarray or empty arrays from root_view
        stiffnesses = wp.to_torch(self.data.joint_stiffness)[0].cpu().tolist()
        dampings = wp.to_torch(self.data.joint_damping)[0].cpu().tolist()
        # -- properties
        armatures = wp.to_torch(self.data.joint_armature)[0].cpu().tolist()
        # For friction, use the individual components from data
        friction_coeff = wp.to_torch(self.data.joint_friction_coeff)[0].cpu()
        dynamic_friction_coeff = wp.to_torch(self.data.joint_dynamic_friction_coeff)[0].cpu()
        viscous_friction_coeff = wp.to_torch(self.data.joint_viscous_friction_coeff)[0].cpu()
        static_frictions = friction_coeff.tolist()
        dynamic_frictions = dynamic_friction_coeff.tolist()
        viscous_frictions = viscous_friction_coeff.tolist()
        # -- limits
        # joint_pos_limits is vec2f array, convert to torch and extract [lower, upper] pairs
        position_limits_torch = wp.to_torch(self.data.joint_pos_limits)[0].cpu()  # shape: (num_joints, 2)
        position_limits = [tuple(pos_limit.tolist()) for pos_limit in position_limits_torch]
        velocity_limits = wp.to_torch(self.data.joint_vel_limits)[0].cpu().tolist()
        effort_limits = wp.to_torch(self.data.joint_effort_limits)[0].cpu().tolist()
        # create table for term information
        joint_table = PrettyTable()
        joint_table.title = f"Simulation Joint Information (Prim path: {self.cfg.prim_path})"
        # build field names based on Isaac Sim version
        field_names = ["Index", "Name", "Stiffness", "Damping", "Armature"]
        field_names.extend(["Static Friction", "Dynamic Friction", "Viscous Friction"])
        field_names.extend(["Position Limits", "Velocity Limits", "Effort Limits"])
        joint_table.field_names = field_names

        # apply custom formatters to numeric columns
        joint_table.custom_format["Stiffness"] = format_large_number
        joint_table.custom_format["Damping"] = format_large_number
        joint_table.custom_format["Armature"] = format_large_number
        joint_table.custom_format["Static Friction"] = format_large_number
        joint_table.custom_format["Dynamic Friction"] = format_large_number
        joint_table.custom_format["Viscous Friction"] = format_large_number
        joint_table.custom_format["Position Limits"] = format_limits
        joint_table.custom_format["Velocity Limits"] = format_large_number
        joint_table.custom_format["Effort Limits"] = format_large_number

        # set alignment of table columns
        joint_table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self.joint_names):
            # build row data based on Isaac Sim version
            row_data = [index, name, stiffnesses[index], dampings[index], armatures[index]]
            if get_isaac_sim_version().major < 5:
                row_data.append(static_frictions[index])
            else:
                row_data.extend([static_frictions[index], dynamic_frictions[index], viscous_frictions[index]])
            row_data.extend([position_limits[index], velocity_limits[index], effort_limits[index]])
            # add row to table
            joint_table.add_row(row_data)
        # convert table to string
        logger.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + joint_table.get_string())

        # read out all fixed tendon parameters from simulation
        if self.num_fixed_tendons > 0:
            # -- gains
            # Use data properties which have already been cloned and stored during initialization
            ft_stiffnesses = wp.to_torch(self.data.fixed_tendon_stiffness)[0].cpu().tolist()
            ft_dampings = wp.to_torch(self.data.fixed_tendon_damping)[0].cpu().tolist()
            # -- limits
            ft_limit_stiffnesses = wp.to_torch(self.data.fixed_tendon_limit_stiffness)[0].cpu().tolist()
            # fixed_tendon_pos_limits is vec2f array
            ft_limits_torch = wp.to_torch(self.data.fixed_tendon_pos_limits)[0].cpu()
            ft_limits = [tuple(limit.tolist()) for limit in ft_limits_torch]
            ft_rest_lengths = wp.to_torch(self.data.fixed_tendon_rest_length)[0].cpu().tolist()
            ft_offsets = wp.to_torch(self.data.fixed_tendon_offset)[0].cpu().tolist()
            # create table for term information
            tendon_table = PrettyTable()
            tendon_table.title = f"Simulation Fixed Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Limits",
                "Rest Length",
                "Offset",
            ]
            tendon_table.float_format = ".3"

            # apply custom formatters to tendon table columns
            tendon_table.custom_format["Stiffness"] = format_large_number
            tendon_table.custom_format["Damping"] = format_large_number
            tendon_table.custom_format["Limit Stiffness"] = format_large_number
            tendon_table.custom_format["Limits"] = format_limits
            tendon_table.custom_format["Rest Length"] = format_large_number
            tendon_table.custom_format["Offset"] = format_large_number

            # add info on each term
            for index in range(self.num_fixed_tendons):
                tendon_table.add_row(
                    [
                        index,
                        ft_stiffnesses[index],
                        ft_dampings[index],
                        ft_limit_stiffnesses[index],
                        ft_limits[index],
                        ft_rest_lengths[index],
                        ft_offsets[index],
                    ]
                )
            # convert table to string
            logger.info(
                f"Simulation parameters for fixed tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

        if self.num_spatial_tendons > 0:
            # -- gains
            # Use data properties which have already been cloned and stored during initialization
            st_stiffnesses = wp.to_torch(self.data.spatial_tendon_stiffness)[0].cpu().tolist()
            st_dampings = wp.to_torch(self.data.spatial_tendon_damping)[0].cpu().tolist()
            # -- limits
            st_limit_stiffnesses = wp.to_torch(self.data.spatial_tendon_limit_stiffness)[0].cpu().tolist()
            st_offsets = wp.to_torch(self.data.spatial_tendon_offset)[0].cpu().tolist()
            # create table for term information
            tendon_table = PrettyTable()
            tendon_table.title = f"Simulation Spatial Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Offset",
            ]
            tendon_table.float_format = ".3"
            # add info on each term
            for index in range(self.num_spatial_tendons):
                tendon_table.add_row(
                    [
                        index,
                        st_stiffnesses[index],
                        st_dampings[index],
                        st_limit_stiffnesses[index],
                        st_offsets[index],
                    ]
                )
            # convert table to string
            logger.info(
                f"Simulation parameters for spatial tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        """Resolve environment indices to a warp array.

        .. note::
            We need to convert torch tensors to warp arrays since the TensorAPI views only support warp arrays.

        Args:
            env_ids: Environment indices. If None, then all indices are used.

        Returns:
            A warp array of environment indices.
        """
        if (env_ids is None) or (env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            # Convert int64 to int32 if needed, as warp expects int32
            if env_ids.dtype == torch.int64:
                env_ids = env_ids.to(torch.int32)
            return wp.from_torch(env_ids, dtype=wp.int32)
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        return env_ids

    def _resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array | torch.Tensor:
        """Resolve joint indices to a warp array or tensor.

        .. note::
            We do not need to convert torch tensors to warp arrays since they never get passed to the TensorAPI views.

        Args:
            joint_ids: Joint indices. If None, then all indices are used.

        Returns:
            A warp array of joint indices or a tensor of joint indices.
        """
        if isinstance(joint_ids, list):
            return wp.array(joint_ids, dtype=wp.int32, device=self.device)
        if (joint_ids is None) or (joint_ids == slice(None)):
            return self._ALL_JOINT_INDICES
        return joint_ids

    def _resolve_body_ids(self, body_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array | torch.Tensor:
        """Resolve body indices to a warp array or tensor.

        Args:
            body_ids: Body indices. If None, then all indices are used.

        Returns:
            A warp array of body indices or a tensor of body indices.
        """
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        if (body_ids is None) or (body_ids == slice(None)):
            return self._ALL_BODY_INDICES
        return body_ids

    def _resolve_fixed_tendon_ids(
        self, tendon_ids: Sequence[int] | torch.Tensor | wp.array | None
    ) -> wp.array | torch.Tensor:
        """Resolve tendon indices to a warp array or tensor.

        Args:
            tendon_ids: Tendon indices. If None, then all indices are used.

        Returns:
            A warp array of tendon indices or a tensor of tendon indices.
        """
        if isinstance(tendon_ids, list):
            return wp.array(tendon_ids, dtype=wp.int32, device=self.device)
        if (tendon_ids is None) or (tendon_ids == slice(None)):
            return self._ALL_FIXED_TENDON_INDICES
        return tendon_ids

    def _resolve_spatial_tendon_ids(
        self, spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None
    ) -> wp.array | torch.Tensor:
        """Resolve spatial tendon indices to a warp array or tensor.

        Args:
            spatial_tendon_ids: Spatial tendon indices. If None, then all indices are used.

        Returns:
            A warp array of spatial tendon indices or a tensor of spatial tendon indices.
        """
        if isinstance(spatial_tendon_ids, list):
            return wp.array(spatial_tendon_ids, dtype=wp.int32, device=self.device)
        if (spatial_tendon_ids is None) or (spatial_tendon_ids == slice(None)):
            return self._ALL_SPATIAL_TENDON_INDICES
        return spatial_tendon_ids

    """
    Deprecated methods.
    """

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Deprecated property. Please use :attr:`root_view` instead."""
        warnings.warn(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_view

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | wp.array | float,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ):
        """Deprecated, same as :meth:`write_joint_friction_coefficient_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_friction_coefficient_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_friction_coefficient_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff,
            joint_dynamic_friction_coeff=joint_dynamic_friction_coeff,
            joint_viscous_friction_coeff=joint_viscous_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
            full_data=full_data,
        )

    def write_joint_viscous_friction_coefficient_to_sim(
        self,
        joint_viscous_friction_coeff: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_viscous_friction_coefficient_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_viscous_friction_coefficient_to_sim' will be deprecated in a future release. "
            "Please use 'write_joint_viscous_friction_coefficient_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data
        )

    def write_joint_dynamic_friction_coefficient_to_sim(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_dynamic_friction_coefficient_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_dynamic_friction_coefficient_to_sim' will be deprecated in a future release. "
            "Please use 'write_joint_dynamic_friction_coefficient_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_dynamic_friction_coefficient_to_sim_index(
            joint_dynamic_friction_coeff, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data
        )

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_com_pose_to_sim_index(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim_index(root_state[:, 7:], env_ids=env_ids)
