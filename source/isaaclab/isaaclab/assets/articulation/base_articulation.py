# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from ..asset_base import AssetBase

if TYPE_CHECKING:
    from isaaclab.utils.wrench_composer import WrenchComposer

    from .articulation_cfg import ArticulationCfg
    from .articulation_data import ArticulationData


class BaseArticulation(AssetBase):
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

    __backend_name__: str = "base"
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
    @abstractmethod
    def data(self) -> ArticulationData:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_shapes_per_body(self) -> list[int]:
        """Number of shapes per body in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def spatial_tendon_names(self) -> list[str]:
        """Ordered names of spatial tendons in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self):
        """Root view for the asset.

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer for the articulation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer for the articulation."""
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | torch.Tensor | None = None):
        """Reset the articulation.

        Note: If both env_ids and env_mask are provided, then env_mask will be used. For performance reasons, it is
        recommended to use the env_mask instead of env_ids.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float):
        raise NotImplementedError()

    """
    Operations - Finders.
    """

    @abstractmethod
    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor | wp.array, list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[torch.Tensor | wp.array, list[int], list[str]]:
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
        raise NotImplementedError()

    @abstractmethod
    def find_fixed_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[torch.Tensor | wp.array, list[int], list[str]]:
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
        raise NotImplementedError()

    @abstractmethod
    def find_spatial_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[torch.Tensor | wp.array, list[int], list[str]]:
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
        raise NotImplementedError()

    """
    Operations - State Writers.
    """

    @abstractmethod
    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's frame rather than the roots center of mass.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_state_to_sim(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then position should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then position should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_position_to_sim(
        self,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint positions to the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then position should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then position should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint velocities to the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then velocity should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then velocity should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    """
    Operations - Simulation Parameters Writers.
    """

    @abstractmethod
    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint stiffness into the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then stiffness should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then stiffness should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        raise NotImplementedError()

    @abstractmethod
    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint damping into the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then damping should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then damping should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the damping for. Defaults to None (all joints).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_position_limit_to_sim(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint position limits into the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then limits should be of shape
        (len(env_ids), len(joint_ids), 2). If env_mask is provided, then limits should be of shape
        (num_instances, num_joints, 2).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2) or (num_instances, num_joints, 2).
            joint_ids: The joint indices to set the limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then limits should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then limits should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the max velocity for. Defaults to None (all joints).
            env_ids: The environment indices to set the max velocity for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then limits should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then limits should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then armature should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then armature should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | wp.array | float,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        r"""Write joint static friction coefficients into the simulation.

        The joint static friction is a unitless quantity. It relates the magnitude of the spatial force transmitted
        from the parent body to the child body to the maximal static friction force that may be applied by the solver
        to resist the joint motion.

        Mathematically, this means that: :math:`F_{resist} \leq \mu F_{spatial}`, where :math:`F_{resist}`
        is the resisting force applied by the solver and :math:`F_{spatial}` is the spatial force
        transmitted from the parent body to the child body. The simulated static friction effect is therefore
        similar to static and Coulomb static friction.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and joint_ids are provided, then joint_friction_coeff should be of shape
        (len(env_ids), len(joint_ids)). If env_mask is provided, then joint_friction_coeff should be of shape
        (num_instances, num_joints).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            joint_friction_coeff: Joint static friction coefficient. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_dynamic_friction_coeff: Joint dynamic friction coefficient. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_viscous_friction_coeff: Joint viscous friction coefficient. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_dynamic_friction_coefficient_to_sim(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def write_joint_viscous_friction_coefficient_to_sim(
        self,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        raise NotImplementedError()

    """
    Operations - Setters.
    """

    @abstractmethod
    def set_masses(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set masses of all bodies in the simulation world frame.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_coms(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set center of mass positions of all bodies in the simulation world frame.

        Args:
            coms: Center of mass positions of all bodies. Shape is (num_instances, num_bodies, 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_inertias(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set inertias of all bodies in the simulation world frame.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 3, 3).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        is_global: bool = False,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. caution::
            If the function is called consecutively with and with different values for ``is_global``, then the
            all the external wrenches will be applied in the frame specified by the last call.

            .. code-block:: python

                # example of setting external wrench in the global frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[0], is_global=True)
                # example of setting external wrench in the link frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[1], is_global=False)
                # Both environments will have the external wrenches applied in the link frame

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3).
            positions: Positions to apply external wrench. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3). Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the articulations' bodies.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_joint_position_target(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_joint_velocity_target(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_joint_effort_target(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    """
    Operations - Tendons.
    """

    @abstractmethod
    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_damping(
        self,
        damping: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon limit stiffness efforts into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_position_limit(
        self,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

         Args:
             limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
             fixed_tendon_ids: The tendon indices to set the limit for. Defaults to None (all fixed tendons).
             env_ids: The environment indices to set the limit for. Defaults to None (all environments).
             fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
             env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_rest_length(
        self,
        rest_length: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon rest length efforts into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_offset(
        self,
        offset: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write fixed tendon properties into the simulation.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the limits for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_stiffness(
        self,
        stiffness: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_damping(
        self,
        damping: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The tendon indices to set the damping for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_offset(
        self,
        offset: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set spatial tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids and spatial_tendon_ids are provided, then offset should be of shape
        (len(env_ids), len(spatial_tendon_ids)). If env_mask is provided, then offset should be of shape
        (num_instances, num_spatial_tendons).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write spatial tendon properties into the simulation.

        ..caution:: Do not mix ids and masks. If a mask is provided, then ids will be ignored.

        Args:
            spatial_tendon_ids: The spatial tendon indices to set the properties for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the properties for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    """
    Internal helper.
    """

    @abstractmethod
    def _initialize_impl(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_buffers(self):
        raise NotImplementedError()

    @abstractmethod
    def _process_cfg(self):
        """Post processing of configuration parameters."""
        raise NotImplementedError()

    """
    Internal simulation callbacks.
    """

    @abstractmethod
    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        raise NotImplementedError()

    """
    Internal helpers -- Actuators.
    """

    @abstractmethod
    def _process_actuators_cfg(self):
        """Process and apply articulation joint properties."""
        raise NotImplementedError()

    @abstractmethod
    def _process_tendons(self):
        """Process fixed and spatial tendons."""
        raise NotImplementedError()

    @abstractmethod
    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        raise NotImplementedError()

    """
    Internal helpers -- Debugging.
    """

    @abstractmethod
    def _validate_cfg(self):
        """Validate the configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        raise NotImplementedError()

    @abstractmethod
    def _log_articulation_info(self):
        """Log information about the articulation.

        Note: We purposefully read the values from the simulator to ensure that the values are configured as expected.
        """
        raise NotImplementedError()

    """
    Deprecated methods.
    """

    @abstractmethod
    def write_joint_friction_to_sim(
        self,
        joint_friction: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint friction coefficients into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_friction_coefficient_to_sim` instead.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_limits_to_sim(
        self,
        limits: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        warn_limit_violation: bool = True,
        joint_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Write joint limits into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_position_limit_to_sim` instead.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_limit(
        self,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ):
        """Set fixed tendon position limits into internal buffers.

        .. deprecated:: 2.1.0
            Please use :meth:`set_fixed_tendon_position_limit` instead.
        """
        raise NotImplementedError()
