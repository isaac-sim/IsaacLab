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
from typing import TYPE_CHECKING, Literal

import omni.log
import warp as wp
from isaacsim.core.simulation_manager import SimulationManager
from newton import JointMode, JointType, Model
from newton.solvers import SolverNotifyFlags
from newton.selection import ArticulationView as NewtonArticulationView
from newton.solvers import SolverMuJoCo
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg, ImplicitActuator
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.types import ArticulationActions
from .kernels import *
from .utils import warn_overhead_cost

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
        return self._root_newton_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self._root_newton_view.is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self._root_newton_view.joint_dof_count

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
        return self._root_newton_view.link_count

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self._root_newton_view.joint_dof_names

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
        return self._root_newton_view.body_names

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

    def reset(self, env_ids: wp.array | Sequence[int] | None = None):
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # reset actuators
        for actuator in self.actuators.values():
            actuator.reset(env_ids)
        # reset external wrench
        wp.launch(
            update_wrench_array_with_value,
            dim=(self.num_instances,),
            inputs=[
                wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                self._external_wrench,
                env_ids,
                self._ALL_BODY_INDICES,
            ]
        )

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
        # write actions into simulation
        self._root_newton_view.set_attribute("joint_f", NewtonManager.get_control(), self._joint_effort_target_sim)
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:
            # Sets the position or velocity target for the implicit actuators depending on the actuator type.
            self._root_newton_view.set_attribute("joint_target", NewtonManager.get_control(), self._joint_target_sim)

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
        return string_utils.resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    """
    Operations - State Writers.
    """

    @warn_overhead_cost
    def write_root_state_to_sim(self, root_state: wp.array, env_ids: wp.array | Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """

        # set into simulation
        target_root_pose = wp.zeros((self.num_instances), dtype=wp.transformf, device=self.device)
        target_root_velocity = wp.zeros((self.num_instances), dtype=wp.spatial_vectorf, device=self.device)

        wp.launch(
            split_root_state,
            dim=(self.num_instances,),
            inputs=[
                root_state,
                target_root_pose,
                target_root_velocity,
                env_ids,
            ]
        )
        self.write_root_link_pose_to_sim(target_root_pose, env_ids=env_ids)
        self.write_root_com_velocity_to_sim(target_root_velocity, env_ids=env_ids)

    @warn_overhead_cost
    def write_root_com_state_to_sim(self, root_state: wp.array, env_ids: wp.array | Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """

        target_root_pose = wp.zeros((self.num_instances), dtype=wp.transformf, device=self.device)
        target_root_velocity = wp.zeros((self.num_instances), dtype=wp.spatial_vectorf, device=self.device)

        wp.launch(
            split_root_state,
            dim=(self.num_instances,),
            inputs=[
                root_state,
                target_root_pose,
                target_root_velocity,
                env_ids,
            ]
        )
        self.write_root_com_pose_to_sim(target_root_pose, env_ids=env_ids)
        self.write_root_com_velocity_to_sim(target_root_velocity, env_ids=env_ids)

    @warn_overhead_cost
    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        target_root_pose = wp.zeros((self.num_instances), dtype=wp.transformf, device=self.device)
        target_root_velocity = wp.zeros((self.num_instances), dtype=wp.spatial_vectorf, device=self.device)

        wp.launch(
            split_root_state,
            dim=(self.num_instances,),
            inputs=[
                root_state,
                target_root_pose,
                target_root_velocity,
                env_ids,
            ]
        )
        self.write_root_link_pose_to_sim(target_root_pose, env_ids=env_ids)
        self.write_root_link_velocity_to_sim(target_root_velocity, env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: wp.array, env_ids: wp.array | Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(self, pose: wp.array, env_ids: wp.array | Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.


        The root pose ``wp.transformf`` comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)

        # set into internal buffers
        wp.launch(
            update_transforms_array,
            dim=self._root_newton_view.count,
            inputs=[
                self._data.root_link_pose_w,
                env_ids,
                pose,
            ]
        )
        # Need to invalidate the buffer to trigger the update with the new state.
        self._data._root_com_pose_w.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0


    def write_root_com_pose_to_sim(self, root_pose: wp.array, env_ids: wp.array | Sequence[int] | None = None) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)

        # set into internal buffers
        wp.launch(
            update_transforms_array,
            dim=self._root_newton_view.count,
            inputs=[
                root_pose,
                env_ids,
                self._data.root_com_pose_w,
            ]
        )
        # set link frame poses
        wp.launch(
            transform_CoM_pose_to_link_frame,
            dim=self._root_newton_view.count,
            inputs=[
                self._data.root_com_pose_w,
                self._data.body_com_pos_b,
                self._data.body_link_pose_w,
            ]
        )
        self._data._body_com_pose_w.timestamp = -1.0

    def write_root_velocity_to_sim(self, root_velocity: wp.array, env_ids: wp.array | Sequence[int] | None = None) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: wp.array, env_ids: wp.array | Sequence[int] | None = None) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)

        # set into internal buffers
        wp.launch(
            update_velocity_array,
            dim=self._root_newton_view.count,
            inputs=[
                root_velocity,
                env_ids,
                self._data.root_com_vel_w,
            ]
        )
        self._data._root_link_vel_w.timestamp = -1.0

    def write_root_link_velocity_to_sim(self, root_velocity: wp.array, env_ids: wp.array | Sequence[int] | None = None) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        wp.launch(
            project_link_velocity_to_com_frame_indexed,
            dim=self._root_newton_view.count,
            inputs=[
                root_velocity,
                self._data.body_link_pose_w,
                self._data.body_com_pos_b,
                self._data.root_com_vel_w,
                env_ids,
            ]
        )

    def write_joint_state_to_sim(
        self,
        position: wp.array,
        velocity: wp.array,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # set into simulation
        self.write_joint_position_to_sim(position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim(velocity, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_position_to_sim(
        self,
        position: wp.array,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint positions to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                position,
                env_ids,
                joint_ids,
                self._data._joint_pos.data,
            ]
        )
        # invalidate buffers to trigger the update with the new root pose.
        self._data._body_com_pose_w.timestamp = -1.0

    def write_joint_velocity_to_sim(
        self,
        velocity: wp.array,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint velocities to the simulation.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # update joint velocity
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                velocity,
                env_ids,
                joint_ids,
                self._data._joint_vel.data,
            ]
        )
        # update previous joint velocity
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                velocity,
                env_ids,
                joint_ids,
                self._data._previous_joint_vel
            ]
        )
        # Set joint acceleration to 0.0
        wp.launch(
            update_joint_array_with_value,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                0.0,
                env_ids,
                joint_ids,
                self._data._joint_acc.data,
            ]
        )
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self._data._body_link_vel_w.timestamp = -1.0
        self._data._joint_acc.timestamp = -1.0

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_control_mode_to_sim(
        self,
        control_mode: Literal["position", "velocity", "none"] | None,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint control mode into the simulation.

        Args:
            control_mode: Joint control mode. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the control mode for. Defaults to None (all joints).
            env_ids: The environment indices to set the control mode for. Defaults to None (all environments).

        Raises:
            ValueError: If the control mode is invalid.
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if control_mode == "position":
            value = JointMode.TARGET_POSITION
        elif control_mode == "velocity":
            value = JointMode.TARGET_VELOCITY
        elif (control_mode is None) or (control_mode == "none"):
            # Set the control mode to None when using explicit actuators
            value = JointMode.NONE
        else:
            raise ValueError(f"Invalid control mode: {control_mode}")
        wp.launch(
            update_joint_array_with_value_int,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                value,
                env_ids,
                joint_ids,
                self._root_newton_view.get_attribute("joint_dof_mode", NewtonManager.get_model())
            ]
        )
                

    def write_joint_stiffness_to_sim(
        self,
        stiffness: wp.array | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if isinstance(stiffness, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    stiffness,
                    env_ids,
                    joint_ids,
                    self._data.joint_stiffness,
                ]
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    stiffness,
                    env_ids,
                    joint_ids,
                    self._data.joint_stiffness,
                ]
            )
        # tell the physics engine to use the new stiffness
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)


    def write_joint_damping_to_sim(
        self,
        damping: wp.array | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the damping for. Defaults to None (all joints).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if isinstance(damping, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    damping,
                    env_ids,
                    joint_ids,
                    self._data.joint_damping,
                ]
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    damping,
                    env_ids,
                    joint_ids,
                    self._data.joint_damping,
                ]
            )
        # tell the physics engine to use the new damping
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_position_limit_to_sim(
        self,
        limits: wp.array | tuple[float, float] | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint position limits into the simulation.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2).
            joint_ids: The joint indices to set the limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        if isinstance(limits, float):
            # update default joint pos to stay within the new limits
            wp.launch(
                update_joint_pos_with_limits_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    self._data.default_joint_pos,
                    limits,
                    env_ids,
                    joint_ids,
                ]
            )
            # set into simulation
            wp.launch(
                update_joint_limits_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    self.cfg.soft_joint_pos_limit_factor,
                    env_ids,
                    joint_ids,
                    self._data.joint_pos_limits_lower,
                    self._data.joint_pos_limits_upper,
                    self._data.soft_joint_pos_limits,
                ]
            )
        elif isinstance(limits, tuple) or isinstance(limits, list):
            # update default joint pos to stay within the new limits
            wp.launch(
                update_joint_pos_with_limits_value_vec2f,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    self._data.default_joint_pos,
                    wp.vec2f(limits[0], limits[1]),
                    env_ids,
                    joint_ids,
                ]
            )
            # set into simulation
            wp.launch(
                update_joint_limits_value_vec2f,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    wp.vec2f(limits[0], limits[1]),
                    self.cfg.soft_joint_pos_limit_factor,
                    env_ids,
                    joint_ids,
                    self._data.joint_pos_limits_lower,
                    self._data.joint_pos_limits_upper,
                    self._data.soft_joint_pos_limits,
                ]
            )
        else:
            # update default joint pos to stay within the new limits
            wp.launch(
                update_joint_pos_with_limits,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    self._data.default_joint_pos,
                    limits,
                    env_ids,
                    joint_ids,
                ]
            )
            # set into simulation
            wp.launch(
                update_joint_limits,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    self.cfg.soft_joint_pos_limit_factor,
                    env_ids,
                    joint_ids,
                    self._data.joint_pos_limits_lower,
                    self._data.joint_pos_limits_upper,
                    self._data.soft_joint_pos_limits,
                ]
            )
        # tell the physics engine to use the new limits
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_velocity_limit_to_sim(
        self,
        limits: wp.array | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. warn:: This function is ignored when using the Mujoco solver.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the max velocity for. Defaults to None (all joints).
            env_ids: The environment indices to set the max velocity for. Defaults to None (all environments).
        """
        # Warn if using Mujoco solver
        if isinstance(NewtonManager._solver, SolverMuJoCo):
            omni.log.warn("write_joint_velocity_limit_to_sim is ignored when using the Mujoco solver.")

        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if isinstance(limits, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                    self._data.joint_vel_limits,
                ]
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                    self._data.joint_vel_limits,
                ]
            )
        # tell the physics engine to use the new limits
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_effort_limit_to_sim(
        self,
        limits: wp.array | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if isinstance(limits, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                    self._data.joint_effort_limits,
                ]
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    env_ids,
                    joint_ids,
                    self._data.joint_effort_limits,
                ]
            )
        # tell the physics engine to use the new limits
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_armature_to_sim(
        self,
        armature: wp.array | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if isinstance(armature, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    armature,
                    env_ids,
                    joint_ids,
                    self._data.joint_armature,
                ]
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    armature,
                    env_ids,
                    joint_ids,
                    self._data.joint_armature,
                ]
            )
        # tell the physics engine to use the new armature
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: wp.array | float,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        r"""Write joint friction coefficients into the simulation.

        The joint friction is a unitless quantity. It relates the magnitude of the spatial force transmitted
        from the parent body to the child body to the maximal friction force that may be applied by the solver
        to resist the joint motion.

        Mathematically, this means that: :math:`F_{resist} \leq \mu F_{spatial}`, where :math:`F_{resist}`
        is the resisting force applied by the solver and :math:`F_{spatial}` is the spatial force
        transmitted from the parent body to the child body. The simulated friction effect is therefore
        similar to static and Coulomb friction.

        Args:
            joint_friction: Joint friction. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set into internal buffers
        if isinstance(joint_friction_coeff, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    joint_friction_coeff,
                    env_ids,
                    joint_ids,
                    self._data.joint_friction_coeff,
                ]
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    joint_friction_coeff,
                    env_ids,
                    joint_ids,
                    self._data.joint_friction_coeff,
                ]
            )
        # tell the physics engine to use the new friction
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: wp.array,
        torques: wp.array,
        body_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
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
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES
        if not isinstance(body_ids, wp.array):
            omni.log.warn("Passing body_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        # Check if there are any external forces or torques
        if (forces is not None) or (torques is not None):
            self.has_external_wrench = True
            if forces is not None:
                wp.launch(
                    update_wrench_array_with_force,
                    dim=(self.num_instances, self.num_bodies),
                    inputs=[
                        forces,
                        self._external_wrench,
                        env_ids,
                        body_ids,
                    ]
                )
            if torques is not None:
                wp.launch(
                    update_wrench_array_with_torque,
                    dim=(self.num_instances, self.num_bodies),
                    inputs=[
                        torques,
                        self._external_wrench,
                        env_ids,
                        body_ids,
                    ]
                )

    def set_joint_position_target(
        self,
        target: wp.array,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set targets
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                env_ids,
                joint_ids,
                self._data.joint_target,
            ]
        )

    def set_joint_velocity_target(
        self,
        target: wp.array,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set targets
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                env_ids,
                joint_ids,
                self._data.joint_target,
            ]
        )

    def set_joint_effort_target(
        self,
        target: wp.array,
        joint_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if not isinstance(env_ids, wp.array):
            omni.log.warn("Passing env_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if joint_ids is None:
            joint_ids = self._ALL_JOINT_INDICES
        if not isinstance(joint_ids, wp.array):
            omni.log.warn("Passing joint_ids as a list or torch tensor will degrade performance. Use wp.array instead.")
            joint_ids = wp.array(joint_ids, dtype=wp.int32, device=self.device)
        # set targets
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                env_ids,
                joint_ids,
                self._data.joint_effort_target,
            ]
        )

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness(
        self,
        stiffness: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon stiffness is not supported in Newton.")

    def set_fixed_tendon_damping(
        self,
        damping: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon damping is not supported in Newton.")

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set fixed tendon limit stiffness efforts into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon limit stiffness is not supported in Newton.")

    def set_fixed_tendon_position_limit(
        self,
        limit: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
             limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)).
             fixed_tendon_ids: The tendon indices to set the limit for. Defaults to None (all fixed tendons).
             env_ids: The environment indices to set the limit for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon position limit is not supported in Newton.")

    def set_fixed_tendon_rest_length(
        self,
        rest_length: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set fixed tendon rest length efforts into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon rest length is not supported in Newton.")

    def set_fixed_tendon_offset(
        self,
        offset: wp.array,
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
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
        fixed_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write fixed tendon properties into the simulation.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the limits for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
        """
        raise NotImplementedError("Fixed tendon properties are not supported in Newton.")

    def set_spatial_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        spatial_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        raise NotImplementedError("Spatial tendon stiffness is not supported in Newton.")

    def set_spatial_tendon_damping(
        self,
        damping: wp.array,
        spatial_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the damping for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        raise NotImplementedError("Spatial tendon damping is not supported in Newton.")

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: wp.array,
        spatial_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """
        raise NotImplementedError("Spatial tendon limit stiffness is not supported in Newton.")

    def set_spatial_tendon_offset(
        self,
        offset: wp.array,
        spatial_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Set spatial tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """
        raise NotImplementedError("Spatial tendon offset is not supported in Newton.")

    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_ids: wp.array | Sequence[int] | None = None,
        env_ids: wp.array | Sequence[int] | None = None,
    ):
        """Write spatial tendon properties into the simulation.

        Args:
            spatial_tendon_ids: The spatial tendon indices to set the properties for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the properties for. Defaults to None (all environments).
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

        self._root_newton_view = NewtonArticulationView(
            NewtonManager.get_model(), prim_path, verbose=True, exclude_joint_types=[JointType.FREE, JointType.FIXED]
        )

        # container for data access
        self._data = ArticulationData(self._root_newton_view, self.device)

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

    def _create_buffers(self):
        # constants
        self._ALL_ENV_INDICES = wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)
        self._ALL_BODY_INDICES = wp.array(list(range(self.num_bodies)), dtype=wp.int32, device=self.device)
        self._ALL_JOINT_INDICES = wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)
        # asset named data
        self._data.joint_names = self.joint_names
        self._data.body_names = self.body_names
        #  -- external forces and torques
        self._external_wrench = self._root_newton_view.get_attribute("body_f", NewtonManager.get_state_0())
        # -- root properties
        self._data.root_link_pose_w = self._root_newton_view.get_root_transforms(NewtonManager.get_state_0())
        self._data.root_com_vel_w = self._root_newton_view.get_root_velocities(NewtonManager.get_state_0())
        # -- body properties
        self._data.body_link_pose_w = self._root_newton_view.get_link_transforms(NewtonManager.get_state_0())
        self._data.body_com_vel_w = self._root_newton_view.get_link_velocities(NewtonManager.get_state_0())
        self._data.body_com_pos_b = self._root_newton_view.get_attribute("body_com", NewtonManager.get_model())
        # -- joint properties
        self._data.joint_pos_limits_lower = self._root_newton_view.get_attribute("joint_limit_lower", NewtonManager.get_model())
        self._data.joint_pos_limits_upper = self._root_newton_view.get_attribute("joint_limit_upper", NewtonManager.get_model())
        self._data.joint_stiffness = self._root_newton_view.get_attribute("joint_target_ke", NewtonManager.get_model())
        self._data.joint_damping = self._root_newton_view.get_attribute("joint_target_kd", NewtonManager.get_model())
        self._data.joint_armature = self._root_newton_view.get_attribute("joint_armature", NewtonManager.get_model())
        self._data.joint_friction_coeff = self._root_newton_view.get_attribute("joint_friction", NewtonManager.get_model())
        self._data.joint_vel_limits = self._root_newton_view.get_attribute("joint_velocity_limit", NewtonManager.get_model())
        self._data.joint_effort_limits = self._root_newton_view.get_attribute("joint_effort_limit", NewtonManager.get_model())
        self._data.joint_control_mode = self._root_newton_view.get_attribute("joint_control_mode", NewtonManager.get_model())
        # -- joint commands (sent to the simulation after actuator processing)
        self._joint_target_sim = self._root_newton_view.get_attribute("joint_f", NewtonManager.get_control())
        self._joint_effort_target_sim = self._root_newton_view.get_attribute("joint_target", NewtonManager.get_control())
        # -- joint commands (sent to the actuator from the user)
        self._data.joint_target = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        self._data.joint_effort_target = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        # -- computed joint efforts from the actuator models
        self._data.computed_torque = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        self._data.applied_torque = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        # -- other data that are filled based on explicit actuator models
        self._data.soft_joint_vel_limits = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        self._data.gear_ratio = wp.ones((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        # -- update the soft joint position limits
        wp.launch(
            update_soft_joint_pos_limits,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                self._data.soft_joint_pos_limits,
                self._data.joint_pos_limits_lower,
                self._data.joint_pos_limits_upper,
                self.cfg.soft_joint_pos_limit_factor,
            ]
        )


    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # -- root state
        self._data.default_root_pose = wp.zeros((self.num_instances), dtype=wp.transformf, device=self.device)
        self._data.default_root_vel = wp.zeros((self.num_instances), dtype=wp.spatial_vectorf, device=self.device)
        # default pose
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        wp.launch(
            update_transforms_array_with_value,
            dim=(self.num_instances,),
            inputs=[
                self._data.default_root_pose,
                self._ALL_ENV_INDICES,
                wp.transformf(default_root_pose),
            ]
        )        
        # default velocity
        default_root_velocity = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        wp.launch(
            update_spatial_vector_array_with_value,
            dim=(self.num_instances,),
            inputs=[
                self._data.default_root_vel,
                self._ALL_ENV_INDICES,
                wp.spatial_vectorf(default_root_velocity),
            ]
        )
        # -- joint state
        self._data.default_joint_pos = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        self._data.default_joint_vel = wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device)
        # joint pos
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_pos, self.joint_names
        )
        wp.launch(
            update_joint_array_with_value_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                self._data.default_joint_pos,
                self._ALL_ENV_INDICES,
                wp.array(indices_list, dtype=wp.int32, device=self.device),
                wp.array(values_list, dtype=wp.float32, device=self.device),
            ]
        )
        # joint vel
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_vel, self.joint_names
        )
        wp.launch(
            update_joint_array_with_value_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                self._data.default_joint_vel,
                self._ALL_ENV_INDICES,
                wp.array(indices_list, dtype=wp.int32, device=self.device),
                wp.array(values_list, dtype=wp.float32, device=self.device),
            ]
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
                joint_ids = torch.tensor(joint_ids, device=self.device)
            # create actuator collection
            # note: for efficiency avoid indexing when over all indices
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=self._data.joint_stiffness,
                damping=self._data.joint_damping,
                armature=self._data.joint_armature,
                friction=self._data.joint_friction_coeff,
                effort_limit=self._data.joint_effort_limits,
                velocity_limit=self._data.joint_vel_limits,
            )
            # log information on actuator groups
            model_type = "implicit" if actuator.is_implicit_model else "explicit"
            omni.log.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (type: {model_type}) and joint names: {joint_names} [{joint_ids}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # set the passed gains and limits into the simulation
            # TODO: write out all joint parameters from simulation
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(actuator.damping, joint_ids=actuator.joint_indices)
                # Sets the control mode for the implicit actuators
                self.write_joint_control_mode_to_sim(actuator.control_mode, joint_ids=actuator.joint_indices)
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.write_joint_stiffness_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(0.0, joint_ids=actuator.joint_indices)
                # Set the control mode to None when using explicit actuators
                self.write_joint_control_mode_to_sim(None, joint_ids=actuator.joint_indices)

            # Set common properties into the simulation
            self.write_joint_effort_limit_to_sim(actuator.effort_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_velocity_limit_to_sim(actuator.velocity_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_armature_to_sim(actuator.armature, joint_ids=actuator.joint_indices)
            self.write_joint_friction_coefficient_to_sim(actuator.friction, joint_ids=actuator.joint_indices)

            # Store the configured values from the actuator model
            # note: this is the value configured in the actuator model (for implicit and explicit actuators)
            #self._data.default_joint_stiffness[:, actuator.joint_indices] = actuator.stiffness
            #self._data.default_joint_damping[:, actuator.joint_indices] = actuator.damping
            #self._data.default_joint_armature[:, actuator.joint_indices] = actuator.armature
            #self._data.default_joint_friction_coeff[:, actuator.joint_indices] = actuator.friction

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
            # prepare input for actuator model based on cached data
            # TODO : A tensor dict would be nice to do the indexing of all tensors together
            control_action = ArticulationActions(
                joint_targets=self._data.joint_target[:, actuator.joint_indices],
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
            if control_action.joint_targets is not None:
                self._joint_target_sim[:, actuator.joint_indices] = control_action.joint_targets
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
        joint_pos_limits = torch.stack(
            (
                wp.to_torch(self._root_newton_view.get_attribute("joint_limit_lower", NewtonManager.get_model())),
                wp.to_torch(self._root_newton_view.get_attribute("joint_limit_upper", NewtonManager.get_model())),
            ),
            dim=2,
        )[0].to(self.device)
        out_of_range = self._data.default_joint_pos[0] < joint_pos_limits[:, 0]
        out_of_range |= self._data.default_joint_pos[0] > joint_pos_limits[:, 1]
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