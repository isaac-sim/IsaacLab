# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import torch
import warnings
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.actuators import ActuatorBase, ImplicitActuator
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.utils.shared import find_bodies, find_joints
from isaaclab_newton.kernels import (
    project_link_velocity_to_com_frame_masked_root,
    split_state_to_pose_and_velocity,
    transform_CoM_pose_to_link_frame_masked_root,
    update_default_joint_pos,
    update_soft_joint_pos_limits,
    update_wrench_array_with_force,
    update_wrench_array_with_torque,
    vec13f,
)
from newton import JointType, Model
from newton.selection import ArticulationView as NewtonArticulationView
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.helpers import deprecated
from isaaclab.utils.warp.update_kernels import (
    update_array1D_with_array1D_masked,
    update_array1D_with_value,
    update_array1D_with_value_indexed,
    update_array1D_with_value_masked,
    update_array2D_with_array1D_indexed,
    update_array2D_with_array2D_masked,
    update_array2D_with_value_indexed,
    update_array2D_with_value_masked,
)
from isaaclab.utils.warp.utils import (
    make_complete_data_from_torch_dual_index,
    make_complete_data_from_torch_single_index,
    make_mask_from_torch_ids,
)
from isaaclab.utils.wrench_composer import WrenchComposer

if TYPE_CHECKING:
    from isaaclab.actuators.actuator_cfg import ActuatorBaseCfg
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

logger = logging.getLogger(__name__)
warnings.simplefilter("once", UserWarning)
logging.captureWarnings(True)


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

    __backend_name__: str = "newton"
    """The name of the backend for the articulation."""

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
        return self._root_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self._root_view.is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self._root_view.joint_dof_count

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
        return self._root_view.link_count

    @property
    def num_shapes_per_body(self) -> list[int]:
        """Number of collision shapes per body in the articulation.

        This property returns a list where each element represents the number of collision
        shapes for the corresponding body in the articulation. This is cached for efficient
        access during material property randomization and other operations.

        Returns:
            List of integers representing the number of shapes per body.
        """
        if not hasattr(self, "_num_shapes_per_body"):
            self._num_shapes_per_body = []
            for shapes in self._root_view.body_shapes:
                self._num_shapes_per_body.append(len(shapes))
        return self._num_shapes_per_body

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self._root_view.joint_dof_names

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
        return self._root_view.body_names

    @property
    def root_view(self) -> NewtonArticulationView:
        """Articulation view for the asset (Newton).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_view

    @property
    def root_newton_model(self) -> Model:
        """Newton model for the asset."""
        return self._root_view.model

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.

        Note:
            Permanent wrenches are composed into the instantaneous wrench before the instantaneous wrenches are
            applied to the simulation.
        """
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are persistent and are applied to the simulation at every step. This is useful to apply forces that
        are constant over a period of time, things like the thrust of a motor for instance.

        Note:
            Permanent wrenches are composed into the instantaneous wrench before the instantaneous wrenches are
            applied to the simulation.
        """
        return self._permanent_wrench_composer

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset the articulation.

        Note: If both env_ids and env_mask are provided, then env_mask will be used. For performance reasons, it is
        recommended to use the env_mask instead of env_ids.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        if env_ids is not None and env_mask is None:
            env_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
            env_mask[env_ids] = True
            env_mask = wp.from_torch(env_mask, dtype=wp.bool)
        elif env_mask is not None:
            if isinstance(env_mask, torch.Tensor):
                env_mask = wp.from_torch(env_mask, dtype=wp.bool)

        # reset external wrenches.
        self._instantaneous_wrench_composer.reset(env_mask=env_mask)
        self._permanent_wrench_composer.reset(env_mask=env_mask)

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        if self._has_implicit_actuators:
            self._root_view.set_attribute(
                "joint_target_pos", NewtonManager.get_control(), self._actuator_position_target_reshaped
            )
            self._root_view.set_attribute(
                "joint_target_vel", NewtonManager.get_control(), self._actuator_velocity_target_expanded
            )
        # write external wrench
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench
                self._instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                wp.launch(
                    update_wrench_array_with_force,
                    dim=(self.num_instances, self.num_bodies),
                    device=self.device,
                    inputs=[
                        self._instantaneous_wrench_composer.composed_force,
                        self._data._sim_bind_body_external_wrench,
                        self._data.ALL_ENV_MASK,
                        self._data.ALL_BODY_MASK,
                    ],
                )
                wp.launch(
                    update_wrench_array_with_torque,
                    dim=(self.num_instances, self.num_bodies),
                    device=self.device,
                    inputs=[
                        self._instantaneous_wrench_composer.composed_torque,
                        self._data._sim_bind_body_external_wrench,
                        self._data.ALL_ENV_MASK,
                        self._data.ALL_BODY_MASK,
                    ],
                )
            else:
                # Apply permanent wrench to the simulation
                wp.launch(
                    update_wrench_array_with_force,
                    dim=(self.num_instances, self.num_bodies),
                    device=self.device,
                    inputs=[
                        self._permanent_wrench_composer.composed_force,
                        self._data._sim_bind_body_external_wrench,
                        self._data.ALL_ENV_MASK,
                        self._data.ALL_BODY_MASK,
                    ],
                )
                wp.launch(
                    update_wrench_array_with_torque,
                    dim=(self.num_instances, self.num_bodies),
                    device=self.device,
                    inputs=[
                        self._permanent_wrench_composer.composed_torque,
                        self._data._sim_bind_body_external_wrench,
                        self._data.ALL_ENV_MASK,
                        self._data.ALL_BODY_MASK,
                    ],
                )
        self._instantaneous_wrench_composer.reset()
        # apply actuator models. Actuator models automatically write the joint efforts into the simulation.
        self._apply_actuator_model()
        # Write the actuator targets into the simulation
        # TODO: Move this to the implicit actuator model.

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names, and indices.
        """
        return find_bodies(self.body_names, name_keys, preserve_order, self.device)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
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
        return find_joints(self.joint_names, name_keys, joint_subset, preserve_order, self.device)

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

    @deprecated("write_root_link_pose_to_sim", "write_root_com_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_state_to_sim(
        self,
        root_state: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            # Lazy initialization of the temporary root state.
            if self._temp_root_state is None:
                self._temp_root_state = wp.zeros((self.num_instances,), dtype=vec13f, device=self.device)
            root_state = make_complete_data_from_torch_single_index(
                root_state, self.num_instances, ids=env_ids, dtype=vec13f, device=self.device, out=self._temp_root_state
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        # write the pose and velocity to the simulation
        self.write_root_link_pose_to_sim(pose, env_mask=env_mask)
        self.write_root_com_velocity_to_sim(velocity, env_mask=env_mask)

    @deprecated("write_root_com_state_to_sim", "write_root_com_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_com_state_to_sim(
        self,
        root_state: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            # Lazy initialization of the temporary root state.
            if self._temp_root_state is None:
                self._temp_root_state = wp.zeros((self.num_instances,), dtype=vec13f, device=self.device)
            root_state = make_complete_data_from_torch_single_index(
                root_state, self.num_instances, ids=env_ids, dtype=vec13f, device=self.device, out=self._temp_root_state
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        # write the pose and velocity to the simulation
        self.write_root_com_pose_to_sim(pose, env_mask=env_mask)
        self.write_root_com_velocity_to_sim(velocity, env_mask=env_mask)

    @deprecated("write_root_link_pose_to_sim", "write_root_link_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_link_state_to_sim(
        self,
        root_state: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            # Lazy initialization of the temporary root state.
            if self._temp_root_state is None:
                self._temp_root_state = wp.zeros((self.num_instances,), dtype=vec13f, device=self.device)
            root_state = make_complete_data_from_torch_single_index(
                root_state, self.num_instances, ids=env_ids, dtype=vec13f, device=self.device, out=self._temp_root_state
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        # write the pose and velocity to the simulation
        self.write_root_link_pose_to_sim(pose, env_mask=env_mask)
        self.write_root_link_velocity_to_sim(velocity, env_mask=env_mask)

    def write_root_pose_to_sim(
        self,
        root_pose: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids, env_mask)

    def write_root_link_pose_to_sim(
        self,
        pose: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.


        The root pose ``wp.transformf`` comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(pose, torch.Tensor):
            if self._temp_root_pose is None:
                self._temp_root_pose = wp.zeros((self.num_instances,), dtype=wp.transformf, device=self.device)
            pose = make_complete_data_from_torch_single_index(
                pose, self.num_instances, ids=env_ids, dtype=wp.transformf, device=self.device, out=self._temp_root_pose
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        self._update_array_with_array_masked(pose, self._data.root_link_pose_w, env_mask, self.num_instances)
        # invalidate the root com pose
        self._data._root_com_pose_w.timestamp = -1.0

    def write_root_com_pose_to_sim(
        self,
        root_pose: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_pose, torch.Tensor):
            root_pose = make_complete_data_from_torch_single_index(
                root_pose, self.num_instances, ids=env_ids, dtype=wp.transformf, device=self.device
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # Write to Newton using warp
        self._update_array_with_array_masked(root_pose, self._data._root_com_pose_w.data, env_mask, self.num_instances)
        # set link frame poses
        wp.launch(
            transform_CoM_pose_to_link_frame_masked_root,
            dim=self.num_instances,
            device=self.device,
            inputs=[
                self._data._root_com_pose_w.data,
                self._data.body_com_pos_b,
                self._data.root_link_pose_w,
                env_mask,
            ],
        )
        # Force update the timestamp
        self._data._root_com_pose_w.timestamp = self._data._sim_timestamp

    def write_root_velocity_to_sim(
        self,
        root_velocity: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_com_velocity_to_sim(root_velocity, env_ids, env_mask)

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_velocity, torch.Tensor):
            if self._temp_root_velocity is None:
                self._temp_root_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self.device)
            root_velocity = make_complete_data_from_torch_single_index(
                root_velocity,
                self.num_instances,
                ids=env_ids,
                dtype=wp.spatial_vectorf,
                device=self.device,
                out=self._temp_root_velocity,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        self._update_array_with_array_masked(root_velocity, self._data.root_com_vel_w, env_mask, self.num_instances)
        # invalidate the derived velocities
        self._data._root_link_vel_w.timestamp = -1.0
        self._data._root_link_vel_b.timestamp = -1.0
        self._data._root_com_vel_b.timestamp = -1.0

    def write_root_link_velocity_to_sim(
        self, root_velocity: wp.array, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_velocity, torch.Tensor):
            if self._temp_root_velocity is None:
                self._temp_root_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self.device)
            root_velocity = make_complete_data_from_torch_single_index(
                root_velocity,
                self.num_instances,
                ids=env_ids,
                dtype=wp.spatial_vectorf,
                device=self.device,
                out=self._temp_root_velocity,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # update the root link velocity
        self._update_array_with_array_masked(
            root_velocity, self._data._root_link_vel_w.data, env_mask, self.num_instances
        )
        # set into simulation
        wp.launch(
            project_link_velocity_to_com_frame_masked_root,
            dim=self.num_instances,
            device=self.device,
            inputs=[
                root_velocity,
                self._data.root_link_pose_w,
                self._data.body_com_pos_b,
                self._data.root_com_vel_w,
                env_mask,
            ],
        )
        # Force update the timestamp
        self._data._root_link_vel_w.timestamp = self._data._sim_timestamp
        # invalidate the derived velocities
        self._data._root_link_vel_b.timestamp = -1.0
        self._data._root_com_vel_b.timestamp = -1.0

    def write_joint_state_to_sim(
        self,
        position: wp.array,
        velocity: wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(position, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            if self._temp_joint_vel is None:
                self._temp_joint_vel = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            position = make_complete_data_from_torch_dual_index(
                position,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
            velocity = make_complete_data_from_torch_dual_index(
                velocity,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_vel,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        self._update_batched_array_with_batched_array_masked(
            position, self._data.joint_pos, env_mask, joint_mask, (self.num_instances, self.num_joints)
        )
        self._update_batched_array_with_batched_array_masked(
            velocity, self._data.joint_vel, env_mask, joint_mask, (self.num_instances, self.num_joints)
        )

    def write_joint_position_to_sim(
        self,
        position: wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint positions to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(position, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            position = make_complete_data_from_torch_dual_index(
                position,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        self._update_batched_array_with_batched_array_masked(
            position, self._data.joint_pos, env_mask, joint_mask, (self.num_instances, self.num_joints)
        )

    def write_joint_velocity_to_sim(
        self,
        velocity: wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint velocities to the simulation.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(velocity, torch.Tensor):
            if self._temp_joint_vel is None:
                self._temp_joint_vel = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            velocity = make_complete_data_from_torch_dual_index(
                velocity,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_vel,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        self._update_batched_array_with_batched_array_masked(
            velocity, self._data.joint_vel, env_mask, joint_mask, (self.num_instances, self.num_joints)
        )

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_stiffness_to_sim(
        self,
        stiffness: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(stiffness, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            stiffness = make_complete_data_from_torch_dual_index(
                stiffness,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(stiffness, float):
            self._update_batched_array_with_value_masked(
                stiffness, self._data.joint_stiffness, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                stiffness, self._data.joint_stiffness, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_damping_to_sim(
        self,
        damping: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(damping, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            damping = make_complete_data_from_torch_dual_index(
                damping,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(damping, float):
            self._update_batched_array_with_value_masked(
                damping, self._data.joint_damping, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                damping, self._data.joint_damping, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_position_limit_to_sim(
        self,
        lower_limits: wp.array | float,
        upper_limits: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint position limits into the simulation.

        Args:
            lower_limits: Joint lower limits. Shape is (num_instances, num_joints).
            upper_limits: Joint upper limits. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(upper_limits, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            upper_limits = make_complete_data_from_torch_dual_index(
                upper_limits,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        if isinstance(lower_limits, torch.Tensor):
            if self._temp_joint_vel is None:
                self._temp_joint_vel = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            lower_limits = make_complete_data_from_torch_dual_index(
                lower_limits,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_vel,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        self._write_joint_position_limit_to_sim(lower_limits, upper_limits, joint_mask, env_mask)
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_velocity_limit_to_sim(
        self,
        limits: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. warn:: This function is ignored when using the Mujoco solver.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Warn if using Mujoco solver
        if isinstance(NewtonManager._solver, SolverMuJoCo):
            warnings.warn("write_joint_velocity_limit_to_sim is ignored by the solver when using Mujoco.")
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(limits, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            limits = make_complete_data_from_torch_dual_index(
                limits,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(limits, float):
            self._update_batched_array_with_value_masked(
                limits, self._data.joint_vel_limits, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                limits, self._data.joint_vel_limits, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_effort_limit_to_sim(
        self,
        limits: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(limits, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            limits = make_complete_data_from_torch_dual_index(
                limits,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(limits, float):
            self._update_batched_array_with_value_masked(
                limits, self._data.joint_effort_limits, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                limits, self._data.joint_effort_limits, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_armature_to_sim(
        self,
        armature: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(armature, torch.Tensor):
            if self._temp_joint_pos is None:
                self._temp_joint_pos = wp.zeros(
                    (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
                )
            armature = make_complete_data_from_torch_dual_index(
                armature,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
                out=self._temp_joint_pos,
            )
        env_mask = make_mask_from_torch_ids(
            self.num_instances, env_ids, env_mask, device=self.device, out=self._data.ENV_MASK
        )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(
            self.num_joints, joint_ids, joint_mask, device=self.device, out=self._data.JOINT_MASK
        )
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(armature, float):
            self._update_batched_array_with_value_masked(
                armature, self._data.joint_armature, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                armature, self._data.joint_armature, env_mask, joint_mask, (self.num_instances, self.num_joints)
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    # FIXME: What do we do of the dynamic and viscous friction coefficients?
    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: wp.array | float,
        joint_dynamic_friction_coeff: wp.array | float | None = None,
        joint_viscous_friction_coeff: wp.array | float | None = None,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
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
            joint_friction_coeff: Joint friction coefficients. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_dynamic_friction_coeff: Joint dynamic friction coefficients. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_viscous_friction_coeff: Joint viscous friction coefficients. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(joint_friction_coeff, torch.Tensor):
            joint_friction_coeff = make_complete_data_from_torch_dual_index(
                joint_friction_coeff,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(self.num_joints, joint_ids, joint_mask, device=self.device)
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(joint_friction_coeff, float):
            self._update_batched_array_with_value_masked(
                joint_friction_coeff,
                self._data.joint_friction_coeff,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                joint_friction_coeff,
                self._data.joint_friction_coeff,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        if joint_dynamic_friction_coeff is not None:
            self.write_joint_dynamic_friction_coefficient_to_sim(
                joint_dynamic_friction_coeff, joint_ids, env_ids, joint_mask, env_mask
            )
        if joint_viscous_friction_coeff is not None:
            self.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff, joint_ids, env_ids, joint_mask, env_mask
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    # FIXME: This is not implemented in Newton.
    def write_joint_dynamic_friction_coefficient_to_sim(
        self,
        joint_dynamic_friction_coeff: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint dynamic friction coefficients into the simulation.

        Warning: Setting joint dynamic friction coefficients are not supported in Newton. This operation will
        update the internal buffers, but not the simulation.

        Args:
            joint_dynamic_friction_coeff: Joint dynamic friction coefficients. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        logger.warning(
            "Setting joint dynamic friction coefficients are not supported in Newton. This operation will"
            "update the internal buffers, but not the simulation."
        )
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(joint_dynamic_friction_coeff, torch.Tensor):
            joint_dynamic_friction_coeff = make_complete_data_from_torch_dual_index(
                joint_dynamic_friction_coeff,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(self.num_joints, joint_ids, joint_mask, device=self.device)
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(joint_dynamic_friction_coeff, float):
            self._update_batched_array_with_value_masked(
                joint_dynamic_friction_coeff,
                self._data.joint_dynamic_friction_coeff,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                joint_dynamic_friction_coeff,
                self._data.joint_dynamic_friction_coeff,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    # FIXME: This is not implemented in Newton.
    def write_joint_viscous_friction_coefficient_to_sim(
        self,
        joint_viscous_friction_coeff: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint viscous friction coefficients into the simulation.

        Warning: Setting joint viscous friction coefficients are not supported in Newton. This operation will
        update the internal buffers, but not the simulation.

        Args:
            joint_viscous_friction_coeff: Joint viscous friction coefficients. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        logger.warning(
            "Setting joint viscous friction coefficients are not supported in Newton. This operation will"
            "update the internal buffers, but not the simulation."
        )
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(joint_viscous_friction_coeff, torch.Tensor):
            joint_viscous_friction_coeff = make_complete_data_from_torch_dual_index(
                joint_viscous_friction_coeff,
                self.num_instances,
                self.num_joints,
                env_ids,
                joint_ids,
                dtype=wp.float32,
                device=self.device,
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        joint_mask = make_mask_from_torch_ids(self.num_joints, joint_ids, joint_mask, device=self.device)
        if joint_mask is None:
            joint_mask = self._data.ALL_JOINT_MASK
        # None masks are handled within the kernel.
        # set into simulation
        if isinstance(joint_viscous_friction_coeff, float):
            self._update_batched_array_with_value_masked(
                joint_viscous_friction_coeff,
                self._data.joint_viscous_friction_coeff,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                joint_viscous_friction_coeff,
                self._data.joint_viscous_friction_coeff,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        # tell the physics engine that some of the joint properties have been updated
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    @deprecated("write_joint_friction_coefficient_to_sim", since="2.1.0", remove_in="4.0.0")
    def write_joint_friction_to_sim(
        self,
        joint_friction_coeff: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
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
            joint_friction_coeff: Joint friction coefficients. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        self.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff, joint_ids=joint_ids, env_ids=env_ids, joint_mask=joint_mask, env_mask=env_mask
        )

    @deprecated("write_joint_position_limit_to_sim", since="2.1.0", remove_in="4.0.0")
    def write_joint_limits_to_sim(
        self,
        upper_limits: wp.array | float,
        lower_limits: wp.array | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint position limits into the simulation.

        Args:
            upper_limits: Joint upper limits. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            lower_limits: Joint lower limits. Shape is (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        self.write_joint_position_limit_to_sim(upper_limits, lower_limits, joint_ids, env_ids, joint_mask, env_mask)

    """
    Operations - Setters.
    """

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
        # raise NotImplementedError()
        if isinstance(masses, torch.Tensor):
            masses = make_complete_data_from_torch_dual_index(
                masses, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.float32, device=self.device
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        body_mask = make_mask_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        if body_mask is None:
            body_mask = self._data.ALL_BODY_MASK
        # None masks are handled within the kernel.
        self._update_batched_array_with_batched_array_masked(
            masses, self._data.body_mass, env_mask, body_mask, (self.num_instances, self.num_bodies)
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

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
        if isinstance(coms, torch.Tensor):
            coms = make_complete_data_from_torch_dual_index(
                coms, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.vec3f, device=self.device
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        body_mask = make_mask_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        if body_mask is None:
            body_mask = self._data.ALL_BODY_MASK
        # None masks are handled within the kernel.
        self._update_batched_array_with_batched_array_masked(
            coms, self._data.body_com_pos_b, env_mask, body_mask, (self.num_instances, self.num_bodies)
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

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
        if isinstance(inertias, torch.Tensor):
            inertias = make_complete_data_from_torch_dual_index(
                inertias, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.mat33f, device=self.device
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        body_mask = make_mask_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        if body_mask is None:
            body_mask = self._data.ALL_BODY_MASK
        # None masks are handled within the kernel.
        self._update_batched_array_with_batched_array_masked(
            inertias, self._data.body_inertia, env_mask, body_mask, (self.num_instances, self.num_bodies)
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    # TODO: Plug-in the Wrench code from Isaac Lab once the PR gets in.
    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        positions: torch.Tensor | wp.array | None = None,
        is_global: bool = False,
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
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3).
            body_ids: The body indices to set the targets for. Defaults to None (all bodies).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
                Defaults to None. If None, the external wrench is applied at the center of mass of the body.
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the articulations' bodies.
        """

        # Write to wrench composer
        self._permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            env_ids=env_ids,
            body_mask=body_mask,
            env_mask=env_mask,
            is_global=is_global,
        )

    def set_joint_position_target(
        self,
        target: wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(target, torch.Tensor):
            target = make_complete_data_from_torch_dual_index(
                target, self.num_instances, self.num_joints, env_ids, joint_ids, dtype=wp.float32
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        joint_mask = make_mask_from_torch_ids(self.num_joints, joint_ids, joint_mask, device=self.device)
        # set into the actuator target buffer
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                self._data.actuator_position_target,
                env_mask,
                joint_mask,
            ],
            device=self.device,
        )

    def set_joint_velocity_target(
        self,
        target: wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(target, torch.Tensor):
            target = make_complete_data_from_torch_dual_index(
                target, self.num_instances, self.num_joints, env_ids, joint_ids, dtype=wp.float32, device=self.device
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        joint_mask = make_mask_from_torch_ids(self.num_joints, joint_ids, joint_mask, device=self.device)
        # set into the actuator target buffer
        self._update_batched_array_with_batched_array_masked(
            target, self._data.actuator_velocity_target, env_mask, joint_mask, (self.num_instances, self.num_joints)
        )

    def set_joint_effort_target(
        self,
        target: wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)) or (num_instances, num_joints).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(target, torch.Tensor):
            target = make_complete_data_from_torch_dual_index(
                target, self.num_instances, self.num_joints, env_ids, joint_ids, dtype=wp.float32, device=self.device
            )
        env_mask = make_mask_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        joint_mask = make_mask_from_torch_ids(self.num_joints, joint_ids, joint_mask, device=self.device)
        # set into the actuator effort target buffer
        self._update_batched_array_with_batched_array_masked(
            target, self._data.actuator_effort_target, env_mask, joint_mask, (self.num_instances, self.num_joints)
        )

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness(
        self,
        stiffness: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The fixed tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon stiffness is not supported in Newton.")

    def set_fixed_tendon_damping(
        self,
        damping: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The fixed tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon damping is not supported in Newton.")

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness efforts into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The fixed tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon limit stiffness is not supported in Newton.")

    def set_fixed_tendon_position_limit(
        self,
        limit: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
            limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The fixed tendon indices to set the limit for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon position limit is not supported in Newton.")

    @deprecated("set_fixed_tendon_position_limit", since="2.1.0", remove_in="4.0.0")
    def set_fixed_tendon_limit(
        self,
        limit: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
            limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The fixed tendon indices to set the limit for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        self.set_fixed_tendon_position_limit(
            limit,
            fixed_tendon_ids=fixed_tendon_ids,
            env_ids=env_ids,
            fixed_tendon_mask=fixed_tendon_mask,
            env_mask=env_mask,
        )

    def set_fixed_tendon_rest_length(
        self,
        rest_length: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length efforts into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The fixed tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon rest length is not supported in Newton.")

    def set_fixed_tendon_offset(
        self,
        offset: wp.array,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)) or (num_instances, num_fixed_tendons).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon offset is not supported in Newton.")

    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the properties for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the properties for. Defaults to None (all environments).
            fixed_tendon_mask: The fixed tendon mask. Shape is (num_fixed_tendons,).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Fixed tendon properties are not supported in Newton.")

    def set_spatial_tendon_stiffness(
        self,
        stiffness: wp.array,
        spatial_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The spatial tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons,).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon stiffness is not supported in Newton.")

    def set_spatial_tendon_damping(
        self,
        damping: wp.array,
        spatial_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The spatial tendon indices to set the damping for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons,).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon damping is not supported in Newton.")

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: wp.array,
        spatial_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The spatial tendon indices to set the limit stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons,).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon limit stiffness is not supported in Newton.")

    def set_spatial_tendon_offset(
        self,
        offset: wp.array,
        spatial_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)) or (num_instances, num_spatial_tendons).
            spatial_tendon_ids: The spatial tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons,).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon offset is not supported in Newton.")

    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation.

        Args:
            spatial_tendon_ids: The spatial tendon indices to set the properties for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the properties for. Defaults to None (all environments).
            spatial_tendon_mask: The spatial tendon mask. Shape is (num_spatial_tendons,).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError("Spatial tendon properties are not supported in Newton.")

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
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

        self._root_view = NewtonArticulationView(
            NewtonManager.get_model(), prim_path, verbose=True, exclude_joint_types=[JointType.FREE, JointType.FIXED]
        )

        # container for data access
        self._data = ArticulationData(self._root_view, self.device)

        # process configuration
        self._create_buffers()
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
        self._data.is_primed = True

    def _create_buffers(self, *args, **kwargs):
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        wp.launch(
            update_soft_joint_pos_limits,
            dim=(self.num_instances, self.num_joints),
            device=self.device,
            inputs=[
                self._data.joint_pos_limits_lower,
                self._data.joint_pos_limits_upper,
                self._data.soft_joint_pos_limits,
                self.cfg.soft_joint_pos_limit_factor,
            ],
        )

        # Assign joint and body names to the data
        self._data.joint_names = self.joint_names
        self._data.body_names = self.body_names

        # external wrench composers
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # Temp buffers for torch-to-warp conversion (lazy allocation - only created when needed)
        # These are reused to avoid per-call allocations when users pass torch tensors with indices
        self._temp_root_state: wp.array | None = None
        self._temp_root_pose: wp.array | None = None
        self._temp_root_velocity: wp.array | None = None
        self._temp_joint_pos: wp.array | None = None
        self._temp_joint_vel: wp.array | None = None
        self._temp_body_data_float: wp.array | None = None
        self._temp_body_data_vec3: wp.array | None = None
        self._temp_body_data_mat33: wp.array | None = None

        # Expanded target buffers
        self._actuator_position_target_reshaped = wp.array(
            ptr=self.data.actuator_position_target.ptr,
            shape=(self.num_instances, 1, self.num_joints),
            dtype=wp.float32,
            device=self.device,
        )
        self._actuator_velocity_target_expanded = wp.array(
            ptr=self.data.actuator_velocity_target.ptr,
            shape=(self.num_instances, 1, self.num_joints),
            dtype=wp.float32,
            device=self.device,
        )

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default pose with quaternion already in (x, y, z, w) format
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        # update the default root pose
        self._update_array_with_value(
            wp.transformf(*default_root_pose), self._data.default_root_pose, self.num_instances
        )
        # default velocity
        default_root_velocity = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        self._update_array_with_value(
            wp.spatial_vectorf(*default_root_velocity), self._data.default_root_vel, self.num_instances
        )
        # -- joint pos
        if self.num_joints > 0:
            # joint pos
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.init_state.joint_pos, self.joint_names
            )
            # Compute the mask once and use it for all joint operations
            wp.launch(
                update_array2D_with_array1D_indexed,
                dim=(self.num_instances, len(indices_list)),
                device=self.device,
                inputs=[
                    wp.array(values_list, dtype=wp.float32, device=self.device),
                    self._data.default_joint_pos,
                    None,
                    wp.array(indices_list, dtype=wp.int32, device=self.device),
                ],
            )
            # joint vel
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.init_state.joint_vel, self.joint_names
            )
            wp.launch(
                update_array2D_with_array1D_indexed,
                dim=(self.num_instances, len(indices_list)),
                device=self.device,
                inputs=[
                    wp.array(values_list, dtype=wp.float32, device=self.device),
                    self._data.default_joint_vel,
                    None,
                    wp.array(indices_list, dtype=wp.int32, device=self.device),
                ],
            )
        self._process_parameter_override()

    def _process_parameter_override(self):
        model = NewtonManager.get_model()
        for param_name, (param_value, param_expr) in self.cfg.model_parameter_override.items():
            # Check that the parameter exists in the model.
            if getattr(model, param_name, None) is None:
                raise ValueError(f"Parameter '{param_name}' is not found in the model.")
            # Check that there is a frequency for this parameter.
            frequency = model.attribute_frequency.get(param_name)
            if frequency is None:
                # No frequency, so we can't resolve the value.
                raise ValueError(
                    f"Parameter '{param_name}' has no frequency, so it cannot be resolved. "
                    "Please provide a scalar value instead."
                )
            # Get the attribute through the selection API
            # A frequency exists for this field, so we can resolve the indices if an expression is provided.
            if frequency == Model.AttributeFrequency.BODY:
                # 1D flattened array
                param = getattr(NewtonManager.get_model(), param_name)
                # Search over all bodies as organized in the environment
                body_subset = NewtonManager.get_model().body_key
                param_expr = ".*" if param_expr is None else param_expr
                indices, _ = string_utils.resolve_matching_names(param_expr, body_subset, False)
                indices = wp.array(indices, dtype=wp.int32, device=self.device)
            elif frequency == Model.AttributeFrequency.JOINT_DOF:
                # 3D array from selection API
                param = self.root_view.get_attribute(param_name, NewtonManager.get_model())
                # Make 2D (for now assume only 1 articulation)
                param = param.reshape(param.shape[0], param.shape[2])
                # Search over all joint DOFs as organized in the articulation
                joint_dof_subset = self.root_view.joint_dof_names
                param_expr = ".*" if param_expr is None else param_expr
                indices, _ = string_utils.resolve_matching_names(param_expr, joint_dof_subset, False)
                indices = wp.array(indices, dtype=wp.int32, device=self.device)
            elif frequency == Model.AttributeFrequency.SHAPE:
                # 1D flattened array
                param = getattr(NewtonManager.get_model(), param_name)
                # Search over all shapes as organized in the environment
                all_shapes = NewtonManager.get_model().shape_key
                param_expr = ".*" if param_expr is None else param_expr
                indices, _ = string_utils.resolve_matching_names(param_expr, all_shapes, False)
                indices = wp.array(indices, dtype=wp.int32, device=self.device)
            elif frequency == Model.AttributeFrequency.JOINT:
                # 1D flattened array
                param = getattr(NewtonManager.get_model(), param_name)
                # Search over all joints as organized in the environment
                all_joints = NewtonManager.get_model().joint_key
                param_expr = ".*" if param_expr is None else param_expr
                indices, _ = string_utils.resolve_matching_names(param_expr, all_joints, False)
                indices = wp.array(indices, dtype=wp.int32, device=self.device)
            else:
                raise ValueError(f"Parameter '{param_name}' has an unsupported frequency: {frequency}.")

            if param.ndim == 1:
                wp.launch(
                    update_array1D_with_value_indexed,
                    dim=(len(indices),),
                    inputs=[
                        param_value,
                        param,
                        indices,
                    ],
                    device=self.device,
                )
            elif param.ndim == 2:
                wp.launch(
                    update_array2D_with_value_indexed,
                    dim=(param.shape[0], len(indices)),
                    inputs=[
                        param_value,
                        param,
                        None,
                        indices,
                    ],
                    device=self.device,
                )
            else:
                raise ValueError(
                    f"Parameter '{param_name}' has an unsupported number of dimensions: {param.ndim}. "
                    "Only 1D and 2D arrays are supported."
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

        # Hack to ensure the limits are not too large.
        # Only set joint limits if there are joints (fixed-base articulations with 0 DOF skip this)
        if self._root_view.joint_dof_count > 0:
            self._root_view.get_attribute("joint_limit_ke", NewtonManager.get_model()).fill_(2500.0)
            self._root_view.get_attribute("joint_limit_kd", NewtonManager.get_model()).fill_(100.0)

        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # type annotation for type checkers
            actuator_cfg: ActuatorBaseCfg
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
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_mask=joint_mask,
                joint_indices=joint_indices,
                env_mask=self._data.ALL_ENV_MASK,
                data=self._data,
                device=self.device,
            )
            # log information on actuator groups
            model_type = "implicit" if actuator.is_implicit_model else "explicit"
            logger.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (type: {model_type}) and joint names: {joint_names} [{joint_mask}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # set the passed gains and limits into the simulation
            # TODO: write out all joint parameters from simulation
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self._update_batched_array_with_batched_array_masked(
                    self._data.actuator_stiffness,
                    self._data.joint_stiffness,
                    self._data.ALL_ENV_MASK,
                    actuator._joint_mask,
                    (self.num_instances, self.num_joints),
                )
                self._update_batched_array_with_batched_array_masked(
                    self._data.actuator_damping,
                    self._data.joint_damping,
                    self._data.ALL_ENV_MASK,
                    actuator._joint_mask,
                    (self.num_instances, self.num_joints),
                )
                # When using implicit actuators, we bind the commands sent from the user to the simulation.
                # We only run the actuator model to compute the estimated joint efforts.
                # self.data._actuator_target = self.data.joint_target
                # self.data._actuator_effort_target = self.data.joint_effort
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self._update_batched_array_with_value_masked(
                    0.0,
                    self._data.joint_stiffness,
                    self._data.ALL_ENV_MASK,
                    actuator._joint_mask,
                    (self.num_instances, self.num_joints),
                )
                self._update_batched_array_with_value_masked(
                    0.0,
                    self._data.joint_damping,
                    self._data.ALL_ENV_MASK,
                    actuator._joint_mask,
                    (self.num_instances, self.num_joints),
                )
                # Bind the applied effort to the simulation effort
                # self.data._applied_effort = self.data.actuator_effort_target

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != (self.num_joints - self.num_fixed_tendons):
            warnings.warn(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints - self.num_fixed_tendons}.",
                UserWarning,
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
        # Skip validation if there are no joints (e.g., fixed-base articulation with 0 DOF)
        if self._root_view.joint_dof_count == 0:
            return

        # check that the default values are within the limits
        joint_pos_limits = torch.stack(
            (
                wp.to_torch(self._root_view.get_attribute("joint_limit_lower", NewtonManager.get_model()))[:, 0],
                wp.to_torch(self._root_view.get_attribute("joint_limit_upper", NewtonManager.get_model()))[:, 0],
            ),
            dim=2,
        )[0].to(self.device)
        out_of_range = wp.to_torch(self._data._default_joint_pos)[0] < joint_pos_limits[:, 0]
        out_of_range |= wp.to_torch(self._data._default_joint_pos)[0] > joint_pos_limits[:, 1]
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        # throw error if any of the default joint positions are out of the limits
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default positions out of the limits: \n"
            default_joint_pos = wp.to_torch(self._data._default_joint_pos)
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = joint_pos_limits[idx]
                joint_pos = default_joint_pos[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

        # check that the default joint velocities are within the limits
        joint_max_vel = wp.to_torch(self._root_view.get_attribute("joint_velocity_limit", NewtonManager.get_model()))[
            :, 0
        ]
        out_of_range = torch.abs(wp.to_torch(self.data.default_joint_vel)) > joint_max_vel
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default velocities out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = [-joint_max_vel[idx], joint_max_vel[idx]]
                joint_vel = wp.to_torch(self._data._default_joint_vel)[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
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
        logger.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + joint_table.get_string())

        # read out all fixed tendon parameters from simulation
        if self.num_fixed_tendons > 0:
            raise NotImplementedError("Tendons are not implemented yet")

        if self.num_spatial_tendons > 0:
            raise NotImplementedError("Tendons are not implemented yet")

    """
    Internal Warp helpers.
    """

    def _update_array_with_value(
        self,
        source: float | int | wp.vec2f | wp.vec3f | wp.quatf | wp.transformf | wp.spatial_vectorf,
        target: wp.array,
        dim: int,
    ):
        """Update an array with a value.

        Args:
            source: The source value.
            target: The target array. Shape is (dim,). Must be pre-allocated, is modified in place.
            dim: The dimension of the array.
        """
        wp.launch(
            update_array1D_with_value,
            dim=(dim,),
            inputs=[
                source,
                target,
            ],
            device=self.device,
        )

    def _update_array_with_value_masked(
        self,
        source: float | int | wp.vec2f | wp.vec3f | wp.quatf | wp.transformf | wp.spatial_vectorf,
        target: wp.array,
        mask: wp.array,
        dim: int,
    ):
        """Update an array with a value using a mask.

        Args:
            source: The source value.
            target: The target array. Shape is (dim,). Must be pre-allocated, is modified in place.
            mask: The mask to use. Shape is (dim,).
            dim: The dimension of the array.
        """
        wp.launch(
            update_array1D_with_value_masked,
            dim=(dim,),
            inputs=[
                source,
                target,
                mask,
            ],
            device=self.device,
        )

    def _update_array_with_array_masked(self, source: wp.array, target: wp.array, mask: wp.array, dim: int):
        """Update an array with an array using a mask.

        Args:
            source: The source array. Shape is (dim,).
            target: The target array. Shape is (dim,). Must be pre-allocated, is modified in place.
            mask: The mask to use. Shape is (dim,).
        """
        wp.launch(
            update_array1D_with_array1D_masked,
            dim=(dim,),
            inputs=[
                source,
                target,
                mask,
            ],
            device=self.device,
        )

    def _update_batched_array_with_batched_array_masked(
        self, source: wp.array, target: wp.array, mask_1: wp.array, mask_2: wp.array, dim: tuple[int, int]
    ):
        """Update a batched array with a batched array using a mask.

        Args:
            source: The source array. Shape is (dim[0], dim[1]).
            target: The target array. Shape is (dim[0], dim[1]). Must be pre-allocated, is modified in place.
            mask_1: The mask to use. Shape is (dim[0],).
            mask_2: The mask to use. Shape is (dim[1],).
            dim: The dimension of the arrays.
        """
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=dim,
            inputs=[
                source,
                target,
                mask_1,
                mask_2,
            ],
            device=self.device,
        )

    def _update_batched_array_with_value_masked(
        self,
        source: float | int | wp.vec2f | wp.vec3f | wp.quatf | wp.transformf | wp.spatial_vectorf,
        target: wp.array,
        mask_1: wp.array,
        mask_2: wp.array,
        dim: tuple[int, int],
    ):
        """Update a batched array with a value using a mask.

        Args:
            source: The source value.
            target: The target array. Shape is (dim[0], dim[1]). Must be pre-allocated, is modified in place.
            mask_1: The mask to use. Shape is (dim[0],).
            mask_2: The mask to use. Shape is (dim[1],).
            dim: The dimension of the arrays.
        """
        wp.launch(
            update_array2D_with_value_masked,
            dim=dim,
            inputs=[
                source,
                target,
                mask_1,
                mask_2,
            ],
            device=self.device,
        )

    def _write_joint_position_limit_to_sim(
        self,
        lower_limits: wp.array | float,
        upper_limits: wp.array | float,
        joint_mask: wp.array,
        env_mask: wp.array,
    ) -> None:
        """Write joint position limits into the simulation.

        Args:
            upper_limits: Joint upper limits. Shape is (num_instances, num_joints).
            lower_limits: Joint lower limits. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices

        if isinstance(lower_limits, float):
            self._update_batched_array_with_value_masked(
                lower_limits,
                self._data.joint_pos_limits_lower,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                lower_limits,
                self._data.joint_pos_limits_lower,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        if isinstance(upper_limits, float):
            self._update_batched_array_with_value_masked(
                upper_limits,
                self._data.joint_pos_limits_upper,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )
        else:
            self._update_batched_array_with_batched_array_masked(
                upper_limits,
                self._data.joint_pos_limits_upper,
                env_mask,
                joint_mask,
                (self.num_instances, self.num_joints),
            )

        # Update default joint position
        wp.launch(
            update_default_joint_pos,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                self._data.joint_pos_limits_lower,
                self._data.joint_pos_limits_upper,
                self._data.default_joint_pos,
            ],
            device=self.device,
        )

        # Update soft joint limits
        wp.launch(
            update_soft_joint_pos_limits,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                self._data.joint_pos_limits_lower,
                self._data.joint_pos_limits_upper,
                self._data.soft_joint_pos_limits,
                self.cfg.soft_joint_pos_limit_factor,
            ],
            device=self.device,
        )

    def _split_state(
        self,
        state: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Split the state into pose and velocity.

        Args:
            state: State in simulation frame. Shape is (num_instances, 13).

        Returns:
            A tuple of pose and velocity. Shape is (num_instances, 7) and (num_instances, 6) respectively.
        """
        if self._temp_root_pose is None:
            self._temp_root_pose = wp.zeros((self.num_instances,), dtype=wp.transformf, device=self.device)
        if self._temp_root_velocity is None:
            self._temp_root_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self.device)

        wp.launch(
            split_state_to_pose_and_velocity,
            dim=self.num_instances,
            inputs=[
                state,
                self._temp_root_pose,
                self._temp_root_velocity,
            ],
            device=self.device,
        )
        return self._temp_root_pose, self._temp_root_velocity
