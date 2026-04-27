# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation implementation backed by ovphysx TensorBindingsAPI."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.physics import PhysicsManager
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager

from .articulation_data import ArticulationData
from .kernels import _body_wrench_to_world, _scatter_rows_partial, update_soft_joint_pos_limits

if TYPE_CHECKING:
    from isaaclab.actuators import ActuatorBase
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

logger = logging.getLogger(__name__)


class Articulation(BaseArticulation):
    """Articulation backed by the ovphysx TensorBindingsAPI.

    Reads and writes simulation state through ovphysx.TensorBinding objects created
    from the OvPhysxManager's PhysX instance.
    """

    __backend_name__ = "ovphysx"

    cfg: ArticulationCfg

    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> ArticulationData:
        """Data container with simulation state for this articulation."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of articulation instances (environments)."""
        return self._num_instances

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self._is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in the articulation."""
        return self._num_joints

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in the articulation."""
        return getattr(self, "_num_fixed_tendons", 0)

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in the articulation."""
        return getattr(self, "_num_spatial_tendons", 0)

    @property
    def num_bodies(self) -> int:
        """Number of bodies (links) in the articulation."""
        return self._num_bodies

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in the articulation."""
        return self._joint_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in the articulation."""
        return getattr(self, "_fixed_tendon_names", [])

    @property
    def spatial_tendon_names(self) -> list[str]:
        """Ordered names of spatial tendons in the articulation."""
        return getattr(self, "_spatial_tendon_names", [])

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies (links) in the articulation."""
        return self._body_names

    @property
    def root_view(self) -> Any:
        """Root articulation view (not available for ovphysx backend)."""
        return None

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer | None:
        """Wrench composer for forces applied only during the current step."""
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer | None:
        """Wrench composer for forces applied persistently every step."""
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
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
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

    def write_data_to_sim(self) -> None:
        """Apply external wrenches, actuator model, and write commands into the simulation."""
        # Apply external wrenches (before actuators, same as PhysX backend).
        self._apply_external_wrenches()

        self._apply_actuator_model()
        # Write effort tensor to simulation.
        if self._effort_binding is not None:
            self._effort_binding.write(self._effort_write_view)
        # Write position and velocity targets in one shot (not per-actuator).
        if self._has_implicit_actuators:
            if self._pos_target_binding is not None:
                self._pos_target_binding.write(self._pos_target_write_view)
            if self._vel_target_binding is not None:
                self._vel_target_binding.write(self._vel_target_write_view)

    def update(self, dt: float) -> None:
        """Update internal data buffers after a simulation step.

        Args:
            dt: The simulation time step [s] used for finite-difference quantities.
        """
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :func:`isaaclab.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return resolve_matching_names(name_keys, self._body_names, preserve_order)

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Please check the :func:`isaaclab.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self._joint_names
        return resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_fixed_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find fixed tendons in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions
                to match the joint names with fixed tendons.
            tendon_subsets: A subset of joints with fixed tendons to search
                for. Defaults to None, which means all joints in the
                articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in
                the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        if tendon_subsets is None:
            tendon_subsets = self.fixed_tendon_names
        return resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    def find_spatial_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find spatial tendons in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions
                to match the tendon names.
            tendon_subsets: A subset of tendons to search for. Defaults to
                None, which means all tendons in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in
                the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        if tendon_subsets is None:
            tendon_subsets = self.spatial_tendon_names
        return resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    """
    Operations - State Writers.
    """

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")
        self._write_root_state(TT.ROOT_POSE, root_pose, env_ids)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over masked environments into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        self._write_root_state(TT.ROOT_POSE, root_pose, mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")
        self._write_root_state(TT.ROOT_POSE, root_pose, env_ids)

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over masked environments into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        self._write_root_state(TT.ROOT_POSE, root_pose, mask=env_mask)

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids),)
                with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")
        self._write_root_state(TT.ROOT_POSE, root_pose, env_ids)

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over masked environments into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances,)
                with dtype wp.transformf.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        self._write_root_state(TT.ROOT_POSE, root_pose, mask=env_mask)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set the root velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        Args:
            root_velocity: Root velocities in simulation world frame. Shape is (len(env_ids),)
                with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")
        self._write_root_state(TT.ROOT_VELOCITY, root_velocity, env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root velocity over masked environments into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        Args:
            root_velocity: Root velocities in simulation world frame. Shape is (num_instances,)
                with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        self._write_root_state(TT.ROOT_VELOCITY, root_velocity, mask=env_mask)

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame.
                Shape is (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")
        self._write_root_state(TT.ROOT_VELOCITY, root_velocity, env_ids)

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over masked environments into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame.
                Shape is (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        self._write_root_state(TT.ROOT_VELOCITY, root_velocity, mask=env_mask)

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's frame rather than the root's center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame.
                Shape is (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")
        self._write_root_state(TT.ROOT_VELOCITY, root_velocity, env_ids)

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over masked environments into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's frame rather than the root's center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame.
                Shape is (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        self._write_root_state(TT.ROOT_VELOCITY, root_velocity, mask=env_mask)

    """
    Operations - Joint State Writers.
    """

    def write_joint_state_to_sim_mask(
        self,
        joint_pos: torch.Tensor | wp.array,
        joint_vel: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        joint_mask: wp.array | None = None,
    ) -> None:
        """Write joint positions and velocities over masked environments into the simulation.

        Args:
            joint_pos: Joint positions. Shape is (num_instances, num_joints).
            joint_vel: Joint velocities. Shape is (num_instances, num_joints).
            env_mask: Environment mask. If None, then all instances are updated.
            joint_mask: Joint mask. If None, then all joints are updated.
        """
        self.write_joint_position_to_sim_mask(position=joint_pos, env_mask=env_mask, joint_mask=joint_mask)
        self.write_joint_velocity_to_sim_mask(velocity=joint_vel, env_mask=env_mask, joint_mask=joint_mask)

    def write_joint_position_to_sim_index(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint positions over selected environment and joint indices into the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(position, (n, d), wp.float32, "position")
        self._write_flat_tensor(TT.DOF_POSITION, position, env_ids, joint_ids)
        self.data._joint_pos_buf.timestamp = -1.0

    def write_joint_position_to_sim_mask(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint positions over masked environments into the simulation.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(position, (self._num_instances, self._num_joints), wp.float32, "position")
        self._write_flat_tensor_mask(TT.DOF_POSITION, position, env_mask, joint_mask)
        self.data._joint_pos_buf.timestamp = -1.0

    def write_joint_velocity_to_sim_index(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint velocities over selected environment and joint indices into the simulation.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(velocity, (n, d), wp.float32, "velocity")
        self._write_flat_tensor(TT.DOF_VELOCITY, velocity, env_ids, joint_ids)
        self.data._joint_vel_buf.timestamp = -1.0

    def write_joint_velocity_to_sim_mask(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint velocities over masked environments into the simulation.

        Args:
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(velocity, (self._num_instances, self._num_joints), wp.float32, "velocity")
        self._write_flat_tensor_mask(TT.DOF_VELOCITY, velocity, env_mask, joint_mask)
        self.data._joint_vel_buf.timestamp = -1.0

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_stiffness_to_sim_index(
        self,
        *,
        stiffness: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint stiffness over selected indices into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(stiffness, (n, d), wp.float32, "stiffness")
        self._write_flat_tensor(TT.DOF_STIFFNESS, stiffness, env_ids, joint_ids)

    def write_joint_stiffness_to_sim_mask(
        self,
        *,
        stiffness: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint stiffness over masked environments into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._num_joints), wp.float32, "stiffness")
        self._write_flat_tensor_mask(TT.DOF_STIFFNESS, stiffness, env_mask, joint_mask)

    def write_joint_damping_to_sim_index(
        self,
        *,
        damping: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint damping over selected indices into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(damping, (n, d), wp.float32, "damping")
        self._write_flat_tensor(TT.DOF_DAMPING, damping, env_ids, joint_ids)

    def write_joint_damping_to_sim_mask(
        self,
        *,
        damping: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint damping over masked environments into the simulation.

        Args:
            damping: Joint damping. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(damping, (self._num_instances, self._num_joints), wp.float32, "damping")
        self._write_flat_tensor_mask(TT.DOF_DAMPING, damping, env_mask, joint_mask)

    def write_joint_position_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Write joint position limits over selected environment indices into the simulation.

        Args:
            limits: Joint position limits [rad or m]. Shape is (len(env_ids), len(joint_ids))
                with dtype wp.vec2f.
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
            warn_limit_violation: Whether to use warning or info level logging when default joint
                positions exceed the new limits. Defaults to True.
        """
        if isinstance(limits, (int, float)):
            raise ValueError("Float scalars are not supported for position limits (vec2f dtype)")
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(limits, (n, d), wp.vec2f, "limits")
        self._write_flat_tensor(TT.DOF_LIMIT, limits, env_ids, joint_ids)

    def write_joint_position_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Write joint position limits over masked environments into the simulation.

        Args:
            limits: Joint position limits [rad or m]. Shape is (num_instances, num_joints)
                with dtype wp.vec2f.
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
            warn_limit_violation: Whether to use warning or info level logging when default joint
                positions exceed the new limits. Defaults to True.
        """
        if isinstance(limits, (int, float)):
            raise ValueError("Float scalars are not supported for position limits (vec2f dtype)")
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.vec2f, "limits")
        self._write_flat_tensor_mask(TT.DOF_LIMIT, limits, env_mask, joint_mask)

    def write_joint_velocity_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint max velocity over selected environment indices into the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine.

        Args:
            limits: Joint max velocity [rad/s or m/s]. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(limits, (n, d), wp.float32, "limits")
        self._write_flat_tensor(TT.DOF_MAX_VELOCITY, limits, env_ids, joint_ids)

    def write_joint_velocity_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint max velocity over masked environments into the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine.

        Args:
            limits: Joint max velocity [rad/s or m/s]. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.float32, "limits")
        self._write_flat_tensor_mask(TT.DOF_MAX_VELOCITY, limits, env_mask, joint_mask)

    def write_joint_effort_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint effort limits over selected environment indices into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine.

        Args:
            limits: Joint effort limits [N or N*m]. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(limits, (n, d), wp.float32, "limits")
        self._write_flat_tensor(TT.DOF_MAX_FORCE, limits, env_ids, joint_ids)

    def write_joint_effort_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint effort limits over masked environments into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine.

        Args:
            limits: Joint effort limits [N or N*m]. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.float32, "limits")
        self._write_flat_tensor_mask(TT.DOF_MAX_FORCE, limits, env_mask, joint_mask)

    def write_joint_armature_to_sim_index(
        self,
        *,
        armature: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint armature over selected environment indices into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature [kg*m^2 or kg]. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(armature, (n, d), wp.float32, "armature")
        self._write_flat_tensor(TT.DOF_ARMATURE, armature, env_ids, joint_ids)

    def write_joint_armature_to_sim_mask(
        self,
        *,
        armature: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint armature over masked environments into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature [kg*m^2 or kg]. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(armature, (self._num_instances, self._num_joints), wp.float32, "armature")
        self._write_flat_tensor_mask(TT.DOF_ARMATURE, armature, env_mask, joint_mask)

    def write_joint_friction_coefficient_to_sim_index(
        self,
        *,
        joint_friction_coeff: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Write joint friction coefficients over selected indices into the simulation.

        Args:
            joint_friction_coeff: Joint friction coefficients. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(joint_friction_coeff, (n, d), wp.float32, "joint_friction_coeff")
        self._write_friction_column(joint_friction_coeff, env_ids, joint_ids)

    def write_joint_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_friction_coeff: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint friction coefficients over masked environments into the simulation.

        Args:
            joint_friction_coeff: Joint friction coefficients. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self.assert_shape_and_dtype(
            joint_friction_coeff, (self._num_instances, self._num_joints), wp.float32, "joint_friction_coeff"
        )
        self._write_friction_column_mask(joint_friction_coeff, env_mask, joint_mask)

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using indices.

        Args:
            masses: Masses of all bodies [kg]. Shape is (len(env_ids), len(body_ids)).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        n = self._n_envs_index(env_ids)
        b = len(body_ids) if body_ids is not None else self._num_bodies
        self.assert_shape_and_dtype(masses, (n, b), wp.float32, "masses")
        self._write_flat_tensor(TT.BODY_MASS, masses, env_ids, body_ids)

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using masks.

        Args:
            masses: Masses of all bodies [kg]. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        self._write_flat_tensor_mask(TT.BODY_MASS, masses, env_mask, body_mask)

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids))
                with dtype wp.transformf.
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for.
                Defaults to None (all environments).
        """
        n = self._n_envs_index(env_ids)
        b = len(body_ids) if body_ids is not None else self._num_bodies
        self.assert_shape_and_dtype(coms, (n, b), wp.transformf, "coms")
        self._write_flat_tensor(TT.BODY_COM_POSE, coms, env_ids, body_ids)

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using masks.

        Args:
            coms: Center of mass pose of all bodies. Shape is (num_instances, num_bodies)
                with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        self._write_flat_tensor_mask(TT.BODY_COM_POSE, coms, env_mask, body_mask)

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using indices.

        Args:
            inertias: Inertias of all bodies [kg*m^2]. Shape is (len(env_ids), len(body_ids), 9).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
        """
        n = self._n_envs_index(env_ids)
        b = len(body_ids) if body_ids is not None else self._num_bodies
        self.assert_shape_and_dtype(inertias, (n, b, 9), wp.float32, "inertias")
        self._write_flat_tensor(TT.BODY_INERTIA, inertias, env_ids, body_ids)

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using masks.

        Args:
            inertias: Inertias of all bodies [kg*m^2]. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        self._write_flat_tensor_mask(TT.BODY_INERTIA, inertias, env_mask, body_mask)

    """
    Operations - Target Setters.
    """

    def set_joint_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets [rad or m]. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._set_target_into_buffer(self._data._joint_pos_target, target, env_ids, joint_ids)

    def set_joint_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers using masks.

        Args:
            target: Joint position targets [rad or m]. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self._set_target_into_buffer_mask(self._data._joint_pos_target, target, env_mask, joint_mask)

    def set_joint_velocity_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets [rad/s or m/s]. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._set_target_into_buffer(self._data._joint_vel_target, target, env_ids, joint_ids)

    def set_joint_velocity_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers using masks.

        Args:
            target: Joint velocity targets [rad/s or m/s]. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self._set_target_into_buffer_mask(self._data._joint_vel_target, target, env_mask, joint_mask)

    def set_joint_effort_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers using indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets [N or N*m]. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: Joint indices. If None, then all joints are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._set_target_into_buffer(self._data._joint_effort_target, target, env_ids, joint_ids)

    def set_joint_effort_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers using masks.

        Args:
            target: Joint effort targets [N or N*m]. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all joints are updated.
            env_mask: Environment mask. If None, then all instances are updated.
        """
        self._set_target_into_buffer_mask(self._data._joint_effort_target, target, env_mask, joint_mask)

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers using indices.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` method.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(stiffness, (n, t), wp.float32, "stiffness")
        if self._data._fixed_tendon_stiffness is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_stiffness, stiffness, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers using masks.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._nft()), wp.float32, "stiffness")
        if self._data._fixed_tendon_stiffness is not None:
            self._set_target_into_buffer_mask(
                self._data._fixed_tendon_stiffness, stiffness, env_mask, fixed_tendon_mask
            )

    def set_fixed_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers using indices.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(damping, (n, t), wp.float32, "damping")
        if self._data._fixed_tendon_damping is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_damping, damping, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers using masks.

        Args:
            damping: Fixed tendon damping. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(damping, (self._num_instances, self._nft()), wp.float32, "damping")
        if self._data._fixed_tendon_damping is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_damping, damping, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness into internal buffers using indices.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(limit_stiffness, (n, t), wp.float32, "limit_stiffness")
        if self._data._fixed_tendon_limit_stiffness is not None:
            self._set_target_into_buffer(
                self._data._fixed_tendon_limit_stiffness, limit_stiffness, env_ids, fixed_tendon_ids
            )

    def set_fixed_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness into internal buffers using masks.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(limit_stiffness, (self._num_instances, self._nft()), wp.float32, "limit_stiffness")
        if self._data._fixed_tendon_limit_stiffness is not None:
            self._set_target_into_buffer_mask(
                self._data._fixed_tendon_limit_stiffness, limit_stiffness, env_mask, fixed_tendon_mask
            )

    def set_fixed_tendon_position_limit_index(
        self,
        *,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon position limits into internal buffers using indices.

        Args:
            limit: Fixed tendon position limits. Shape is (len(env_ids), len(fixed_tendon_ids))
                with dtype wp.vec2f.
            fixed_tendon_ids: The tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(limit, (n, t), wp.vec2f, "limit")
        if self._data._fixed_tendon_pos_limits is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_pos_limits, limit, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_position_limit_mask(
        self,
        *,
        limit: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon position limits into internal buffers using masks.

        Args:
            limit: Fixed tendon position limits. Shape is (num_instances, num_fixed_tendons)
                with dtype wp.vec2f.
            fixed_tendon_mask: Tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(limit, (self._num_instances, self._nft()), wp.vec2f, "limit")
        if self._data._fixed_tendon_pos_limits is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_pos_limits, limit, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_rest_length_index(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length into internal buffers using indices.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(rest_length, (n, t), wp.float32, "rest_length")
        if self._data._fixed_tendon_rest_length is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_rest_length, rest_length, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_rest_length_mask(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length into internal buffers using masks.

        Args:
            rest_length: Fixed tendon rest length. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(rest_length, (self._num_instances, self._nft()), wp.float32, "rest_length")
        if self._data._fixed_tendon_rest_length is not None:
            self._set_target_into_buffer_mask(
                self._data._fixed_tendon_rest_length, rest_length, env_mask, fixed_tendon_mask
            )

    def set_fixed_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon offset into internal buffers using indices.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(offset, (n, t), wp.float32, "offset")
        if self._data._fixed_tendon_offset is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_offset, offset, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon offset into internal buffers using masks.

        Args:
            offset: Fixed tendon offset. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(offset, (self._num_instances, self._nft()), wp.float32, "offset")
        if self._data._fixed_tendon_offset is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_offset, offset, env_mask, fixed_tendon_mask)

    def write_fixed_tendon_properties_to_sim_index(
        self,
        *,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation using indices.

        Args:
            fixed_tendon_ids: The fixed tendon indices to write the properties for. Defaults to None
                (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if self._nft() == 0:
            return
        for tt, buf in [
            (TT.FIXED_TENDON_STIFFNESS, self._data._fixed_tendon_stiffness),
            (TT.FIXED_TENDON_DAMPING, self._data._fixed_tendon_damping),
            (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._data._fixed_tendon_limit_stiffness),
            (TT.FIXED_TENDON_LIMIT, self._data._fixed_tendon_pos_limits),
            (TT.FIXED_TENDON_REST_LENGTH, self._data._fixed_tendon_rest_length),
            (TT.FIXED_TENDON_OFFSET, self._data._fixed_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor(tt, buf, env_ids, fixed_tendon_ids)

    def write_fixed_tendon_properties_to_sim_mask(
        self,
        *,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation using masks.

        Args:
            fixed_tendon_mask: Fixed tendon mask. If None, then all fixed tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        if self._nft() == 0:
            return
        for tt, buf in [
            (TT.FIXED_TENDON_STIFFNESS, self._data._fixed_tendon_stiffness),
            (TT.FIXED_TENDON_DAMPING, self._data._fixed_tendon_damping),
            (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._data._fixed_tendon_limit_stiffness),
            (TT.FIXED_TENDON_LIMIT, self._data._fixed_tendon_pos_limits),
            (TT.FIXED_TENDON_REST_LENGTH, self._data._fixed_tendon_rest_length),
            (TT.FIXED_TENDON_OFFSET, self._data._fixed_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor_mask(tt, buf, env_mask, fixed_tendon_mask)

    def set_spatial_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers using indices.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(stiffness, (n, t), wp.float32, "stiffness")
        if self._data._spatial_tendon_stiffness is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_stiffness, stiffness, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers using masks.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._nst()), wp.float32, "stiffness")
        if self._data._spatial_tendon_stiffness is not None:
            self._set_target_into_buffer_mask(
                self._data._spatial_tendon_stiffness, stiffness, env_mask, spatial_tendon_mask
            )

    def set_spatial_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers using indices.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(damping, (n, t), wp.float32, "damping")
        if self._data._spatial_tendon_damping is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_damping, damping, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers using masks.

        Args:
            damping: Spatial tendon damping. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(damping, (self._num_instances, self._nst()), wp.float32, "damping")
        if self._data._spatial_tendon_damping is not None:
            self._set_target_into_buffer_mask(
                self._data._spatial_tendon_damping, damping, env_mask, spatial_tendon_mask
            )

    def set_spatial_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers using indices.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(limit_stiffness, (n, t), wp.float32, "limit_stiffness")
        if self._data._spatial_tendon_limit_stiffness is not None:
            self._set_target_into_buffer(
                self._data._spatial_tendon_limit_stiffness, limit_stiffness, env_ids, spatial_tendon_ids
            )

    def set_spatial_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers using masks.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(limit_stiffness, (self._num_instances, self._nst()), wp.float32, "limit_stiffness")
        if self._data._spatial_tendon_limit_stiffness is not None:
            self._set_target_into_buffer_mask(
                self._data._spatial_tendon_limit_stiffness, limit_stiffness, env_mask, spatial_tendon_mask
            )

    def set_spatial_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset into internal buffers using indices.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        n = self._n_envs_index(env_ids)
        t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(offset, (n, t), wp.float32, "offset")
        if self._data._spatial_tendon_offset is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_offset, offset, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset into internal buffers using masks.

        Args:
            offset: Spatial tendon offset. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        self.assert_shape_and_dtype(offset, (self._num_instances, self._nst()), wp.float32, "offset")
        if self._data._spatial_tendon_offset is not None:
            self._set_target_into_buffer_mask(self._data._spatial_tendon_offset, offset, env_mask, spatial_tendon_mask)

    def write_spatial_tendon_properties_to_sim_index(
        self,
        *,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation using indices.

        Args:
            spatial_tendon_ids: The spatial tendon indices to write the properties for. Defaults to None
                (all spatial tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        if self._nst() == 0:
            return
        for tt, buf in [
            (TT.SPATIAL_TENDON_STIFFNESS, self._data._spatial_tendon_stiffness),
            (TT.SPATIAL_TENDON_DAMPING, self._data._spatial_tendon_damping),
            (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._data._spatial_tendon_limit_stiffness),
            (TT.SPATIAL_TENDON_OFFSET, self._data._spatial_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor(tt, buf, env_ids, spatial_tendon_ids)

    def write_spatial_tendon_properties_to_sim_mask(
        self,
        *,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation using masks.

        Args:
            spatial_tendon_mask: Spatial tendon mask. If None, then all spatial tendons are used.
            env_mask: Environment mask. If None, then all the instances are updated.
        """
        if self._nst() == 0:
            return
        for tt, buf in [
            (TT.SPATIAL_TENDON_STIFFNESS, self._data._spatial_tendon_stiffness),
            (TT.SPATIAL_TENDON_DAMPING, self._data._spatial_tendon_damping),
            (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._data._spatial_tendon_limit_stiffness),
            (TT.SPATIAL_TENDON_OFFSET, self._data._spatial_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor_mask(tt, buf, env_mask, spatial_tendon_mask)

    """
    Deprecated methods.
    """

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Deprecated in base class. Use :meth:`write_root_pose_to_sim_index` and
        :meth:`write_root_velocity_to_sim_index` instead."""
        self._write_root_state(TT.ROOT_POSE, root_state[:, :7], env_ids)
        self._write_root_state(TT.ROOT_VELOCITY, root_state[:, 7:], env_ids)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Deprecated in base class. Use :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index` instead."""
        self._write_root_state(TT.ROOT_POSE, root_state[:, :7], env_ids)
        self._write_root_state(TT.ROOT_VELOCITY, root_state[:, 7:], env_ids)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Deprecated in base class. Use :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index` instead."""
        self._write_root_state(TT.ROOT_POSE, root_state[:, :7], env_ids)
        self._write_root_state(TT.ROOT_VELOCITY, root_state[:, 7:], env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Deprecated in base class. Use :meth:`write_joint_position_to_sim_index` and
        :meth:`write_joint_velocity_to_sim_index` instead."""
        self.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)

    """
    Internal helper.
    """

    def _initialize_impl(self) -> None:
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

        prim_path = self.cfg.prim_path
        # Convert IsaacLab prim-path notation to the glob patterns ovphysx expects.
        # IsaacLab uses two conventions:
        #   /World/envs/env_.*/Robot  -- regex dot-star for "any env index"
        #   /World/envs/{ENV_REGEX_NS}/Robot -- explicit placeholder
        # ovphysx create_tensor_binding() uses fnmatch-style globs, so both map to '*'.
        pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", prim_path)
        pattern = re.sub(r"\.\*", "*", pattern)  # env_.* -> env_*

        # The pattern above points to the ArticulationCfg prim (e.g. /World/envs/env_*/Robot).
        # However, PhysicsArticulationRootAPI may be on a CHILD prim (e.g. /Robot/torso)
        # rather than on the prim itself.  create_tensor_binding() only matches prims that
        # *have* PhysicsArticulationRootAPI, so we need to extend the pattern to the actual
        # articulation root.  Mirror the PhysX backend's discovery logic: find the first
        # matching prim in the USD stage, walk its subtree for the articulation root, and
        # append the relative suffix to the glob pattern.
        from pxr import UsdPhysics

        from isaaclab.sim.utils.queries import find_first_matching_prim, get_all_matching_child_prims

        stage = PhysicsManager._sim.stage
        first_prim = find_first_matching_prim(prim_path, stage=stage)
        if first_prim is None:
            raise RuntimeError(f"OvPhysxManager: no prim found for path '{prim_path}'.")
        first_prim_path = first_prim.GetPath().pathString

        root_prims = get_all_matching_child_prims(
            first_prim_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"No prim with PhysicsArticulationRootAPI found under '{first_prim_path}'."
                " Check that the articulation has 'PhysicsArticulationRootAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Multiple articulation roots found under '{first_prim_path}': {root_prims}."
                " There must be exactly one articulation root per prim path."
            )
        self._articulation_root_path = root_prims[0].GetPath().pathString
        root_relative = self._articulation_root_path[len(first_prim_path) :]
        if root_relative:
            # e.g. first_prim_path=/World/envs/env_0/Robot, root_relative=/torso
            # pattern becomes /World/envs/env_*/Robot/torso
            pattern = pattern + root_relative
            logger.info("OvPhysxManager: articulation root at '%s' (pattern extended to '%s')", root_relative, pattern)

        # Bindings are created lazily (on first access) to avoid allocating
        # handles for tensor types the user never queries.  Only the root-pose
        # binding is created eagerly because we need it to read articulation
        # metadata (joint count, body count, names, fixed-base flag).
        self._bindings: dict[int, Any] = {}
        self._physx_instance = physx_instance
        self._binding_pattern = pattern

        eager_types = [
            TT.ROOT_POSE,
            TT.DOF_POSITION,
            TT.DOF_STIFFNESS,
            TT.DOF_DAMPING,
            TT.DOF_LIMIT,
            TT.DOF_MAX_VELOCITY,
            TT.DOF_MAX_FORCE,
            TT.DOF_ARMATURE,
            TT.DOF_FRICTION_PROPERTIES,
            TT.BODY_MASS,
            TT.BODY_COM_POSE,
            TT.BODY_INERTIA,
        ]
        for tt in eager_types:
            try:
                self._bindings[tt] = physx_instance.create_tensor_binding(pattern=pattern, tensor_type=tt)
            except Exception:
                logger.debug("Could not create tensor binding for type %s on pattern %s", tt, pattern)

        sample = next(iter(self._bindings.values()))
        self._num_instances = sample.count
        self._num_joints = sample.dof_count
        self._num_bodies = sample.body_count
        self._is_fixed_base = sample.is_fixed_base
        self._joint_names = list(sample.dof_names)
        self._body_names = list(sample.body_names)

        # Create data container.
        self._data = ArticulationData(self._bindings, self._device, binding_getter=self._get_binding)

        # Discover tendon counts/names before buffer allocation so that
        # _create_buffers can size the tendon property arrays.
        self._process_tendons()

        self._create_buffers()

        self._process_cfg()
        self._process_actuators_cfg()
        self._validate_cfg()
        self._log_articulation_info()

        # Cache the effort binding and a stable float32 view of the applied_torque
        # buffer for write_data_to_sim(). The binding's internal write cache
        # (keyed on object identity) handles the fast path automatically.
        self._effort_binding = self._get_binding(TT.DOF_ACTUATION_FORCE)
        if self._effort_binding is not None:
            torque = self._data.applied_torque
            shape = self._effort_binding.shape
            self._effort_write_view = wp.array(
                ptr=torque.ptr,
                shape=shape,
                dtype=wp.float32,
                device=str(torque.device),
                copy=False,
            )
        else:
            self._effort_write_view = None

        # Cache position/velocity target bindings + views for one-shot writes.
        def _make_write_view(tt, buf):
            b = self._get_binding(tt)
            if b is None or buf is None:
                return None, None
            v = wp.array(ptr=buf.ptr, shape=b.shape, dtype=wp.float32, device=str(buf.device), copy=False)
            return b, v

        self._pos_target_binding, self._pos_target_write_view = _make_write_view(
            TT.DOF_POSITION_TARGET, self._data.joint_pos_target
        )
        self._vel_target_binding, self._vel_target_write_view = _make_write_view(
            TT.DOF_VELOCITY_TARGET, self._data.joint_vel_target
        )

        # Let the articulation data know that it is fully instantiated and ready to use.
        self.data.is_primed = True

    def _create_buffers(self) -> None:
        self._data._create_buffers()

        self._ALL_INDICES = wp.array(np.arange(self._num_instances, dtype=np.int32), device=self._device)

        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)
        self._wrench_buf = wp.zeros((self._num_instances, self._num_bodies, 9), dtype=wp.float32, device=self._device)

        # Joint-index arrays for each actuator (filled by _process_actuators_cfg).
        self._joint_ids_per_actuator: dict[str, list[int]] = {}
        self._write_scratch: dict[int, wp.array] = {}

    def _process_cfg(self) -> None:
        """Process the articulation configuration (initial state, soft limits, etc.)."""
        cfg = self.cfg
        N = self._num_instances
        D = self._num_joints
        dev = self._device

        # Default root state from config (matching PhysX pattern).
        default_root_pose = tuple(cfg.init_state.pos) + tuple(cfg.init_state.rot)
        default_root_vel = tuple(cfg.init_state.lin_vel) + tuple(cfg.init_state.ang_vel)
        np_pose = np.tile(np.array(default_root_pose, dtype=np.float32), (N, 1))
        np_vel = np.tile(np.array(default_root_vel, dtype=np.float32), (N, 1))
        wp.copy(
            self._data._default_root_pose,
            wp.from_numpy(np_pose, dtype=wp.transformf, device=dev),
        )
        wp.copy(
            self._data._default_root_vel,
            wp.from_numpy(np_vel, dtype=wp.spatial_vectorf, device=dev),
        )

        # Default joint positions / velocities from config patterns.
        self._resolve_joint_values(cfg.init_state.joint_pos, self._data._default_joint_pos)
        self._resolve_joint_values(cfg.init_state.joint_vel, self._data._default_joint_vel)

        # Keep soft-limit computation on-device, matching the PhysX/Newton path.
        wp.launch(
            update_soft_joint_pos_limits,
            dim=(N, D),
            inputs=[self._data.joint_pos_limits, cfg.soft_joint_pos_limit_factor],
            outputs=[self._data._soft_joint_pos_limits],
            device=dev,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        self._is_initialized = False

    def _process_actuators_cfg(self) -> None:
        """Build actuator instances from the config and write drive properties to PhysX.

        Mirrors what the legacy PhysX backend does in its own _process_actuators_cfg:
        - For ImplicitActuator: write the configured stiffness / damping to the PhysX
          drive so the solver uses exactly the values from the actuator config.
        - For all explicit actuators: zero out PhysX stiffness / damping so the
          USD-authored drive gains cannot interfere with the explicit torque path.
        - For all actuators: write effort_limit_sim and velocity_limit_sim.

        These writes happen via TensorBinding (GPU-resident) after warmup has
        allocated the GPU buffers (MODEL_INIT fires post-warmup).
        """
        from isaaclab.actuators import ImplicitActuator

        self.actuators: dict[str, ActuatorBase] = {}
        self._has_implicit_actuators = False
        for name, act_cfg in self.cfg.actuators.items():
            joint_ids, joint_names = self.find_joints(act_cfg.joint_names_expr)
            if not joint_ids:
                logger.warning("Actuator '%s': no joints matched '%s'", name, act_cfg.joint_names_expr)
                continue
            act_cfg_copy = act_cfg.copy()
            act = act_cfg_copy.class_type(
                act_cfg_copy,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self._num_instances,
                device=self._device,
            )
            self.actuators[name] = act
            self._joint_ids_per_actuator[name] = joint_ids

            # Write drive gains and limits to PhysX to match the actuator config.
            # Without this, PhysX retains whatever stiffness/damping was authored in the
            # USD file, which can produce large restoring forces if the USD gains differ
            # from the actuator config (e.g. a position-controlled robot exported with
            # non-zero drive stiffness but configured with ImplicitActuator(stiffness=0)).
            jids = list(joint_ids)
            if isinstance(act, ImplicitActuator):
                self._has_implicit_actuators = True
                stiffness = act.stiffness  # torch (N, J)
                damping = act.damping  # torch (N, J)
            else:
                stiffness = wp.zeros((self._num_instances, len(jids)), dtype=wp.float32, device=self._device)
                damping = wp.zeros((self._num_instances, len(jids)), dtype=wp.float32, device=self._device)
            self.write_joint_stiffness_to_sim_index(stiffness=stiffness, joint_ids=jids)
            self.write_joint_damping_to_sim_index(damping=damping, joint_ids=jids)
            self.write_joint_effort_limit_to_sim_index(limits=act.effort_limit_sim, joint_ids=jids)
            self.write_joint_velocity_limit_to_sim_index(limits=act.velocity_limit_sim, joint_ids=jids)

    def _process_tendons(self) -> None:
        """Discover tendon counts from binding metadata and names from USD.

        Tendon counts come from the ovphysx binding metadata. Tendon names are
        recovered from the exported USD articulation subtree because ovphysx
        exposes joint names/counts, but not the per-joint USD paths that the
        PhysX backend can query directly.
        """
        self._fixed_tendon_names = []
        self._spatial_tendon_names = []

        sample = next(iter(self._bindings.values()))
        self._num_fixed_tendons = getattr(sample, "fixed_tendon_count", 0)
        self._num_spatial_tendons = getattr(sample, "spatial_tendon_count", 0)

        if self._num_fixed_tendons > 0 or self._num_spatial_tendons > 0:
            stage_path = OvPhysxManager._stage_path
            if stage_path is not None:
                try:
                    from pxr import Usd, UsdPhysics

                    from isaaclab.sim.utils.queries import get_all_matching_child_prims

                    stage = Usd.Stage.Open(stage_path)
                    articulation_root_path = getattr(self, "_articulation_root_path", None)
                    if articulation_root_path is None:
                        joint_prims = stage.Traverse()
                    else:
                        joint_prims = get_all_matching_child_prims(
                            articulation_root_path,
                            predicate=lambda p: p.IsA(UsdPhysics.Joint),
                            stage=stage,
                            traverse_instance_prims=False,
                        )
                    for prim in joint_prims:
                        if not prim.IsA(UsdPhysics.Joint):
                            continue
                        schema_names = list(prim.GetAppliedSchemas())
                        metadata = prim.GetMetadata("apiSchemas")
                        if metadata is not None:
                            for field in ("prependedItems", "appendedItems", "explicitItems"):
                                items = getattr(metadata, field, None)
                                if items:
                                    schema_names.extend(str(item) for item in items)
                        schemas_str = " ".join(schema_names)
                        name = prim.GetPath().name
                        if "PhysxTendonAxisRootAPI" in schemas_str:
                            self._fixed_tendon_names.append(name)
                        elif (
                            "PhysxTendonAttachmentRootAPI" in schemas_str
                            or "PhysxTendonAttachmentLeafAPI" in schemas_str
                        ):
                            self._spatial_tendon_names.append(name)
                except Exception:
                    logger.debug("Could not parse USD stage for tendon names at %s", stage_path)

        self._data._num_fixed_tendons = self._num_fixed_tendons
        self._data._num_spatial_tendons = self._num_spatial_tendons
        self._data.fixed_tendon_names = self._fixed_tendon_names
        self._data.spatial_tendon_names = self._spatial_tendon_names

    def _apply_external_wrenches(self) -> None:
        """Compose and write external wrenches to the LINK_WRENCH binding.

        WrenchComposer accumulates forces/torques in body (link) frame.
        The LINK_WRENCH binding expects world-frame [fx,fy,fz,tx,ty,tz,px,py,pz].
        We rotate the body-frame vectors to world frame using the link quaternion
        and pack them into the [N, L, 9] tensor with application position = origin.
        """
        inst = self._instantaneous_wrench_composer
        perm = self._permanent_wrench_composer
        if not inst.active and not perm.active:
            return
        if inst.active:
            if perm.active:
                inst.add_forces_and_torques_index(
                    forces=perm.composed_force,
                    torques=perm.composed_torque,
                    body_ids=list(range(self._num_bodies)),
                    env_ids=list(range(self._num_instances)),
                )
            force_b = inst.composed_force
            torque_b = inst.composed_torque
        else:
            force_b = perm.composed_force
            torque_b = perm.composed_torque

        poses = self._data.body_link_pose_w
        wp.launch(
            _body_wrench_to_world,
            dim=(self._num_instances, self._num_bodies),
            inputs=[force_b, torque_b, poses],
            outputs=[self._wrench_buf],
            device=self._device,
        )
        wrench_binding = self._get_binding(TT.LINK_WRENCH)
        if wrench_binding is not None:
            wrench_binding.write(self._wrench_buf)
        inst.reset()

    def _apply_actuator_model(self) -> None:
        """Run the actuator model to compute torques from user targets.

        IsaacLab actuators are torch-based.  We convert warp -> torch via
        DLPack (zero-copy on GPU), run the actuator, then write results back.
        """
        from isaaclab.utils.types import ArticulationActions

        for name, act in self.actuators.items():
            jids = act.joint_indices
            if jids is None:
                continue
            jids_t = jids if isinstance(jids, list) else list(jids)
            all_joints = len(jids_t) == self._num_joints

            # warp -> torch (zero-copy on same device via DLPack)
            jp_target_full = self._data.joint_pos_target.torch
            jv_target_full = self._data.joint_vel_target.torch
            je_target_full = self._data.joint_effort_target.torch
            jp_target = jp_target_full if all_joints else jp_target_full[:, jids_t]
            jv_target = jv_target_full if all_joints else jv_target_full[:, jids_t]
            je_target = je_target_full if all_joints else je_target_full[:, jids_t]

            control_action = ArticulationActions(
                joint_positions=jp_target,
                joint_velocities=jv_target,
                joint_efforts=je_target,
            )

            jp_cur_full = self._data.joint_pos.torch
            jv_cur_full = self._data.joint_vel.torch
            jp_cur = jp_cur_full if all_joints else jp_cur_full[:, jids_t]
            jv_cur = jv_cur_full if all_joints else jv_cur_full[:, jids_t]

            control_action = act.compute(control_action, jp_cur, jv_cur)

            if act.computed_effort is not None:
                ct = wp.to_torch(self._data._computed_torque)
                at = wp.to_torch(self._data._applied_torque)
                if all_joints:
                    ct[:] = act.computed_effort
                    at[:] = act.applied_effort
                else:
                    ct[:, jids_t] = act.computed_effort
                    at[:, jids_t] = act.applied_effort

    def _validate_cfg(self) -> None:
        pass

    def _log_articulation_info(self) -> None:
        """Log information about the articulation.

        .. note:: We purposefully read the values from the simulator to ensure that the values are configured as
            expected.
        """
        from prettytable import PrettyTable

        def format_large_number(_, v: float) -> str:
            if abs(v) >= 1e3:
                return f"{v:.1e}"
            return f"{v:.3f}"

        def format_limits(_, v: tuple[float, float]) -> str:
            if abs(v[0]) >= 1e3 or abs(v[1]) >= 1e3:
                return f"[{v[0]:.1e}, {v[1]:.1e}]"
            return f"[{v[0]:.3f}, {v[1]:.3f}]"

        stiffnesses = self.data.joint_stiffness.warp.numpy()[0].tolist()
        dampings = self.data.joint_damping.warp.numpy()[0].tolist()
        armatures = self.data.joint_armature.warp.numpy()[0].tolist()
        frictions = self.data.joint_friction_coeff.warp.numpy()[0].tolist()
        pos_limits_np = self.data.joint_pos_limits.warp.numpy().reshape(self._num_instances, self._num_joints, 2)
        position_limits = [tuple(pos_limits_np[0, j].tolist()) for j in range(self._num_joints)]
        velocity_limits = self.data.joint_vel_limits.warp.numpy()[0].tolist()
        effort_limits = self.data.joint_effort_limits.warp.numpy()[0].tolist()

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
        joint_table.custom_format["Stiffness"] = format_large_number
        joint_table.custom_format["Damping"] = format_large_number
        joint_table.custom_format["Armature"] = format_large_number
        joint_table.custom_format["Friction"] = format_large_number
        joint_table.custom_format["Position Limits"] = format_limits
        joint_table.custom_format["Velocity Limits"] = format_large_number
        joint_table.custom_format["Effort Limits"] = format_large_number
        joint_table.align["Name"] = "l"

        for index, name in enumerate(self.joint_names):
            joint_table.add_row(
                [
                    index,
                    name,
                    stiffnesses[index],
                    dampings[index],
                    armatures[index],
                    frictions[index],
                    position_limits[index],
                    velocity_limits[index],
                    effort_limits[index],
                ]
            )
        logger.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + joint_table.get_string())

        if self.num_fixed_tendons > 0:
            ft_stiffnesses = self.data.fixed_tendon_stiffness.warp.numpy()[0].tolist()
            ft_dampings = self.data.fixed_tendon_damping.warp.numpy()[0].tolist()
            ft_limit_stiffnesses = self.data.fixed_tendon_limit_stiffness.warp.numpy()[0].tolist()
            ft_limits_np = self.data.fixed_tendon_pos_limits.warp.numpy().reshape(
                self._num_instances, self.num_fixed_tendons, 2
            )
            ft_limits = [tuple(ft_limits_np[0, t].tolist()) for t in range(self.num_fixed_tendons)]
            ft_rest_lengths = self.data.fixed_tendon_rest_length.warp.numpy()[0].tolist()
            ft_offsets = self.data.fixed_tendon_offset.warp.numpy()[0].tolist()

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
            tendon_table.custom_format["Stiffness"] = format_large_number
            tendon_table.custom_format["Damping"] = format_large_number
            tendon_table.custom_format["Limit Stiffness"] = format_large_number
            tendon_table.custom_format["Limits"] = format_limits
            tendon_table.custom_format["Rest Length"] = format_large_number
            tendon_table.custom_format["Offset"] = format_large_number
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
            logger.info(
                f"Simulation parameters for fixed tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

        if self.num_spatial_tendons > 0:
            st_stiffnesses = self.data.spatial_tendon_stiffness.warp.numpy()[0].tolist()
            st_dampings = self.data.spatial_tendon_damping.warp.numpy()[0].tolist()
            st_limit_stiffnesses = self.data.spatial_tendon_limit_stiffness.warp.numpy()[0].tolist()
            st_offsets = self.data.spatial_tendon_offset.warp.numpy()[0].tolist()

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
            logger.info(
                f"Simulation parameters for spatial tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

    """
    Internal helpers -- Bindings.
    """

    def _get_binding(self, tensor_type: int):
        """Return a cached TensorBinding, creating it on first access.

        Bindings are lightweight handles (a pointer + shape metadata into
        PhysX's shared GPU buffer).  Creating one does NOT allocate new GPU
        memory -- the underlying simulation buffers are allocated once by PhysX
        regardless of how many bindings point into them.  Still, we defer
        creation so that tensor types the user never queries are never looked up.
        """
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        try:
            binding = self._physx_instance.create_tensor_binding(pattern=self._binding_pattern, tensor_type=tensor_type)
            self._bindings[tensor_type] = binding
            return binding
        except Exception:
            logger.debug("Could not create tensor binding for type %s", tensor_type)
            return None

    """
    Internal helpers -- Write.
    """

    def _to_flat_f32(self, data, target_shape: tuple[int, ...] | None = None) -> wp.array | np.ndarray:
        """Ensure data is a contiguous float32 tensor suitable for binding I/O.

        State tensor bindings (positions, velocities, poses) live on the
        simulation device (GPU in GPU mode).  We always return data on
        self._device so the binding device check passes.

        For structured warp dtypes (transformf, spatial_vectorf, etc.) a
        zero-copy flat float32 view is created instead of roundtripping
        through CPU numpy.
        """
        dev = self._device
        if isinstance(data, wp.array):
            if str(data.device) != dev:
                data = wp.clone(data, device=dev)
            if data.dtype == wp.float32:
                return data
            # Structured dtype: zero-copy flat float32 view.
            # transformf -> [N, 7], spatial_vectorf -> [N, 6], etc.
            floats_per_elem = data.strides[0] // 4
            return wp.array(
                ptr=data.ptr,
                shape=(data.shape[0], floats_per_elem),
                dtype=wp.float32,
                device=dev,
                copy=False,
            )
        elif isinstance(data, torch.Tensor):
            if data.is_cuda and dev.startswith("cuda"):
                return wp.from_torch(data.detach().contiguous().float())
            np_data = data.detach().cpu().numpy().astype(np.float32)
            return wp.from_numpy(np_data, dtype=wp.float32, device=dev)
        elif isinstance(data, np.ndarray):
            return wp.from_numpy(data.astype(np.float32), dtype=wp.float32, device=dev)
        elif isinstance(data, (int, float)):
            return wp.from_numpy(np.array(data, dtype=np.float32), dtype=wp.float32, device=dev)
        return wp.from_numpy(np.asarray(data, dtype=np.float32), dtype=wp.float32, device=dev)

    def _as_gpu_f32_2d(self, data, cols: int) -> wp.array:
        """View/convert data as 2D [rows, cols] float32 on self._device.

        For warp arrays with structured dtypes (transformf, spatial_vectorf),
        creates a zero-copy flat float32 view.  For torch/numpy, converts to
        warp on the simulation device.
        """
        dev = self._device
        if isinstance(data, wp.array):
            if str(data.device) != dev:
                data = wp.clone(data, device=dev)
            if data.dtype == wp.float32 and data.ndim == 2:
                return data
            n = data.shape[0]
            return wp.array(
                ptr=data.ptr,
                shape=(n, cols),
                dtype=wp.float32,
                device=dev,
                copy=False,
            )
        if isinstance(data, torch.Tensor) and data.is_cuda and dev.startswith("cuda"):
            return wp.from_torch(data.detach().contiguous().float().reshape(-1, cols))
        np_data = self._to_cpu_numpy(data).reshape(-1, cols)
        return wp.from_numpy(np_data, dtype=wp.float32, device=dev)

    def _get_write_scratch(self, tensor_type: int, binding) -> wp.array:
        """Return a cached GPU scratch buffer for read-modify-write."""
        if not hasattr(self, "_write_scratch"):
            self._write_scratch = {}
        buf = self._write_scratch.get(tensor_type)
        if buf is None:
            buf = wp.zeros(binding.shape, dtype=wp.float32, device=self._device)
            self._write_scratch[tensor_type] = buf
        return buf

    def _write_root_state(self, tensor_type: int, data, env_ids=None, mask=None, _ids_gpu=None) -> None:
        """GPU-native write for root pose [N,7] or velocity [N,6].

        Three paths, fastest first:
        - Full write (no env_ids, no mask): zero-copy DLPack.
        - Indexed write with full-size data: zero-copy view + indices.
          The binding API only copies the indexed rows from the full buffer,
          so no read-modify-write is needed when data is already [N,...].
        - Indexed write with partial data [K,...]: scatter kernel into a GPU
          scratch buffer, then write with indices.
        - Masked write: data is always full [N,...], pass directly with mask.

        Args:
            _ids_gpu: Pre-converted GPU warp int32 array of env indices.
                When provided, skips the per-call GPU->CPU->GPU conversion
                of env_ids.
        """
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        N, C = binding.shape

        if env_ids is None and _ids_gpu is None and mask is None:
            binding.write(self._to_flat_f32(data))
            self._invalidate_root_caches(tensor_type)
            return

        src = self._as_gpu_f32_2d(data, C)

        if env_ids is not None or _ids_gpu is not None:
            if _ids_gpu is None:
                _ids_gpu = self._env_ids_to_gpu_warp(env_ids)
            K = _ids_gpu.shape[0]
            if src.shape[0] == N:
                binding.write(src, indices=_ids_gpu)
            else:
                scratch = self._get_write_scratch(tensor_type, binding)
                binding.read(scratch)
                wp.launch(
                    _scatter_rows_partial,
                    dim=(K, C),
                    inputs=[scratch, src, _ids_gpu],
                    device=self._device,
                )
                binding.write(scratch, indices=_ids_gpu)
        else:
            mask_u8 = wp.from_numpy(
                self._to_cpu_numpy(mask).astype(np.uint8),
                device=self._device,
            )
            binding.write(src, mask=mask_u8)
        self._invalidate_root_caches(tensor_type)

    def _invalidate_root_caches(self, tensor_type: int) -> None:
        """Force re-read from GPU on next property access after a binding write."""
        if tensor_type == TT.ROOT_POSE:
            self.data._root_link_pose_w.timestamp = -1.0
            self.data._root_com_pose_w.timestamp = -1.0
        elif tensor_type == TT.ROOT_VELOCITY:
            self.data._root_link_vel_w.timestamp = -1.0
            self.data._root_com_vel_w.timestamp = -1.0

    def _write_flat_tensor(self, tensor_type: int, data, env_ids=None, joint_ids=None, _ids_gpu=None) -> None:
        """Write a 2-D tensor to a binding, with optional env/joint index subsetting."""
        if isinstance(data, (int, float)):
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        from isaaclab_ovphysx.tensor_types import _CPU_ONLY_TYPES

        is_cpu_only = tensor_type in _CPU_ONLY_TYPES

        # CPU-only types or column scatter must go through numpy.
        if is_cpu_only or joint_ids is not None:
            target_device = "cpu" if is_cpu_only else self._device
            np_data = self._to_cpu_numpy(data)
            if joint_ids is not None:
                if is_cpu_only:
                    full = np.zeros(binding.shape, dtype=np.float32)
                    binding.read(full)
                else:
                    scratch = self._get_write_scratch(tensor_type, binding)
                    binding.read(scratch)
                    full = scratch.numpy()
                jids = self._to_cpu_indices(joint_ids, np.intp)
                if env_ids is not None:
                    eids = self._to_cpu_indices(env_ids, np.intp)
                    full[np.ix_(eids, jids)] = np_data.reshape(len(eids), len(jids), *np_data.shape[2:])
                else:
                    full[:, jids] = np_data.reshape(full.shape[0], len(jids), *np_data.shape[2:])
                binding.write(wp.from_numpy(full, dtype=wp.float32, device=target_device))
            elif env_ids is not None:
                if is_cpu_only:
                    full = np.zeros(binding.shape, dtype=np.float32)
                    binding.read(full)
                else:
                    scratch = self._get_write_scratch(tensor_type, binding)
                    binding.read(scratch)
                    full = scratch.numpy()
                eids = self._to_cpu_indices(env_ids, np.intp)
                full[eids] = np_data if np_data.shape[0] == len(eids) else np_data[eids]
                flat = wp.from_numpy(full.astype(np.float32), dtype=wp.float32, device=target_device)
                idx = _ids_gpu if _ids_gpu is not None else self._env_ids_to_gpu_warp(env_ids)
                binding.write(flat, indices=idx)
            else:
                binding.write(wp.from_numpy(np_data.astype(np.float32), dtype=wp.float32, device=target_device))
            return

        # GPU path: data stays on device.
        if env_ids is None and _ids_gpu is None:
            binding.write(self._to_flat_f32(data))
            return

        N, C = binding.shape[0], binding.shape[1]
        src = self._as_gpu_f32_2d(data, C)
        if _ids_gpu is None:
            _ids_gpu = self._env_ids_to_gpu_warp(env_ids)
        K = _ids_gpu.shape[0]
        if src.shape[0] == N:
            binding.write(src, indices=_ids_gpu)
        else:
            scratch = self._get_write_scratch(tensor_type, binding)
            binding.read(scratch)
            wp.launch(
                _scatter_rows_partial,
                dim=(K, C),
                inputs=[scratch, src, _ids_gpu],
                device=self._device,
            )
            binding.write(scratch, indices=_ids_gpu)

    def _write_flat_tensor_mask(self, tensor_type: int, data, env_mask=None, joint_mask=None) -> None:
        """Write a 2-D tensor to a binding, with optional env/joint mask subsetting."""
        if isinstance(data, (int, float)):
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        from isaaclab_ovphysx.tensor_types import _CPU_ONLY_TYPES

        is_cpu_only = tensor_type in _CPU_ONLY_TYPES

        # CPU-only types or column-mask scatter must go through numpy.
        if is_cpu_only or joint_mask is not None:
            target_device = "cpu" if is_cpu_only else self._device
            np_data = self._to_cpu_numpy(data)
            if joint_mask is not None:
                # GPU bindings cannot read into numpy directly; read into GPU
                # scratch first, then pull to CPU for column scatter.
                if is_cpu_only:
                    full = np.zeros(binding.shape, dtype=np.float32)
                    binding.read(full)
                else:
                    scratch = self._get_write_scratch(tensor_type, binding)
                    binding.read(scratch)
                    full = scratch.numpy()
                jmask = self._to_cpu_numpy(joint_mask).astype(bool)
                cols = np.where(jmask)[0]
                if env_mask is not None:
                    emask = self._to_cpu_numpy(env_mask).astype(bool)
                    rows = np.where(emask)[0]
                    full[rows[:, None], cols] = np_data[rows[:, None], cols]
                else:
                    full[:, cols] = np_data[:, cols]
                binding.write(wp.from_numpy(full.astype(np.float32), dtype=wp.float32, device=target_device))
            elif env_mask is not None:
                flat = wp.from_numpy(np_data.astype(np.float32), dtype=wp.float32, device=target_device)
                mask_u8 = wp.from_numpy(
                    self._to_cpu_numpy(env_mask).astype(np.uint8),
                    device=target_device,
                )
                binding.write(flat, mask=mask_u8)
            else:
                binding.write(wp.from_numpy(np_data.astype(np.float32), dtype=wp.float32, device=target_device))
            return

        # GPU path: data stays on device.
        if env_mask is None:
            binding.write(self._to_flat_f32(data))
            return

        # Data is full [N, D], the binding API selects rows via the mask.
        mask_u8 = wp.from_numpy(
            self._to_cpu_numpy(env_mask).astype(np.uint8),
            device=self._device,
        )
        binding.write(self._to_flat_f32(data), mask=mask_u8)

    def _write_friction_column(self, data, env_ids=None, joint_ids=None) -> None:
        """Write static friction coefficient into column 0 of DOF_FRICTION_PROPERTIES [N,D,3]."""
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        if binding is None:
            return
        full = np.zeros(binding.shape, dtype=np.float32)
        binding.read(full)
        if isinstance(data, (int, float)):
            if env_ids is not None and joint_ids is not None:
                eids = self._to_cpu_numpy(env_ids).astype(np.intp)
                jids = self._to_cpu_indices(joint_ids, np.intp)
                full[np.ix_(eids, jids, [0])] = data
            elif env_ids is not None:
                eids = self._to_cpu_numpy(env_ids).astype(np.intp)
                full[eids, :, 0] = data
            elif joint_ids is not None:
                jids = self._to_cpu_indices(joint_ids, np.intp)
                full[:, jids, 0] = data
            else:
                full[..., 0] = data
            binding.write(wp.from_numpy(full.astype(np.float32), dtype=wp.float32, device="cpu"))
            return
        np_data = self._to_cpu_numpy(data)
        if env_ids is not None and joint_ids is not None:
            eids = self._to_cpu_numpy(env_ids).astype(np.intp)
            jids = self._to_cpu_indices(joint_ids, np.intp)
            full[np.ix_(eids, jids, [0])] = np_data.reshape(len(eids), len(jids), 1)
        elif env_ids is not None:
            eids = self._to_cpu_numpy(env_ids).astype(np.intp)
            full[eids, :, 0] = np_data.reshape(len(eids), -1)
        elif joint_ids is not None:
            jids = self._to_cpu_indices(joint_ids, np.intp)
            full[:, jids, 0] = np_data.reshape(full.shape[0], len(jids))
        else:
            full[..., 0] = np_data.reshape(full.shape[0], full.shape[1])
        binding.write(wp.from_numpy(full.astype(np.float32), dtype=wp.float32, device="cpu"))

    def _write_friction_column_mask(self, data, env_mask=None, joint_mask=None) -> None:
        """Write static friction coefficient via mask into column 0 of DOF_FRICTION_PROPERTIES."""
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        if binding is None:
            return
        full = np.zeros(binding.shape, dtype=np.float32)
        binding.read(full)
        if isinstance(data, (int, float)):
            new_col = np.full((full.shape[0], full.shape[1]), data, dtype=np.float32)
        else:
            new_col = self._to_cpu_numpy(data).reshape(full.shape[0], full.shape[1])
        if env_mask is not None:
            emask = self._to_cpu_numpy(env_mask).astype(bool)
            if joint_mask is not None:
                jmask = self._to_cpu_numpy(joint_mask).astype(bool)
                rows = np.where(emask)[0]
                cols = np.where(jmask)[0]
                full[rows[:, None], cols, 0] = new_col[rows[:, None], cols]
            else:
                full[emask, :, 0] = new_col[emask]
        elif joint_mask is not None:
            jmask = self._to_cpu_numpy(joint_mask).astype(bool)
            full[:, jmask, 0] = new_col[:, jmask]
        else:
            full[..., 0] = new_col
        binding.write(wp.from_numpy(full.astype(np.float32), dtype=wp.float32, device="cpu"))

    def _write_joint_subset(self, tensor_type: int, buffer: wp.array, joint_ids: list[int]) -> None:
        """Write a full-width joint buffer into the simulation for an actuator's joints."""
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        if not hasattr(self, "_write_dltensor_cache"):
            self._write_dltensor_cache = {}
        cache_key = (tensor_type, buffer.ptr)
        cached = self._write_dltensor_cache.get(cache_key)
        if cached is None:
            flat = self._to_flat_f32(buffer)
            from ovphysx._dlpack_utils import acquire_dltensor

            dl, keepalive = acquire_dltensor(flat)
            self._write_dltensor_cache[cache_key] = (dl, keepalive, flat)
            cached = self._write_dltensor_cache[cache_key]
        binding.write(cached[0])

    @staticmethod
    def _to_cpu_numpy(data) -> np.ndarray:
        """Convert data (warp, torch, numpy, scalar) to a CPU numpy array."""
        if isinstance(data, wp.array):
            return data.numpy().astype(np.float32)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _to_cpu_indices(data, dtype=np.int32) -> np.ndarray:
        """Convert index array (warp, torch, list, numpy) to CPU numpy int array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(dtype)
        if isinstance(data, wp.array):
            return data.numpy().astype(dtype)
        return np.asarray(data, dtype=dtype)

    def _env_ids_to_gpu_warp(self, env_ids) -> wp.array:
        """Convert env_ids to a GPU int32 warp array, with single-entry caching.

        The cache avoids repeated GPU -> CPU -> GPU round-trips when the same
        ``env_ids`` object is passed to multiple binding writes in a single step
        (e.g. reset writes root_pose, root_vel, joint_pos, joint_vel).  A new
        object identity (``id()``) or shape change invalidates the cache.
        """
        if hasattr(env_ids, "data_ptr"):
            key = (env_ids.data_ptr(), env_ids.shape[0])
        elif isinstance(env_ids, wp.array):
            key = (env_ids.ptr, env_ids.shape[0])
        else:
            key = None

        if key is not None and hasattr(self, "_ids_cache_key") and self._ids_cache_key == key:
            return self._ids_cache_val

        result = wp.array(self._to_cpu_indices(env_ids, np.int32), device=self._device)
        if key is not None:
            self._ids_cache_key = key
            self._ids_cache_val = result
        return result

    def _set_target_into_buffer(self, buffer: wp.array, data, env_ids=None, joint_ids=None) -> None:
        """Set user-provided target data into a warp command buffer.

        For the common case (no index subset), this uses wp.copy to stay on
        the simulation device.  Subset writes (specific env_ids or joint_ids)
        fall back to CPU because warp does not support scatter indexing.
        """
        # Fast path: all-joints shortcut.  When joint_ids covers every joint
        # and env_ids is None, the subset is equivalent to a full copy.
        if joint_ids is not None and env_ids is None:
            n_joints = buffer.shape[1] if len(buffer.shape) > 1 else 1
            if hasattr(joint_ids, "__len__") and len(joint_ids) == n_joints:
                joint_ids = None
        if env_ids is None and joint_ids is None:
            src = self._to_flat_f32(data)
            if isinstance(src, np.ndarray):
                src = wp.from_numpy(src, dtype=wp.float32, device=buffer.device)
            wp.copy(buffer, src)
        else:
            np_data = self._to_cpu_numpy(data)
            buf_np = buffer.numpy()
            env_idx = self._to_cpu_numpy(env_ids).astype(np.intp) if env_ids is not None else None
            jnt_idx = self._to_cpu_numpy(joint_ids).astype(np.intp) if joint_ids is not None else None
            if env_idx is not None and jnt_idx is not None:
                buf_np[np.ix_(env_idx, jnt_idx)] = np_data
            elif env_idx is not None:
                buf_np[env_idx] = np_data
            else:
                buf_np[:, jnt_idx] = np_data
            wp.copy(buffer, wp.from_numpy(buf_np, dtype=wp.float32, device=buffer.device))

    def _set_target_into_buffer_mask(self, buffer: wp.array, data, env_mask=None, joint_mask=None) -> None:
        """Set user-provided target data into a warp command buffer using masks."""
        if env_mask is None:
            src = self._to_flat_f32(data)
            if isinstance(src, np.ndarray):
                src = wp.from_numpy(src, dtype=wp.float32, device=buffer.device)
            wp.copy(buffer, src)
        else:
            np_data = self._to_cpu_numpy(data)
            buf_np = buffer.numpy()
            mask_np = self._to_cpu_numpy(env_mask).astype(bool)
            buf_np[mask_np] = np_data[mask_np]
            wp.copy(buffer, wp.from_numpy(buf_np, dtype=wp.float32, device=buffer.device))

    """
    Internal helpers -- Utilities.
    """

    def _n_envs_index(self, env_ids):
        """Return the number of environments from an env_ids argument."""
        if env_ids is None:
            return self._num_instances
        if isinstance(env_ids, (list, tuple)):
            return len(env_ids)
        return env_ids.shape[0] if hasattr(env_ids, "shape") else len(env_ids)

    def _nft(self):
        """Return the number of fixed tendons (0 if none)."""
        return getattr(self, "_num_fixed_tendons", 0)

    def _nst(self):
        """Return the number of spatial tendons (0 if none)."""
        return getattr(self, "_num_spatial_tendons", 0)

    def _resolve_joint_values(self, pattern_dict: dict[str, float], buffer: wp.array) -> None:
        """Resolve a {pattern: value} dict into a per-joint buffer.

        Builds values on CPU then copies to buffer's device (GPU arrays'
        .numpy() returns a read-only copy, not a writable view).
        """
        buf_np = buffer.numpy()
        modified = False
        for pattern, value in pattern_dict.items():
            for j, name in enumerate(self._joint_names):
                if re.fullmatch(pattern, name):
                    buf_np[:, j] = value
                    modified = True
        if modified:
            wp.copy(buffer, wp.from_numpy(buf_np, dtype=buffer.dtype, device=str(buffer.device)))
