# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.cloner import queue_usd_replication
from isaaclab.physics import PhysicsManager
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.kernels import _body_wrench_to_world
from isaaclab_ovphysx.cloner import queue_ovphysx_replication
from isaaclab_ovphysx.physics import OvPhysxManager

from .articulation_data import ArticulationData
from .kernels import (
    clamp_default_joint_pos_and_update_soft_limits_index,
    clamp_default_joint_pos_and_update_soft_limits_mask,
    update_soft_joint_pos_limits,
    write_joint_friction_data_to_buffer_index,
    write_joint_friction_data_to_buffer_mask,
)

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

    OVPhysX exposes per-tensor-type :class:`ovphysx.TensorBinding` objects rather than a single
    opaque view; binding handles are created eagerly in :meth:`_initialize_impl` and reused across
    reads and writes. CPU-only bindings (mass, CoM, inertia, joint properties, tendon properties)
    are routed through pinned-host staging buffers managed by :class:`ArticulationData`.

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html

    """

    cfg: ArticulationCfg
    """Configuration instance for the articulation."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the articulation."""

    def __init__(self, cfg: ArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        queue_usd_replication(cfg)
        queue_ovphysx_replication(cfg)
        # bindings are populated eagerly in ``_initialize_impl``; the dict
        # also caches any tensor type the user explicitly queries later
        self._bindings: dict[int, Any] = {}

    """
    Properties
    """

    @property
    def data(self) -> ArticulationData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self._is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self._num_joints

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        return self._num_fixed_tendons

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in articulation."""
        return self._num_spatial_tendons

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self._num_bodies

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self._joint_names

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
        return self._body_names

    @property
    def root_view(self) -> dict[int, Any]:
        """Root view for the asset.

        OVPhysX exposes per-tensor-type bindings rather than a single opaque view object
        as used by the PhysX and Newton backends. Callers that need low-level binding
        access should call :meth:`_get_binding` rather than iterating this dict directly.
        For high-level state access (instance counts, prim paths, transforms), use the
        :attr:`num_instances`, :attr:`body_names`, and :attr:`~ArticulationData.root_link_pose_w`
        accessors instead.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._bindings

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.
        """
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are persistent and are applied to the simulation at every step. This is useful to apply forces that
        are constant over a period of time, things like the thrust of a motor for instance.
        """
        return self._permanent_wrench_composer

    """
    Operations.
    """

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Reset the articulation.

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = slice(None)
        # reset external wrenches.
        self._instantaneous_wrench_composer.reset(env_ids, env_mask)
        self._permanent_wrench_composer.reset(env_ids, env_mask)

    def write_data_to_sim(self) -> None:
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        inst = self._instantaneous_wrench_composer
        perm = self._permanent_wrench_composer
        if inst.active or perm.active:
            if inst.active:
                if perm.active:
                    inst.add_raw_buffers_from(perm)
                force_b = inst.out_force_b.warp
                torque_b = inst.out_torque_b.warp
            else:
                force_b = perm.out_force_b.warp
                torque_b = perm.out_torque_b.warp

            # rotate body-frame wrenches into the world frame expected by ``LINK_WRENCH``
            poses = self._data.body_link_pose_w.warp
            wp.launch(
                _body_wrench_to_world,
                dim=(self._num_instances, self._num_bodies),
                inputs=[force_b, torque_b, poses],
                outputs=[self._wrench_buf],
                device=self._device,
            )
            binding = self._get_binding(TT.LINK_WRENCH)
            if binding is not None:
                binding.write(self._wrench_buf)
            inst.reset()

        # apply actuator models
        self._apply_actuator_model()
        # write actions into simulation (zeros are safe when no actuators are active)
        if self._effort_binding is not None:
            self._effort_binding.write(self._effort_write_view)
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:
            if self._pos_target_binding is not None:
                self._pos_target_binding.write(self._pos_target_write_view)
            if self._vel_target_binding is not None:
                self._vel_target_binding.write(self._vel_target_write_view)

    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
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
        return resolve_matching_names(name_keys, self.body_names, preserve_order)

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: list[str] | None = None,
        preserve_order: bool = False,
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
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find fixed tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the
                joint names with fixed tendons.
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
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
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
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.write_root_link_pose_to_sim_mask(root_pose=root_pose, env_mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_pose, (env_ids.shape[0],), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[root_pose, env_ids],
            outputs=[self.data.root_link_pose_w],
            device=self._device,
        )
        # invalidate dependent timestamps: root link pose changes the body
        # kinematics chain, so all body-pose buffers go stale
        self.data._root_com_pose_w.timestamp = -1.0
        self.data._body_link_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        # push cache to the simulation via an indexed write
        binding = self._get_binding(TT.ROOT_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_pose, env_mask_wp],
            outputs=[self.data.root_link_pose_w],
            device=self._device,
        )
        # invalidate dependent timestamps (see :meth:`write_root_link_pose_to_sim_index`)
        self.data._root_com_pose_w.timestamp = -1.0
        self.data._body_link_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        binding = self._get_binding(TT.ROOT_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), mask=env_mask_wp)

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_pose, (env_ids.shape[0],), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[root_pose, self.data.body_com_pose_b, env_ids],
            outputs=[self.data.root_com_pose_w, self.data.root_link_pose_w],
            device=self._device,
        )
        # writing the root CoM pose updates the inferred root link pose, which
        # in turn invalidates the body kinematics chain
        self.data._body_link_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        binding = self._get_binding(TT.ROOT_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_pose, self.data.body_com_pose_b, env_mask_wp],
            outputs=[self.data.root_com_pose_w, self.data.root_link_pose_w],
            device=self._device,
        )
        # invalidate dependent timestamps (see :meth:`write_root_com_pose_to_sim_index`)
        self.data._body_link_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        binding = self._get_binding(TT.ROOT_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), mask=env_mask_wp)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's center of mass rather than the root's frame.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.write_root_com_velocity_to_sim_mask(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's center of mass rather than the root's frame.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_velocity, (env_ids.shape[0],), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[root_velocity, env_ids, self._num_bodies],
            outputs=[self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        # Invalidate dependent root_link_vel timestamp.
        self.data._root_link_vel_w.timestamp = -1.0
        binding = self._get_binding(TT.ROOT_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's center of mass rather than the root's frame.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_velocity, env_mask_wp, self._num_bodies],
            outputs=[self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        self.data._root_link_vel_w.timestamp = -1.0
        binding = self._get_binding(TT.ROOT_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), mask=env_mask_wp)

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's frame rather than the root's center of mass.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_velocity, (env_ids.shape[0],), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_link_velocity_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[
                root_velocity,
                self.data.body_com_pose_b,
                self.data.root_link_pose_w,
                env_ids,
                self._num_bodies,
            ],
            outputs=[self.data.root_link_vel_w, self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        binding = self._get_binding(TT.ROOT_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's frame rather than the root's center of mass.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_link_velocity_to_sim_mask,
            dim=self._num_instances,
            inputs=[
                root_velocity,
                self.data.body_com_pose_b,
                self.data.root_link_pose_w,
                env_mask_wp,
                self._num_bodies,
            ],
            outputs=[self.data.root_link_vel_w, self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        binding = self._get_binding(TT.ROOT_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), mask=env_mask_wp)

    def write_joint_position_to_sim_index(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint positions over selected env / joint indices into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            position: Joint positions [m or rad, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices.  Defaults to None (all joints).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        self.assert_shape_and_dtype(position, (env_ids.shape[0], joint_ids.shape[0]), wp.float32, "position")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[position, env_ids, joint_ids],
            outputs=[self._data._joint_pos_buf.data],
            device=self._device,
        )
        # invalidate body-state buffers so the next read re-fetches FK from the
        # wheel using the new joint positions
        self._data._body_com_vel_w.timestamp = -1.0
        self._data._body_link_vel_w.timestamp = -1.0
        self._data._body_com_pose_b.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0
        self._data._body_link_pose_w.timestamp = -1.0
        self._data._joint_acc.timestamp = -1.0
        binding = self._get_binding(TT.DOF_POSITION)
        binding.write(self._data._joint_pos_buf.data, indices=env_ids)

    def write_joint_position_to_sim_mask(
        self,
        *,
        position: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        joint_mask: wp.array | None = None,
    ) -> None:
        """Set joint positions over selected env / joint masks into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            position: Joint positions [m or rad, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            env_mask: Environment mask.  If None, all instances are updated.  Shape is
                (num_instances,).
            joint_mask: Joint mask.  If None, all joints are updated.  Shape is
                (num_joints,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        self.assert_shape_and_dtype(position, (self._num_instances, self._num_joints), wp.float32, "position")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[position, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_pos_buf.data],
            device=self._device,
        )
        # invalidate body-state buffers (see :meth:`write_joint_position_to_sim_index`)
        self._data._body_com_vel_w.timestamp = -1.0
        self._data._body_link_vel_w.timestamp = -1.0
        self._data._body_com_pose_b.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0
        self._data._body_link_pose_w.timestamp = -1.0
        self._data._joint_acc.timestamp = -1.0
        binding = self._get_binding(TT.DOF_POSITION)
        binding.write(self._data._joint_pos_buf.data, mask=env_mask_wp)

    def write_joint_velocity_to_sim_index(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint velocities over selected env / joint indices into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            velocity: Joint velocities [m/s or rad/s, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices.  Defaults to None (all joints).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        self.assert_shape_and_dtype(velocity, (env_ids.shape[0], joint_ids.shape[0]), wp.float32, "velocity")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[velocity, env_ids, joint_ids],
            outputs=[self._data._joint_vel_buf.data],
            device=self._device,
        )
        # Sync previous_joint_vel to the new values so the next FD step does not
        # produce a spurious acceleration spike.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[velocity, env_ids, joint_ids],
            outputs=[self._data._previous_joint_vel],
            device=self._device,
        )
        self._data._joint_acc.timestamp = -1.0
        binding = self._get_binding(TT.DOF_VELOCITY)
        binding.write(self._data._joint_vel_buf.data, indices=env_ids)

    def write_joint_velocity_to_sim_mask(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        joint_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocities over selected env / joint masks into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            velocity: Joint velocities [m/s or rad/s, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            env_mask: Environment mask.  If None, all instances are updated.  Shape is
                (num_instances,).
            joint_mask: Joint mask.  If None, all joints are updated.  Shape is
                (num_joints,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        self.assert_shape_and_dtype(velocity, (self._num_instances, self._num_joints), wp.float32, "velocity")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[velocity, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_vel_buf.data],
            device=self._device,
        )
        # Sync previous_joint_vel so the next FD step does not produce a spurious spike.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[velocity, env_mask_wp, joint_mask_wp],
            outputs=[self._data._previous_joint_vel],
            device=self._device,
        )
        self._data._joint_acc.timestamp = -1.0
        binding = self._get_binding(TT.DOF_VELOCITY)
        binding.write(self._data._joint_vel_buf.data, mask=env_mask_wp)

    def write_joint_state_to_sim_mask(
        self,
        *,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint positions and velocities over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            position: Joint positions [m or rad, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            velocity: Joint velocities [m/s or rad/s, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask.  If None, all joints are updated.  Shape is
                (num_joints,).
            env_mask: Environment mask.  If None, all instances are updated.  Shape is
                (num_instances,).
        """
        self.write_joint_position_to_sim_mask(position=position, env_mask=env_mask, joint_mask=joint_mask)
        self.write_joint_velocity_to_sim_mask(velocity=velocity, env_mask=env_mask, joint_mask=joint_mask)

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_stiffness_to_sim_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint stiffness over selected env / joint indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            stiffness: Joint stiffness [N/m or N·m/rad, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        stiffness = self._broadcast_scalar_to_2d(stiffness, shape)
        self.assert_shape_and_dtype(stiffness, shape, wp.float32, "stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[stiffness, env_ids, joint_ids],
            outputs=[self._data._joint_stiffness.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_joint_stiffness, self._data._joint_stiffness.data)
        binding = self._get_binding(TT.DOF_STIFFNESS)
        binding.write(self.data._cpu_joint_stiffness, indices=cpu_env_ids)

    def write_joint_stiffness_to_sim_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint stiffness over selected env / joint masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            stiffness: Joint stiffness [N/m or N·m/rad, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask. If None, all joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        stiffness = self._broadcast_scalar_to_2d(stiffness, shape)
        self.assert_shape_and_dtype(stiffness, shape, wp.float32, "stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[stiffness, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_stiffness.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_joint_stiffness, self._data._joint_stiffness.data)
        binding = self._get_binding(TT.DOF_STIFFNESS)
        binding.write(self.data._cpu_joint_stiffness, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_damping_to_sim_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint damping over selected env / joint indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_DAMPING`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            damping: Joint damping [N·s/m or N·m·s/rad, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        damping = self._broadcast_scalar_to_2d(damping, shape)
        self.assert_shape_and_dtype(damping, shape, wp.float32, "damping")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[damping, env_ids, joint_ids],
            outputs=[self._data._joint_damping.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_joint_damping, self._data._joint_damping.data)
        binding = self._get_binding(TT.DOF_DAMPING)
        binding.write(self.data._cpu_joint_damping, indices=cpu_env_ids)

    def write_joint_damping_to_sim_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint damping over selected env / joint masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_DAMPING`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            damping: Joint damping [N·s/m or N·m·s/rad, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask. If None, all joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        damping = self._broadcast_scalar_to_2d(damping, shape)
        self.assert_shape_and_dtype(damping, shape, wp.float32, "damping")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[damping, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_damping.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_joint_damping, self._data._joint_damping.data)
        binding = self._get_binding(TT.DOF_DAMPING)
        binding.write(self.data._cpu_joint_damping, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_position_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Set joint position limits over selected env / joint indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_LIMIT`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limits: Joint position limits ``[lower, upper]``
                [m or rad, depending on joint type]. Either shape
                (len(env_ids), len(joint_ids), 2) with dtype wp.float32, or
                shape (len(env_ids), len(joint_ids)) with dtype wp.vec2f.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
            warn_limit_violation: If True, log a warning when the provided limits
                are inconsistent (lower > upper). Defaults to True.
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        # Position limits cannot be scalar-broadcast (they pair lower/upper);
        # match PhysX which explicitly rejects floats here.
        if isinstance(limits, float):
            raise ValueError("Joint position limits must be a tensor or array, not a float.")
        # Accept both wp.vec2f shape (N, J) and the legacy (N, J, 2) wp.float32
        # form (canonical PhysX/Newton layout uses vec2f).
        if isinstance(limits, wp.array) and limits.dtype == wp.vec2f:
            self.assert_shape_and_dtype(limits, (env_ids.shape[0], joint_ids.shape[0]), wp.vec2f, "limits")
            # Reinterpret the vec2f input as a (N, J, 2) float32 view for the kernel.
            kernel_limits = wp.array(
                ptr=limits.ptr,
                shape=(env_ids.shape[0], joint_ids.shape[0], 2),
                dtype=wp.float32,
                device=str(limits.device),
                copy=False,
            )
        else:
            self.assert_shape_and_dtype(limits, (env_ids.shape[0], joint_ids.shape[0], 2), wp.float32, "limits")
            kernel_limits = limits
        # Scatter [lower, upper] pairs into the vec2f cache buffer.
        wp.launch(
            shared_kernels.write_joint_position_limit_to_buffer_index,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[kernel_limits, env_ids, joint_ids],
            outputs=[self._data._joint_pos_limits.data],
            device=self._device,
        )
        # Clamp default_joint_pos to the new limits and refresh soft_joint_pos_limits.
        clamped_count = wp.zeros(1, dtype=wp.int32, device=self._device)
        wp.launch(
            clamp_default_joint_pos_and_update_soft_limits_index,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                self._data._joint_pos_limits.data,
                env_ids,
                joint_ids,
                self.cfg.soft_joint_pos_limit_factor,
            ],
            outputs=[
                self._data._default_joint_pos,
                self._data._soft_joint_pos_limits,
                clamped_count,
            ],
            device=self._device,
        )
        if clamped_count.numpy()[0] > 0:
            violation_message = (
                "Some default joint positions are outside of the range of the new joint limits. Default joint"
                " positions will be clamped to be within the new joint limits."
            )
            if warn_limit_violation:
                logger.warning(violation_message)
            else:
                logger.info(violation_message)
        # Stage to pinned-host CPU: flatten the vec2f buffer to float32 view.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        flat_src = wp.array(
            ptr=self._data._joint_pos_limits.data.ptr,
            shape=(self._num_instances, self._num_joints, 2),
            dtype=wp.float32,
            device=self._device,
            copy=False,
        )
        wp.copy(self.data._cpu_joint_position_limit, flat_src)
        binding = self._get_binding(TT.DOF_LIMIT)
        binding.write(self.data._cpu_joint_position_limit, indices=cpu_env_ids)

    def write_joint_position_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Set joint position limits over selected env / joint masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_LIMIT`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limits: Joint position limits ``[lower, upper]``
                [m or rad, depending on joint type]. Either shape
                (num_instances, num_joints, 2) with dtype wp.float32, or shape
                (num_instances, num_joints) with dtype wp.vec2f.
            joint_mask: Joint mask. If None, all joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
            warn_limit_violation: If True, log a warning when the provided limits
                are inconsistent (lower > upper). Defaults to True.
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        # Position limits cannot be scalar-broadcast (they pair lower/upper);
        # match PhysX which explicitly rejects floats here.
        if isinstance(limits, float):
            raise ValueError("Joint position limits must be a tensor or array, not a float.")
        # Accept both wp.vec2f shape (N, J) and the legacy (N, J, 2) wp.float32
        # form (canonical PhysX/Newton layout uses vec2f).
        if isinstance(limits, wp.array) and limits.dtype == wp.vec2f:
            self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.vec2f, "limits")
            kernel_limits = wp.array(
                ptr=limits.ptr,
                shape=(self._num_instances, self._num_joints, 2),
                dtype=wp.float32,
                device=str(limits.device),
                copy=False,
            )
        else:
            self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints, 2), wp.float32, "limits")
            kernel_limits = limits
        wp.launch(
            shared_kernels.write_joint_position_limit_to_buffer_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[kernel_limits, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_pos_limits.data],
            device=self._device,
        )
        # Clamp default_joint_pos to the new limits and refresh soft_joint_pos_limits.
        clamped_count = wp.zeros(1, dtype=wp.int32, device=self._device)
        wp.launch(
            clamp_default_joint_pos_and_update_soft_limits_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[
                self._data._joint_pos_limits.data,
                env_mask_wp,
                joint_mask_wp,
                self.cfg.soft_joint_pos_limit_factor,
            ],
            outputs=[
                self._data._default_joint_pos,
                self._data._soft_joint_pos_limits,
                clamped_count,
            ],
            device=self._device,
        )
        if clamped_count.numpy()[0] > 0:
            violation_message = (
                "Some default joint positions are outside of the range of the new joint limits. Default joint"
                " positions will be clamped to be within the new joint limits."
            )
            if warn_limit_violation:
                logger.warning(violation_message)
            else:
                logger.info(violation_message)
        flat_src = wp.array(
            ptr=self._data._joint_pos_limits.data.ptr,
            shape=(self._num_instances, self._num_joints, 2),
            dtype=wp.float32,
            device=self._device,
            copy=False,
        )
        wp.copy(self.data._cpu_joint_position_limit, flat_src)
        binding = self._get_binding(TT.DOF_LIMIT)
        binding.write(self.data._cpu_joint_position_limit, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_velocity_limit_to_sim_index(
        self,
        *,
        limits: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint velocity limits over selected env / joint indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_MAX_VELOCITY`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limits: Joint velocity limits [m/s or rad/s, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        limits = self._broadcast_scalar_to_2d(limits, shape)
        self.assert_shape_and_dtype(limits, shape, wp.float32, "limits")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[limits, env_ids, joint_ids],
            outputs=[self._data._joint_vel_limits.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_joint_velocity_limit, self._data._joint_vel_limits.data)
        binding = self._get_binding(TT.DOF_MAX_VELOCITY)
        binding.write(self.data._cpu_joint_velocity_limit, indices=cpu_env_ids)

    def write_joint_velocity_limit_to_sim_mask(
        self,
        *,
        limits: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity limits over selected env / joint masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_MAX_VELOCITY`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limits: Joint velocity limits [m/s or rad/s, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask. If None, all joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        limits = self._broadcast_scalar_to_2d(limits, shape)
        self.assert_shape_and_dtype(limits, shape, wp.float32, "limits")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[limits, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_vel_limits.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_joint_velocity_limit, self._data._joint_vel_limits.data)
        binding = self._get_binding(TT.DOF_MAX_VELOCITY)
        binding.write(self.data._cpu_joint_velocity_limit, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_effort_limit_to_sim_index(
        self,
        *,
        limits: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint effort limits over selected env / joint indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_MAX_FORCE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limits: Joint effort limits [N or N·m, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        limits = self._broadcast_scalar_to_2d(limits, shape)
        self.assert_shape_and_dtype(limits, shape, wp.float32, "limits")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[limits, env_ids, joint_ids],
            outputs=[self._data._joint_effort_limits.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_joint_effort_limit, self._data._joint_effort_limits.data)
        binding = self._get_binding(TT.DOF_MAX_FORCE)
        binding.write(self.data._cpu_joint_effort_limit, indices=cpu_env_ids)

    def write_joint_effort_limit_to_sim_mask(
        self,
        *,
        limits: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint effort limits over selected env / joint masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_MAX_FORCE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limits: Joint effort limits [N or N·m, depending on joint type].
                May be a scalar :class:`float` (broadcast), or shape
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask. If None, all joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        limits = self._broadcast_scalar_to_2d(limits, shape)
        self.assert_shape_and_dtype(limits, shape, wp.float32, "limits")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[limits, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_effort_limits.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_joint_effort_limit, self._data._joint_effort_limits.data)
        binding = self._get_binding(TT.DOF_MAX_FORCE)
        binding.write(self.data._cpu_joint_effort_limit, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_armature_to_sim_index(
        self,
        *,
        armature: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint armature over selected env / joint indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_ARMATURE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            armature: Joint armature [kg·m²]. May be a scalar :class:`float`
                (broadcast), or shape (len(env_ids), len(joint_ids)) with
                dtype wp.float32.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        armature = self._broadcast_scalar_to_2d(armature, shape)
        self.assert_shape_and_dtype(armature, shape, wp.float32, "armature")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[armature, env_ids, joint_ids],
            outputs=[self._data._joint_armature.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_joint_armature, self._data._joint_armature.data)
        binding = self._get_binding(TT.DOF_ARMATURE)
        binding.write(self.data._cpu_joint_armature, indices=cpu_env_ids)

    def write_joint_armature_to_sim_mask(
        self,
        *,
        armature: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint armature over selected env / joint masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``DOF_ARMATURE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            armature: Joint armature [kg·m²]. May be a scalar :class:`float`
                (broadcast), or shape (num_instances, num_joints) with dtype
                wp.float32.
            joint_mask: Joint mask. If None, all joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        armature = self._broadcast_scalar_to_2d(armature, shape)
        self.assert_shape_and_dtype(armature, shape, wp.float32, "armature")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[armature, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_armature.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_joint_armature, self._data._joint_armature.data)
        binding = self._get_binding(TT.DOF_ARMATURE)
        binding.write(self.data._cpu_joint_armature, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_friction_coefficient_to_sim_index(
        self,
        *,
        joint_friction_coeff: float | torch.Tensor | wp.array,
        joint_dynamic_friction_coeff: float | torch.Tensor | wp.array | None = None,
        joint_viscous_friction_coeff: float | torch.Tensor | wp.array | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        r"""Write joint friction coefficients over selected env / joint indices into the simulation.

        Mirrors :meth:`isaaclab_physx.assets.Articulation.write_joint_friction_coefficient_to_sim_index`:
        Coulomb (static & dynamic) friction with an optional viscous term.  Any of the three
        components can be left unset by passing ``None``; the corresponding slot in the
        combined ``DOF_FRICTION_PROPERTIES`` ``(N, J, 3)`` binding is preserved.

        ``DOF_FRICTION_PROPERTIES`` is a CPU-only OVPhysX binding, so the
        write is routed through pinned-host staging.

        .. note::
            This method expects partial data.  Each component, if provided,
            may be a scalar :class:`float` (broadcast to
            ``(len(env_ids), len(joint_ids))``) or a 2D tensor / warp array.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            joint_friction_coeff: Static friction coefficient :math:`\mu_s` [dimensionless].
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient
                :math:`\mu_d`.  If ``None``, the dynamic component is preserved.
            joint_viscous_friction_coeff: Viscous friction coefficient :math:`c_v`.
                If ``None``, the viscous component is preserved.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        joint_friction_coeff = self._broadcast_scalar_to_2d(joint_friction_coeff, shape)
        if joint_dynamic_friction_coeff is not None:
            joint_dynamic_friction_coeff = self._broadcast_scalar_to_2d(joint_dynamic_friction_coeff, shape)
        if joint_viscous_friction_coeff is not None:
            joint_viscous_friction_coeff = self._broadcast_scalar_to_2d(joint_viscous_friction_coeff, shape)
        self.assert_shape_and_dtype(joint_friction_coeff, shape, wp.float32, "joint_friction_coeff")
        if joint_dynamic_friction_coeff is not None:
            self.assert_shape_and_dtype(joint_dynamic_friction_coeff, shape, wp.float32, "joint_dynamic_friction_coeff")
        if joint_viscous_friction_coeff is not None:
            self.assert_shape_and_dtype(joint_viscous_friction_coeff, shape, wp.float32, "joint_viscous_friction_coeff")
        # refresh the combined (N, J, 3) buffer from the binding so unchanged
        # components are preserved on the round-trip
        self._data._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._data._joint_friction_props_buf)
        wp.launch(
            write_joint_friction_data_to_buffer_index,
            dim=shape,
            inputs=[
                joint_friction_coeff,
                joint_dynamic_friction_coeff,
                joint_viscous_friction_coeff,
                env_ids,
                joint_ids,
            ],
            outputs=[self._data._joint_friction_props_buf.data],
            device=self._device,
        )
        # Stage the combined (N, J, 3) buffer to pinned-host CPU and write to the binding.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        cpu_friction = self._data._stage_to_pinned_cpu(
            TT.DOF_FRICTION_PROPERTIES, "write", self._data._joint_friction_props_buf.data
        )
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        binding.write(cpu_friction, indices=cpu_env_ids)

    def write_joint_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_friction_coeff: float | torch.Tensor | wp.array,
        joint_dynamic_friction_coeff: float | torch.Tensor | wp.array | None = None,
        joint_viscous_friction_coeff: float | torch.Tensor | wp.array | None = None,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        r"""Mask variant of :meth:`write_joint_friction_coefficient_to_sim_index`.

        Args:
            joint_friction_coeff: Static friction coefficient :math:`\mu_s`. Full data,
                shape ``(num_instances, num_joints)``.  May be a scalar :class:`float`.
            joint_dynamic_friction_coeff: Dynamic friction.  ``None`` to preserve.
            joint_viscous_friction_coeff: Viscous friction.  ``None`` to preserve.
            joint_mask: Joint mask. If None, all joints are updated.
            env_mask: Environment mask. If None, all instances are updated.
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        joint_friction_coeff = self._broadcast_scalar_to_2d(joint_friction_coeff, shape)
        if joint_dynamic_friction_coeff is not None:
            joint_dynamic_friction_coeff = self._broadcast_scalar_to_2d(joint_dynamic_friction_coeff, shape)
        if joint_viscous_friction_coeff is not None:
            joint_viscous_friction_coeff = self._broadcast_scalar_to_2d(joint_viscous_friction_coeff, shape)
        self.assert_shape_and_dtype(joint_friction_coeff, shape, wp.float32, "joint_friction_coeff")
        if joint_dynamic_friction_coeff is not None:
            self.assert_shape_and_dtype(joint_dynamic_friction_coeff, shape, wp.float32, "joint_dynamic_friction_coeff")
        if joint_viscous_friction_coeff is not None:
            self.assert_shape_and_dtype(joint_viscous_friction_coeff, shape, wp.float32, "joint_viscous_friction_coeff")
        # refresh the (N, J, 3) buffer first (see ``_index`` variant)
        self._data._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._data._joint_friction_props_buf)
        wp.launch(
            write_joint_friction_data_to_buffer_mask,
            dim=shape,
            inputs=[
                joint_friction_coeff,
                joint_dynamic_friction_coeff,
                joint_viscous_friction_coeff,
                env_mask_wp,
                joint_mask_wp,
            ],
            outputs=[self._data._joint_friction_props_buf.data],
            device=self._device,
        )
        cpu_friction = self._data._stage_to_pinned_cpu(
            TT.DOF_FRICTION_PROPERTIES, "write", self._data._joint_friction_props_buf.data
        )
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        binding.write(cpu_friction, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_dynamic_friction_coefficient_to_sim_index(
        self,
        *,
        joint_dynamic_friction_coeff: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        r"""Write joint dynamic friction coefficients over selected env / joint indices into the simulation.

        Mirrors :meth:`isaaclab_physx.assets.Articulation.write_joint_dynamic_friction_coefficient_to_sim_index`:
        updates only the dynamic (Coulomb) slot of the combined ``DOF_FRICTION_PROPERTIES`` ``(N, J, 3)``
        binding; the static and viscous components are preserved.

        ``DOF_FRICTION_PROPERTIES`` is a CPU-only OVPhysX binding, so the
        write is routed through pinned-host staging.

        .. note::
            This method expects partial data.  ``joint_dynamic_friction_coeff`` may be a
            scalar :class:`float` (broadcast to ``(len(env_ids), len(joint_ids))``) or a
            2D tensor / warp array.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient
                :math:`\mu_d` [dimensionless]. Shape is ``(len(env_ids), len(joint_ids))``
                with dtype wp.float32, or a scalar that is broadcast.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        joint_dynamic_friction_coeff = self._broadcast_scalar_to_2d(joint_dynamic_friction_coeff, shape)
        self.assert_shape_and_dtype(joint_dynamic_friction_coeff, shape, wp.float32, "joint_dynamic_friction_coeff")
        # refresh the combined (N, J, 3) buffer from the binding so unchanged
        # components are preserved on the round-trip
        self._data._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._data._joint_friction_props_buf)
        wp.launch(
            write_joint_friction_data_to_buffer_index,
            dim=shape,
            inputs=[
                None,  # in_static — preserved
                joint_dynamic_friction_coeff,
                None,  # in_viscous — preserved
                env_ids,
                joint_ids,
            ],
            outputs=[self._data._joint_friction_props_buf.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        cpu_friction = self._data._stage_to_pinned_cpu(
            TT.DOF_FRICTION_PROPERTIES, "write", self._data._joint_friction_props_buf.data
        )
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        binding.write(cpu_friction, indices=cpu_env_ids)

    def write_joint_dynamic_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_dynamic_friction_coeff: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        r"""Mask variant of :meth:`write_joint_dynamic_friction_coefficient_to_sim_index`.

        Updates only the dynamic (Coulomb) slot of the combined ``DOF_FRICTION_PROPERTIES``
        ``(N, J, 3)`` binding; the static and viscous components are preserved.

        Args:
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient
                :math:`\mu_d` [dimensionless]. Full data, shape
                ``(num_instances, num_joints)``.  May be a scalar :class:`float`.
            joint_mask: Joint mask. If None, all joints are updated.
            env_mask: Environment mask. If None, all instances are updated.
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        joint_dynamic_friction_coeff = self._broadcast_scalar_to_2d(joint_dynamic_friction_coeff, shape)
        self.assert_shape_and_dtype(joint_dynamic_friction_coeff, shape, wp.float32, "joint_dynamic_friction_coeff")
        # refresh the (N, J, 3) buffer first (see ``_index`` variant)
        self._data._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._data._joint_friction_props_buf)
        wp.launch(
            write_joint_friction_data_to_buffer_mask,
            dim=shape,
            inputs=[
                None,  # in_static — preserved
                joint_dynamic_friction_coeff,
                None,  # in_viscous — preserved
                env_mask_wp,
                joint_mask_wp,
            ],
            outputs=[self._data._joint_friction_props_buf.data],
            device=self._device,
        )
        cpu_friction = self._data._stage_to_pinned_cpu(
            TT.DOF_FRICTION_PROPERTIES, "write", self._data._joint_friction_props_buf.data
        )
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        binding.write(cpu_friction, mask=self._get_cpu_env_mask(env_mask_wp))

    def write_joint_viscous_friction_coefficient_to_sim_index(
        self,
        *,
        joint_viscous_friction_coeff: float | torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        r"""Write joint viscous friction coefficients over selected env / joint indices into the simulation.

        Mirrors :meth:`isaaclab_physx.assets.Articulation.write_joint_viscous_friction_coefficient_to_sim_index`:
        updates only the viscous slot of the combined ``DOF_FRICTION_PROPERTIES`` ``(N, J, 3)``
        binding; the static and dynamic components are preserved.

        ``DOF_FRICTION_PROPERTIES`` is a CPU-only OVPhysX binding, so the
        write is routed through pinned-host staging.

        .. note::
            This method expects partial data.  ``joint_viscous_friction_coeff`` may be a
            scalar :class:`float` (broadcast to ``(len(env_ids), len(joint_ids))``) or a
            2D tensor / warp array.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            joint_viscous_friction_coeff: Viscous friction coefficient
                :math:`c_v` [N·m·s/rad or N·s/m, depending on joint type].
                Shape is ``(len(env_ids), len(joint_ids))`` with dtype wp.float32, or
                a scalar that is broadcast.
            joint_ids: Joint indices. Defaults to None (all joints).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        shape = (env_ids.shape[0], joint_ids.shape[0])
        joint_viscous_friction_coeff = self._broadcast_scalar_to_2d(joint_viscous_friction_coeff, shape)
        self.assert_shape_and_dtype(joint_viscous_friction_coeff, shape, wp.float32, "joint_viscous_friction_coeff")
        # refresh the combined (N, J, 3) buffer from the binding so unchanged
        # components are preserved on the round-trip
        self._data._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._data._joint_friction_props_buf)
        wp.launch(
            write_joint_friction_data_to_buffer_index,
            dim=shape,
            inputs=[
                None,  # in_static — preserved
                None,  # in_dynamic — preserved
                joint_viscous_friction_coeff,
                env_ids,
                joint_ids,
            ],
            outputs=[self._data._joint_friction_props_buf.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        cpu_friction = self._data._stage_to_pinned_cpu(
            TT.DOF_FRICTION_PROPERTIES, "write", self._data._joint_friction_props_buf.data
        )
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        binding.write(cpu_friction, indices=cpu_env_ids)

    def write_joint_viscous_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_viscous_friction_coeff: float | torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        r"""Mask variant of :meth:`write_joint_viscous_friction_coefficient_to_sim_index`.

        Updates only the viscous slot of the combined ``DOF_FRICTION_PROPERTIES``
        ``(N, J, 3)`` binding; the static and dynamic components are preserved.

        Args:
            joint_viscous_friction_coeff: Viscous friction coefficient
                :math:`c_v` [N·m·s/rad or N·s/m, depending on joint type].
                Full data, shape ``(num_instances, num_joints)``.  May be a
                scalar :class:`float`.
            joint_mask: Joint mask. If None, all joints are updated.
            env_mask: Environment mask. If None, all instances are updated.
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        shape = (self._num_instances, self._num_joints)
        joint_viscous_friction_coeff = self._broadcast_scalar_to_2d(joint_viscous_friction_coeff, shape)
        self.assert_shape_and_dtype(joint_viscous_friction_coeff, shape, wp.float32, "joint_viscous_friction_coeff")
        # refresh the (N, J, 3) buffer first (see ``_index`` variant)
        self._data._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._data._joint_friction_props_buf)
        wp.launch(
            write_joint_friction_data_to_buffer_mask,
            dim=shape,
            inputs=[
                None,  # in_static — preserved
                None,  # in_dynamic — preserved
                joint_viscous_friction_coeff,
                env_mask_wp,
                joint_mask_wp,
            ],
            outputs=[self._data._joint_friction_props_buf.data],
            device=self._device,
        )
        cpu_friction = self._data._stage_to_pinned_cpu(
            TT.DOF_FRICTION_PROPERTIES, "write", self._data._joint_friction_props_buf.data
        )
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        binding.write(cpu_friction, mask=self._get_cpu_env_mask(env_mask_wp))

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set body masses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_MASS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized
            implementations.  Performance is similar for both.  However, to
            allow graphed pipelines, the mask method must be used.

        Args:
            masses: Body masses [kg].  Shape is (len(env_ids), len(body_ids))
                with dtype wp.float32.
            body_ids: Body indices.  Defaults to None (all bodies).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(masses, (env_ids.shape[0], body_ids.shape[0]), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[masses, env_ids, body_ids],
            outputs=[self._data._body_mass.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_mass, self._data._body_mass.data)
        binding = self._get_binding(TT.BODY_MASS)
        binding.write(self.data._cpu_body_mass, indices=cpu_env_ids)

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set body masses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_MASS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized
            implementations.  Performance is similar for both.  However, to
            allow graphed pipelines, the mask method must be used.

        Args:
            masses: Body masses [kg].  Shape is (num_instances, num_bodies)
                with dtype wp.float32.
            body_mask: Body mask.  If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask.  If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        body_mask_wp = self._resolve_body_mask(body_mask)
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[masses, env_mask_wp, body_mask_wp],
            outputs=[self._data._body_mass.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_body_mass, self._data._body_mass.data)
        binding = self._get_binding(TT.BODY_MASS)
        binding.write(self.data._cpu_body_mass, mask=self._get_cpu_env_mask(env_mask_wp))

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set body center-of-mass poses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_COM_POSE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized
            implementations.  Performance is similar for both.  However, to
            allow graphed pipelines, the mask method must be used.

        Args:
            coms: Body center-of-mass poses [m, quaternion (w, x, y, z)].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices.  Defaults to None (all bodies).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(coms, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[coms, env_ids, body_ids],
            outputs=[self._data._body_com_pose_b.data],
            device=self._device,
        )
        # Invalidate derived buffers that depend on body_com_pose_b.
        self.data._root_com_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_coms, self._data._body_com_pose_b.data)
        binding = self._get_binding(TT.BODY_COM_POSE)
        binding.write(self.data._cpu_body_coms, indices=cpu_env_ids)

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set body center-of-mass poses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_COM_POSE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized
            implementations.  Performance is similar for both.  However, to
            allow graphed pipelines, the mask method must be used.

        Args:
            coms: Body center-of-mass poses [m, quaternion (w, x, y, z)].
                Shape is (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask.  If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask.  If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        body_mask_wp = self._resolve_body_mask(body_mask)
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[coms, env_mask_wp, body_mask_wp],
            outputs=[self._data._body_com_pose_b.data],
            device=self._device,
        )
        # Invalidate derived buffers that depend on body_com_pose_b.
        self.data._root_com_pose_w.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        wp.copy(self.data._cpu_body_coms, self._data._body_com_pose_b.data)
        binding = self._get_binding(TT.BODY_COM_POSE)
        binding.write(self.data._cpu_body_coms, mask=self._get_cpu_env_mask(env_mask_wp))

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set body inertia tensors over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_INERTIA`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized
            implementations.  Performance is similar for both.  However, to
            allow graphed pipelines, the mask method must be used.

        Args:
            inertias: Body inertia tensors [kg·m²].  Shape is
                (len(env_ids), len(body_ids), 9) with dtype wp.float32.
            body_ids: Body indices.  Defaults to None (all bodies).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(inertias, (env_ids.shape[0], body_ids.shape[0], 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[inertias, env_ids, body_ids],
            outputs=[self._data._body_inertia.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_inertia, self._data._body_inertia.data)
        binding = self._get_binding(TT.BODY_INERTIA)
        binding.write(self.data._cpu_body_inertia, indices=cpu_env_ids)

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set body inertia tensors over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_INERTIA`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized
            implementations.  Performance is similar for both.  However, to
            allow graphed pipelines, the mask method must be used.

        Args:
            inertias: Body inertia tensors [kg·m²].  Shape is
                (num_instances, num_bodies, 9) with dtype wp.float32.
            body_mask: Body mask.  If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask.  If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        body_mask_wp = self._resolve_body_mask(body_mask)
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[inertias, env_mask_wp, body_mask_wp],
            outputs=[self._data._body_inertia.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_body_inertia, self._data._body_inertia.data)
        binding = self._get_binding(TT.BODY_INERTIA)
        binding.write(self.data._cpu_body_inertia, mask=self._get_cpu_env_mask(env_mask_wp))

    def set_joint_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation.  It only fills the
        buffers with the desired values.  To apply the joint targets, call
        :meth:`write_data_to_sim`.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            target: Joint position targets [m or rad, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices.  Defaults to None (all joints).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        self.assert_shape_and_dtype(target, (env_ids.shape[0], joint_ids.shape[0]), wp.float32, "target")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids],
            outputs=[self._data._joint_pos_target],
            device=self._device,
        )
        binding = self._get_binding(TT.DOF_POSITION_TARGET)
        binding.write(self._data._joint_pos_target, indices=env_ids)

    def set_joint_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            target: Joint position targets [m or rad, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask.  If None, all joints are updated.  Shape is (num_joints,).
            env_mask: Environment mask.  If None, all instances are updated.  Shape is
                (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        self.assert_shape_and_dtype(target, (self._num_instances, self._num_joints), wp.float32, "target")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[target, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_pos_target],
            device=self._device,
        )
        binding = self._get_binding(TT.DOF_POSITION_TARGET)
        binding.write(self._data._joint_pos_target, mask=env_mask_wp)

    def set_joint_velocity_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation.  It only fills the
        buffers with the desired values.  To apply the joint targets, call
        :meth:`write_data_to_sim`.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            target: Joint velocity targets [m/s or rad/s, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices.  Defaults to None (all joints).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        self.assert_shape_and_dtype(target, (env_ids.shape[0], joint_ids.shape[0]), wp.float32, "target")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids],
            outputs=[self._data._joint_vel_target],
            device=self._device,
        )
        binding = self._get_binding(TT.DOF_VELOCITY_TARGET)
        binding.write(self._data._joint_vel_target, indices=env_ids)

    def set_joint_velocity_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            target: Joint velocity targets [m/s or rad/s, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask.  If None, all joints are updated.  Shape is (num_joints,).
            env_mask: Environment mask.  If None, all instances are updated.  Shape is
                (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        self.assert_shape_and_dtype(target, (self._num_instances, self._num_joints), wp.float32, "target")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[target, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_vel_target],
            device=self._device,
        )
        binding = self._get_binding(TT.DOF_VELOCITY_TARGET)
        binding.write(self._data._joint_vel_target, mask=env_mask_wp)

    def set_joint_effort_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint effort targets into internal buffers using indices.

        This function does not apply the joint targets to the simulation.  It only fills the
        buffers with the desired values.  To apply the joint targets, call
        :meth:`write_data_to_sim`.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            target: Joint effort targets [N or N·m, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices.  Defaults to None (all joints).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)
        self.assert_shape_and_dtype(target, (env_ids.shape[0], joint_ids.shape[0]), wp.float32, "target")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids],
            outputs=[self._data._joint_effort_target],
            device=self._device,
        )
        binding = self._get_binding(TT.DOF_ACTUATION_FORCE)
        binding.write(self._data._joint_effort_target, indices=env_ids)

    def set_joint_effort_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint effort targets into internal buffers using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            target: Joint effort targets [N or N·m, depending on joint type].  Shape is
                (num_instances, num_joints) with dtype wp.float32.
            joint_mask: Joint mask.  If None, all joints are updated.  Shape is (num_joints,).
            env_mask: Environment mask.  If None, all instances are updated.  Shape is
                (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        joint_mask_wp = self._resolve_joint_mask(joint_mask)
        self.assert_shape_and_dtype(target, (self._num_instances, self._num_joints), wp.float32, "target")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_joints),
            inputs=[target, env_mask_wp, joint_mask_wp],
            outputs=[self._data._joint_effort_target],
            device=self._device,
        )
        binding = self._get_binding(TT.DOF_ACTUATION_FORCE)
        binding.write(self._data._joint_effort_target, mask=env_mask_wp)

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
        """Set fixed-tendon stiffness over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            stiffness: Fixed-tendon stiffness [N/m]. May be a scalar
                :class:`float` (broadcast), or shape
                (len(env_ids), len(fixed_tendon_ids)) with dtype wp.float32.
            fixed_tendon_ids: Fixed-tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        shape = (env_ids.shape[0], tendon_ids.shape[0])
        stiffness = self._broadcast_scalar_to_2d(stiffness, shape)
        self.assert_shape_and_dtype(stiffness, shape, wp.float32, "stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[stiffness, env_ids, tendon_ids],
            outputs=[self._data._fixed_tendon_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_STIFFNESS)
        binding.write(self._data._fixed_tendon_stiffness.data, indices=env_ids)

    def set_fixed_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed-tendon stiffness over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            stiffness: Fixed-tendon stiffness [N/m]. May be a scalar
                :class:`float` (broadcast), or shape
                (num_instances, num_fixed_tendons) with dtype wp.float32.
            fixed_tendon_mask: Fixed-tendon mask. If None, all fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_fixed_tendon_mask(fixed_tendon_mask)
        shape = (self._num_instances, self._num_fixed_tendons)
        stiffness = self._broadcast_scalar_to_2d(stiffness, shape)
        self.assert_shape_and_dtype(stiffness, shape, wp.float32, "stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[stiffness, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._fixed_tendon_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_STIFFNESS)
        binding.write(self._data._fixed_tendon_stiffness.data, mask=env_mask_wp)

    def set_fixed_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed-tendon damping over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_DAMPING`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            damping: Fixed-tendon damping [N·s/m]. May be a scalar :class:`float`
                (broadcast), or shape (len(env_ids), len(fixed_tendon_ids)) with
                dtype wp.float32.
            fixed_tendon_ids: Fixed-tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        shape = (env_ids.shape[0], tendon_ids.shape[0])
        damping = self._broadcast_scalar_to_2d(damping, shape)
        self.assert_shape_and_dtype(damping, shape, wp.float32, "damping")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[damping, env_ids, tendon_ids],
            outputs=[self._data._fixed_tendon_damping.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_DAMPING)
        binding.write(self._data._fixed_tendon_damping.data, indices=env_ids)

    def set_fixed_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed-tendon damping over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_DAMPING`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            damping: Fixed-tendon damping [N·s/m]. May be a scalar :class:`float`
                (broadcast), or shape (num_instances, num_fixed_tendons) with
                dtype wp.float32.
            fixed_tendon_mask: Fixed-tendon mask. If None, all fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_fixed_tendon_mask(fixed_tendon_mask)
        shape = (self._num_instances, self._num_fixed_tendons)
        damping = self._broadcast_scalar_to_2d(damping, shape)
        self.assert_shape_and_dtype(damping, shape, wp.float32, "damping")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[damping, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._fixed_tendon_damping.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_DAMPING)
        binding.write(self._data._fixed_tendon_damping.data, mask=env_mask_wp)

    def set_fixed_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed-tendon limit stiffness over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_LIMIT_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limit_stiffness: Fixed-tendon limit stiffness [N/m]. May be a
                scalar :class:`float` (broadcast), or shape
                (len(env_ids), len(fixed_tendon_ids)) with dtype wp.float32.
            fixed_tendon_ids: Fixed-tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        shape = (env_ids.shape[0], tendon_ids.shape[0])
        limit_stiffness = self._broadcast_scalar_to_2d(limit_stiffness, shape)
        self.assert_shape_and_dtype(limit_stiffness, shape, wp.float32, "limit_stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[limit_stiffness, env_ids, tendon_ids],
            outputs=[self._data._fixed_tendon_limit_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_LIMIT_STIFFNESS)
        binding.write(self._data._fixed_tendon_limit_stiffness.data, indices=env_ids)

    def set_fixed_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed-tendon limit stiffness over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_LIMIT_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limit_stiffness: Fixed-tendon limit stiffness [N/m]. May be a
                scalar :class:`float` (broadcast), or shape
                (num_instances, num_fixed_tendons) with dtype wp.float32.
            fixed_tendon_mask: Fixed-tendon mask. If None, all fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_fixed_tendon_mask(fixed_tendon_mask)
        shape = (self._num_instances, self._num_fixed_tendons)
        limit_stiffness = self._broadcast_scalar_to_2d(limit_stiffness, shape)
        self.assert_shape_and_dtype(limit_stiffness, shape, wp.float32, "limit_stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[limit_stiffness, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._fixed_tendon_limit_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_LIMIT_STIFFNESS)
        binding.write(self._data._fixed_tendon_limit_stiffness.data, mask=env_mask_wp)

    def set_fixed_tendon_position_limit_index(
        self,
        *,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed-tendon position limits over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_LIMIT`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limit: Fixed-tendon position limits ``[lower, upper]`` [m].
                Shape is (len(env_ids), len(fixed_tendon_ids), 2) with dtype wp.float32.
            fixed_tendon_ids: Fixed-tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        self.assert_shape_and_dtype(limit, (env_ids.shape[0], tendon_ids.shape[0], 2), wp.float32, "limit")
        # Scatter [lower, upper] pairs into the vec2f cache buffer.
        wp.launch(
            shared_kernels.write_joint_position_limit_to_buffer_index,
            dim=(env_ids.shape[0], tendon_ids.shape[0]),
            inputs=[limit, env_ids, tendon_ids],
            outputs=[self._data._fixed_tendon_pos_limits.data],
            device=self._device,
        )
        # reinterpret the vec2f buffer as a (N, T, 2) float32 view for the binding
        flat_src = wp.array(
            ptr=self._data._fixed_tendon_pos_limits.data.ptr,
            shape=(self._num_instances, self._num_fixed_tendons, 2),
            dtype=wp.float32,
            device=self._device,
            copy=False,
        )
        binding = self._get_binding(TT.FIXED_TENDON_LIMIT)
        binding.write(flat_src, indices=env_ids)

    def set_fixed_tendon_position_limit_mask(
        self,
        *,
        limit: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed-tendon position limits over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_LIMIT`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limit: Fixed-tendon position limits ``[lower, upper]`` [m].
                Shape is (num_instances, num_fixed_tendons, 2) with dtype wp.float32.
            fixed_tendon_mask: Fixed-tendon mask. If None, all fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_fixed_tendon_mask(fixed_tendon_mask)
        self.assert_shape_and_dtype(limit, (self._num_instances, self._num_fixed_tendons, 2), wp.float32, "limit")
        wp.launch(
            shared_kernels.write_joint_position_limit_to_buffer_mask,
            dim=(self._num_instances, self._num_fixed_tendons),
            inputs=[limit, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._fixed_tendon_pos_limits.data],
            device=self._device,
        )
        flat_src = wp.array(
            ptr=self._data._fixed_tendon_pos_limits.data.ptr,
            shape=(self._num_instances, self._num_fixed_tendons, 2),
            dtype=wp.float32,
            device=self._device,
            copy=False,
        )
        binding = self._get_binding(TT.FIXED_TENDON_LIMIT)
        binding.write(flat_src, mask=env_mask_wp)

    def set_fixed_tendon_rest_length_index(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed-tendon rest lengths over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_REST_LENGTH`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            rest_length: Fixed-tendon rest lengths [m]. May be a scalar
                :class:`float` (broadcast), or shape
                (len(env_ids), len(fixed_tendon_ids)) with dtype wp.float32.
            fixed_tendon_ids: Fixed-tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        shape = (env_ids.shape[0], tendon_ids.shape[0])
        rest_length = self._broadcast_scalar_to_2d(rest_length, shape)
        self.assert_shape_and_dtype(rest_length, shape, wp.float32, "rest_length")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[rest_length, env_ids, tendon_ids],
            outputs=[self._data._fixed_tendon_rest_length.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_REST_LENGTH)
        binding.write(self._data._fixed_tendon_rest_length.data, indices=env_ids)

    def set_fixed_tendon_rest_length_mask(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed-tendon rest lengths over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_REST_LENGTH`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            rest_length: Fixed-tendon rest lengths [m]. May be a scalar
                :class:`float` (broadcast), or shape
                (num_instances, num_fixed_tendons) with dtype wp.float32.
            fixed_tendon_mask: Fixed-tendon mask. If None, all fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_fixed_tendon_mask(fixed_tendon_mask)
        shape = (self._num_instances, self._num_fixed_tendons)
        rest_length = self._broadcast_scalar_to_2d(rest_length, shape)
        self.assert_shape_and_dtype(rest_length, shape, wp.float32, "rest_length")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[rest_length, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._fixed_tendon_rest_length.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_REST_LENGTH)
        binding.write(self._data._fixed_tendon_rest_length.data, mask=env_mask_wp)

    def set_fixed_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed-tendon offsets over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_OFFSET`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            offset: Fixed-tendon offsets [m]. May be a scalar :class:`float`
                (broadcast), or shape (len(env_ids), len(fixed_tendon_ids))
                with dtype wp.float32.
            fixed_tendon_ids: Fixed-tendon indices. Defaults to None (all fixed tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_fixed_tendon_ids(fixed_tendon_ids)
        shape = (env_ids.shape[0], tendon_ids.shape[0])
        offset = self._broadcast_scalar_to_2d(offset, shape)
        self.assert_shape_and_dtype(offset, shape, wp.float32, "offset")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=shape,
            inputs=[offset, env_ids, tendon_ids],
            outputs=[self._data._fixed_tendon_offset.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_OFFSET)
        binding.write(self._data._fixed_tendon_offset.data, indices=env_ids)

    def set_fixed_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed-tendon offsets over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``FIXED_TENDON_OFFSET`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            offset: Fixed-tendon offsets [m]. May be a scalar :class:`float`
                (broadcast), or shape (num_instances, num_fixed_tendons) with
                dtype wp.float32.
            fixed_tendon_mask: Fixed-tendon mask. If None, all fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_fixed_tendon_mask(fixed_tendon_mask)
        shape = (self._num_instances, self._num_fixed_tendons)
        offset = self._broadcast_scalar_to_2d(offset, shape)
        self.assert_shape_and_dtype(offset, shape, wp.float32, "offset")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[offset, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._fixed_tendon_offset.data],
            device=self._device,
        )
        binding = self._get_binding(TT.FIXED_TENDON_OFFSET)
        binding.write(self._data._fixed_tendon_offset.data, mask=env_mask_wp)

    def write_fixed_tendon_properties_to_sim_index(
        self,
        *,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Push the cached fixed-tendon properties to the simulation in a single batch.

        PhysX exposes a single ``root_view.set_fixed_tendon_properties`` that writes all
        six tendon property buffers at once. OVPhysX has no such batch setter, so this
        method writes each ``FIXED_TENDON_*`` binding individually from the matching
        ``self._data._fixed_tendon_*`` buffer.

        .. note::
            Only env indices apply to the simulation write; ``fixed_tendon_ids`` is
            accepted for API parity with PhysX but is unused (the simulation
            writes all tendons of the selected envs).

        Args:
            fixed_tendon_ids: Accepted for PhysX API parity; ignored.
            env_ids: Environment indices. If None, all environments are written.
        """
        env_ids = self._resolve_env_ids(env_ids)
        for tt, buf in (
            (TT.FIXED_TENDON_STIFFNESS, self._data._fixed_tendon_stiffness),
            (TT.FIXED_TENDON_DAMPING, self._data._fixed_tendon_damping),
            (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._data._fixed_tendon_limit_stiffness),
            (TT.FIXED_TENDON_REST_LENGTH, self._data._fixed_tendon_rest_length),
            (TT.FIXED_TENDON_OFFSET, self._data._fixed_tendon_offset),
        ):
            binding = self._get_binding(tt)
            if binding is not None:
                binding.write(buf.data, indices=env_ids)
        # Position-limit binding consumes a flat (N, T, 2) float32 view.
        binding = self._get_binding(TT.FIXED_TENDON_LIMIT)
        if binding is not None:
            flat_src = wp.array(
                ptr=self._data._fixed_tendon_pos_limits.data.ptr,
                shape=(self._num_instances, self._num_fixed_tendons, 2),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            binding.write(flat_src, indices=env_ids)

    def write_fixed_tendon_properties_to_sim_mask(
        self,
        *,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Mask variant of :meth:`write_fixed_tendon_properties_to_sim_index`.

        Args:
            fixed_tendon_mask: Accepted for PhysX API parity; ignored.
            env_mask: Environment mask.  If None, all environments are written.
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        for tt, buf in (
            (TT.FIXED_TENDON_STIFFNESS, self._data._fixed_tendon_stiffness),
            (TT.FIXED_TENDON_DAMPING, self._data._fixed_tendon_damping),
            (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._data._fixed_tendon_limit_stiffness),
            (TT.FIXED_TENDON_REST_LENGTH, self._data._fixed_tendon_rest_length),
            (TT.FIXED_TENDON_OFFSET, self._data._fixed_tendon_offset),
        ):
            binding = self._get_binding(tt)
            if binding is not None:
                binding.write(buf.data, mask=env_mask_wp)
        binding = self._get_binding(TT.FIXED_TENDON_LIMIT)
        if binding is not None:
            flat_src = wp.array(
                ptr=self._data._fixed_tendon_pos_limits.data.ptr,
                shape=(self._num_instances, self._num_fixed_tendons, 2),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            binding.write(flat_src, mask=env_mask_wp)

    def set_spatial_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial-tendon stiffness over selected env / tendon indices into the simulation.

        ``SPATIAL_TENDON_STIFFNESS`` is a sim-device binding on OVPhysX
        (tendon properties are applied without a CPU clone), so the write
        goes directly from the sim-device buffer to the binding.

        .. note::
            This method expects partial data.  A scalar :class:`float` is
            broadcast to ``(len(env_ids), len(spatial_tendon_ids))``.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            stiffness: Spatial-tendon stiffness [N/m]. Scalar :class:`float`,
                or shape ``(len(env_ids), len(spatial_tendon_ids))`` with
                dtype wp.float32.
            spatial_tendon_ids: Spatial-tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        stiffness = self._broadcast_scalar_to_2d(stiffness, (env_ids.shape[0], tendon_ids.shape[0]))
        self.assert_shape_and_dtype(stiffness, (env_ids.shape[0], tendon_ids.shape[0]), wp.float32, "stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], tendon_ids.shape[0]),
            inputs=[stiffness, env_ids, tendon_ids],
            outputs=[self._data._spatial_tendon_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_STIFFNESS)
        binding.write(self._data._spatial_tendon_stiffness.data, indices=env_ids)

    def set_spatial_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial-tendon stiffness over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``SPATIAL_TENDON_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            stiffness: Spatial-tendon stiffness [N/m]. May be a scalar
                :class:`float` (broadcast), or shape
                (num_instances, num_spatial_tendons) with dtype wp.float32.
            spatial_tendon_mask: Spatial-tendon mask. If None, all spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_spatial_tendon_mask(spatial_tendon_mask)
        shape = (self._num_instances, self._num_spatial_tendons)
        stiffness = self._broadcast_scalar_to_2d(stiffness, shape)
        self.assert_shape_and_dtype(stiffness, shape, wp.float32, "stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[stiffness, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._spatial_tendon_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_STIFFNESS)
        binding.write(self._data._spatial_tendon_stiffness.data, mask=env_mask_wp)

    def set_spatial_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial-tendon damping over selected env / tendon indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``SPATIAL_TENDON_DAMPING`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            damping: Spatial-tendon damping [N·s/m]. Shape is
                (len(env_ids), len(spatial_tendon_ids)) with dtype wp.float32.
            spatial_tendon_ids: Spatial-tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        damping = self._broadcast_scalar_to_2d(damping, (env_ids.shape[0], tendon_ids.shape[0]))
        self.assert_shape_and_dtype(damping, (env_ids.shape[0], tendon_ids.shape[0]), wp.float32, "damping")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], tendon_ids.shape[0]),
            inputs=[damping, env_ids, tendon_ids],
            outputs=[self._data._spatial_tendon_damping.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_DAMPING)
        binding.write(self._data._spatial_tendon_damping.data, indices=env_ids)

    def set_spatial_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial-tendon damping over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``SPATIAL_TENDON_DAMPING`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            damping: Spatial-tendon damping [N·s/m]. May be a scalar
                :class:`float` (broadcast), or shape
                (num_instances, num_spatial_tendons) with dtype wp.float32.
            spatial_tendon_mask: Spatial-tendon mask. If None, all spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_spatial_tendon_mask(spatial_tendon_mask)
        shape = (self._num_instances, self._num_spatial_tendons)
        damping = self._broadcast_scalar_to_2d(damping, shape)
        self.assert_shape_and_dtype(damping, shape, wp.float32, "damping")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[damping, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._spatial_tendon_damping.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_DAMPING)
        binding.write(self._data._spatial_tendon_damping.data, mask=env_mask_wp)

    def set_spatial_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial-tendon limit stiffness over selected env / tendon indices into the simulation.

        ``SPATIAL_TENDON_LIMIT_STIFFNESS`` is a sim-device binding on OVPhysX;
        the write goes directly from the sim-device buffer to the binding.

        .. note::
            This method expects partial data.  A scalar :class:`float` is
            broadcast to ``(len(env_ids), len(spatial_tendon_ids))``.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limit_stiffness: Spatial-tendon limit stiffness [N/m]. Scalar
                :class:`float`, or shape ``(len(env_ids), len(spatial_tendon_ids))``
                with dtype wp.float32.
            spatial_tendon_ids: Spatial-tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        limit_stiffness = self._broadcast_scalar_to_2d(limit_stiffness, (env_ids.shape[0], tendon_ids.shape[0]))
        self.assert_shape_and_dtype(
            limit_stiffness, (env_ids.shape[0], tendon_ids.shape[0]), wp.float32, "limit_stiffness"
        )
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], tendon_ids.shape[0]),
            inputs=[limit_stiffness, env_ids, tendon_ids],
            outputs=[self._data._spatial_tendon_limit_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_LIMIT_STIFFNESS)
        binding.write(self._data._spatial_tendon_limit_stiffness.data, indices=env_ids)

    def set_spatial_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial-tendon limit stiffness over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``SPATIAL_TENDON_LIMIT_STIFFNESS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            limit_stiffness: Spatial-tendon limit stiffness [N/m]. May be a
                scalar :class:`float` (broadcast), or shape
                (num_instances, num_spatial_tendons) with dtype wp.float32.
            spatial_tendon_mask: Spatial-tendon mask. If None, all spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_spatial_tendon_mask(spatial_tendon_mask)
        shape = (self._num_instances, self._num_spatial_tendons)
        limit_stiffness = self._broadcast_scalar_to_2d(limit_stiffness, shape)
        self.assert_shape_and_dtype(limit_stiffness, shape, wp.float32, "limit_stiffness")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[limit_stiffness, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._spatial_tendon_limit_stiffness.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_LIMIT_STIFFNESS)
        binding.write(self._data._spatial_tendon_limit_stiffness.data, mask=env_mask_wp)

    def set_spatial_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial-tendon offsets over selected env / tendon indices into the simulation.

        ``SPATIAL_TENDON_OFFSET`` is a sim-device binding on OVPhysX; the
        write goes directly from the sim-device buffer to the binding.

        .. note::
            This method expects partial data.  A scalar :class:`float` is
            broadcast to ``(len(env_ids), len(spatial_tendon_ids))``.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            offset: Spatial-tendon offsets [m]. Scalar :class:`float`, or
                shape ``(len(env_ids), len(spatial_tendon_ids))`` with
                dtype wp.float32.
            spatial_tendon_ids: Spatial-tendon indices. Defaults to None (all spatial tendons).
            env_ids: Environment indices. Defaults to None (all environments).
        """
        env_ids = self._resolve_env_ids(env_ids)
        tendon_ids = self._resolve_spatial_tendon_ids(spatial_tendon_ids)
        offset = self._broadcast_scalar_to_2d(offset, (env_ids.shape[0], tendon_ids.shape[0]))
        self.assert_shape_and_dtype(offset, (env_ids.shape[0], tendon_ids.shape[0]), wp.float32, "offset")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], tendon_ids.shape[0]),
            inputs=[offset, env_ids, tendon_ids],
            outputs=[self._data._spatial_tendon_offset.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_OFFSET)
        binding.write(self._data._spatial_tendon_offset.data, indices=env_ids)

    def set_spatial_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial-tendon offsets over selected env / tendon masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``SPATIAL_TENDON_OFFSET`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations.
            Performance is similar for both.  However, to allow graphed pipelines, the
            mask method must be used.

        Args:
            offset: Spatial-tendon offsets [m]. May be a scalar :class:`float`
                (broadcast), or shape (num_instances, num_spatial_tendons) with
                dtype wp.float32.
            spatial_tendon_mask: Spatial-tendon mask. If None, all spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        tendon_mask_wp = self._resolve_spatial_tendon_mask(spatial_tendon_mask)
        shape = (self._num_instances, self._num_spatial_tendons)
        offset = self._broadcast_scalar_to_2d(offset, shape)
        self.assert_shape_and_dtype(offset, shape, wp.float32, "offset")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=shape,
            inputs=[offset, env_mask_wp, tendon_mask_wp],
            outputs=[self._data._spatial_tendon_offset.data],
            device=self._device,
        )
        binding = self._get_binding(TT.SPATIAL_TENDON_OFFSET)
        binding.write(self._data._spatial_tendon_offset.data, mask=env_mask_wp)

    def write_spatial_tendon_properties_to_sim_index(
        self,
        *,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Push the cached spatial-tendon properties to the simulation in a single batch.

        Mirrors :meth:`write_fixed_tendon_properties_to_sim_index` for
        spatial tendons.  Only the four wheel-supported tensor types are
        written; ``ARTICULATION_SPATIAL_TENDON_LIMIT`` and
        ``ARTICULATION_SPATIAL_TENDON_REST_LENGTH`` are forward-compat
        stubs (see ``docs/superpowers/specs/2026-04-28-ovphysx-wheel-gaps-for-marco.md``).

        Args:
            spatial_tendon_ids: Accepted for PhysX API parity; ignored.
            env_ids: Environment indices.  If None, all environments are written.
        """
        env_ids = self._resolve_env_ids(env_ids)
        for tt, buf in (
            (TT.SPATIAL_TENDON_STIFFNESS, self._data._spatial_tendon_stiffness),
            (TT.SPATIAL_TENDON_DAMPING, self._data._spatial_tendon_damping),
            (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._data._spatial_tendon_limit_stiffness),
            (TT.SPATIAL_TENDON_OFFSET, self._data._spatial_tendon_offset),
        ):
            binding = self._get_binding(tt)
            if binding is not None:
                binding.write(buf.data, indices=env_ids)

    def write_spatial_tendon_properties_to_sim_mask(
        self,
        *,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Mask variant of :meth:`write_spatial_tendon_properties_to_sim_index`."""
        env_mask_wp = self._resolve_env_mask(env_mask)
        for tt, buf in (
            (TT.SPATIAL_TENDON_STIFFNESS, self._data._spatial_tendon_stiffness),
            (TT.SPATIAL_TENDON_DAMPING, self._data._spatial_tendon_damping),
            (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._data._spatial_tendon_limit_stiffness),
            (TT.SPATIAL_TENDON_OFFSET, self._data._spatial_tendon_offset),
        ):
            binding = self._get_binding(tt)
            if binding is not None:
                binding.write(buf.data, mask=env_mask_wp)

    """
    Internal helper.
    """

    def _initialize_impl(self) -> None:
        """Initialize the articulation from the OVPhysX simulation backend."""
        # obtain global simulation view
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        self._device = OvPhysxManager.get_device()

        # Resolve the articulation root expression.
        if self.cfg.articulation_root_prim_path is not None:
            root_prim_path_expr = self.cfg.prim_path + self.cfg.articulation_root_prim_path
        else:

            def has_articulation_root_api(prim) -> bool:
                return bool(prim.HasAPI(UsdPhysics.ArticulationRootAPI))

            asset_prim, root_expr = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)[0]
            walk_root = asset_prim.GetPath().pathString
            root_prims = sim_utils.get_all_matching_child_prims(
                walk_root, has_articulation_root_api, expected_num_matches=1
            )
            root_prim_path_expr = root_expr + root_prims[0].GetPath().pathString[len(walk_root) :]
        # Validate the prim exists on the live stage -- ``create_tensor_binding`` silently
        # returns a 0-count binding when the pattern matches nothing, surfacing as obscure
        # AttributeErrors deep in property accessors. Also stash the concrete source-side
        # root path for tendon discovery downstream.
        stage = PhysicsManager._sim.stage
        first_match = sim_utils.find_first_matching_prim(root_prim_path_expr, stage=stage)
        if first_match is None:
            raise RuntimeError(f"Failed to find articulation root prim at '{root_prim_path_expr}'.")
        self._articulation_root_path = first_match.GetPath().pathString

        # IsaacLab paths may use ``.*`` regex or ``{ENV_REGEX_NS}`` placeholder; ovphysx
        # ``create_tensor_binding`` expects fnmatch globs.
        pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", root_prim_path_expr)
        pattern = re.sub(r"\.\*", "*", pattern)
        self._binding_pattern = pattern

        # eagerly create every binding the data container reads at init, so
        # failures surface here rather than as KeyError downstream
        eager_types = [
            TT.ROOT_POSE,
            TT.ROOT_VELOCITY,
            TT.LINK_POSE,
            TT.LINK_VELOCITY,
            TT.LINK_ACCELERATION,
            TT.LINK_INCOMING_JOINT_FORCE,
            TT.DOF_POSITION,
            TT.DOF_VELOCITY,
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

        if not self._bindings:
            raise RuntimeError(
                f"OVPhysX could not create any articulation bindings for pattern {pattern!r}. "
                f"Check that prim_path={self.cfg.prim_path!r} matches at least one "
                "UsdPhysics.ArticulationRootAPI prim."
            )

        # read metadata from the first available binding
        sample = next(iter(self._bindings.values()))
        self._num_instances = sample.count
        self._num_joints = sample.dof_count
        self._num_bodies = sample.body_count
        self._is_fixed_base = sample.is_fixed_base
        self._joint_names = list(sample.dof_names)
        self._body_names = list(sample.body_names)

        # tendon counts/names must be resolved before buffer allocation
        self._process_tendons()

        # eagerly create tendon bindings now that the counts are known; this keeps
        # ArticulationData's _get_binding a simple dict lookup (no lazy callback).
        if self._num_fixed_tendons > 0:
            for tt in (
                TT.FIXED_TENDON_STIFFNESS,
                TT.FIXED_TENDON_DAMPING,
                TT.FIXED_TENDON_LIMIT_STIFFNESS,
                TT.FIXED_TENDON_LIMIT,
                TT.FIXED_TENDON_REST_LENGTH,
                TT.FIXED_TENDON_OFFSET,
            ):
                try:
                    self._bindings[tt] = physx_instance.create_tensor_binding(pattern=pattern, tensor_type=tt)
                except Exception:
                    logger.debug("Could not create tensor binding for type %s on pattern %s", tt, pattern)
        if self._num_spatial_tendons > 0:
            for tt in (
                TT.SPATIAL_TENDON_STIFFNESS,
                TT.SPATIAL_TENDON_DAMPING,
                TT.SPATIAL_TENDON_LIMIT_STIFFNESS,
                TT.SPATIAL_TENDON_OFFSET,
            ):
                try:
                    self._bindings[tt] = physx_instance.create_tensor_binding(pattern=pattern, tensor_type=tt)
                except Exception:
                    logger.debug("Could not create tensor binding for type %s on pattern %s", tt, pattern)

        # construct the data container; counts come from the bindings
        self._data = ArticulationData(self._bindings, self._device)
        self._data.body_names = self._body_names
        self._data.joint_names = self._joint_names
        self._data.fixed_tendon_names = self._fixed_tendon_names
        self._data.spatial_tendon_names = self._spatial_tendon_names

        # allocate asset-side buffers
        self._create_buffers()

        # apply initial state from config
        self._process_cfg()

        # build actuator instances and write drive properties to PhysX
        self._process_actuators_cfg()

        # cache effort / target bindings and write-views for write_data_to_sim().
        # The effort view aliases applied_torque so the binding gets the actuator
        # output without an extra copy.
        self._effort_binding = self._get_binding(TT.DOF_ACTUATION_FORCE)
        if self._effort_binding is not None:
            torque = self._data._applied_torque
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

        def _make_write_view(tt, buf):
            b = self._get_binding(tt)
            if b is None or buf is None:
                return None, None
            v = wp.array(ptr=buf.ptr, shape=b.shape, dtype=wp.float32, device=str(buf.device), copy=False)
            return b, v

        self._pos_target_binding, self._pos_target_write_view = _make_write_view(
            TT.DOF_POSITION_TARGET, self._data._joint_pos_target
        )
        self._vel_target_binding, self._vel_target_write_view = _make_write_view(
            TT.DOF_VELOCITY_TARGET, self._data._joint_vel_target
        )

        # validate the resolved configuration AFTER actuator/tendon processing
        # so the values reflect any overrides applied by the actuator models
        self._validate_cfg()

        # prime the data by performing the first read
        self.update(0.0)

        # mark data as ready
        self._data.is_primed = True

    def _create_buffers(self) -> None:
        """Allocate asset-side buffers (index/mask constants, wrench buf, pinned CPU staging)."""
        N = self._num_instances
        B = self._num_bodies
        J = self._num_joints
        FT = self._num_fixed_tendons
        ST = self._num_spatial_tendons
        device = self._device

        # Index constants.
        self._ALL_INDICES = wp.array(np.arange(N, dtype=np.int32), device=device)
        self._ALL_BODY_INDICES = wp.array(np.arange(B, dtype=np.int32), device=device)
        self._ALL_JOINT_INDICES = wp.array(np.arange(J, dtype=np.int32), device=device)
        self._ALL_FIXED_TENDON_INDICES = wp.array(np.arange(FT, dtype=np.int32), device=device)
        self._ALL_SPATIAL_TENDON_INDICES = wp.array(np.arange(ST, dtype=np.int32), device=device)

        # All-true masks.
        self._ALL_TRUE_ENV_MASK = wp.array(np.ones(N, dtype=bool), dtype=wp.bool, device=device)
        self._ALL_TRUE_BODY_MASK = wp.array(np.ones(B, dtype=bool), dtype=wp.bool, device=device)
        self._ALL_TRUE_JOINT_MASK = wp.array(np.ones(J, dtype=bool), dtype=wp.bool, device=device)
        self._ALL_TRUE_FIXED_TENDON_MASK = wp.array(np.ones(FT, dtype=bool), dtype=wp.bool, device=device)
        self._ALL_TRUE_SPATIAL_TENDON_MASK = wp.array(np.ones(ST, dtype=bool), dtype=wp.bool, device=device)

        # Wrench buffer (force, torque, position) per body, written by the
        # ``_body_wrench_to_world`` kernel and consumed by the
        # ``LINK_WRENCH`` binding which expects the 3D ``(N, B, 9)`` shape.
        self._wrench_buf = wp.zeros((N, B, 9), dtype=wp.float32, device=device)

        # Wrench composers.
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # Wrench scratch buffer (used by _apply_external_wrenches, not yet allocated above).
        # Joint-index arrays for each actuator (populated by _process_actuators_cfg).
        self._joint_ids_per_actuator: dict[str, list[int]] = {}

        # Pinned-host CPU staging for env ids/masks (PR #5329 pattern).
        self._cpu_env_ids_all = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_env_ids_all, self._ALL_INDICES)
        self._cpu_env_mask = wp.zeros(N, dtype=wp.bool, device="cpu", pinned=True)

    def _process_cfg(self) -> None:
        """Populate default state buffers from the config (mirrors RigidObject and Newton Articulation)."""
        cfg = self.cfg
        N = self._num_instances
        D = self._num_joints
        dev = self._device

        # Default root state from config (matching PhysX pattern).
        default_root_pose = tuple(cfg.init_state.pos) + tuple(cfg.init_state.rot)
        default_root_vel = tuple(cfg.init_state.lin_vel) + tuple(cfg.init_state.ang_vel)
        np_pose = np.tile(np.array(default_root_pose, dtype=np.float32), (N, 1))
        np_vel = np.tile(np.array(default_root_vel, dtype=np.float32), (N, 1))
        self._data.default_root_pose = wp.array(np_pose, dtype=wp.transformf, device=dev)
        self._data.default_root_vel = wp.array(np_vel, dtype=wp.spatial_vectorf, device=dev)

        # Default joint positions / velocities from config patterns.
        # cfg.init_state.joint_pos is a dict[str, float] where keys are regex patterns
        # matching joint names.  We expand this into a (N, D) buffer.
        self._resolve_joint_values(cfg.init_state.joint_pos, self._data._default_joint_pos)
        self._resolve_joint_values(cfg.init_state.joint_vel, self._data._default_joint_vel)

        # Compute soft joint position limits from the hard limits read from the binding
        # (or zeros if no joints).  This matches the PhysX/Newton path.
        if D > 0:
            wp.launch(
                update_soft_joint_pos_limits,
                dim=(N, D),
                inputs=[self._data.joint_pos_limits, cfg.soft_joint_pos_limit_factor],
                outputs=[self._data._soft_joint_pos_limits],
                device=dev,
            )

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
                    from pxr import Usd

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

    def _get_binding(self, tensor_type: int):
        """Return a cached TensorBinding, creating it on first access.

        Bindings are lightweight handles (a pointer + shape metadata into
        PhysX's shared GPU buffer).  Creating one does NOT allocate new GPU
        memory -- the underlying simulation buffers are allocated once by PhysX
        regardless of how many bindings point into them.  Still, we defer
        creation so that tensor types the user never queries are never looked up.

        Args:
            tensor_type: The TensorType constant identifying which simulation
                buffer to bind (e.g. :attr:`~isaaclab_ovphysx.tensor_types.ROOT_POSE`).

        Returns:
            A TensorBinding object, or ``None`` if the binding could not be created.
        """
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        try:
            binding = self._ovphysx.create_tensor_binding(pattern=self._binding_pattern, tensor_type=tensor_type)
            self._bindings[tensor_type] = binding
            return binding
        except Exception:
            logger.debug("Could not create tensor binding for type %s", tensor_type)
            return None

    def _resolve_joint_values(self, pattern_dict: dict[str, float], buffer: wp.array) -> None:
        """Resolve a ``{pattern: value}`` dict into a per-joint buffer.

        Builds values on CPU then copies to buffer's device (GPU arrays'
        ``.numpy()`` returns a read-only copy, not a writable view).

        Args:
            pattern_dict: A mapping from regex pattern strings to scalar values.
                Matches joint names returned by :attr:`joint_names`.
            buffer: Target warp array of shape ``(num_instances, num_joints)``
                to populate.
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

    def _n_envs_index(self, env_ids) -> int:
        """Return the number of environments from an ``env_ids`` argument."""
        if env_ids is None:
            return self._num_instances
        if isinstance(env_ids, (list, tuple)):
            return len(env_ids)
        return env_ids.shape[0] if hasattr(env_ids, "shape") else len(env_ids)

    def _nft(self) -> int:
        """Return the number of fixed tendons (0 if none)."""
        return self._num_fixed_tendons

    def _nst(self) -> int:
        """Return the number of spatial tendons (0 if none)."""
        return self._num_spatial_tendons

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidate the asset on simulation reset."""
        super()._invalidate_initialize_callback(event)

    """
    Internal helpers -- Actuators.
    """

    def _process_actuators_cfg(self) -> None:
        """Build actuator instances from the config and write drive properties to PhysX.

        Mirrors the PhysX backend's ``_process_actuators_cfg``:

        * For :class:`~isaaclab.actuators.ImplicitActuator`: write the configured
          stiffness/damping to the PhysX drive so the solver uses exactly those values.
        * For all explicit actuators: zero out PhysX stiffness/damping so USD-authored
          drive gains cannot interfere with the explicit torque path.
        * For all actuators: write :attr:`~isaaclab.actuators.ActuatorBase.effort_limit_sim`
          and :attr:`~isaaclab.actuators.ActuatorBase.velocity_limit_sim`.
        """
        from isaaclab.actuators import ImplicitActuator

        self.actuators: dict[str, Any] = {}
        self._has_implicit_actuators = False
        for name, act_cfg in self.cfg.actuators.items():
            joint_ids, joint_names = self.find_joints(act_cfg.joint_names_expr)
            if not joint_ids:
                logger.warning("Actuator '%s': no joints matched '%s'", name, act_cfg.joint_names_expr)
                continue
            act_cfg_copy = act_cfg.copy()
            # seed the actuator with the simulation's already-correct DOF defaults
            # (USD-authored ``physxJoint:maxJointVelocity`` etc. parsed at scene-load).
            # Without these the ActuatorBase constructor falls back to ``inf`` for unset
            # cfg fields, and the ``write_joint_*_to_sim_index`` calls below then
            # overwrite the correct values with ``inf``.
            act = act_cfg_copy.class_type(
                act_cfg_copy,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self._num_instances,
                device=self._device,
                stiffness=self._data.joint_stiffness.torch[:, joint_ids],
                damping=self._data.joint_damping.torch[:, joint_ids],
                armature=self._data.joint_armature.torch[:, joint_ids],
                friction=self._data.joint_friction_coeff.torch[:, joint_ids],
                dynamic_friction=self._data.joint_dynamic_friction_coeff.torch[:, joint_ids],
                viscous_friction=self._data.joint_viscous_friction_coeff.torch[:, joint_ids],
                effort_limit=self._data.joint_effort_limits.torch[:, joint_ids].clone(),
                velocity_limit=self._data.joint_vel_limits.torch[:, joint_ids],
            )
            self.actuators[name] = act
            self._joint_ids_per_actuator[name] = joint_ids

            # Write drive gains and limits to PhysX to match the actuator config.
            # Without this, PhysX retains whatever stiffness/damping was authored in the
            # USD file, which can produce large restoring forces when the USD gains differ
            # from the actuator config.
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

    def _apply_actuator_model(self) -> None:
        """Run the actuator model to compute joint torques from user-supplied targets.

        IsaacLab actuators are torch-based. The method converts Warp buffers to
        torch via DLPack (zero-copy on GPU), runs each actuator's
        :meth:`~isaaclab.actuators.ActuatorBase.compute` method, then writes the
        computed effort back to the private ``_computed_torque`` / ``_applied_torque``
        buffers of the data container. :meth:`write_data_to_sim` then pushes
        ``_applied_torque`` to the ``DOF_ACTUATION_FORCE`` binding in one shot.
        """
        from isaaclab.utils.types import ArticulationActions

        for name, act in self.actuators.items():
            jids = act.joint_indices
            if jids is None:
                continue
            jids_t = jids if isinstance(jids, list) else list(jids)
            all_joints = len(jids_t) == self._num_joints

            # Warp -> torch (zero-copy on same device via DLPack).
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

    """
    Internal helpers -- Debugging.
    """

    def _validate_cfg(self) -> None:
        """Validate the configuration after processing.

        Mirrors :meth:`isaaclab_physx.assets.Articulation._validate_cfg` (raises
        ``ValueError`` with a per-joint message when any default joint position
        is outside ``[lower, upper]`` or any default joint velocity exceeds the
        per-joint max velocity).  Reads come from :attr:`ArticulationData`
        accessors instead of PhysX's ``root_view.get_dof_limits`` /
        ``get_dof_max_velocities`` because OVPhysX's ``root_view`` is the
        per-tensor-type bindings dict.

        .. note::
            Must be called only after :meth:`_create_buffers` /
            :meth:`_process_cfg` / :meth:`_process_actuators_cfg`, otherwise
            limits and defaults may not yet reflect the final values.
        """
        # check that the default joint positions are within the limits
        joint_pos_limits = self._data.joint_pos_limits.torch[0]  # (num_joints, 2)
        default_joint_pos = self._data.default_joint_pos.torch[0]  # (num_joints,)
        out_of_range = default_joint_pos < joint_pos_limits[:, 0]
        out_of_range |= default_joint_pos > joint_pos_limits[:, 1]
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:
            msg = "The following joints have default positions out of the limits: \n"
            for idx in violated_indices:
                joint_name = self._data.joint_names[idx]
                joint_limit = joint_pos_limits[idx]
                joint_pos = default_joint_pos[idx]
                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

        # check that the default joint velocities are within the limits
        joint_max_vel = self._data.joint_vel_limits.torch[0]  # (num_joints,)
        default_joint_vel = self._data.default_joint_vel.torch[0]  # (num_joints,)
        out_of_range = torch.abs(default_joint_vel) > joint_max_vel
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:
            msg = "The following joints have default velocities out of the limits: \n"
            for idx in violated_indices:
                joint_name = self._data.joint_names[idx]
                joint_limit = [-joint_max_vel[idx], joint_max_vel[idx]]
                joint_vel = default_joint_vel[idx]
                msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

    def _log_articulation_info(self) -> None:
        pass

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp int32 array on ``self._device`` (mirrors PhysX).

        Tests sometimes hand us indices on CPU even when the sim runs on GPU; we move the
        resolved array onto ``self._device`` so kernel launches don't fail on a device
        mismatch.
        """
        if env_ids is None or env_ids == slice(None):
            return self._ALL_INDICES
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self._device)
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(env_ids, wp.array) and str(env_ids.device) != self._device:
            env_ids = wp.clone(env_ids, device=self._device)
        return env_ids

    def _resolve_body_ids(self, body_ids) -> wp.array:
        """Resolve body indices to a warp int32 array on ``self._device`` (mirrors PhysX)."""
        if body_ids is None or body_ids == slice(None):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self._device)
        if isinstance(body_ids, torch.Tensor):
            return wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(body_ids, wp.array) and str(body_ids.device) != self._device:
            body_ids = wp.clone(body_ids, device=self._device)
        return body_ids

    def _resolve_joint_ids(self, joint_ids) -> wp.array:
        """Resolve joint indices to a warp int32 array on ``self._device``."""
        if joint_ids is None or joint_ids == slice(None):
            return self._ALL_JOINT_INDICES
        if isinstance(joint_ids, list):
            return wp.array(joint_ids, dtype=wp.int32, device=self._device)
        if isinstance(joint_ids, torch.Tensor):
            return wp.from_torch(joint_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(joint_ids, wp.array) and str(joint_ids.device) != self._device:
            joint_ids = wp.clone(joint_ids, device=self._device)
        return joint_ids

    def _resolve_fixed_tendon_ids(self, tendon_ids) -> wp.array:
        """Resolve fixed-tendon indices to a warp int32 array on ``self._device``."""
        if tendon_ids is None or tendon_ids == slice(None):
            return self._ALL_FIXED_TENDON_INDICES
        if isinstance(tendon_ids, list):
            return wp.array(tendon_ids, dtype=wp.int32, device=self._device)
        if isinstance(tendon_ids, torch.Tensor):
            return wp.from_torch(tendon_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(tendon_ids, wp.array) and str(tendon_ids.device) != self._device:
            tendon_ids = wp.clone(tendon_ids, device=self._device)
        return tendon_ids

    def _resolve_spatial_tendon_ids(self, tendon_ids) -> wp.array:
        """Resolve spatial-tendon indices to a warp int32 array on ``self._device``."""
        if tendon_ids is None or tendon_ids == slice(None):
            return self._ALL_SPATIAL_TENDON_INDICES
        if isinstance(tendon_ids, list):
            return wp.array(tendon_ids, dtype=wp.int32, device=self._device)
        if isinstance(tendon_ids, torch.Tensor):
            return wp.from_torch(tendon_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(tendon_ids, wp.array) and str(tendon_ids.device) != self._device:
            tendon_ids = wp.clone(tendon_ids, device=self._device)
        return tendon_ids

    def _broadcast_scalar_to_2d(
        self, value: float | torch.Tensor | wp.array, shape: tuple[int, int]
    ) -> torch.Tensor | wp.array:
        """Broadcast a scalar :class:`float` to a ``(rows, cols)`` torch ``float32`` tensor.

        Tendon and joint setters accept ``float | torch.Tensor | wp.array``; the
        underlying ``shared_kernels.write_2d_data_to_buffer_*`` kernels only
        accept 2D arrays.  This helper expands a Python float into a constant
        tensor on :attr:`_device`; tensor / warp inputs are returned as-is.

        Mirrors the PhysX backend's ``isinstance(value, float)`` branching,
        which dispatches to ``articulation_kernels.float_data_to_buffer_with_*``.
        OVPhysX does not have those scalar kernels, so we materialize the
        broadcast on the Python side.

        Args:
            value: Scalar float or 2D tensor / warp array.
            shape: ``(rows, cols)`` target shape used when broadcasting a
                scalar.

        Returns:
            A 2D :class:`torch.Tensor` on ``self._device`` if *value* was a
            float; otherwise *value* unchanged.
        """
        if isinstance(value, float):
            return torch.full(shape, value, dtype=torch.float32, device=self._device)
        return value

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve an environment mask to a ``wp.bool`` array on ``self._device``.

        OVPhysX (like Newton) uses the binding's native ``binding.write(mask=...)`` path,
        so the mask is preserved end-to-end; no ``torch.nonzero`` conversion is needed.
        ``None`` returns the pre-allocated all-true mask.
        """
        if env_mask is None:
            return self._ALL_TRUE_ENV_MASK
        if isinstance(env_mask, torch.Tensor):
            return wp.from_torch(env_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(env_mask, wp.array) and str(env_mask.device) != self._device:
            env_mask = wp.clone(env_mask, device=self._device)
        return env_mask

    def _resolve_body_mask(self, body_mask: wp.array | None) -> wp.array:
        """Resolve a body mask to a ``wp.bool`` array on ``self._device`` (Newton-style)."""
        if body_mask is None:
            return self._ALL_TRUE_BODY_MASK
        if isinstance(body_mask, torch.Tensor):
            return wp.from_torch(body_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(body_mask, wp.array) and str(body_mask.device) != self._device:
            body_mask = wp.clone(body_mask, device=self._device)
        return body_mask

    def _resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        """Resolve a joint mask to a ``wp.bool`` array on ``self._device``."""
        if joint_mask is None:
            return self._ALL_TRUE_JOINT_MASK
        if isinstance(joint_mask, torch.Tensor):
            return wp.from_torch(joint_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(joint_mask, wp.array) and str(joint_mask.device) != self._device:
            joint_mask = wp.clone(joint_mask, device=self._device)
        return joint_mask

    def _resolve_fixed_tendon_mask(self, tendon_mask: wp.array | None) -> wp.array:
        """Resolve a fixed-tendon mask to a ``wp.bool`` array on ``self._device``."""
        if tendon_mask is None:
            return self._ALL_TRUE_FIXED_TENDON_MASK
        if isinstance(tendon_mask, torch.Tensor):
            return wp.from_torch(tendon_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(tendon_mask, wp.array) and str(tendon_mask.device) != self._device:
            tendon_mask = wp.clone(tendon_mask, device=self._device)
        return tendon_mask

    def _resolve_spatial_tendon_mask(self, tendon_mask: wp.array | None) -> wp.array:
        """Resolve a spatial-tendon mask to a ``wp.bool`` array on ``self._device``."""
        if tendon_mask is None:
            return self._ALL_TRUE_SPATIAL_TENDON_MASK
        if isinstance(tendon_mask, torch.Tensor):
            return wp.from_torch(tendon_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(tendon_mask, wp.array) and str(tendon_mask.device) != self._device:
            tendon_mask = wp.clone(tendon_mask, device=self._device)
        return tendon_mask

    def _get_cpu_env_mask(self, env_mask: wp.array) -> wp.array:
        """Return a pinned-host CPU copy of :paramref:`env_mask` for a CPU-only binding write.

        :paramref:`env_mask` is normally on ``self._device``; ``binding.write(mask=...)``
        requires the mask on the binding's device, which is CPU for mass / CoMs / inertia.
        Reuses the pre-allocated ``_cpu_env_mask`` pinned buffer.
        """
        wp.copy(self._cpu_env_mask, env_mask)
        return self._cpu_env_mask

    def _get_cpu_env_ids(self, env_ids: wp.array | torch.Tensor) -> wp.array:
        """Return CPU int32 indices, using the pre-allocated pinned ``_cpu_env_ids_all``
        fast path when *env_ids* matches ``_ALL_INDICES`` (PR #5329 pattern).
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids, dtype=wp.int32)
        if env_ids.ptr == self._ALL_INDICES.ptr:
            return self._cpu_env_ids_all
        return wp.clone(env_ids, device="cpu")

    """
    Deprecated methods.
    """

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated; use :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index` instead.

        Args:
            root_state: Root state [m, m, m, qw, qx, qy, qz, m/s, m/s, m/s, rad/s, rad/s, rad/s].
                Shape is (len(env_ids), 13) with dtype wp.float32.
            env_ids: Environment indices. Defaults to None (all environments).
        """
        warnings.warn(
            "The function 'write_root_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated; use :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index` instead.

        Args:
            root_state: Root CoM state [m, m, m, qw, qx, qy, qz, m/s, m/s, m/s, rad/s, rad/s, rad/s].
                Shape is (len(env_ids), 13) with dtype wp.float32.
            env_ids: Environment indices. Defaults to None (all environments).
        """
        warnings.warn(
            "The function 'write_root_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_com_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated; use :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index` instead.

        Args:
            root_state: Root link state [m, m, m, qw, qx, qy, qz, m/s, m/s, m/s, rad/s, rad/s, rad/s].
                Shape is (len(env_ids), 13) with dtype wp.float32.
            env_ids: Environment indices. Defaults to None (all environments).
        """
        warnings.warn(
            "The function 'write_root_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated combined joint-state write; use :meth:`write_joint_position_to_sim_index`
        and :meth:`write_joint_velocity_to_sim_index` instead.

        Args:
            position: Joint positions [m or rad, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            velocity: Joint velocities [m/s or rad/s, depending on joint type].  Shape is
                (len(env_ids), len(joint_ids)) with dtype wp.float32.
            joint_ids: Joint indices.  Defaults to None (all joints).
            env_ids: Environment indices.  Defaults to None (all environments).
        """
        warnings.warn(
            "write_joint_state_to_sim is deprecated; use write_joint_position_to_sim_index"
            " and write_joint_velocity_to_sim_index instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | wp.array | float,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | float | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_friction_coefficient_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_friction_coefficient_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_friction_coefficient_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=joint_friction_coeff,
            joint_dynamic_friction_coeff=joint_dynamic_friction_coeff,
            joint_viscous_friction_coeff=joint_viscous_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

    def write_joint_dynamic_friction_coefficient_to_sim(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_dynamic_friction_coefficient_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_dynamic_friction_coefficient_to_sim' will be deprecated in a future release. "
            "Please use 'write_joint_dynamic_friction_coefficient_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_dynamic_friction_coefficient_to_sim_index(
            joint_dynamic_friction_coeff=joint_dynamic_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

    def write_joint_viscous_friction_coefficient_to_sim(
        self,
        joint_viscous_friction_coeff: torch.Tensor | wp.array | float,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_viscous_friction_coefficient_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_viscous_friction_coefficient_to_sim' will be deprecated in a future release. "
            "Please use 'write_joint_viscous_friction_coefficient_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff=joint_viscous_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
