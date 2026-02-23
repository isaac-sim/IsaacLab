# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.wrench_composer import WrenchComposer

from ..asset_base import AssetBase

if TYPE_CHECKING:
    from .rigid_object_collection_cfg import RigidObjectCollectionCfg
    from .rigid_object_collection_data import RigidObjectCollectionData


class BaseRigidObjectCollection(AssetBase):
    """A rigid object collection class.

    This class represents a collection of rigid objects in the simulation, where the state of the
    rigid objects can be accessed and modified using a batched ``(env_ids, object_ids)`` API.

    For each rigid body in the collection, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid bodies. On playing the
    simulation, the physics engine will automatically register the rigid bodies and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_view` attribute.

    Rigid objects in the collection are uniquely identified via the key of the dictionary
    :attr:`~isaaclab.assets.RigidObjectCollectionCfg.rigid_objects` in the
    :class:`~isaaclab.assets.RigidObjectCollectionCfg` configuration class.
    This differs from the :class:`~isaaclab.assets.RigidObject` class, where a rigid object is identified by
    the name of the Xform where the `USD RigidBodyAPI`_ is applied. This would not be possible for the rigid
    object collection since the :attr:`~isaaclab.assets.RigidObjectCollectionCfg.rigid_objects` dictionary
    could contain the same rigid object multiple times, leading to ambiguity.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCollectionCfg
    """Configuration instance for the rigid object."""

    __backend_name__: str = "base"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        pass

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> RigidObjectCollectionData:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in the rigid object collection."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object collection."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self):
        """Root view for the rigid object collection.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are persistent and are applied to the simulation at every step. This is useful to apply forces that
        are constant over a period of time, things like the thrust of a motor for instance.
        """
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        object_ids: slice | torch.Tensor | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Resets all internal buffers of selected environments and objects.

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        raise NotImplementedError()

    """
    Operations - Finders.
    """

    @abstractmethod
    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Find bodies in the rigid body collection based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        raise NotImplementedError()

    """
    Operations - Write to simulation.
    """

    @abstractmethod
    def write_body_pose_to_sim_index(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body poses over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_pose_to_sim_mask(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body poses over selected environment and body mask into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body poses in simulation frame. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_pose_to_sim_index(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body link pose over selected environment and body indices into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body link poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_pose_to_sim_mask(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body link pose over selected environment and body mask into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body link poses in simulation frame. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_pose_to_sim_index(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body center of mass poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_pose_to_sim_mask(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body center of mass pose over selected environment and body mask into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body center of mass poses in simulation frame. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_velocity_to_sim_index(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body velocities in simulation frame. Shape is (len(env_ids), len(body_ids), 6)
                or (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_velocity_to_sim_mask(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body velocity over selected environment and body mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body velocities in simulation frame. Shape is (num_instances, num_bodies, 6)
                or (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_velocity_to_sim_index(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body center of mass velocities in simulation frame. Shape is
                (len(env_ids), len(body_ids), 6) or (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_velocity_to_sim_mask(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body center of mass velocities in simulation frame. Shape is
                (num_instances, num_bodies, 6) or (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_velocity_to_sim_index(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body link velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body link velocities in simulation frame. Shape is (len(env_ids), len(body_ids), 6)
                or (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_velocity_to_sim_mask(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body link velocity over selected environment and body mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body link velocities in simulation frame. Shape is (num_instances, num_bodies, 6)
                or (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    """
    Operations - Setters.
    """

    @abstractmethod
    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set masses of all bodies.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set center of mass positions of all bodies.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            coms: Center of mass positions of all bodies. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None
                (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass positions of all bodies.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            coms: Center of mass positions of all bodies. Shape is (num_instances, num_bodies, 3)
                or (num_instances, num_bodies) with dtype wp.vec3f.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    """
    Internal helper.
    """

    @abstractmethod
    def _initialize_impl(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _create_buffers(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        raise NotImplementedError()

    """
    Internal simulation callbacks.
    """

    @abstractmethod
    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)

    """
    Deprecated properties and methods.
    """

    @property
    def num_objects(self) -> int:
        """Deprecated property. Please use :attr:`num_bodies` instead."""
        warnings.warn(
            "The `num_objects` property will be deprecated in a future release. Please use `num_bodies` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_bodies

    @property
    def object_names(self) -> list[str]:
        """Deprecated property. Please use :attr:`body_names` instead."""
        warnings.warn(
            "The `object_names` property will be deprecated in a future release. Please use `body_names` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_names

    @abstractmethod
    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_com_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_link_velocity_to_sim_index`."""
        raise NotImplementedError()

    def write_body_pose_to_sim(
        self,
        body_poses: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_pose_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_pose_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_pose_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_pose_to_sim_index(body_poses=body_poses, env_ids=env_ids, body_ids=body_ids)

    def write_body_link_pose_to_sim(
        self,
        body_poses: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_link_pose_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim_index(body_poses=body_poses, env_ids=env_ids, body_ids=body_ids)

    def write_body_com_pose_to_sim(
        self,
        body_poses: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_com_pose_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_com_pose_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_com_pose_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_pose_to_sim_index(body_poses=body_poses, env_ids=env_ids, body_ids=body_ids)

    def write_body_velocity_to_sim(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_velocity_to_sim_index(body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids)

    def write_body_com_velocity_to_sim(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_com_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_velocity_to_sim_index(body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids)

    def write_body_link_velocity_to_sim(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_link_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_velocity_to_sim_index(body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids)

    def set_masses(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_masses_index`."""
        warnings.warn(
            "The function 'set_masses' will be deprecated in a future release. Please use 'set_masses_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_masses_index(masses=masses, body_ids=body_ids, env_ids=env_ids)

    def set_coms(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_coms_index`."""
        warnings.warn(
            "The function 'set_coms' will be deprecated in a future release. Please use 'set_coms_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_coms_index(coms=coms, body_ids=body_ids, env_ids=env_ids)

    def set_inertias(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_inertias_index`."""
        warnings.warn(
            "The function 'set_inertias' will be deprecated in a future release. Please"
            " use 'set_inertias_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_inertias_index(inertias=inertias, body_ids=body_ids, env_ids=env_ids)

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        positions: torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        is_global: bool = False,
    ) -> None:
        """Deprecated, same as :meth:`permanent_wrench_composer.set_forces_and_torques`."""
        warnings.warn(
            "The function 'set_external_force_and_torque' will be deprecated in a future release. Please"
            " use 'permanent_wrench_composer.set_forces_and_torques' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )

    def write_object_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_pose_to_sim_index` and
        :meth:`write_body_link_velocity_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_state_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_pose_to_sim_index` and `write_body_link_velocity_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_state_to_sim(object_state, env_ids=env_ids, body_ids=object_ids)

    def write_object_com_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_pose_to_sim_index` and
        :meth:`write_body_velocity_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_com_state_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_pose_to_sim_index` and `write_body_velocity_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_state_to_sim(object_state, env_ids=env_ids, body_ids=object_ids)

    def write_object_link_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_pose_to_sim_index` and
        :meth:`write_body_link_velocity_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_link_state_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_pose_to_sim_index` and `write_body_link_velocity_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_state_to_sim(object_state, env_ids=env_ids, body_ids=object_ids)

    def write_object_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_pose_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_pose_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_pose_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_pose_to_sim_index(body_poses=object_pose, env_ids=env_ids, body_ids=object_ids)

    def write_object_link_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_link_pose_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_link_pose_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_link_pose_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim_index(body_poses=object_pose, env_ids=env_ids, body_ids=object_ids)

    def write_object_com_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_pose_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_com_pose_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_pose_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_pose_to_sim_index(body_poses=object_pose, env_ids=env_ids, body_ids=object_ids)

    def write_object_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_velocity_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_velocity_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_velocity_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_velocity_to_sim_index(body_velocities=object_velocity, env_ids=env_ids, body_ids=object_ids)

    def write_object_com_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_velocity_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_com_velocity_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_velocity_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_velocity_to_sim_index(body_velocities=object_velocity, env_ids=env_ids, body_ids=object_ids)

    def write_object_link_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_link_velocity_to_sim_index` instead."""
        warnings.warn(
            "The `write_object_link_velocity_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_link_velocity_to_sim_index` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_velocity_to_sim_index(
            body_velocities=object_velocity, env_ids=env_ids, body_ids=object_ids
        )

    def find_objects(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Deprecated method. Please use :meth:`find_bodies` instead."""
        warnings.warn(
            "The `find_objects` method will be deprecated in a future release. Please use `find_bodies` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_bodies(name_keys, preserve_order)
