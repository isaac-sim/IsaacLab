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
    from .rigid_object_cfg import RigidObjectCfg
    from .rigid_object_data import RigidObjectData


class BaseRigidObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCfg
    """Configuration instance for the rigid object."""

    __backend_name__: str = "base"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> RigidObjectData:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self):
        """Root view for the asset.

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
        self, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Reset the rigid object.

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
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
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7) or
                (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (num_instances, 7) or
                (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7) or
                (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_velocity_to_sim_mask(
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
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
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
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
    Deprecated.
    """

    @abstractmethod
    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_pose_to_sim_index` and :meth:`write_root_velocity_to_sim_index`."""
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index` and :meth:`write_root_velocity_to_sim_index`."""
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_pose_to_sim_index`
        and :meth:`write_root_link_velocity_to_sim_index`."""
        raise NotImplementedError()

    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_pose_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_pose_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_pose_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_link_pose_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_com_pose_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_pose_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_com_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_com_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_link_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def set_masses(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | slice | None = None,
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
            forces, torques, positions=positions, body_ids=body_ids, env_ids=env_ids, is_global=is_global
        )
