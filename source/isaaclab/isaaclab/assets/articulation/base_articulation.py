# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from ...sim import SimulationContext
from ...utils.buffers import TimestampedBufferWarp
from ...utils.leapp.leapp_semantics import OutputKindEnum, joint_names_resolver, leapp_tensor_semantics
from ...utils.warp import ProxyArray
from ..asset_base import AssetBase
from . import ordering_kernels
from .ordering import ArticulationNameMap, ArticulationOrderingConvention, build_articulation_name_map
from .ordering_resolvers import _resolve_articulation_ordering_names

if TYPE_CHECKING:
    from isaaclab.utils.wrench_composer import WrenchComposer

    from .articulation_cfg import ArticulationCfg
    from .base_articulation_data import BaseArticulationData

logger = logging.getLogger(__name__)


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

    .. note::
        Index-based writer selectors must contain unique environment, joint, and
        body indices. Repeated selector entries issue concurrent writes to the
        same simulation cell, so the winning value is undefined. Use a mask when
        a selection may contain duplicates.

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html

    """

    cfg: ArticulationCfg
    """Configuration instance for the articulations."""

    __backend_name__: str = "base"
    """The name of the backend for the articulation."""

    __backend_native_orderings__: tuple[str, ...] = ()
    """Symbolic convention names this backend's native order already satisfies.

    A convention listed here takes the identity fast path in ordering
    resolution: requesting it returns backend names without any cross-backend
    discovery. The base default is empty; concrete backends declare the
    :class:`ArticulationOrderingConvention` values (e.g. ``("physx",)``) their
    solver-view order matches.
    """

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
        sim_ctx = SimulationContext.instance()
        self._sim_cfg = sim_ctx.cfg if sim_ctx is not None else None
        # Per-articulation cache of resolved cross-backend convention name orderings,
        # populated lazily while ordering maps are resolved. A single resolution may
        # query the same convention for both joints and bodies, so results are keyed
        # by ``(convention, kind)``.
        self._ordering_convention_name_cache: dict[
            tuple[ArticulationOrderingConvention, Literal["joint", "body"]], tuple[str, ...]
        ] = {}

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseArticulationData:
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
    def joint_names(self) -> list[str]:
        """Joint names in public API order.

        The order follows :attr:`ArticulationCfg.joint_ordering` when configured
        and otherwise matches :attr:`backend_joint_names`. Once the articulation
        installs its resolved names on :attr:`data`, those are returned directly;
        before that, the property falls back to :attr:`backend_joint_names`.
        """
        names = self.data.joint_names
        if names is not None:
            return list(names)
        return list(self.backend_joint_names)

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
    def body_names(self) -> list[str]:
        """Body names in public API order.

        The order follows :attr:`ArticulationCfg.body_ordering` when configured
        and otherwise matches :attr:`backend_body_names`. Once the articulation
        installs its resolved names on :attr:`data`, those are returned directly;
        before that, the property falls back to :attr:`backend_body_names`.
        """
        names = self.data.body_names
        if names is not None:
            return list(names)
        return list(self.backend_body_names)

    @property
    def backend_joint_names(self) -> list[str]:
        """Joint names in active backend solver-view order.

        Concrete backends must override this property so its order matches
        ``root_view`` metadata and joint-indexed solver arrays even when
        :attr:`joint_names` uses another public order.

        The inherited compatibility fallback emits :class:`DeprecationWarning`
        and returns :attr:`joint_names`. A subclass relying on that fallback
        therefore receives public order and cannot expose a distinct solver order.

        Raises:
            NotImplementedError: If the subclass overrides neither
                :attr:`joint_names` nor this property, since the two inherited
                fallbacks delegate to each other and cannot produce names.
        """
        if type(self).joint_names is BaseArticulation.joint_names:
            raise NotImplementedError(f"{type(self).__name__} must override joint_names or backend_joint_names.")
        warnings.warn(
            f"{type(self).__name__} must override backend_joint_names before it becomes abstract in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_names

    @property
    def backend_body_names(self) -> list[str]:
        """Body names in active backend solver-view order.

        Concrete backends must override this property so its order matches
        ``root_view`` metadata and body-indexed solver arrays even when
        :attr:`body_names` uses another public order.

        The inherited compatibility fallback emits :class:`DeprecationWarning`
        and returns :attr:`body_names`. A subclass relying on that fallback
        therefore receives public order and cannot expose a distinct solver order.

        Raises:
            NotImplementedError: If the subclass overrides neither
                :attr:`body_names` nor this property, since the two inherited
                fallbacks delegate to each other and cannot produce names.
        """
        if type(self).body_names is BaseArticulation.body_names:
            raise NotImplementedError(f"{type(self).__name__} must override body_names or backend_body_names.")
        warnings.warn(
            f"{type(self).__name__} must override backend_body_names before it becomes abstract in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_names

    @property
    def joint_ordering(self) -> ArticulationNameMap | None:
        """Bidirectional map between backend and public joint order.

        The map is ``None`` whenever the public and backend orders coincide:
        either no ordering is configured, or the configured ordering resolved
        to the backend's native order. A non-``None`` map always denotes an
        actual permutation.
        """
        return self.data.joint_ordering

    @property
    def body_ordering(self) -> ArticulationNameMap | None:
        """Bidirectional map between backend and public body order.

        The map is ``None`` whenever the public and backend orders coincide:
        either no ordering is configured, or the configured ordering resolved
        to the backend's native order. A non-``None`` map always denotes an
        actual permutation.
        """
        return self.data.body_ordering

    def map_joint_ids_to_backend(self, joint_ids: Sequence[int] | slice) -> Sequence[int] | slice:
        """Translate public joint indices to active-backend joint indices.

        Backend solver views expose joint metadata and joint-indexed arrays in
        :attr:`backend_joint_names` order, which can differ from the public
        :attr:`joint_names` order selected by :attr:`joint_ordering`. Consumers
        that pick joints with public indices (for example event terms) must
        convert those indices before addressing backend arrays.

        When :attr:`joint_ordering` is ``None`` the public and backend orders
        coincide and :paramref:`joint_ids` is returned unchanged without any
        per-index lookup.

        Args:
            joint_ids: Joint indices in public :attr:`joint_names` order, or a
                slice selecting them.

        Returns:
            The selected joint indices expressed in :attr:`backend_joint_names`
            order, or :paramref:`joint_ids` unchanged when the orders coincide.
            A slice is expanded to its backend indices under a permutation.
        """
        ordering = self.joint_ordering
        if ordering is None:
            return joint_ids
        if isinstance(joint_ids, slice):
            return list(ordering.user_to_backend_indices[joint_ids])
        return [ordering.user_to_backend_indices[joint_id] for joint_id in joint_ids]

    def map_body_ids_to_backend(self, body_ids: Sequence[int] | slice) -> Sequence[int] | slice:
        """Translate public body indices to active-backend body indices.

        Backend solver views expose body metadata and body-indexed arrays in
        :attr:`backend_body_names` order, which can differ from the public
        :attr:`body_names` order selected by :attr:`body_ordering`. Consumers
        that pick bodies with public indices (for example event terms) must
        convert those indices before addressing backend arrays.

        When :attr:`body_ordering` is ``None`` the public and backend orders
        coincide and :paramref:`body_ids` is returned unchanged without any
        per-index lookup.

        Args:
            body_ids: Body indices in public :attr:`body_names` order, or a
                slice selecting them.

        Returns:
            The selected body indices expressed in :attr:`backend_body_names`
            order, or :paramref:`body_ids` unchanged when the orders coincide.
            A slice is expanded to its backend indices under a permutation.
        """
        ordering = self.body_ordering
        if ordering is None:
            return body_ids
        if isinstance(body_ids, slice):
            return list(ordering.user_to_backend_indices[body_ids])
        return [ordering.user_to_backend_indices[body_id] for body_id in body_ids]

    @property
    @abstractmethod
    def root_view(self):
        """Root articulation view in active backend order.

        Name metadata and joint- or body-indexed arrays exposed by this view always
        use backend solver-view order, regardless of the configured public order.
        Use :attr:`joint_ordering` or :attr:`body_ordering` when converting axes.

        .. note::
            Use this view with caution. It requires handling backend tensors in
            the backend-specific way.
        """
        raise NotImplementedError()

    def _resolve_and_install_ordering_maps(self) -> None:
        """Resolve configured articulation name orderings and store maps on :attr:`data`.

        This is the single install site for ordering state: the resolved maps live
        only on :attr:`data` (:attr:`~BaseArticulationData.joint_ordering` and
        :attr:`~BaseArticulationData.body_ordering`) and every read path checks
        them there. Orderings that resolve to the backend's native order are
        normalized to ``None``, so a non-``None`` map always denotes an actual
        permutation and ``is not None`` checks alone decide whether reordering is
        active.
        """
        joint_names, joint_ordering = self._resolve_axis_ordering("joint")
        self.data.joint_names = joint_names
        self.data.joint_ordering = joint_ordering

        body_names, body_ordering = self._resolve_axis_ordering("body")
        if body_ordering is not None and self.is_fixed_base:
            root_body_name = self.backend_body_names[0]
            requested_index = body_ordering.backend_to_user_indices[0]
            if requested_index != 0:
                raise ValueError(
                    f"Invalid body_ordering for fixed-base articulation '{self.cfg.prim_path}': root body "
                    f"'{root_body_name}' must remain at public index 0, but was requested at index "
                    f"{requested_index}. Put '{root_body_name}' first; all remaining bodies may be reordered "
                    "freely."
                )
        self.data.body_names = body_names
        self.data.body_ordering = body_ordering

        self.data._apply_ordering_maps_after_resolve()

    def _resolve_axis_ordering(self, kind: Literal["joint", "body"]) -> tuple[list[str], ArticulationNameMap | None]:
        """Resolve one axis's public names and permutation from the configuration.

        Args:
            kind: Articulation element kind to resolve.

        Returns:
            The public names for the axis and its permutation map, where ``None``
            means public order equals backend order (no ordering configured, or a
            configured ordering that resolved to the backend's native order).
        """
        backend_names = tuple(self.backend_joint_names if kind == "joint" else self.backend_body_names)
        cfg_ordering = self.cfg.joint_ordering if kind == "joint" else self.cfg.body_ordering
        if cfg_ordering is None:
            return list(backend_names), None

        user_names = _resolve_articulation_ordering_names(
            kind=kind,
            backend_names=backend_names,
            ordering=cfg_ordering,
            active_backend_name=self.__backend_name__,
            articulation=self,
        )
        ordering = build_articulation_name_map(
            kind=kind, backend_names=backend_names, user_names=user_names, device=self.device
        )
        if ordering is None:
            logger.info(
                "Configured %s_ordering for '%s' resolves to the backend's native order; no reordering"
                " will be applied.",
                kind,
                self.cfg.prim_path,
            )
        return list(user_names), ordering

    @property
    def num_base_dofs(self) -> int:
        """Number of free DoFs of the floating base.

        A floating-base articulation can translate and rotate freely in space, so
        its base contributes 6 DoFs (3 linear, 3 angular). A fixed-base articulation
        is bolted to the world and contributes 0.

        Use this to map an actuated-joint index ``j`` to its column in the Jacobian
        / mass matrix / gravity vector: ``column = j + num_base_dofs``.
        """
        return 0 if self.is_fixed_base else 6

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
        """Reset the articulation.

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

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
        self,
        name_keys: str | Sequence[str],
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :func:`isaaclab.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.
            as_proxy: Whether to return cached, immutable :class:`ProxyArray` indices. Use ``.warp`` or
                ``.torch`` and reacquire after invalidation. Defaults to False.

        Returns:
            Matched body indices and names. Indices are a list by default or a cached proxy when requested.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: list[str] | None = None,
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.
            as_proxy: Whether to return proxy indices. See :meth:`find_bodies`. Subsets use asset-global
                proxy indices.

        Returns:
            Matched joint indices and names. Indices are a list by default or a cached proxy when requested.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_fixed_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        """Find fixed tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint
                names with fixed tendons.
            tendon_subsets: A subset of joints with fixed tendons to search for. Defaults to None, which means
                all joints in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.
            as_proxy: Whether to return proxy indices. See :meth:`find_bodies`. Subsets use asset-global
                proxy indices.

        Returns:
            Matched fixed-tendon indices and names. Indices are a list by default or a cached proxy when requested.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_spatial_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        """Find spatial tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the tendon names.
            tendon_subsets: A subset of tendons to search for. Defaults to None, which means all tendons
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.
            as_proxy: Whether to return proxy indices. See :meth:`find_bodies`. Subsets use asset-global
                proxy indices.

        Returns:
            Matched spatial-tendon indices and names. Indices are a list by default or a cached proxy when requested.
        """
        raise NotImplementedError()

    """
    Operations - State Writers.
    """

    @abstractmethod
    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
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
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
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
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
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
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
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
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
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
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
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
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
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
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
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
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_position_to_sim_index(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Write joint positions to the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all instances).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_position_to_sim_mask(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Write joint positions to the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_velocity_to_sim_index(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Write joint velocities to the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all instances).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_velocity_to_sim_mask(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Write joint velocities to the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        raise NotImplementedError()

    """
    Operations - Simulation Parameters Writers.
    """

    @abstractmethod
    def write_joint_stiffness_to_sim_index(
        self,
        *,
        stiffness: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint stiffness into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_stiffness_to_sim_mask(
        self,
        *,
        stiffness: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint stiffness into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            stiffness: Joint stiffness. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_damping_to_sim_index(
        self,
        *,
        damping: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint damping into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the damping for. Defaults to None (all joints).
            env_ids: The environment indices to set the damping for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_damping_to_sim_mask(
        self,
        *,
        damping: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint damping into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            damping: Joint damping. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_position_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Write joint position limits into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2) or (len(env_ids), len(joint_ids)) with
                dtype wp.vec2f.
            joint_ids: The joint indices to set the limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the limits for. Defaults to None (all instances).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_position_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Write joint position limits into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limits: Joint limits. Shape is (num_instances, num_joints, 2) or (num_instances, num_joints) with dtype
                wp.vec2f.
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_velocity_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the max velocity for. Defaults to None (all joints).
            env_ids: The environment indices to set the max velocity for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_velocity_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limits: Joint max velocity. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_effort_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_effort_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limits: Joint torque limits. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_armature_to_sim_index(
        self,
        *,
        armature: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_armature_to_sim_mask(
        self,
        *,
        armature: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            armature: Joint armature. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_friction_coefficient_to_sim_index(
        self,
        *,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        r"""Write backend-specific joint friction values into the simulation.

        .. warning::
            The physical meaning and units of joint friction depend on the concrete backend and solver. Do not assume
            values are comparable across backends; check the backend-specific implementation before interpreting or
            reusing them.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            joint_friction_coeff: Backend-specific joint friction values. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_joint_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        r"""Write backend-specific joint friction values into the simulation.

        .. warning::
            The physical meaning and units of joint friction depend on the concrete backend and solver. Do not assume
            values are comparable across backends; check the backend-specific implementation before interpreting or
            reusing them.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            joint_friction_coeff: Backend-specific joint friction values. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
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
        """Set masses of all bodies in the simulation world frame.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all instances).
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
        """Set masses of all bodies in the simulation world frame.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all the bodies are updated. Shape is (num_bodies,).
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
        """Set center of mass pose of all bodies in their respective body link frames.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None
                (all instances).
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
        """Set center of mass pose of all bodies in their respective body link frames.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all the bodies are updated. Shape is (num_bodies,).
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
        """Set inertias of all bodies in the simulation world frame.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all instances).
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
        """Set inertias of all bodies in the simulation world frame.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all the bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    @leapp_tensor_semantics(kind=OutputKindEnum.JOINT_POSITION, element_names_resolver=joint_names_resolver)
    def set_joint_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    @leapp_tensor_semantics(kind=OutputKindEnum.JOINT_POSITION, element_names_resolver=joint_names_resolver)
    def set_joint_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    @leapp_tensor_semantics(kind=OutputKindEnum.JOINT_VELOCITY, element_names_resolver=joint_names_resolver)
    def set_joint_velocity_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    @leapp_tensor_semantics(kind=OutputKindEnum.JOINT_VELOCITY, element_names_resolver=joint_names_resolver)
    def set_joint_velocity_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    @leapp_tensor_semantics(kind=OutputKindEnum.JOINT_EFFORT, element_names_resolver=joint_names_resolver)
    def set_joint_effort_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    @leapp_tensor_semantics(kind=OutputKindEnum.JOINT_EFFORT, element_names_resolver=joint_names_resolver)
    def set_joint_effort_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (num_instances, num_joints).
            joint_mask: Joint mask. If None, then all the joints are updated. Shape is (num_joints,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    """
    Operations - Tendons.
    """

    @abstractmethod
    def set_fixed_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            damping: Fixed tendon damping. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_index` function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim_mask` function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_position_limit_index(
        self,
        *,
        limit: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon position limits into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the limit for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_position_limit_mask(
        self,
        *,
        limit: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon position limits into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limit: Fixed tendon limit. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_rest_length_index(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_rest_length_mask(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon rest length into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            rest_length: Fixed tendon rest length. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon offset into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_fixed_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set fixed tendon offset into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            offset: Fixed tendon offset. Shape is (num_instances, num_fixed_tendons).
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_fixed_tendon_properties_to_sim_index(
        self,
        *,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the limits for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limits for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_fixed_tendon_properties_to_sim_mask(
        self,
        *,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write fixed tendon properties into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            fixed_tendon_mask: Fixed tendon mask. If None, then all the fixed tendons are updated.
                Shape is (num_fixed_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all the spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the damping for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            damping: Spatial tendon damping. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all the spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim_index` function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None
                (all spatial tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim_mask` function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all the spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim_index`
        function.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_spatial_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set spatial tendon offset into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim_mask`
        function.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            offset: Spatial tendon offset. Shape is (num_instances, num_spatial_tendons).
            spatial_tendon_mask: Spatial tendon mask. If None, then all the spatial tendons are updated.
                Shape is (num_spatial_tendons,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_spatial_tendon_properties_to_sim_index(
        self,
        *,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            spatial_tendon_ids: The spatial tendon indices to set the properties for. Defaults to None
                (all spatial tendons).
            env_ids: The environment indices to set the properties for. Defaults to None (all instances).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_spatial_tendon_properties_to_sim_mask(
        self,
        *,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write spatial tendon properties into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            spatial_tendon_mask: Spatial tendon mask. If None, then all the spatial tendons are updated.
                Shape is (num_spatial_tendons,).
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
    Internal helpers -- Ordering.
    """

    _ordering_joint_staging_names: tuple[str, ...] = ()
    """Backend-order joint-target staging attributes managed by :meth:`_ordering_configure_backend_staging`.

    Each named attribute is a 2-D ``(num_instances, num_joints)`` ``wp.float32``
    staging buffer, allocated while a nonidentity joint ordering is active and set
    to ``None`` otherwise. Backends override this tuple to declare their staging;
    the default empty tuple keeps identity-only backends at a no-op.
    """

    def _joint_user_to_backend_map(self) -> wp.array:
        """Device public-to-backend joint map, or the identity map when no ordering is active.

        The identity fallback reads ``_ALL_JOINT_INDICES``, an index buffer that
        backends allocate without a base-class default (a backend-owned contract,
        like the staging attributes named by :attr:`_ordering_joint_staging_names`),
        so unconditional launch sites -- for example in-graph publish kernels that
        always gather -- get a valid map either way. Conditional sites should read
        :attr:`~BaseArticulationData.joint_ordering` directly and skip the launch
        when it is ``None``.
        """
        ordering = self.data.joint_ordering
        return ordering.user_to_backend if ordering is not None else self._ALL_JOINT_INDICES

    def _joint_backend_to_user_map(self) -> wp.array:
        """Device backend-to-public joint map, or the identity map when no ordering is active.

        See :meth:`_joint_user_to_backend_map` for the identity-fallback contract.
        """
        ordering = self.data.joint_ordering
        return ordering.backend_to_user if ordering is not None else self._ALL_JOINT_INDICES

    def _body_user_to_backend_map(self) -> wp.array:
        """Device public-to-backend body map, or the identity map when no ordering is active.

        See :meth:`_joint_user_to_backend_map` for the identity-fallback contract;
        the body fallback reads ``_ALL_BODY_INDICES``.
        """
        ordering = self.data.body_ordering
        return ordering.user_to_backend if ordering is not None else self._ALL_BODY_INDICES

    def _body_backend_to_user_map(self) -> wp.array:
        """Device backend-to-public body map, or the identity map when no ordering is active.

        See :meth:`_body_user_to_backend_map` for the identity-fallback contract.
        """
        ordering = self.data.body_ordering
        return ordering.backend_to_user if ordering is not None else self._ALL_BODY_INDICES

    def _ordering_configure_backend_staging(self) -> None:
        """Allocate or release backend-order joint-target staging after ordering maps change.

        Iterates :attr:`_ordering_joint_staging_names`, allocating each declared
        staging buffer while a nonidentity joint ordering is active and clearing it
        to ``None`` otherwise. Backends that also stage body-indexed buffers extend
        this via ``super()`` before handling their own body staging.
        """
        for backend_name in self._ordering_joint_staging_names:
            if not hasattr(self, backend_name):
                continue
            if self.data.joint_ordering is not None:
                if getattr(self, backend_name) is None:
                    setattr(
                        self,
                        backend_name,
                        wp.zeros((self.num_instances, self.num_joints), dtype=wp.float32, device=self.device),
                    )
            else:
                setattr(self, backend_name, None)

    def _get_backend_ordered_joint_buffer(
        self,
        user_buffer: wp.array,
        backend_buffer: wp.array | TimestampedBufferWarp | None,
        *,
        component_count: int | None = None,
    ) -> wp.array:
        """Return a backend-order view or copy of a public-order joint buffer.

        When :paramref:`component_count` is ``None`` the buffer is treated as
        two-dimensional (per joint); otherwise it is treated as three-dimensional
        with that many trailing components per joint.

        Avoid this modify-then-reorder helper on per-step hot paths: it costs a
        full extra kernel launch over the buffer. Prefer the fused
        ``write_*_user_to_backend`` write paths from
        :mod:`isaaclab.assets.articulation.ordering_kernels`, which write the
        public and backend buffers in a single launch. This helper exists for
        infrequent property writers where the extra launch is irrelevant.

        Args:
            user_buffer: Public-order joint buffer to reorder.
            backend_buffer: Backend-order staging destination, either a raw Warp
                array or a :class:`~isaaclab.utils.buffers.TimestampedBufferWarp`.
                Required when the articulation has a non-identity joint ordering.
            component_count: Number of trailing components per joint for a
                three-dimensional buffer, or ``None`` for a two-dimensional buffer.

        Returns:
            :paramref:`user_buffer` when the joint ordering is identity; otherwise
            the backend-order staging array populated from :paramref:`user_buffer`.

        Raises:
            RuntimeError: If a non-identity joint ordering is active but no backend
                staging buffer was provided.
        """
        ordering = self.data.joint_ordering
        if ordering is None:
            return user_buffer
        if backend_buffer is None:
            detail = "backend staging" if component_count is None else "3-D backend staging"
            raise RuntimeError(f"{self.__backend_name__} joint ordering requires {detail}.")
        backend_data = backend_buffer.data if isinstance(backend_buffer, TimestampedBufferWarp) else backend_buffer
        if component_count is None:
            wp.launch(
                ordering_kernels.reorder_2d_user_to_backend,
                dim=(self.num_instances, self.num_joints),
                inputs=[user_buffer, ordering.backend_to_user],
                outputs=[backend_data],
                device=self.device,
            )
        else:
            wp.launch(
                ordering_kernels.reorder_3d_user_to_backend,
                dim=(self.num_instances, self.num_joints, component_count),
                inputs=[user_buffer, ordering.backend_to_user],
                outputs=[backend_data],
                device=self.device,
            )
        return backend_data

    @abstractmethod
    def _process_actuators_cfg(self) -> None:
        """Process and apply articulation joint properties."""
        raise NotImplementedError()

    @abstractmethod
    def _process_tendons(self) -> None:
        """Process fixed and spatial tendons."""
        raise NotImplementedError()

    @abstractmethod
    def _apply_actuator_model(self) -> None:
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        raise NotImplementedError()

    """
    Internal helpers -- Debugging.
    """

    @abstractmethod
    def _validate_cfg(self) -> None:
        """Validate the configuration after processing.

        .. note::
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        raise NotImplementedError()

    @abstractmethod
    def _log_articulation_info(self) -> None:
        """Log information about the articulation.

        .. note::
            We purposefully read the values from the simulator to ensure that the values are configured as expected.
        """
        raise NotImplementedError()

    """
    Deprecated methods.
    """

    def write_joint_friction_to_sim(
        self,
        joint_friction: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint friction coefficients into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_friction_coefficient_to_sim` instead.
        """
        warnings.warn(
            "The function 'write_joint_friction_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_friction_coefficient_to_sim' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_friction_coefficient_to_sim(joint_friction, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_limits_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Write joint limits into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_position_limit_to_sim` instead.
        """
        warnings.warn(
            "The function 'write_joint_limits_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_position_limit_to_sim' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_position_limit_to_sim(
            limits,
            joint_ids=joint_ids,
            env_ids=env_ids,
            warn_limit_violation=warn_limit_violation,
        )

    def set_fixed_tendon_limit(
        self,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set fixed tendon position limits into internal buffers.

        .. deprecated:: 2.1.0
            Please use :meth:`set_fixed_tendon_position_limit` instead.
        """
        warnings.warn(
            "The function 'set_fixed_tendon_limit' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_position_limit' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_position_limit(
            limit,
            fixed_tendon_ids=fixed_tendon_ids,
            env_ids=env_ids,
        )

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
        """Deprecated, same as :meth:`write_root_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index`."""
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

    @abstractmethod
    def write_joint_state_to_sim(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_position_to_sim_index` and
        :meth:`write_joint_velocity_to_sim_index`."""
        raise NotImplementedError()

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_position_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_position_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_position_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_velocity_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_stiffness_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_stiffness_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_stiffness_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_stiffness_to_sim_index(stiffness=stiffness, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_damping_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_damping_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_damping_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_damping_to_sim_index(damping=damping, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_position_limit_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_position_limit_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_position_limit_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_position_limit_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_position_limit_to_sim_index(
            limits=limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=warn_limit_violation
        )

    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_velocity_limit_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_velocity_limit_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_velocity_limit_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_velocity_limit_to_sim_index(limits=limits, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_effort_limit_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_effort_limit_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_effort_limit_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_effort_limit_to_sim_index(limits=limits, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_joint_armature_to_sim_index`."""
        warnings.warn(
            "The function 'write_joint_armature_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_armature_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
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
            joint_friction_coeff=joint_friction_coeff, joint_ids=joint_ids, env_ids=env_ids
        )

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
        """Deprecated. Resets target environments, then adds forces and torques via the permanent wrench composer."""
        warnings.warn(
            "The function 'set_external_force_and_torque' is deprecated. Please use"
            " 'permanent_wrench_composer.reset' followed by 'permanent_wrench_composer.add_forces_and_torques'"
            " instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Reset only target env_ids then add (not set which clears all envs globally)
        self.permanent_wrench_composer.reset(env_ids=env_ids)
        self.permanent_wrench_composer.add_forces_and_torques(
            forces, torques, positions=positions, body_ids=body_ids, env_ids=env_ids, is_global=is_global
        )

    def set_joint_position_target(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_joint_position_target_index`."""
        warnings.warn(
            "The function 'set_joint_position_target' will be deprecated in a future release. Please"
            " use 'set_joint_position_target_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)

    def set_joint_velocity_target(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_joint_velocity_target_index`."""
        warnings.warn(
            "The function 'set_joint_velocity_target' will be deprecated in a future release. Please"
            " use 'set_joint_velocity_target_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_joint_velocity_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)

    def set_joint_effort_target(
        self,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_joint_effort_target_index`."""
        warnings.warn(
            "The function 'set_joint_effort_target' will be deprecated in a future release. Please"
            " use 'set_joint_effort_target_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_joint_effort_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)

    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_fixed_tendon_stiffness_index`."""
        warnings.warn(
            "The function 'set_fixed_tendon_stiffness' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_stiffness_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_stiffness_index(stiffness=stiffness, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids)

    def set_fixed_tendon_damping(
        self,
        damping: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_fixed_tendon_damping_index`."""
        warnings.warn(
            "The function 'set_fixed_tendon_damping' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_damping_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_damping_index(damping=damping, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids)

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_fixed_tendon_limit_stiffness_index`."""
        warnings.warn(
            "The function 'set_fixed_tendon_limit_stiffness' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_limit_stiffness_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_limit_stiffness_index(
            limit_stiffness=limit_stiffness, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids
        )

    def set_fixed_tendon_position_limit(
        self,
        limit: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_fixed_tendon_position_limit_index`."""
        warnings.warn(
            "The function 'set_fixed_tendon_position_limit' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_position_limit_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_position_limit_index(limit=limit, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids)

    def set_fixed_tendon_rest_length(
        self,
        rest_length: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_fixed_tendon_rest_length_index`."""
        warnings.warn(
            "The function 'set_fixed_tendon_rest_length' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_rest_length_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_rest_length_index(
            rest_length=rest_length, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids
        )

    def set_fixed_tendon_offset(
        self,
        offset: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_fixed_tendon_offset_index`."""
        warnings.warn(
            "The function 'set_fixed_tendon_offset' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_offset_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fixed_tendon_offset_index(offset=offset, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids)

    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_fixed_tendon_properties_to_sim_index`."""
        warnings.warn(
            "The function 'write_fixed_tendon_properties_to_sim' will be deprecated in a future release. Please"
            " use 'write_fixed_tendon_properties_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Removing the fixed tendon ids argument as it is not used.
        self.write_fixed_tendon_properties_to_sim_index(env_ids=env_ids)

    def set_spatial_tendon_stiffness(
        self,
        stiffness: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_spatial_tendon_stiffness_index`."""
        warnings.warn(
            "The function 'set_spatial_tendon_stiffness' will be deprecated in a future release. Please"
            " use 'set_spatial_tendon_stiffness_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_spatial_tendon_stiffness_index(
            stiffness=stiffness, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids
        )

    def set_spatial_tendon_damping(
        self,
        damping: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_spatial_tendon_damping_index`."""
        warnings.warn(
            "The function 'set_spatial_tendon_damping' will be deprecated in a future release. Please"
            " use 'set_spatial_tendon_damping_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_spatial_tendon_damping_index(damping=damping, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids)

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_spatial_tendon_limit_stiffness_index`."""
        warnings.warn(
            "The function 'set_spatial_tendon_limit_stiffness' will be deprecated in a future release. Please"
            " use 'set_spatial_tendon_limit_stiffness_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_spatial_tendon_limit_stiffness_index(
            limit_stiffness=limit_stiffness, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids
        )

    def set_spatial_tendon_offset(
        self,
        offset: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`set_spatial_tendon_offset_index`."""
        warnings.warn(
            "The function 'set_spatial_tendon_offset' will be deprecated in a future release. Please"
            " use 'set_spatial_tendon_offset_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_spatial_tendon_offset_index(offset=offset, spatial_tendon_ids=spatial_tendon_ids, env_ids=env_ids)

    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_spatial_tendon_properties_to_sim_index`."""
        warnings.warn(
            "The function 'write_spatial_tendon_properties_to_sim' will be deprecated in a future release. Please"
            " use 'write_spatial_tendon_properties_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Removing the spatial tendon ids argument as it is not used.
        self.write_spatial_tendon_properties_to_sim_index(env_ids=env_ids)
