# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from ... import cloner
from ... import sim as sim_utils
from ...utils import string as string_utils
from ...utils.warp import ProxyArray
from ..sensor_base import SensorBase
from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg

_GLOB_WILDCARD = re.compile(r"(?<![.\])])\*")
"""Matches a glob ``*``: one that does not repeat a regex ``.``, character class, or group."""


class BaseContactSensor(SensorBase):
    """A contact reporting sensor.

    The contact sensor reports the normal contact forces on a rigid body in the world frame.
    It relies on the physics engine's contact reporting API to be activated on the rigid bodies.

    To enable the contact reporter on a rigid body, please make sure to enable the
    :attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sensors` on your
    asset spawner configuration. This will enable the contact reporter on all the rigid bodies
    in the asset.

    The sensor can be configured to report the contact forces on a set of bodies with a given
    filter pattern using the :attr:`ContactSensorCfg.filter_prim_paths_expr`. This is useful
    when you want to report the contact forces between the sensor bodies and a specific set of
    bodies in the scene. The data can be accessed using the :attr:`ContactSensorData.normal_force_matrix_w`.

    Filtered contact reporting is many-to-many: every sensor body reports its contact forces
    against every filter body.
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the contact sensor."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        if cfg.history_length < 0:
            raise ValueError(f"History length must be greater than 0! Received: {cfg.history_length}")

    def update(self, dt: float, force_recompute: bool = False) -> None:
        """Advance the sensor timestamp and update history-bearing buffers.

        Args:
            dt: Time elapsed since the previous sensor update [s].
            force_recompute: Whether to recompute the sensor buffers regardless of their
                configured update period. Defaults to False.
        """
        super().update(dt, force_recompute=force_recompute)
        if self._is_initialized and self.cfg.history_length > 0:
            self._update_outdated_buffers()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self.num_sensors}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    @abstractmethod
    def num_instances(self) -> int | None:
        """Number of instances of the sensor."""
        raise NotImplementedError(f"Num instances is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def data(self) -> BaseContactSensorData:
        """Data from the sensor."""
        raise NotImplementedError(f"Data is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def num_sensors(self) -> int:
        """Number of sensors per environment."""
        raise NotImplementedError(f"Num sensors is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def body_names(self) -> list[str] | None:
        """Ordered names of shapes or bodies with contact sensors attached."""
        raise NotImplementedError(f"Body names is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def contact_view(self) -> None:
        """View for the contact forces captured.

        .. note::
            None if there is no view associated with the sensor.
        """
        raise NotImplementedError(f"Contact view is not implemented for {self.__class__.__name__}.")

    """
    Operations
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array(dtype=wp.bool) | None = None):
        """Resets the sensor.

        Args:
            env_ids: The indices of the environments to reset. Defaults to None: all the environments are reset.
            env_mask: The masks of the environments to reset. Defaults to None: all the environments are reset.
        """
        # reset the timers and counters
        super().reset(env_ids, env_mask)

    def find_sensors(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find sensors in the contact sensor based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the sensor indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def _resolve_first_transition_tolerance(self, abs_tol: float | None) -> float:
        """Resolves the tolerance used to detect a first contact or first air transition.

        Valid air and contact timers are integer multiples of the sensor update interval, so half an
        interval is the midpoint between "one update ago" and "two updates ago". Using it as the
        tolerance keeps the comparison robust to the float32 rounding of the timers, which exceeds
        any small fixed tolerance for periods of a few tenths of a second.

        Args:
            abs_tol: The caller-provided tolerance [s]. If None, half the sensor update interval
                is used.

        Returns:
            The absolute tolerance to add to the queried time period [s].
        """
        if abs_tol is not None:
            return abs_tol
        # An update period of 0.0 means the sensor is updated on every physics step.
        return 0.5 * max(self.cfg.update_period, self._sim_physics_dt)

    def _resolve_filter_body_exprs(self) -> list[str]:
        """Resolves :attr:`ContactSensorCfg.filter_prim_paths_expr` into one path expression per filter body.

        The physics views turn each expression into one filter column, so an expression must match a single
        prim per environment. An expression whose last segment is a regular expression and which has no glob
        ``*`` is therefore resolved from the stage, clone-plan aware like the sensor bodies: the segment is
        matched against the names of the prims at any depth below the prims matched by the rest of the
        expression, which are not candidates themselves. Rigid bodies match, and so do colliders without a
        rigid-body ancestor (static geometry such as a ground plane); colliders, visuals, and other prims inside
        a rigid body do not. If several prims match, each gets its own expression, which keeps the environment
        segment of the original.

        Other expressions, and expressions that match at most one such prim, are returned unchanged.

        Returns:
            The expressions in the configured order, the bodies of one expression in breadth-first order.
        """
        from pxr import UsdPhysics  # noqa: PLC0415

        def is_rigid_body(prim) -> bool:
            return prim.HasAPI(UsdPhysics.RigidBodyAPI)

        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        body_exprs = []
        for path_expr in self.cfg.filter_prim_paths_expr:
            *parent_segments, leaf_pattern = sim_utils.split_path_expr(path_expr)
            parent_expr = "/".join(parent_segments)
            # the physics views match a leaf name at any depth themselves, and globs are their syntax
            if leaf_pattern.isidentifier() or not parent_expr.startswith("/") or _GLOB_WILDCARD.search(path_expr):
                body_exprs.append(path_expr)
                continue
            name_pattern = re.compile(leaf_pattern)

            def is_partner(prim) -> bool:
                if not name_pattern.fullmatch(prim.GetName()):
                    return False
                body = sim_utils.get_first_matching_ancestor_prim(prim.GetPath(), is_rigid_body, prim.GetStage())
                return body == prim if body is not None else prim.HasAPI(UsdPhysics.CollisionAPI)

            bodies = {}
            parents = sim_utils.resolve_matching_prims_from_source(parent_expr, raise_if_no_matches=False)
            for parent, parent_dest in parents:
                root = parent.GetPath().pathString
                prims = sim_utils.get_all_matching_child_prims(root, is_partner, traverse_instance_prims=False)
                # only the prims below the parent, each once
                for path in (prim.GetPath().pathString for prim in prims):
                    if path != root:
                        bodies.setdefault(path, parent_dest + path[len(root) :])
            exprs = list(bodies.values())
            if len(exprs) < 2:
                # at most one body: the expression stays one pattern, matched by the views
                body_exprs.append(path_expr)
                continue
            world = None if plan is None else cloner.path.match(path_expr, plan.env_template)
            if world is not None:
                # the clone plan resolves the environment segment to a wildcard; keep the expression's own
                source_root, own_root = plan.env_template.format("[^/]+"), plan.env_template.format(world.instance)
                exprs = [cloner.path.rebase(expr, source_root, own_root) for expr in exprs]
            body_exprs.extend(exprs)
        return body_exprs

    @abstractmethod
    def compute_first_contact(self, dt: float, abs_tol: float | None = None) -> ProxyArray:
        """Checks if bodies that have established contact within the last :attr:`dt` seconds.

        This function checks if the bodies have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the bodies are considered to be in contact. Outdated sensor
        buffers are refreshed before the comparison.

        .. note::
            The function assumes that :attr:`dt` is a factor of the sensor update time-step. In other
            words :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contact was established.
            abs_tol: The absolute tolerance for the comparison [s]. Defaults to None, in which case
                half the sensor update interval is used.

        Returns:
            A boolean tensor indicating the bodies that have established contact within the last
            :attr:`dt` seconds. Shape is (N, B), where N is the number of sensors and B is the
            number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        raise NotImplementedError(f"Compute first contact is not implemented for {self.__class__.__name__}.")

    @abstractmethod
    def compute_first_air(self, dt: float, abs_tol: float | None = None) -> ProxyArray:
        """Checks if bodies that have broken contact within the last :attr:`dt` seconds.

        This function checks if the bodies have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the bodies are considered to not be in contact. Outdated sensor
        buffers are refreshed before the comparison.

        .. note::
            It assumes that :attr:`dt` is a factor of the sensor update time-step. In other words,
            :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true if
            the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison [s]. Defaults to None, in which case
                half the sensor update interval is used.

        Returns:
            A boolean tensor indicating the bodies that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        raise NotImplementedError(f"Compute first air is not implemented for {self.__class__.__name__}.")

    """
    Implementation.
    """

    @abstractmethod
    def _initialize_impl(self):
        super()._initialize_impl()

    @abstractmethod
    def _create_buffers(self):
        """Creates the buffers for the sensor data."""
        raise NotImplementedError(f"Create buffers is not implemented for {self.__class__.__name__}.")

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array | None):
        """Fills the buffers of the sensor data.

        Args:
            env_mask: Mask of the environments to update. None: update all environments.
        """
        raise NotImplementedError(f"Update buffers is not implemented for {self.__class__.__name__}.")

    @property
    def num_bodies(self) -> int:
        """Deprecated property. Please use `num_sensors` instead."""
        warnings.warn(
            "The `num_bodies` property will be deprecated in a future release. Please use `num_sensors` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_sensors

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Deprecated method. Please use `find_sensors` instead."""
        warnings.warn(
            "The `find_bodies` method will be deprecated in a future release. Please use `find_sensors` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_sensors(name_keys, preserve_order)
