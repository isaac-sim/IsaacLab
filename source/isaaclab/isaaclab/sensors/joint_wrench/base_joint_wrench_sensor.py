# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from pxr import UsdPhysics

from ...assets.articulation.ordering import ArticulationNameMap, build_articulation_name_map
from ...sim.utils.queries import resolve_matching_prims_from_source
from ...utils import string as string_utils
from ..sensor_base import SensorBase
from .base_joint_wrench_sensor_data import BaseJointWrenchSensorData

if TYPE_CHECKING:
    from ...assets.articulation import BaseArticulation
    from .joint_wrench_sensor_cfg import JointWrenchSensorCfg


class BaseJointWrenchSensor(SensorBase):
    """The joint reaction wrench sensor.

    Reports incoming joint wrenches for the bodies selected by the backend as
    split force [N] / torque [N·m] pairs expressed in the
    ``INCOMING_JOINT_FRAME`` convention (child-side joint frame, child-side
    joint anchor reference point). Backends convert from their native
    representation to this convention internally. In an interactive scene,
    entries follow the owning articulation's public body order, restricted to
    reportable bodies. Standalone sensors retain native sensor order. Use :attr:`body_names` or
    :meth:`find_bodies` to map entries to articulation bodies.
    """

    cfg: JointWrenchSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the joint wrench sensor."""

    def __init__(self, cfg: JointWrenchSensorCfg) -> None:
        """Initializes the joint wrench sensor object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        self._scene_articulations: tuple[BaseArticulation, ...] = ()
        self._body_ordering: ArticulationNameMap | None = None

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseJointWrenchSensorData:
        """The sensor data container, populated after simulation initialization."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of the bodies whose incoming joint wrench is reported."""
        raise NotImplementedError

    @property
    def num_bodies(self) -> int:
        """Number of bodies whose incoming joint wrench is reported."""
        return len(self.body_names)

    """
    Operations
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find reported bodies based on name keys.

        Args:
            name_keys: A regular expression or list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            The matching body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def set_scene_articulations(self, articulations: Sequence[BaseArticulation]) -> None:
        """Set the articulations whose public body order the sensor can follow.

        :class:`~isaaclab.scene.InteractiveScene` calls this before initialization. The sensor follows
        the articulation whose single root is the sensor's articulation root.

        Args:
            articulations: Candidate articulations of the scene.
        """
        self._scene_articulations = tuple(articulations)

    """
    Implementation - Abstract methods to be implemented by backend-specific subclasses.
    """

    @abstractmethod
    def _initialize_impl(self) -> None:
        """Initialize the sensor handles and internal buffers.

        Subclasses should call ``super()._initialize_impl()`` first to
        initialize the common sensor infrastructure from
        :class:`~isaaclab.sensors.SensorBase`.
        """
        super()._initialize_impl()

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        raise NotImplementedError

    """
    Helper functions.
    """

    def _initialize_body_ordering(self, backend_names: list[str], root_prim_path_expr: str) -> None:
        """Order reportable bodies by the scene articulation at the same USD root.

        Args:
            backend_names: Reported body names in backend sensor order.
            root_prim_path_expr: Articulation root expression resolved by the sensor.
        """
        self._body_ordering = None
        self._data._body_names = backend_names
        if not self._scene_articulations:
            return
        sensor_root = resolve_matching_prims_from_source(root_prim_path_expr, expected_num_matches=1)[0][0]
        for articulation in self._scene_articulations:
            root_expr = articulation.cfg.prim_path + (articulation.cfg.articulation_root_prim_path or "")
            roots = resolve_matching_prims_from_source(
                root_expr,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                raise_if_no_matches=False,
            )
            # multi-root articulations never own a single sensor root
            if [root.GetPath() for root, _ in roots] != [sensor_root.GetPath()]:
                continue
            if not articulation.is_initialized:
                raise RuntimeError("The owning articulation must initialize before its joint-wrench sensor.")
            reportable_names = set(backend_names)
            user_names = [name for name in articulation.body_names if name in reportable_names]
            if user_names != backend_names:
                self._body_ordering = build_articulation_name_map(
                    kind="body", backend_names=backend_names, user_names=user_names, device=self._device
                )
                self._data._body_names = user_names
            return
