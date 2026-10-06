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

    def __init__(self, cfg: JointWrenchSensorCfg):
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

    """
    Implementation - Abstract methods to be implemented by backend-specific subclasses.
    """

    def _initialize_body_ordering(self, native_names: list[str], root_prim_path_expr: str) -> None:
        """Order reportable bodies by the scene articulation at the same USD root."""
        public_names = native_names
        if self._scene_articulations:
            sensor_root = resolve_matching_prims_from_source(root_prim_path_expr, expected_num_matches=1)[0][0]
            for articulation in self._scene_articulations:
                root_expr = articulation.cfg.prim_path
                if articulation.cfg.articulation_root_prim_path is not None:
                    root_expr += articulation.cfg.articulation_root_prim_path
                root = resolve_matching_prims_from_source(
                    root_expr,
                    predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                    expected_num_matches=1,
                )[0][0]
                if root.GetPath() == sensor_root.GetPath():
                    if not articulation.is_initialized:
                        raise RuntimeError("The owning articulation must initialize before its joint-wrench sensor.")
                    reportable_names = set(native_names)
                    public_names = [name for name in articulation.body_names if name in reportable_names]
                    break
        self._body_ordering = build_articulation_name_map(
            kind="body", backend_names=native_names, user_names=public_names, device=self._device
        )
        self._data._body_names = public_names

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
