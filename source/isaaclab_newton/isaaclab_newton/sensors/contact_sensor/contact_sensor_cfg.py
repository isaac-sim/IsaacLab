# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import TYPE_CHECKING

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as BaseContactSensorCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(BaseContactSensorCfg):
    """Configuration for the Newton contact sensor with shape-level support.

    Extends :class:`ContactSensorCfg` with shape-level fields for finer-grained
    contact reporting. A body is a rigid body; a shape is an individual collision geometry
    attached to a body. A single body can have multiple shapes.

    Sensing objects (what to measure forces on):

    - :attr:`sensor_body_prim_expr` — read-only alias for :attr:`prim_path` (body-level sensing).
    - :attr:`sensor_shape_prim_expr` — optional shape-level sensing. If set, takes
      precedence over :attr:`prim_path`.

    Filter partners (what to measure forces against):

    - :attr:`filter_prim_paths_expr` — body-level filter (inherited from :class:`ContactSensorCfg`).
    - :attr:`filter_shape_prim_expr` — shape-level filter.

    An instance can be created from an existing :class:`ContactSensorCfg` via
    :meth:`from_base_cfg`.
    """

    class_type: type["ContactSensor"] | str = "{DIR}.contact_sensor:ContactSensor"

    @property
    def sensor_body_prim_expr(self) -> str:
        """Read-only alias for :attr:`prim_path`."""
        return self.prim_path

    sensor_shape_prim_expr: list[str] = []
    """List of shape prim path expressions for shape-level contact sensing.
    Defaults to empty, meaning sensing is at the body level (via :attr:`prim_path`).

    Mutually exclusive with body-level sensing: if non-empty, :attr:`prim_path` is ignored
    for the sensing objects and these shape expressions are used instead.

    .. note::
        Expressions can contain the environment namespace regex ``{ENV_REGEX_NS}``, which
        is replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot/fingertip_.*`` becomes ``/World/envs/env_.*/Robot/fingertip_.*``.
    """

    filter_shape_prim_expr: list[str] = []
    """List of shape prim path expressions to filter contacts against at the shape level.
    Defaults to empty, meaning filter partners are resolved at the body level only
    (via :attr:`ContactSensorCfg.filter_prim_paths_expr`).

    If provided, the force matrix reports per-shape contact forces between the sensing
    primitives and the filter shapes.

    Mutually exclusive with :attr:`ContactSensorCfg.filter_prim_paths_expr`; only one
    must be set.

    .. note::
        Expressions can contain the environment namespace regex ``{ENV_REGEX_NS}``, which
        is replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` becomes ``/World/envs/env_.*/Object``.
    """

    def __post_init__(self):
        if self.track_contact_points:
            warnings.warn(
                "ContactSensorCfg: 'track_contact_points' is not supported by the Newton backend. Ignoring.",
                stacklevel=2,
            )
            self.track_contact_points = False

        if self.max_contact_data_count_per_prim is not None:
            warnings.warn(
                "ContactSensorCfg: 'max_contact_data_count_per_prim' is not supported by the Newton backend. Ignoring.",
                stacklevel=2,
            )
            self.max_contact_data_count_per_prim = None

        if self.track_friction_forces:
            warnings.warn(
                "ContactSensorCfg: 'track_friction_forces' is not supported by the Newton backend. Ignoring.",
                stacklevel=2,
            )
            self.track_friction_forces = False

    @classmethod
    def from_base_cfg(cls, base_cfg: BaseContactSensorCfg, **kwargs) -> "ContactSensorCfg":
        """Creates a :class:`ContactSensorCfg` from an existing :class:`ContactSensorCfg`.

        Args:
            base_cfg: The base contact sensor configuration to copy from.
            **kwargs: Newton-specific fields, e.g. ``filter_shape_prim_expr=["fingertip_.*"]``.

        Returns:
            A new :class:`ContactSensorCfg` instance.

        Raises:
            ValueError: If ``class_type`` is passed in keyword arguments.
        """
        if "class_type" in kwargs:
            raise ValueError("Cannot override 'class_type' via from_base_cfg.")
        base_fields = {
            field: getattr(base_cfg, field) for field in base_cfg.__dataclass_fields__ if field != "class_type"
        }
        return cls(**base_fields, **kwargs)
