# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import TYPE_CHECKING

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as BaseContactSensorCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(BaseContactSensorCfg):
    """Configuration that pins the contact sensor to the Newton backend.

    :class:`~isaaclab.sensors.ContactSensorCfg` already resolves to this backend automatically under
    Newton physics (including its ``*_shape_prim_expr`` shape-level fields). Use this class directly
    only to force the Newton implementation regardless of the active backend. It warns about and
    disables base fields the Newton backend does not support, and can be built from a base config via
    :meth:`from_base_cfg`.
    """

    class_type: type["ContactSensor"] | str = "{DIR}.contact_sensor:ContactSensor"

    @property
    def sensor_body_prim_expr(self) -> str:
        """Read-only alias for :attr:`prim_path`."""
        return self.prim_path

    def __post_init__(self):
        if self.track_contact_points and not (self.filter_prim_paths_expr or self.filter_shape_prim_expr):
            warnings.warn(
                "ContactSensorCfg: 'track_contact_points' requires filter objects on the Newton backend. Please set"
                " 'filter_prim_paths_expr' or 'filter_shape_prim_expr'. Ignoring.",
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
        # ``kwargs`` override copied base fields (the shape-level fields now live on the base config).
        base_fields.update(kwargs)
        return cls(**base_fields)
