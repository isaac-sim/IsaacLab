# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Categories used by typed preset selection and CLI help."""

import enum

from isaaclab.physics import PhysicsCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sim import SimulationCfg


class PresetTarget(enum.Enum):
    """Classify preset alternatives; untyped values belong to the domain category."""

    PHYSICS = ("physics", (PhysicsCfg,))
    RENDERER = ("renderer", (RendererCfg,))
    DOMAIN = ("presets", ())

    def __new__(cls, label: str, base_classes: tuple[type, ...]):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.base_classes = base_classes
        return obj

    def matches(self, cfg: object) -> bool:
        """Match a config type, including physics bundled with simulation-wide settings.

        Args:
            cfg: Preset alternative to classify.

        Returns:
            Whether the alternative belongs to this category.
        """
        return isinstance(cfg, self.base_classes) or (
            self is PresetTarget.PHYSICS and isinstance(cfg, SimulationCfg) and isinstance(cfg.physics, PhysicsCfg)
        )
