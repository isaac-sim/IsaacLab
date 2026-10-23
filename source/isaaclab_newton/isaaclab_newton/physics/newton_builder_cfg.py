# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the shared Newton model builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from newton import ModelBuilder


@configclass
class NewtonBuilderCfg:
    """Share mutable construction data, populated before model allocation and retained across hard resets."""

    class_type: Callable[[NewtonBuilderCfg], ModelBuilder] | str = "{DIR}.newton_manager:create_newton_builder"
    """Factory function returning a new :class:`newton.ModelBuilder`; :func:`~isaaclab.utils.instantiate` calls it."""
    physics_cfg: PhysicsCfg = field(kw_only=True, metadata={"copy": False})
    """Selected physics settings; non-Newton physics requires a render-only Newton representation."""
