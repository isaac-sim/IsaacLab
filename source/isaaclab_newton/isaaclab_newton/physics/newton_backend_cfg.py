# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton backend."""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend


@configclass
class NewtonBackendCfg(BackendCfg):
    """Allocate model and state from the matching builder; closing them leaves the builder intact."""

    class_type: type[NewtonBackend] | str = "{DIR}.newton_backend:NewtonBackend"
    physics_cfg: PhysicsCfg = field(kw_only=True, metadata={"copy": False})
    """Selected physics settings, also identifying the shared construction builder."""
    device: str = MISSING
    """Device on which to allocate the model and native buffers."""
