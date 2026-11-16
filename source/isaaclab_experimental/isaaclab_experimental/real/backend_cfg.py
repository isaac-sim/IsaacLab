# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the experimental real robot backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .session import DeploymentSession


@configclass
class RealCfg(PhysicsCfg):
    """Select a session supplied by a deployment runtime or a fake driver."""

    class_type: str = "{DIR}.backend:RealManager"

    session_factory: Callable[[], DeploymentSession] = MISSING
    """Construct the one session owned by the real backend manager."""
