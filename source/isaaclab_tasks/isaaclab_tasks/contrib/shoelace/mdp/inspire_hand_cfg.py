# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for bounded Inspire finger targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .inspire_hand import InspireFingerAction


@configclass
class InspireFingerActionCfg(BinaryJointPositionActionCfg):
    """Binary targets for coupled finger chains, with a control-step angular-rate bound."""

    class_type: type[InspireFingerAction] | str = "{DIR}.inspire_hand:InspireFingerAction"
    max_target_velocity: float = 1.0
    """Maximum joint-target change rate [rad/s]; actual speed depends on the drive and contacts."""
