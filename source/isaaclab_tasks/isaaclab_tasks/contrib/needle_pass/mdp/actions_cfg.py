# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for the dVRK needle-pass task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.managers.action_manager import ActionTerm

    from .terminations import HandoffPhaseCfg


@configclass
class WorldFrameDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configure live world-to-root absolute differential IK."""

    class_type: type[ActionTerm] | str = "{DIR}.actions:WorldFrameDifferentialInverseKinematicsAction"
    """Action-term implementation, resolved lazily after the simulation application starts."""


@configclass
class PairedJawJointPositionActionCfg(JointPositionActionCfg):
    """Configure the exact ordered paired-jaw action."""

    class_type: type[ActionTerm] | str = "{DIR}.actions:PairedJawJointPositionAction"
    """Action-term implementation, resolved lazily after the simulation application starts."""


@configclass
class DonorReleaseGuardedPairedJawJointPositionActionCfg(PairedJawJointPositionActionCfg):
    """Configure paired donor jaws with a measured receiver-grasp interlock."""

    class_type: type[ActionTerm] | str = "{DIR}.actions:DonorReleaseGuardedPairedJawJointPositionAction"
    """Action-term implementation, resolved lazily after the simulation application starts."""

    phase_cfg: HandoffPhaseCfg | None = None
    """Shared hand-off phase configuration used by the release interlock."""

    release_aperture_threshold_rad: float = 0.0
    """Minimum outward displacement from the held target that requests release [rad]."""

    hold_jaw_pos: tuple[float, float] = (0.0, 0.0)
    """Ordered donor-jaw positions commanded while release is blocked [rad]."""


__all__ = [
    "DonorReleaseGuardedPairedJawJointPositionActionCfg",
    "PairedJawJointPositionActionCfg",
    "WorldFrameDifferentialInverseKinematicsActionCfg",
]
