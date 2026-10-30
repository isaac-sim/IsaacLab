# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-data configuration for the GR1T2 standalone action boundary."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .gr1t2_kitless_action import GR1T2KitlessPinkInverseKinematicsAction


@configclass
class GR1T2KitlessPinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Pink action with opt-in, backend-specific target conditioning for GR1T2.

    The feature is disabled by default, preserving the regular Kit/PhysX action
    path. The task enables it for kitless OVPhysX or Newton and fills the
    hand metadata while conditioning the spawned USD before physics parses it.
    """

    class_type: type[GR1T2KitlessPinkInverseKinematicsAction] | str = (
        "{DIR}.gr1t2_kitless_action:GR1T2KitlessPinkInverseKinematicsAction"
    )

    kitless_target_conditioning_enabled: bool = False
    """Enable the shared raw-GR1T2 joint/target policy for standalone OVPhysX and Newton."""

    max_ik_joint_delta_rad: float = 0.2
    """Maximum IK target offset from measured joint position on each IK application [rad].

    The bound is re-evaluated when the action is applied. OVPhysX applies actions
    on each environment-managed physics substep; Newton can manage decimation
    within a physics step. This is not a per-policy-step travel limit.
    """

    max_ik_joint_magnitude_rad: float = 3.5
    """Symmetric absolute bound for commanded IK-joint positions [rad]."""

    max_hand_target_step_rad: float = 0.079
    """Maximum hand-target change per policy step [rad]."""

    hand_target_margin_rad: float = 0.5
    """Extra target travel beyond anatomical limits, used to sustain grip effort [rad]."""

    hand_mimic_relations: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    """Mimic joint to ``(reference joint, gearing, offset)`` from the spawned USD."""

    hand_joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Original anatomical hand-joint limits recorded before physical limits are widened [rad]."""
