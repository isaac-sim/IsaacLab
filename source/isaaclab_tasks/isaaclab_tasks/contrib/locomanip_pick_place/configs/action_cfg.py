# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import config_field

if TYPE_CHECKING:
    from ..mdp.actions import AgileBasedLowerBodyAction


@dataclass
class AgileBasedLowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term that is based on Agile lower body RL policy."""

    class_type: type["AgileBasedLowerBodyAction"] | str = config_field(
        "isaaclab_tasks.contrib.locomanip_pick_place.mdp.actions:AgileBasedLowerBodyAction"
    )
    """The class type for the lower body action term."""

    joint_names: list[str] = config_field(MISSING)
    """The names of the joints to control."""

    obs_group_name: str = config_field(MISSING)
    """The name of the observation group to use."""

    policy_path: str = config_field(MISSING)
    """The path to the policy model."""

    policy_output_offset: float = config_field(0.0)
    """Offsets the output of the policy."""

    policy_output_scale: float = config_field(1.0)
    """Scales the output of the policy."""
