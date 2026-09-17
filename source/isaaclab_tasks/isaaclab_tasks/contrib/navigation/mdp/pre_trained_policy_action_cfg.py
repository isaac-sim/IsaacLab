# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from isaaclab.managers import ActionTermCfg, ObservationGroupCfg
from isaaclab.utils import REQUIRED


@dataclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type | str = "isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action:PreTrainedPolicyAction"
    """Class of the action term."""

    asset_name: str = REQUIRED
    """Name of the asset in the environment for which the commands are generated."""

    policy_path: str = REQUIRED
    """Path to the low level policy (.pt files)."""

    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""

    low_level_actions: ActionTermCfg = REQUIRED
    """Low level action configuration."""

    low_level_observations: ObservationGroupCfg = REQUIRED
    """Low level observation configuration."""

    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
