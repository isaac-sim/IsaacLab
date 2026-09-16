# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass

from isaaclab.managers import ActionTermCfg, ObservationGroupCfg
from isaaclab.utils import config_field


@dataclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type | str = config_field("{DIR}.pre_trained_policy_action:PreTrainedPolicyAction")
    """Class of the action term."""

    asset_name: str = config_field(MISSING)
    """Name of the asset in the environment for which the commands are generated."""

    policy_path: str = config_field(MISSING)
    """Path to the low level policy (.pt files)."""

    low_level_decimation: int = config_field(4)
    """Decimation factor for the low level action term."""

    low_level_actions: ActionTermCfg = config_field(MISSING)
    """Low level action configuration."""

    low_level_observations: ObservationGroupCfg = config_field(MISSING)
    """Low level observation configuration."""

    debug_vis: bool = config_field(True)
    """Whether to visualize debug information. Defaults to False."""
