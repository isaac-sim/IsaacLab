# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg, ObservationGroupCfg
from isaaclab.utils import configclass

from .pre_trained_policy_action import PreTrainedPolicyAction


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type | str = PreTrainedPolicyAction
    """Class of the action term."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""

    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""

    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""

    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""

    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
