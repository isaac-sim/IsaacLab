# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the active-gaze action."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .gaze import GazeAction


@configclass
class GazeActionCfg(ActionTermCfg):
    """Configuration of :class:`~isaaclab_fovi.mdp.gaze.GazeAction`."""

    class_type: type[GazeAction] | str = "{DIR}.gaze:GazeAction"
    asset_name: str = MISSING
    """Scene name of the foveated camera whose gaze the action moves."""
    step_size: float = 0.1
    """Change of each gaze coordinate per step at an action of 1, in field radii. Positive and finite."""
    gaze_limit: float = 1.0
    """Radius of the allowed tilt/pan disk, in field radii. Positive and finite."""

    def validate_config(self):
        """Reject gaze bounds that would divide by zero."""
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError(f"GazeActionCfg.step_size must be positive and finite, got {self.step_size!r}.")
        if not math.isfinite(self.gaze_limit) or self.gaze_limit <= 0:
            raise ValueError(f"GazeActionCfg.gaze_limit must be positive and finite, got {self.gaze_limit!r}.")
