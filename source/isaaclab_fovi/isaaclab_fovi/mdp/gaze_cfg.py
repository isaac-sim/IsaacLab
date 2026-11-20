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
    """Largest distance of the gaze from the camera's optical axis, in field radii. Positive and finite."""
    random_reset: bool = False
    """Start episodes at a random gaze the camera sees; otherwise on the optical axis."""
    mount: str = "fixed"
    """Teacher's mount: ``"fixed"`` at the action's creation, or ``"tracked"`` on a moving body (Newton, OVRTX)."""

    def validate_config(self):
        """Reject gaze bounds that would divide by zero and unknown mounts."""
        if not _positive_finite(self.step_size):
            raise ValueError(f"GazeActionCfg.step_size must be positive and finite, got {self.step_size!r}.")
        if not _positive_finite(self.gaze_limit):
            raise ValueError(f"GazeActionCfg.gaze_limit must be positive and finite, got {self.gaze_limit!r}.")
        if self.mount not in ("fixed", "tracked"):
            raise ValueError(f"GazeActionCfg.mount must be 'fixed' or 'tracked', got {self.mount!r}.")


def _positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
