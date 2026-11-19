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

GAZE_MODES = ("software", "physical", "hybrid", "dual")
"""How the gaze action moves the gaze; see :class:`~isaaclab_fovi.mdp.gaze.GazeAction`."""


@configclass
class GazeRigCfg:
    """The articulation that turns the camera, for physical, hybrid and dual gaze."""

    asset_name: str = MISSING
    """Scene name of the articulation whose joints turn the camera."""
    joint_names: tuple[str, str] = ("pan_joint", "tilt_joint")
    """The joints that turn the camera left and up, in that order."""
    joint_signs: tuple[float, float] = (1.0, 1.0)
    """Sign of each joint's angle for a turn of the camera to the left and up."""
    max_step_deg: float = 6.0
    """Largest change of each joint target per step in hybrid gaze [deg]: how fast the head catches up."""


@configclass
class GazeActionCfg(ActionTermCfg):
    """Configuration of :class:`~isaaclab_fovi.mdp.gaze.GazeAction`."""

    class_type: type[GazeAction] | str = "{DIR}.gaze:GazeAction"
    asset_name: str = MISSING
    """Scene name of the foveated camera whose gaze the action moves."""
    mode: str = "software"
    """``"software"``, ``"physical"``, ``"hybrid"``, or ``"dual"`` (experimental): what the actions turn."""
    rig: GazeRigCfg | None = None
    """The articulation that turns the camera; needed by every mode but software gaze."""
    step_size: float = 0.1
    """Change of each gaze coordinate per step at an action of 1, in field radii. Positive and finite."""
    gaze_limit: float = 1.0
    """Largest distance of the software gaze from the camera's optical axis, in field radii. Positive and finite."""
    random_reset: bool = False
    """Start episodes at a random software gaze the camera sees; otherwise on the optical axis."""
    mount: str = "fixed"
    """Teacher's mount: ``"fixed"`` at the action's creation, or ``"tracked"`` on a moving body (Newton, OVRTX)."""

    def validate_config(self):
        """Reject invalid gaze bounds, modes and mounts, and rig settings that a mode cannot use."""
        for name, value in (("step_size", self.step_size), ("gaze_limit", self.gaze_limit)):
            if not _positive_finite(value):
                raise ValueError(f"GazeActionCfg.{name} must be positive and finite, got {value!r}.")
        if self.mount not in ("fixed", "tracked"):
            raise ValueError(f"GazeActionCfg.mount must be 'fixed' or 'tracked', got {self.mount!r}.")
        if self.mode not in GAZE_MODES:
            raise ValueError(f"GazeActionCfg.mode must be one of {GAZE_MODES}, got {self.mode!r}.")
        if self.mode == "software":
            return
        if self.rig is None:
            raise ValueError(f"GazeActionCfg.mode={self.mode!r} turns a rig's joints; set GazeActionCfg.rig.")
        if self.mount != "tracked":
            raise ValueError(f"GazeActionCfg.mode={self.mode!r} moves the camera's mount; set mount='tracked'.")
        if self.random_reset:
            raise ValueError("GazeActionCfg.random_reset applies only to software gaze.")
        if not _positive_finite(self.rig.max_step_deg):
            raise ValueError(f"GazeRigCfg.max_step_deg must be positive and finite, got {self.rig.max_step_deg!r}.")
        if len(self.rig.joint_names) != 2 or len(set(self.rig.joint_names)) != 2:
            raise ValueError(f"GazeRigCfg.joint_names must name two joints, got {self.rig.joint_names!r}.")
        if any(sign not in (-1.0, 1.0) for sign in self.rig.joint_signs):
            raise ValueError(f"GazeRigCfg.joint_signs must each be 1 or -1, got {self.rig.joint_signs!r}.")


def _positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
