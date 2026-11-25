# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Configuration for reference motion commands."""

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import CsvMotionCfg

if TYPE_CHECKING:
    from .commands import MotionCommand


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Named reference motion with failure-adaptive phase sampling."""

    class_type: type["MotionCommand"] | str = "{DIR}.commands:MotionCommand"
    asset_name: str = MISSING
    """Articulation to control."""
    motion_file: str = MISSING
    """NPZ/CSV/NPY reference path or directory; files are read without modifying the source."""
    source_fps: float = 30.0
    """Frame rate for pose-only CSV/NPY input."""
    csv: CsvMotionCfg = CsvMotionCfg()
    sampling_mode: str = "adaptive"
    """Reset phase: failure-adaptive sampling or the start of each clip (``start``)."""
    anchor_body_name: str = MISSING
    """Body defining the reference alignment frame."""
    body_names: list[str] = MISSING
    """Tracked bodies in order, starting with the articulation root."""
    pose_range: dict[str, tuple[float, float]] = {}
    """Reset offsets: x/y/z [m] and roll/pitch/yaw [rad]."""
    velocity_range: dict[str, tuple[float, float]] = {}
    """Reset velocity offsets: x/y/z [m/s] and roll/pitch/yaw [rad/s]."""
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    """Joint reset perturbation [rad]."""
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
