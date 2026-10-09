# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show a physical Rizon--Sharpa pickup and rising pour with water closeups.

Run ``uv run isaaclab demo rizon-sharpa-teapot`` for the interactive Newton GL
view. Record with ``--visualizer newton_rtx --video <path.mp4>`` using the
``ovrtx`` extra and ``imageio-ffmpeg``. The licensed robot asset bundle is
resolved by the shared teapot demo; no assets are redistributed.

This example keeps the shared MPM simulation, stock teapot geometry, default
ground, and physical index-finger grasp. Its 450 g pot uses proportionally scaled
inertia to reduce rocking during pickup. It reaches a 54-degree tilt over four seconds,
then eases to 49 degrees early in its 70 cm rise over twelve seconds. At the default
particle resolution, the initial fill is approximately 168 ml, leaving clearance
below the open rim while giving the water time to drain. The stream stays aimed at
the bowl's center.
Its presentation moves from pickup to the pouring stream, shows reconstructed
water for two seconds, returns to particles
entering the bowl, then frames the robot and bowl together. User arguments
override these demo defaults, including ``--grasp_finger middle``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Run the shared teapot demo with robot and closeup presentation defaults."""
    original_args = sys.argv[:]
    sys.argv[1:1] = [
        "--robot",
        "rizon_sharpa",
        "--grasp_finger",
        "index",
        "--teapot_mass",
        "0.45",
        "--pour_angle",
        "54",
        "--pour_tilt_time",
        "4.0",
        "--pour_upper_angle",
        "49",
        "--fill_level",
        "0.35",
        "--pour_rise_time",
        "12.0",
        "--fluid_render_mode",
        "particles",
        "--presentation",
        "pour_closeups",
    ]
    try:
        runpy.run_path(str(Path(__file__).with_name("teapot_fill.py")), run_name="__main__")
    finally:
        sys.argv[:] = original_args


if __name__ == "__main__":
    main()
