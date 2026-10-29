# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate an Isaac Lab manipulation environment; see :mod:`isaaclab_teleop.entrypoints.teleop_se3_agent`."""

# Warp captures ``enable_backward`` when a module is created, which happens at import
# time, so it has to be set before importing anything that defines Warp kernels.
# Isaac Lab does not use Warp autodiff; skipping adjoint codegen roughly halves the
# time spent building kernels on a cold kernel cache.
import warp as wp

wp.config.enable_backward = False

from isaaclab_teleop.entrypoints import run_teleop_se3_agent_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Teleoperate an Isaac Lab manipulation environment."""
    return run_teleop_se3_agent_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
