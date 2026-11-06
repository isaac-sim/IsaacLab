# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supported launcher for the library-owned training benchmark entrypoint."""

from __future__ import annotations

from isaaclab.benchmark.entrypoints.training import main

if __name__ == "__main__":
    raise SystemExit(main())
