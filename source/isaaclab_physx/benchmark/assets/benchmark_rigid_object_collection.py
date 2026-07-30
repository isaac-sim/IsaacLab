# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the combined rigid_object_collection method and data micro-benchmarks for physx."""

from isaaclab.benchmark.asset_suites.cli import run_asset_benchmark_cli

if __name__ == "__main__":
    run_asset_benchmark_cli("physx", "rigid_object_collection")
