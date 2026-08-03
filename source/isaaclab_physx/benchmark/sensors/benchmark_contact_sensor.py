# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the PhysX contact sensor update cadence.

Measures synchronized sensor work over one environment step while excluding physics simulation.
Each timed cadence advances the sensor by --decimation physics steps and reads the data once.
Use --history_length 0 for lazy on-read updates and a positive value for physics-step history
updates. Warp kernels use CUDA graphs by default; pass --disable_graph for eager execution.

Usage:
    # Lazy update once per environment step
    uv run python source/isaaclab_physx/benchmark/sensors/benchmark_contact_sensor.py
        --num_envs 4096 --history_length 0

    # History update at each of four physics steps
    uv run python source/isaaclab_physx/benchmark/sensors/benchmark_contact_sensor.py
        --num_envs 4096 --history_length 3 --decimation 4
"""

from __future__ import annotations

import argparse
import traceback
from functools import partial

from isaaclab.app import AppLauncher
from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int
from isaaclab.benchmark.sensor_suites import (
    add_sensor_benchmark_args,
    create_contact_sensor_scene_cfg,
    run_contact_sensor_workload,
)

parser = argparse.ArgumentParser(description="Benchmark the PhysX contact sensor update.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("physx",),
    default_physics_variant="physx",
    add_device=False,
)
parser.add_argument("--decimation", type=parse_positive_int, default=4, help="Physics steps per timed sensor cadence.")
parser.add_argument(
    "--history_length", type=parse_non_negative_int, default=0, help="Number of contact history frames."
)
parser.add_argument("--disable_graph", action="store_true", help="Disable CUDA graph capture of the sensor update.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene


def main():
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = create_contact_sensor_scene_cfg(
        history_length=args_cli.history_length,
        num_envs=args_cli.num_envs,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    sensor = scene["contact_sensor"]
    if args_cli.disable_graph:
        sensor._use_graph = False

    synchronize_device = partial(wp.synchronize_device, sim.device)
    mode = "eager" if args_cli.disable_graph else "graph"
    run_contact_sensor_workload(
        benchmark_name="physx_contact_sensor",
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        metadata={
            "physics_variant": args_cli.physics_variant,
            "label": args_cli.label,
            "mode": mode,
            "device": str(sim.device),
            "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps,
            "warmup_steps": args_cli.warmup_steps,
            "decimation": args_cli.decimation,
            "history_length": args_cli.history_length,
        },
        num_steps=args_cli.num_steps,
        warmup_steps=args_cli.warmup_steps,
        decimation=args_cli.decimation,
        expected_contacts=args_cli.num_envs,
        step=lambda: sim.step(render=False),
        update=lambda: sensor.update(sim_dt),
        read=lambda: getattr(sensor, "data"),
        count_contacts=lambda: int((sensor.data.net_forces_w.torch.norm(dim=-1) > 0.1).sum().item()),
        synchronize=synchronize_device,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if simulation_app.config.get("fast_shutdown", False):
            traceback.print_exc()
        simulation_app.close(exit_code=1)
        raise
    else:
        simulation_app.close()
