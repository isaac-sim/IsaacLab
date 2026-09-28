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
from functools import partial

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int
from isaaclab.benchmark.sensor_suites import (
    add_sensor_benchmark_args,
    create_contact_sensor_scene_cfg,
    run_contact_sensor_workload,
)
from isaaclab.utils import instantiate

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

add_launcher_args(parser)
args_cli = parser.parse_args()

import warp as wp

import isaaclab.sim as sim_utils


def main():
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)

        scene_cfg = create_contact_sensor_scene_cfg(
            history_length=args_cli.history_length,
            num_envs=args_cli.num_envs,
        )
        scene = instantiate(scene_cfg)
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
            count_contacts=lambda: int((sensor.data.net_normal_forces_w.torch.norm(dim=-1) > 0.1).sum().item()),
            synchronize=synchronize_device,
        )


if __name__ == "__main__":
    main()
