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

parser = argparse.ArgumentParser(description="Benchmark the PhysX contact sensor update.")
parser.add_argument("--physics_variant", choices=("physx",), default="physx", help="Exact physics variant.")
parser.add_argument("--num_envs", type=parse_positive_int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=parse_positive_int, default=500, help="Number of timed simulation steps.")
parser.add_argument("--warmup_steps", type=parse_non_negative_int, default=50, help="Number of untimed warm-up steps.")
parser.add_argument("--decimation", type=parse_positive_int, default=4, help="Physics steps per timed sensor cadence.")
parser.add_argument(
    "--history_length", type=parse_non_negative_int, default=0, help="Number of contact history frames."
)
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
parser.add_argument("--disable_graph", action="store_true", help="Disable CUDA graph capture of the sensor update.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, LatencySample, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass


@configclass
class ContactSensorBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene with one cube per environment and a contact sensor on the cube."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        track_air_time=True,
        update_period=0.0,
        history_length=args_cli.history_length,
    )


def main():
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = ContactSensorBenchmarkSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, lazy_sensor_update=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    sensor = scene["contact_sensor"]
    if args_cli.disable_graph:
        sensor._use_graph = False

    # Warm up the simulation and capture the sensor graph when enabled.
    for _ in range(args_cli.warmup_steps):
        for _ in range(args_cli.decimation):
            sim.step(render=False)
            sensor.update(sim_dt)
        _ = sensor.data
    wp.synchronize_device(sim.device)

    # Time only sensor work. Physics steps happen outside the timed regions.
    synchronize_device = partial(wp.synchronize_device, sim.device)
    cadence_samples: list[LatencySample] = []
    for _ in range(args_cli.num_steps):
        cadence_synchronized = 0.0
        cadence_submission = 0.0
        for _ in range(args_cli.decimation):
            sim.step(render=False)
            sample = measure_latency(operation=lambda: sensor.update(sim_dt), synchronize=synchronize_device)
            cadence_synchronized += sample.synchronized_s
            cadence_submission += sample.submission_s

        sample = measure_latency(operation=lambda: getattr(sensor, "data"), synchronize=synchronize_device)
        cadence_synchronized += sample.synchronized_s
        cadence_submission += sample.submission_s
        cadence_samples.append(LatencySample(submission_s=cadence_submission, synchronized_s=cadence_synchronized))

    observer_synchronized_s = [
        sum(
            measure_latency(operation=lambda: None, synchronize=synchronize_device).synchronized_s
            for _ in range(args_cli.decimation + 1)
        )
        for _ in range(args_cli.num_steps)
    ]

    mode = "eager" if args_cli.disable_graph else "graph"
    # Cubes rest on the ground, so every sensor must report an upward net force.
    net_forces = sensor.data.net_forces_w.torch
    num_in_contact = int((net_forces.norm(dim=-1) > 0.1).sum().item())

    if num_in_contact != args_cli.num_envs:
        raise RuntimeError(f"Expected {args_cli.num_envs} contacting sensors, received {num_in_contact}.")

    benchmark = LatencyBenchmarkRunner(
        benchmark_name="physx_contact_sensor",
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        metadata={
            "physics_variant": args_cli.physics_variant,
            "mode": mode,
            "device": str(sim.device),
            "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps,
            "warmup_steps": args_cli.warmup_steps,
            "decimation": args_cli.decimation,
            "history_length": args_cli.history_length,
        },
    )
    benchmark.add_latency_samples("sensor_cadence", cadence_samples)
    benchmark.add_synchronized_samples("observer", "Synchronized Observer Floor", observer_synchronized_s)
    benchmark.add_measurement(
        "validation",
        measurement=[
            SingleMeasurement(name="Sensors in Contact", value=num_in_contact, unit="count"),
            SingleMeasurement(name="Expected Sensors", value=args_cli.num_envs, unit="count"),
        ],
    )
    benchmark.finalize()


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
