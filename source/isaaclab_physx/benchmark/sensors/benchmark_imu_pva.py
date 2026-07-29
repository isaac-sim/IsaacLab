# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the PhysX IMU and PVA sensor update paths.

Usage:
    uv run python source/isaaclab_physx/benchmark/sensors/benchmark_imu_pva.py \
        --sensor imu --num_envs 4096
    uv run python source/isaaclab_physx/benchmark/sensors/benchmark_imu_pva.py \
        --sensor pva --num_envs 4096
"""

from __future__ import annotations

import argparse
import traceback
from functools import partial

from isaaclab.app import AppLauncher
from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int

parser = argparse.ArgumentParser(description="Benchmark a PhysX IMU or PVA sensor update path.")
parser.add_argument("--physics_variant", choices=("physx",), default="physx", help="Exact physics variant.")
parser.add_argument("--sensor", choices=("imu", "pva"), required=True, help="Sensor update path to benchmark.")
parser.add_argument(
    "--num_envs", type=parse_positive_int, default=4096, help="Number of environments and sensor instances."
)
parser.add_argument("--num_steps", type=parse_positive_int, default=500, help="Number of timed updates.")
parser.add_argument(
    "--warmup_steps", type=parse_non_negative_int, default=50, help="Number of untimed warm-up updates."
)
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
parser.add_argument(
    "--disable_recorded_launch",
    action="store_true",
    help="Use cached PhysX views with ordinary eager Warp launches.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below follows application launch."""

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ImuCfg, PvaCfg
from isaaclab.utils.configclass import configclass


@configclass
class ImuPvaBenchmarkSceneCfg(InteractiveSceneCfg):
    """One kinematic rigid body and one selected sensor per environment."""

    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    imu: ImuCfg | None = None
    pva: PvaCfg | None = None


def main() -> None:
    """Run the selected sensor benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = ImuPvaBenchmarkSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=1.0,
        lazy_sensor_update=True,
    )
    if args_cli.sensor == "imu":
        scene_cfg.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Body")
    else:
        scene_cfg.pva = PvaCfg(prim_path="{ENV_REGEX_NS}/Body")

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    sensor = scene[args_cli.sensor]

    if args_cli.disable_recorded_launch:
        sensor._use_recorded_launch = False
    for _ in range(args_cli.warmup_steps):
        sim.step(render=False)
        sensor.update(sim_dt, force_recompute=True)
    wp.synchronize_device(sim.device)

    synchronize_device = partial(wp.synchronize_device, sim.device)
    samples = []
    for _ in range(args_cli.num_steps):
        sim.step(render=False)
        samples.append(
            measure_latency(
                operation=lambda: sensor.update(sim_dt, force_recompute=True),
                synchronize=synchronize_device,
            )
        )

    observer_samples = [
        measure_latency(operation=lambda: None, synchronize=synchronize_device) for _ in range(args_cli.num_steps)
    ]

    lin_acc_b = sensor.data.lin_acc_b.torch
    ang_vel_b = sensor.data.ang_vel_b.torch
    finite_instances = int((torch.isfinite(lin_acc_b).all(dim=-1) & torch.isfinite(ang_vel_b).all(dim=-1)).sum())

    if finite_instances != args_cli.num_envs:
        raise RuntimeError(f"Expected {args_cli.num_envs} finite sensor outputs, received {finite_instances}.")

    benchmark_name = f"physx_{args_cli.sensor}_sensor"
    benchmark = LatencyBenchmarkRunner(
        benchmark_name=benchmark_name,
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        metadata={
            "physics_variant": args_cli.physics_variant,
            "label": args_cli.label,
            "sensor": args_cli.sensor,
            "device": str(sim.device),
            "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps,
            "warmup_steps": args_cli.warmup_steps,
        },
    )
    benchmark.add_latency_samples("sensor_update", samples)
    benchmark.add_synchronized_samples(
        "observer", "Synchronized Observer Floor", [s.synchronized_s for s in observer_samples]
    )
    benchmark.add_measurement(
        "validation",
        measurement=[
            SingleMeasurement(name="Finite Sensor Outputs", value=finite_instances, unit="count"),
            SingleMeasurement(name="Expected Sensor Outputs", value=args_cli.num_envs, unit="count"),
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
