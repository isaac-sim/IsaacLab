# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the Newton IMU and PVA sensor update paths.

Usage:
    ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_imu_pva.py \
        --sensor imu --num_envs 4096
    ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_imu_pva.py \
        --sensor pva --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

parser = argparse.ArgumentParser(description="Benchmark a Newton IMU or PVA sensor update path.")
parser.add_argument("--sensor", choices=("imu", "pva"), required=True, help="Sensor update path to benchmark.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments and sensor instances.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
args_cli = parser.parse_args()
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero")
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be greater than zero")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")

import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg

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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    imu: ImuCfg | None = None
    pva: PvaCfg | None = None


def main() -> None:
    """Run the selected sensor benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=NewtonCfg(), gravity=(0.0, 0.0, -9.81))
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
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

        for _ in range(args_cli.warmup_steps):
            sim.step(render=False)
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize = partial(wp.synchronize_device, sim.device)
        samples = []
        for _ in range(args_cli.num_steps):
            sim.step(render=False)
            samples.append(
                measure_latency(
                    operation=lambda: sensor.update(sim_dt, force_recompute=True),
                    synchronize=synchronize,
                )
            )

        observer_samples = [
            measure_latency(operation=lambda: None, synchronize=synchronize) for _ in range(args_cli.num_steps)
        ]

        lin_acc_b = sensor.data.lin_acc_b.torch
        ang_vel_b = sensor.data.ang_vel_b.torch
        finite_instances = int(
            (torch.isfinite(lin_acc_b).all(dim=-1) & torch.isfinite(ang_vel_b).all(dim=-1)).sum().item()
        )
        if finite_instances != args_cli.num_envs:
            raise RuntimeError(f"Expected {args_cli.num_envs} finite sensor outputs, received {finite_instances}.")

        benchmark = LatencyBenchmarkRunner(
            benchmark_name=f"newton_{args_cli.sensor}_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
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
            "observer", "Synchronized Observer Floor", [sample.synchronized_s for sample in observer_samples]
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
    main()
