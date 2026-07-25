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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark a PhysX IMU or PVA sensor update path.")
parser.add_argument("--sensor", choices=("imu", "pva"), required=True, help="Sensor update path to benchmark.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments and sensor instances.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
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

import statistics
import time

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
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


def _percentile(samples: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for scalar samples."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


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

    synchronized_ms: list[float] = []
    submission_ms: list[float] = []
    for _ in range(args_cli.num_steps):
        sim.step(render=False)
        wp.synchronize_device(sim.device)
        start = time.perf_counter()
        sensor.update(sim_dt, force_recompute=True)
        submitted = time.perf_counter()
        wp.synchronize_device(sim.device)
        finished = time.perf_counter()
        synchronized_ms.append((finished - start) * 1000.0)
        submission_ms.append((submitted - start) * 1000.0)

    lin_acc_b = sensor.data.lin_acc_b.torch
    ang_vel_b = sensor.data.ang_vel_b.torch
    finite_instances = int((torch.isfinite(lin_acc_b).all(dim=-1) & torch.isfinite(ang_vel_b).all(dim=-1)).sum())

    print("-" * 80)
    print(f"{args_cli.sensor.upper()} update benchmark (PhysX)")
    print(f"  label                  : {args_cli.label}")
    print(f"  device                 : {sim.device}")
    print(f"  num_envs               : {args_cli.num_envs}")
    print(f"  num_steps              : {args_cli.num_steps}")
    print(f"  synchronized mean      : {statistics.mean(synchronized_ms):.3f} ms")
    print(f"  synchronized p50       : {_percentile(synchronized_ms, 0.50):.3f} ms")
    print(f"  synchronized p95       : {_percentile(synchronized_ms, 0.95):.3f} ms")
    print(f"  submission mean        : {statistics.mean(submission_ms):.3f} ms")
    print(f"  submission p50         : {_percentile(submission_ms, 0.50):.3f} ms")
    print(f"  submission p95         : {_percentile(submission_ms, 0.95):.3f} ms")
    print("-" * 80)
    print(f"  finite sensor outputs  : {finite_instances} / {args_cli.num_envs}")


if __name__ == "__main__":
    main()
    simulation_app.close()
