# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the OVPhysX JointWrench sensor update path.

Mirrors ``isaaclab_physx/benchmark/sensors/benchmark_joint_wrench.py`` but runs
kitless against the OVPhysX backend. Also times the blocking native
``LINK_INCOMING_JOINT_FORCE`` read in isolation.

Usage:
    ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_joint_wrench.py --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX JointWrench sensor update path.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
args_cli = parser.parse_args()
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero")
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be greater than zero")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")

import isaaclab_ovphysx.tensor_types as TT
import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils  # noqa: F401
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

from isaaclab_assets import CARTPOLE_CFG

wp.init()


@configclass
class JointWrenchBenchmarkSceneCfg(InteractiveSceneCfg):
    """One cartpole articulation and JointWrench sensor per environment."""

    robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    """Run the benchmark and print latency and output sanity statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=sim_dt, device=args_cli.device, physics=OvPhysxCfg(), gravity=(0.0, 0.0, -9.81))
    with build_simulation_context(device=args_cli.device, sim_cfg=sim_cfg) as sim:
        scene_cfg = JointWrenchBenchmarkSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=4.0,
            lazy_sensor_update=True,
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensor = scene["joint_wrench"]

        for _ in range(args_cli.warmup_steps):
            sim.step()
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize = partial(wp.synchronize_device, sim.device)
        samples = []
        for _ in range(args_cli.num_steps):
            sim.step()
            samples.append(
                measure_latency(
                    operation=lambda: sensor.update(sim_dt, force_recompute=True),
                    synchronize=synchronize,
                )
            )

        observer_samples = [
            measure_latency(operation=lambda: None, synchronize=synchronize) for _ in range(args_cli.num_steps)
        ]

        # Read-only phase: the blocking native fetch without the Warp kernel tail.
        read_only_samples = [
            measure_latency(
                operation=lambda: sensor._root_view.read_into(TT.LINK_INCOMING_JOINT_FORCE, sensor._wrench_buf),
                synchronize=synchronize,
            )
            for _ in range(args_cli.num_steps)
        ]

        force = sensor.data.force.torch
        torque = sensor.data.torque.torch
        finite_wrenches = int((torch.isfinite(force).all(dim=-1) & torch.isfinite(torque).all(dim=-1)).sum().item())
        nonzero_wrenches = int(((force != 0.0).any(dim=-1) | (torque != 0.0).any(dim=-1)).sum().item())
        expected_wrenches = args_cli.num_envs * len(sensor.body_names)
        if finite_wrenches != expected_wrenches:
            raise RuntimeError(f"Expected {expected_wrenches} finite wrenches, received {finite_wrenches}.")
        if nonzero_wrenches == 0:
            raise RuntimeError("Expected at least one nonzero wrench.")

        benchmark = LatencyBenchmarkRunner(
            benchmark_name="ovphysx_joint_wrench_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "label": args_cli.label,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "bodies_per_env": len(sensor.body_names),
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
            },
        )
        full_stats = benchmark.add_latency_samples("sensor_update", samples)
        read_stats = benchmark.add_latency_samples("native_read", read_only_samples)
        benchmark.add_synchronized_samples(
            "observer", "Synchronized Observer Floor", [s.synchronized_s for s in observer_samples]
        )
        benchmark.add_measurement(
            "sensor_update",
            measurement=SingleMeasurement(
                name="Estimated Synchronized Non-read Time",
                value=(full_stats.mean_s - read_stats.mean_s) * 1000.0,
                unit="ms",
            ),
        )
        benchmark.add_measurement(
            "validation",
            measurement=[
                SingleMeasurement(name="Finite Wrenches", value=finite_wrenches, unit="count"),
                SingleMeasurement(name="Nonzero Wrenches", value=nonzero_wrenches, unit="count"),
                SingleMeasurement(name="Expected Wrenches", value=expected_wrenches, unit="count"),
            ],
        )
        benchmark.finalize()


if __name__ == "__main__":
    main()
