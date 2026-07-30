# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the Newton JointWrench sensor update path.

Each environment contains one cartpole articulation. The sensor reads the
incoming wrench for all three links and transforms it into each child-side
joint frame.

Usage:
    ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_joint_wrench.py --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args

parser = argparse.ArgumentParser(description="Benchmark the Newton JointWrench sensor update path.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("newton_mjwarp", "newton_kamino"),
    default_physics_variant="newton_mjwarp",
    add_device=True,
)
args_cli = parser.parse_args()

import torch
import warp as wp
from isaaclab_newton.benchmark._physics import create_microbenchmark_physics_cfg

import isaaclab.sim as sim_utils
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.sensor_suites import add_sensor_latency_measurements, collect_sensor_latency_samples
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import CARTPOLE_CFG


@configclass
class JointWrenchBenchmarkSceneCfg(InteractiveSceneCfg):
    """One cartpole articulation and JointWrench sensor per environment."""

    robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    """Run the benchmark and print latency and output sanity statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt, device=args_cli.device, physics=create_microbenchmark_physics_cfg(args_cli.physics_variant)
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
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
            sim.step(render=False)
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples = collect_sensor_latency_samples(
            num_steps=args_cli.num_steps,
            step=lambda: sim.step(render=False),
            update=lambda: sensor.update(sim_dt, force_recompute=True),
            synchronize=synchronize_device,
        )

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
            benchmark_name="newton_joint_wrench_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "physics_variant": args_cli.physics_variant,
                "label": args_cli.label,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "bodies_per_env": len(sensor.body_names),
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
            },
        )
        add_sensor_latency_measurements(
            benchmark,
            samples=samples,
            validation=[
                SingleMeasurement(name="Finite Wrenches", value=finite_wrenches, unit="count"),
                SingleMeasurement(name="Nonzero Wrenches", value=nonzero_wrenches, unit="count"),
                SingleMeasurement(name="Expected Wrenches", value=expected_wrenches, unit="count"),
            ],
            update_phase="sensor_update",
            observer_phase="observer",
            validation_phase="validation",
        )
        benchmark.finalize()


if __name__ == "__main__":
    main()
