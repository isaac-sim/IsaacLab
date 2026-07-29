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

from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX JointWrench sensor update path.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("ovphysx",),
    default_physics_variant="ovphysx",
    add_device=True,
)
args_cli = parser.parse_args()

import isaaclab_ovphysx.tensor_types as TT
import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils  # noqa: F401
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.sensor_suites import add_sensor_latency_measurements, collect_sensor_latency_samples
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

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples = collect_sensor_latency_samples(
            num_steps=args_cli.num_steps,
            step=sim.step,
            update=lambda: sensor.update(sim_dt, force_recompute=True),
            synchronize=synchronize_device,
            native_read=lambda: sensor._root_view.read_into(TT.LINK_INCOMING_JOINT_FORCE, sensor._wrench_buf),
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
            benchmark_name="ovphysx_joint_wrench_sensor",
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
