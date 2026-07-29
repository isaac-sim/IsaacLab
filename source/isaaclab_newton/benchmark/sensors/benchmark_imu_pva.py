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

from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args

parser = argparse.ArgumentParser(description="Benchmark a Newton IMU or PVA sensor update path.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("newton_mjwarp", "newton_kamino"),
    default_physics_variant="newton_mjwarp",
    add_device=True,
)
parser.add_argument("--sensor", choices=("imu", "pva"), required=True, help="Sensor update path to benchmark.")
args_cli = parser.parse_args()

import torch
import warp as wp
from isaaclab_newton.benchmark._physics import create_microbenchmark_physics_cfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.sensor_suites import add_sensor_latency_measurements, collect_sensor_latency_samples
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
    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt,
        device=args_cli.device,
        physics=create_microbenchmark_physics_cfg(args_cli.physics_variant),
        gravity=(0.0, 0.0, -9.81),
    )
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

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples = collect_sensor_latency_samples(
            num_steps=args_cli.num_steps,
            step=lambda: sim.step(render=False),
            update=lambda: sensor.update(sim_dt, force_recompute=True),
            synchronize=synchronize_device,
        )

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
                "physics_variant": args_cli.physics_variant,
                "label": args_cli.label,
                "sensor": args_cli.sensor,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
            },
        )
        add_sensor_latency_measurements(
            benchmark,
            samples=samples,
            validation=[
                SingleMeasurement(name="Finite Sensor Outputs", value=finite_instances, unit="count"),
                SingleMeasurement(name="Expected Sensor Outputs", value=args_cli.num_envs, unit="count"),
            ],
            update_phase="sensor_update",
            observer_phase="observer",
            validation_phase="validation",
        )
        benchmark.finalize()


if __name__ == "__main__":
    main()
