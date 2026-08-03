# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the OVPhysX IMU and PVA sensor update paths.

Mirrors ``isaaclab_physx/benchmark/sensors/benchmark_imu_pva.py`` but runs
kitless against the OVPhysX backend. In addition to the full-update timing it
also times the blocking native reads and required staging copies so the derived
processing share can be estimated.

Usage:
    ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_imu_pva.py \
        --sensor imu --num_envs 4096
    ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_imu_pva.py \
        --sensor pva --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args

parser = argparse.ArgumentParser(description="Benchmark an OVPhysX IMU or PVA sensor update path.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("ovphysx",),
    default_physics_variant="ovphysx",
    add_device=True,
)
parser.add_argument("--sensor", choices=("imu", "pva"), required=True, help="Sensor update path to benchmark.")
args_cli = parser.parse_args()

import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.sensor_suites import add_sensor_latency_measurements, collect_sensor_latency_samples
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ImuCfg, PvaCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

wp.init()


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


def _read_only(sensor) -> None:
    """Issue only the native OVPhysX reads performed by the sensor update."""
    import isaaclab_ovphysx.tensor_types as TT

    sensor._root_view.read_into(TT.RIGID_BODY_POSE, sensor._transforms)
    sensor._root_view.read_into(TT.RIGID_BODY_VELOCITY, sensor._velocities)
    if args_cli.sensor == "imu":
        sensor._root_view.read_into(TT.RIGID_BODY_COM_POSE, sensor._coms_read_view)
        if sensor._coms_read_view is not sensor._coms_gpu_view:
            wp.copy(sensor._coms_gpu_view, sensor._coms_read_view)
    else:
        sensor._root_view.read_into(TT.RIGID_BODY_COM_POSE, sensor._coms_read_view)
        if sensor._coms_read_view is not sensor._coms_buffer:
            wp.copy(sensor._coms_buffer, sensor._coms_read_view)


def main() -> None:
    """Run the selected sensor benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=sim_dt, device=args_cli.device, physics=OvPhysxCfg(), gravity=(0.0, 0.0, -9.81))
    with build_simulation_context(device=args_cli.device, sim_cfg=sim_cfg) as sim:
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
            sim.step()
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples = collect_sensor_latency_samples(
            num_steps=args_cli.num_steps,
            step=sim.step,
            update=lambda: sensor.update(sim_dt, force_recompute=True),
            synchronize=synchronize_device,
            native_read=lambda: _read_only(sensor),
        )

        lin_acc_b = sensor.data.lin_acc_b.torch
        ang_vel_b = sensor.data.ang_vel_b.torch
        finite_instances = int((torch.isfinite(lin_acc_b).all(dim=-1) & torch.isfinite(ang_vel_b).all(dim=-1)).sum())
        if finite_instances != args_cli.num_envs:
            raise RuntimeError(f"Expected {args_cli.num_envs} finite sensor outputs, received {finite_instances}.")

        benchmark_name = f"ovphysx_{args_cli.sensor}_sensor"
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
