# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the OVPhysX contact sensor update path.

Mirrors ``isaaclab_physx/benchmark/sensors/benchmark_contact_sensor.py`` but
runs kitless against the OVPhysX backend. Also times the blocking native
``read_net_forces`` fetch in isolation.

Usage:
    ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_contact_sensor.py --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX contact sensor update path.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("ovphysx",),
    default_physics_variant="ovphysx",
    add_device=True,
)
args_cli = parser.parse_args()

import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_ovphysx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.sensor_suites import add_sensor_latency_measurements, collect_sensor_latency_samples
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

wp.init()


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
    )


def main() -> None:
    """Run the benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=sim_dt, device=args_cli.device, physics=OvPhysxCfg(), gravity=(0.0, 0.0, -9.81))
    with build_simulation_context(device=args_cli.device, sim_cfg=sim_cfg) as sim:
        scene_cfg = ContactSensorBenchmarkSceneCfg(
            num_envs=args_cli.num_envs, env_spacing=2.0, lazy_sensor_update=False
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensor = scene["contact_sensor"]

        # warm-up: let the cubes settle on the ground
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
            native_read=lambda: sensor._contact_binding.read_net_forces(sensor._net_forces_flat_buf),
        )

        net_forces = sensor.data.net_forces_w.torch
        num_in_contact = int((net_forces.norm(dim=-1) > 0.1).sum().item())
        if num_in_contact != args_cli.num_envs:
            raise RuntimeError(f"Expected {args_cli.num_envs} contacting sensors, received {num_in_contact}.")

        benchmark = LatencyBenchmarkRunner(
            benchmark_name="ovphysx_contact_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "physics_variant": args_cli.physics_variant,
                "label": args_cli.label,
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
                SingleMeasurement(name="Sensors in Contact", value=num_in_contact, unit="count"),
                SingleMeasurement(name="Expected Sensors", value=args_cli.num_envs, unit="count"),
            ],
            update_phase="sensor_update",
            observer_phase="observer",
            validation_phase="validation",
        )
        benchmark.finalize()


if __name__ == "__main__":
    main()
