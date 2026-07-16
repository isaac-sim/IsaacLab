# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the Newton contact sensor update cadence.

Measures synchronized sensor work over one environment step while excluding physics simulation.
Each timed cadence advances the sensor by --decimation physics steps and reads the data once.
Use --history_length 0 for lazy on-read updates and a positive value for physics-step history
updates.

Usage:
    # Lazy update once per environment step
    ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_contact_sensor.py
        --num_envs 4096 --history_length 0

    # History update at each of four physics steps
    ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_contact_sensor.py
        --num_envs 4096 --history_length 3 --decimation 4
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Benchmark the Newton contact sensor update.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed simulation steps.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up steps.")
parser.add_argument("--decimation", type=int, default=4, help="Physics steps per timed sensor cadence.")
parser.add_argument("--history_length", type=int, default=0, help="Number of contact history frames.")

parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
args_cli = parser.parse_args()

import statistics
import time

import warp as wp
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
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
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=NewtonCfg())
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene_cfg = ContactSensorBenchmarkSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, lazy_sensor_update=True)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensor = scene["contact_sensor"]
        # Warm up the simulation before collecting timings.
        for _ in range(args_cli.warmup_steps):
            for _ in range(args_cli.decimation):
                sim.step(render=False)
                sensor.update(sim_dt)
            _ = sensor.data
        wp.synchronize_device(sim.device)

        # Time only sensor work. Physics steps happen outside the timed regions.
        cadence_times_ms = []
        submit_times_ms = []
        for _ in range(args_cli.num_steps):
            cadence_time_ms = 0.0
            submit_time_ms = 0.0
            for _ in range(args_cli.decimation):
                sim.step(render=False)
                wp.synchronize_device(sim.device)
                t_start = time.perf_counter()
                sensor.update(sim_dt)
                t_submit = time.perf_counter()
                wp.synchronize_device(sim.device)
                cadence_time_ms += (time.perf_counter() - t_start) * 1000.0
                submit_time_ms += (t_submit - t_start) * 1000.0

            wp.synchronize_device(sim.device)
            t_start = time.perf_counter()
            _ = sensor.data
            t_submit = time.perf_counter()
            wp.synchronize_device(sim.device)
            cadence_time_ms += (time.perf_counter() - t_start) * 1000.0
            submit_time_ms += (t_submit - t_start) * 1000.0
            cadence_times_ms.append(cadence_time_ms)
            submit_times_ms.append(submit_time_ms)

        print("-" * 80)
        print("Contact sensor cadence benchmark (Newton)")
        print(f"  device                  : {sim.device}")
        print(f"  num_envs                : {args_cli.num_envs}")
        print(f"  num_steps               : {args_cli.num_steps}")
        print(f"  decimation              : {args_cli.decimation}")
        print(f"  history_length          : {args_cli.history_length}")
        print(f"  cadence mean (sync)     : {statistics.mean(cadence_times_ms):.3f} ms")
        print(f"  cadence p50  (sync)     : {statistics.median(cadence_times_ms):.3f} ms")
        print(f"  cadence min  (sync)     : {min(cadence_times_ms):.3f} ms")
        print(f"  cadence submit mean     : {statistics.mean(submit_times_ms):.3f} ms")
        print("-" * 80)

        # sanity check: cubes rest on the ground, so every sensor must report an upward net force
        net_forces = sensor.data.net_forces_w.torch
        num_in_contact = int((net_forces.norm(dim=-1) > 0.1).sum().item())
        if num_in_contact != args_cli.num_envs:
            raise RuntimeError(f"Expected {args_cli.num_envs} contacting sensors, received {num_in_contact}.")
        print(f"  sensors in contact      : {num_in_contact} / {args_cli.num_envs}")


if __name__ == "__main__":
    main()
