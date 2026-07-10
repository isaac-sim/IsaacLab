# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script for the PhysX contact sensor update.

Measures the time spent in :meth:`ContactSensor.update` for a scene of cubes resting on a
flat ground plane. By default the warp kernels of the sensor update are captured into CUDA
graphs and replayed; pass ``--disable_graph`` to measure the eager (non-graphed) update for
comparison.

Usage:
    # Graphed update (default)
    ./isaaclab.sh -p source/isaaclab_physx/benchmark/sensors/benchmark_contact_sensor.py --num_envs 4096

    # Eager update (baseline)
    ./isaaclab.sh -p source/isaaclab_physx/benchmark/sensors/benchmark_contact_sensor.py --num_envs 4096 --disable_graph
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark the PhysX contact sensor update.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed simulation steps.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up steps.")
parser.add_argument("--disable_graph", action="store_true", help="Disable CUDA graph capture of the sensor update.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import statistics
import time

import warp as wp

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
    )


def main():
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = ContactSensorBenchmarkSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, lazy_sensor_update=False)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    sensor = scene["contact_sensor"]
    if args_cli.disable_graph:
        sensor._use_graph = False

    # warm-up: let the cubes settle on the ground and trigger the graph capture (if enabled)
    for _ in range(args_cli.warmup_steps):
        sim.step(render=False)
        sensor.update(sim_dt, force_recompute=True)
    wp.synchronize_device(sim.device)

    # timed loop: per-call latency of the sensor update (synchronized)
    update_times_ms = []
    total_wall_ms = 0.0
    for _ in range(args_cli.num_steps):
        sim.step(render=False)
        wp.synchronize_device(sim.device)
        t_start = time.perf_counter()
        sensor.update(sim_dt, force_recompute=True)
        t_submit = time.perf_counter()
        wp.synchronize_device(sim.device)
        t_end = time.perf_counter()
        update_times_ms.append((t_end - t_start) * 1000.0)
        total_wall_ms += (t_submit - t_start) * 1000.0

    mode = "eager" if args_cli.disable_graph else "graph"
    print("-" * 80)
    print("Contact sensor update benchmark (PhysX)")
    print(f"  mode              : {mode}")
    print(f"  device            : {sim.device}")
    print(f"  num_envs          : {args_cli.num_envs}")
    print(f"  num_steps         : {args_cli.num_steps}")
    print(f"  update mean (sync): {statistics.mean(update_times_ms):.3f} ms")
    print(f"  update p50  (sync): {statistics.median(update_times_ms):.3f} ms")
    print(f"  update min  (sync): {min(update_times_ms):.3f} ms")
    print(f"  submit mean (cpu) : {total_wall_ms / args_cli.num_steps:.3f} ms")
    print("-" * 80)

    # sanity check: cubes rest on the ground, so every sensor must report an upward net force
    net_forces = sensor.data.net_forces_w.torch
    num_in_contact = int((net_forces.norm(dim=-1) > 0.1).sum().item())
    print(f"  sensors in contact: {num_in_contact} / {args_cli.num_envs}")


if __name__ == "__main__":
    main()
    simulation_app.close()
