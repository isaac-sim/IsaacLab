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

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX contact sensor update path.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed simulation steps.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up steps.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
parser.add_argument(
    "--disable_graph",
    action="store_true",
    default=False,
    help="Run the update kernels eagerly instead of replaying the captured CUDA graph.",
)
args_cli = parser.parse_args()

import statistics
import time

import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_ovphysx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
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


def _percentile(samples: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for sorted scalar samples."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


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
        if args_cli.disable_graph and hasattr(sensor, "_update_graph"):
            sensor._update_graph.enabled = False

        # warm-up: let the cubes settle on the ground
        for _ in range(args_cli.warmup_steps):
            sim.step()
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronized_ms: list[float] = []
        submission_ms: list[float] = []
        for _ in range(args_cli.num_steps):
            sim.step()
            wp.synchronize_device(sim.device)
            start = time.perf_counter()
            sensor.update(sim_dt, force_recompute=True)
            submitted = time.perf_counter()
            wp.synchronize_device(sim.device)
            finished = time.perf_counter()
            synchronized_ms.append((finished - start) * 1000.0)
            submission_ms.append((submitted - start) * 1000.0)

        # read-only phase: only the blocking native fetch, no warp kernel tail
        read_only_ms: list[float] = []
        for _ in range(args_cli.num_steps):
            wp.synchronize_device(sim.device)
            start = time.perf_counter()
            sensor._contact_binding.read_net_forces(sensor._net_forces_flat_buf)
            wp.synchronize_device(sim.device)
            finished = time.perf_counter()
            read_only_ms.append((finished - start) * 1000.0)

        net_forces = sensor.data.net_forces_w.torch
        num_in_contact = int((net_forces.norm(dim=-1) > 0.1).sum().item())

        full_mean = statistics.mean(synchronized_ms)
        read_mean = statistics.mean(read_only_ms)
        print("-" * 80)
        print("Contact sensor update benchmark (OVPhysX)")
        print(f"  label                  : {args_cli.label}")
        print(f"  device                 : {sim.device}")
        print(f"  num_envs               : {args_cli.num_envs}")
        print(f"  num_steps              : {args_cli.num_steps}")
        print(f"  synchronized mean      : {full_mean:.3f} ms")
        print(f"  synchronized p50       : {_percentile(synchronized_ms, 0.50):.3f} ms")
        print(f"  synchronized p95       : {_percentile(synchronized_ms, 0.95):.3f} ms")
        print(f"  submission mean        : {statistics.mean(submission_ms):.3f} ms")
        print(f"  submission p50         : {_percentile(submission_ms, 0.50):.3f} ms")
        print(f"  submission p95         : {_percentile(submission_ms, 0.95):.3f} ms")
        print(f"  read-only mean         : {read_mean:.3f} ms")
        print(f"  read-only p50          : {_percentile(read_only_ms, 0.50):.3f} ms")
        print(f"  read-only p95          : {_percentile(read_only_ms, 0.95):.3f} ms")
        print(f"  implied kernel tail    : {full_mean - read_mean:.3f} ms")
        print("-" * 80)
        print(f"  sensors in contact     : {num_in_contact} / {args_cli.num_envs}")


if __name__ == "__main__":
    main()
