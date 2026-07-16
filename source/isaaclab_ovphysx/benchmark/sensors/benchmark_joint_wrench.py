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

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX JointWrench sensor update path.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
args_cli = parser.parse_args()

import statistics
import time

import isaaclab_ovphysx.tensor_types as TT
import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils  # noqa: F401
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


def _percentile(samples: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for sorted scalar samples."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


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
            sensor._root_view.read_into(TT.LINK_INCOMING_JOINT_FORCE, sensor._wrench_buf)
            wp.synchronize_device(sim.device)
            finished = time.perf_counter()
            read_only_ms.append((finished - start) * 1000.0)

        force = sensor.data.force.torch
        torque = sensor.data.torque.torch
        finite_wrenches = int((torch.isfinite(force).all(dim=-1) & torch.isfinite(torque).all(dim=-1)).sum().item())
        nonzero_wrenches = int(((force != 0.0).any(dim=-1) | (torque != 0.0).any(dim=-1)).sum().item())
        expected_wrenches = args_cli.num_envs * len(sensor.body_names)
        if finite_wrenches != expected_wrenches:
            raise RuntimeError(f"Expected {expected_wrenches} finite wrenches, received {finite_wrenches}.")
        if nonzero_wrenches == 0:
            raise RuntimeError("Expected at least one nonzero wrench.")

        full_mean = statistics.mean(synchronized_ms)
        read_mean = statistics.mean(read_only_ms)
        print("-" * 80)
        print("JointWrench sensor update benchmark (OVPhysX)")
        print(f"  label                  : {args_cli.label}")
        print(f"  device                 : {sim.device}")
        print(f"  num_envs               : {args_cli.num_envs}")
        print(f"  bodies_per_env         : {len(sensor.body_names)}")
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
        print(f"  finite wrenches        : {finite_wrenches} / {expected_wrenches}")
        print(f"  nonzero wrenches       : {nonzero_wrenches} / {expected_wrenches}")


if __name__ == "__main__":
    main()
