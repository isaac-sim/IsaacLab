# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the standard Newton RayCaster update path.

Each environment contains one kinematic body carrying a downward-facing grid
sensor. All sensors cast against one shared ground plane.

Usage:
    ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_ray_caster.py --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

parser = argparse.ArgumentParser(description="Benchmark the standard Newton RayCaster update path.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--grid_size", type=float, default=1.0, help="Width and length [m] of each ray grid.")
parser.add_argument("--grid_resolution", type=float, default=0.25, help="Ray-grid resolution [m].")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
args_cli = parser.parse_args()
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero")
if args_cli.grid_size <= 0.0:
    parser.error("--grid_size must be greater than zero")
if args_cli.grid_resolution <= 0.0:
    parser.error("--grid_resolution must be greater than zero")
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be greater than zero")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")

import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass


@configclass
class RayCasterBenchmarkSceneCfg(InteractiveSceneCfg):
    """One kinematic sensor body per environment above a shared ground plane."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sensor_body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    ray_caster: RayCasterCfg = None


def main() -> None:
    """Run the benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=NewtonCfg(), gravity=(0.0, 0.0, 0.0))
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene_cfg = RayCasterBenchmarkSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=2.0,
            lazy_sensor_update=True,
        )
        scene_cfg.ray_caster = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/SensorBody",
            mesh_prim_paths=["/World/ground"],
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=args_cli.grid_resolution,
                size=(args_cli.grid_size, args_cli.grid_size),
                direction=(0.0, 0.0, -1.0),
            ),
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensor = scene["ray_caster"]

        for _ in range(args_cli.warmup_steps):
            sim.step(render=False)
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize = partial(wp.synchronize_device, sim.device)
        samples = []
        for _ in range(args_cli.num_steps):
            sim.step(render=False)
            samples.append(
                measure_latency(
                    operation=lambda: sensor.update(sim_dt, force_recompute=True),
                    synchronize=synchronize,
                )
            )

        observer_samples = [
            measure_latency(operation=lambda: None, synchronize=synchronize) for _ in range(args_cli.num_steps)
        ]

        ray_hits = sensor.data.ray_hits_w.torch
        finite_hits = int(torch.isfinite(ray_hits).all(dim=-1).sum().item())
        expected_hits = args_cli.num_envs * sensor.num_rays
        if finite_hits != expected_hits:
            raise RuntimeError(f"Expected {expected_hits} finite ray hits, received {finite_hits}.")
        max_hit_height = float(torch.abs(ray_hits[..., 2]).max().item())
        hit_height_tolerance = 1.0e-4
        if max_hit_height > hit_height_tolerance:
            raise RuntimeError(
                f"Expected ray hits within {hit_height_tolerance} m of the z=0 plane, "
                f"received maximum |z| {max_hit_height} m."
            )

        benchmark = LatencyBenchmarkRunner(
            benchmark_name="newton_ray_caster_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "label": args_cli.label,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "rays_per_env": sensor.num_rays,
                "grid_size": args_cli.grid_size,
                "grid_resolution": args_cli.grid_resolution,
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
            },
        )
        benchmark.add_latency_samples("sensor_update", samples)
        benchmark.add_synchronized_samples(
            "observer", "Synchronized Observer Floor", [sample.synchronized_s for sample in observer_samples]
        )
        benchmark.add_measurement(
            "validation",
            measurement=[
                SingleMeasurement(name="Finite Ray Hits", value=finite_hits, unit="count"),
                SingleMeasurement(name="Expected Ray Hits", value=expected_hits, unit="count"),
                SingleMeasurement(name="Maximum Absolute Hit Height", value=max_hit_height, unit="m"),
            ],
        )
        benchmark.finalize()


if __name__ == "__main__":
    main()
