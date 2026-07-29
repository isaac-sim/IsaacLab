# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the OVPhysX RayCaster update path.

Mirrors ``isaaclab_physx/benchmark/sensors/benchmark_ray_caster.py`` but runs
kitless against the OVPhysX backend. Also times the blocking native
``RIGID_BODY_POSE`` binding read in isolation.

Usage:
    ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_ray_caster.py --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX RayCaster update path.")
parser.add_argument("--physics_variant", choices=("ovphysx",), default="ovphysx", help="Exact physics variant.")
parser.add_argument("--num_envs", type=parse_positive_int, default=4096, help="Number of environments.")
parser.add_argument("--grid_size", type=float, default=1.0, help="Width and length [m] of each ray grid.")
parser.add_argument("--grid_resolution", type=float, default=0.25, help="Ray-grid resolution [m].")
parser.add_argument("--num_steps", type=parse_positive_int, default=500, help="Number of timed updates.")
parser.add_argument(
    "--warmup_steps", type=parse_non_negative_int, default=50, help="Number of untimed warm-up updates."
)
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
args_cli = parser.parse_args()
if args_cli.grid_size <= 0:
    parser.error("--grid_size must be greater than zero")
if args_cli.grid_resolution <= 0:
    parser.error("--grid_resolution must be greater than zero")

import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

wp.init()


@configclass
class RayCasterBenchmarkSceneCfg(InteractiveSceneCfg):
    """One kinematic sensor body per environment above a shared ground plane."""

    ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    sensor_body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    ray_caster: RayCasterCfg | None = None


def main() -> None:
    """Run the benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=sim_dt, device=args_cli.device, physics=OvPhysxCfg(), gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=args_cli.device, sim_cfg=sim_cfg) as sim:
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
            sim.step()
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples = []
        for _ in range(args_cli.num_steps):
            sim.step()
            samples.append(
                measure_latency(
                    operation=lambda: sensor.update(sim_dt, force_recompute=True),
                    synchronize=synchronize_device,
                )
            )

        observer_samples = [
            measure_latency(operation=lambda: None, synchronize=synchronize_device) for _ in range(args_cli.num_steps)
        ]

        # Read-only phase: the blocking native pose fetch without raycast kernels.
        read_only_samples = []
        if sensor._ovphysx_body_view is not None:
            read_only_samples = [
                measure_latency(
                    operation=lambda: sensor._ovphysx_body_view.read(sensor._pose_buf),
                    synchronize=synchronize_device,
                )
                for _ in range(args_cli.num_steps)
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
            benchmark_name="ovphysx_ray_caster_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "physics_variant": args_cli.physics_variant,
                "label": args_cli.label,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "rays_per_env": sensor.num_rays,
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
            },
        )
        full_stats = benchmark.add_latency_samples("sensor_update", samples)
        if read_only_samples:
            read_stats = benchmark.add_latency_samples("native_read", read_only_samples)
            benchmark.add_measurement(
                "sensor_update",
                measurement=SingleMeasurement(
                    name="Estimated Synchronized Non-read Time",
                    value=(full_stats.mean_s - read_stats.mean_s) * 1000.0,
                    unit="ms",
                ),
            )
        benchmark.add_synchronized_samples(
            "observer", "Synchronized Observer Floor", [s.synchronized_s for s in observer_samples]
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
