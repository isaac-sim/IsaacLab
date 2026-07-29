# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the OVPhysX RayCaster update path.

Mirrors ``isaaclab_physx/benchmark/sensors/benchmark_ray_caster.py`` but runs
kitless against the OVPhysX backend. By default, matched sensors cast against a
plane and deterministic rough terrain. The script also times each blocking
native ``RIGID_BODY_POSE`` binding read in isolation.

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
parser.add_argument(
    "--terrain",
    choices=("all", "plane", "rough"),
    default="all",
    help="Terrain workload to run. The default runs plane and rough workloads.",
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.seed import configure_seed

wp.init()

_ROUGH_TERRAIN_SEED = 0
_ROUGH_TERRAIN_SIZE = max(3.0, args_cli.grid_size + 1.0)
_WORKLOADS = ("plane", "rough")


def _sensor_body_cfg(prim_path: str, position: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> RigidObjectCfg:
    """Create one collision-free kinematic carrier for a ray-caster workload."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _rough_terrain_cfg() -> TerrainGeneratorCfg:
    """Create deterministic rough terrain large enough for the configured ray grid."""
    return TerrainGeneratorCfg(
        seed=_ROUGH_TERRAIN_SEED,
        size=(_ROUGH_TERRAIN_SIZE, _ROUGH_TERRAIN_SIZE),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        use_cache=False,
        sub_terrains={
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(0.0, 0.2),
                noise_step=0.02,
                border_width=0.25,
            )
        },
    )


@configclass
class RayCasterBenchmarkSceneCfg(InteractiveSceneCfg):
    """Matched plane and rough-terrain ray-caster workloads."""

    terrain = TerrainImporterCfg(
        prim_path="/World/rough_ground",
        terrain_type="generator",
        terrain_generator=_rough_terrain_cfg(),
    )
    plane_ground = AssetBaseCfg(prim_path="/World/plane_ground", spawn=sim_utils.GroundPlaneCfg())
    plane_sensor_body = _sensor_body_cfg(
        "{ENV_REGEX_NS}/PlaneSensorBody", position=(2.0 * _ROUGH_TERRAIN_SIZE, 0.0, 1.0)
    )
    rough_sensor_body = _sensor_body_cfg("{ENV_REGEX_NS}/RoughSensorBody")
    plane_ray_caster: RayCasterCfg | None = None
    rough_ray_caster: RayCasterCfg | None = None


def main() -> None:
    """Run the benchmark and print latency statistics."""
    configure_seed(_ROUGH_TERRAIN_SEED)
    sim_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=sim_dt, device=args_cli.device, physics=OvPhysxCfg(), gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=args_cli.device, sim_cfg=sim_cfg) as sim:
        scene_cfg = RayCasterBenchmarkSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=2.0,
            lazy_sensor_update=True,
        )
        workloads = _WORKLOADS if args_cli.terrain == "all" else (args_cli.terrain,)
        pattern_cfg = patterns.GridPatternCfg(
            resolution=args_cli.grid_resolution,
            size=(args_cli.grid_size, args_cli.grid_size),
            direction=(0.0, 0.0, -1.0),
        )
        for workload in workloads:
            setattr(
                scene_cfg,
                f"{workload}_ray_caster",
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{workload.title()}SensorBody",
                    mesh_prim_paths=[f"/World/{workload}_ground"],
                    ray_alignment="world",
                    pattern_cfg=pattern_cfg,
                    global_world_only=True,
                ),
            )

        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensors = {workload: scene[f"{workload}_ray_caster"] for workload in workloads}
        rays_per_env = next(iter(sensors.values())).num_rays
        if any(sensor.num_rays != rays_per_env for sensor in sensors.values()):
            raise RuntimeError("Plane and rough workloads must cast the same number of rays.")

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples_by_workload = {}
        observer_samples_by_workload = {}
        read_only_samples_by_workload = {}
        validation_by_workload = {}
        for workload, sensor in sensors.items():
            for _ in range(args_cli.warmup_steps):
                sim.step()
                sensor.update(sim_dt, force_recompute=True)
            # Intentionally drain warm-up work before any timed boundary.
            wp.synchronize_device(sim.device)

            samples = []
            for _ in range(args_cli.num_steps):
                sim.step()
                samples.append(
                    measure_latency(
                        operation=lambda sensor=sensor: sensor.update(sim_dt, force_recompute=True),
                        synchronize=synchronize_device,
                    )
                )
            samples_by_workload[workload] = samples
            observer_samples_by_workload[workload] = [
                measure_latency(operation=lambda: None, synchronize=synchronize_device)
                for _ in range(args_cli.num_steps)
            ]

            read_only_samples = []
            if sensor._ovphysx_body_view is not None:
                read_only_samples = [
                    measure_latency(
                        operation=lambda sensor=sensor: sensor._ovphysx_body_view.read(sensor._pose_buf),
                        synchronize=synchronize_device,
                    )
                    for _ in range(args_cli.num_steps)
                ]
            read_only_samples_by_workload[workload] = read_only_samples

            ray_hits = sensor.data.ray_hits_w.torch
            finite_hits = int(torch.isfinite(ray_hits).all(dim=-1).sum().item())
            expected_hits = args_cli.num_envs * sensor.num_rays
            if finite_hits != expected_hits:
                raise RuntimeError(f"Expected {expected_hits} finite {workload} ray hits, received {finite_hits}.")
            validation = [
                SingleMeasurement(name="Finite Ray Hits", value=finite_hits, unit="count"),
                SingleMeasurement(name="Expected Ray Hits", value=expected_hits, unit="count"),
            ]
            if workload == "plane":
                max_hit_height = float(torch.abs(ray_hits[..., 2]).max().item())
                hit_height_tolerance = 1.0e-4
                if max_hit_height > hit_height_tolerance:
                    raise RuntimeError(
                        f"Expected plane ray hits within {hit_height_tolerance} m of z=0, "
                        f"received maximum |z| {max_hit_height} m."
                    )
                validation.append(SingleMeasurement(name="Maximum Absolute Hit Height", value=max_hit_height, unit="m"))
            else:
                validation.extend(
                    [
                        SingleMeasurement(
                            name="Minimum Hit Height", value=float(ray_hits[..., 2].min().item()), unit="m"
                        ),
                        SingleMeasurement(
                            name="Maximum Hit Height", value=float(ray_hits[..., 2].max().item()), unit="m"
                        ),
                    ]
                )
            validation_by_workload[workload] = validation

        benchmark = LatencyBenchmarkRunner(
            benchmark_name="ovphysx_ray_caster_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "physics_variant": args_cli.physics_variant,
                "label": args_cli.label,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "rays_per_env": rays_per_env,
                "grid_size": args_cli.grid_size,
                "grid_resolution": args_cli.grid_resolution,
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
                "terrain": args_cli.terrain,
                "rough_terrain_seed": _ROUGH_TERRAIN_SEED,
            },
        )
        for workload in workloads:
            update_phase = f"{workload}_sensor_update"
            full_stats = benchmark.add_latency_samples(update_phase, samples_by_workload[workload])
            read_only_samples = read_only_samples_by_workload[workload]
            if read_only_samples:
                read_stats = benchmark.add_latency_samples(f"{workload}_native_read", read_only_samples)
                benchmark.add_measurement(
                    update_phase,
                    measurement=SingleMeasurement(
                        name="Estimated Synchronized Non-read Time",
                        value=(full_stats.mean_s - read_stats.mean_s) * 1000.0,
                        unit="ms",
                    ),
                )
            benchmark.add_synchronized_samples(
                f"{workload}_observer",
                "Synchronized Observer Floor",
                [sample.synchronized_s for sample in observer_samples_by_workload[workload]],
            )
            benchmark.add_measurement(
                f"{workload}_validation",
                measurement=validation_by_workload[workload],
            )
        benchmark.finalize()


if __name__ == "__main__":
    main()
