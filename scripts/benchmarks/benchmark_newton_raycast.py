# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark comparing the Newton BVH ray-cast sensor against the warp-mesh ray caster.

Both sensors cast the same grid pattern from the same bodies against the same
rough terrain on the Newton backend:

- **LegacyRayCaster** (baseline): warp-mesh ray cast against the single terrain mesh.
- **NewtonRaycastSensor**: BVH ray cast against the whole Newton scene via
  :func:`newton.intersect_ray`, measured with and without the per-step BVH
  refit (the refit is shared with the tiled-camera renderer in real scenes).

Usage:
    ./isaaclab.sh -p scripts/benchmarks/benchmark_newton_raycast.py --num_envs 1024 --headless
"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Benchmark Newton BVH ray cast vs warp-mesh ray cast.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--num_iterations", type=int, default=300, help="Timed iterations per benchmark.")
parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations per benchmark.")
parser.add_argument("--grid_size", type=float, nargs=2, default=(1.6, 1.0), help="Grid pattern size [m].")
parser.add_argument("--grid_resolution", type=float, default=0.1, help="Grid pattern resolution [m].")
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.sensors import LegacyRayCaster, NewtonRaycastSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass


def _make_terrain_cfg(num_envs: int, env_spacing: float) -> TerrainGeneratorCfg:
    """Rough terrain sized to cover the whole env grid so every ray hits."""
    extent = math.ceil(math.sqrt(num_envs)) * env_spacing
    tiles = max(2, math.ceil(extent / 8.0) + 1)
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=4.0,
        num_rows=tiles,
        num_cols=tiles,
        use_cache=False,
        sub_terrains={
            "random": terrain_gen.HfRandomUniformTerrainCfg(noise_range=(0.0, 0.2), noise_step=0.02),
        },
    )


def _make_scene_cfg(num_envs: int, env_spacing: float = 2.0) -> InteractiveSceneCfg:
    pattern = GridPatternCfg(resolution=args_cli.grid_resolution, size=tuple(args_cli.grid_size))
    offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0))

    @configclass
    class RaycastBenchSceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=_make_terrain_cfg(num_envs, env_spacing),
        )
        body = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SensorBody",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
        ray_caster = RayCasterCfg(
            class_type=LegacyRayCaster,
            prim_path="{ENV_REGEX_NS}/SensorBody",
            offset=offset,
            pattern_cfg=pattern,
            ray_alignment="yaw",
            mesh_prim_paths=["/World/ground"],
        )
        newton_raycast = NewtonRaycastSensorCfg(
            prim_path="{ENV_REGEX_NS}/SensorBody",
            offset=offset,
            pattern_cfg=pattern,
            ray_alignment="yaw",
            global_world_only=True,
        )

    return RaycastBenchSceneCfg(num_envs=num_envs, env_spacing=env_spacing)


def _time_calls(fn, num_iterations: int, warmup: int, device: str) -> float:
    """Return mean milliseconds per call of ``fn`` (GPU-synchronized)."""
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(num_iterations):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - start) / num_iterations * 1e3


def main():
    with launch_simulation(cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1 / 200, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(_make_scene_cfg(args_cli.num_envs))
        sim.reset()
        sim.step()
        scene.update(sim.get_physics_dt())

        device = sim.device
        ray_caster = scene["ray_caster"]
        newton_raycast = scene["newton_raycast"]
        num_rays = ray_caster.num_rays
        assert newton_raycast.num_rays == num_rays
        total_rays = num_rays * args_cli.num_envs

        # Sanity: both sensors must agree on the terrain height under each ray.
        ray_caster._update_buffers_impl(ray_caster._ALL_ENV_MASK)
        newton_raycast._update_buffers_impl(newton_raycast._ALL_ENV_MASK)
        old_hits = ray_caster.data.ray_hits_w.torch
        new_hits = newton_raycast.data.ray_hits_w.torch
        old_hit = old_hits.isfinite().all(dim=-1)
        new_hit = new_hits.isfinite().all(dim=-1)
        both_hit = old_hit & new_hit
        max_err = (old_hits[both_hit] - new_hits[both_hit]).norm(dim=-1).max().item()
        print(
            f"[INFO] hit rate: warp mesh {old_hit.float().mean():.1%}, BVH {new_hit.float().mean():.1%},"
            f" max hit error {max_err:.2e} m"
        )

        def run_warp_mesh():
            ray_caster._update_buffers_impl(ray_caster._ALL_ENV_MASK)

        def run_bvh():
            newton_raycast._update_buffers_impl(newton_raycast._ALL_ENV_MASK)

        def run_bvh_with_refit():
            NewtonManager._mark_sensor_state_dirty()
            newton_raycast._update_buffers_impl(newton_raycast._ALL_ENV_MASK)

        def run_refit_only():
            NewtonManager._mark_sensor_state_dirty()
            NewtonManager._update_sensor_tasks()

        results = {
            "LegacyRayCaster (warp mesh)": run_warp_mesh,
            "NewtonRaycastSensor (BVH, no refit)": run_bvh,
            "NewtonRaycastSensor (BVH + refit)": run_bvh_with_refit,
            "BVH refit alone": run_refit_only,
        }

        print(
            f"\nnum_envs={args_cli.num_envs}, rays/env={num_rays}, total rays={total_rays},"
            f" shapes in BVH={NewtonManager.get_model().bvh_shape_count_enabled},"
            f" iterations={args_cli.num_iterations}, device={device}"
        )
        print(f"{'benchmark':<40} {'ms/call':>10} {'Mrays/s':>10}")
        for name, fn in results.items():
            ms = _time_calls(fn, args_cli.num_iterations, args_cli.warmup, device)
            mrays = total_rays / (ms * 1e-3) / 1e6
            print(f"{name:<40} {ms:>10.3f} {mrays:>10.1f}")

        # free tensors before shutdown
        del old_hits, new_hits, both_hit
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
