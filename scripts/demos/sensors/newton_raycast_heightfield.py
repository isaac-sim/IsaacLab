# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: Newton BVH ray-cast sensor scanning a heightfield terrain.

A grid ray-cast sensor rides a body that circles, bobs, and tumbles above a
wave heightfield. Rays are drawn live in the Newton viewer — red where they
hit the terrain (with sphere markers at the hit points), gray where they miss.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/sensors/newton_raycast_heightfield.py

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Newton BVH ray-cast sensor scanning a heightfield.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()

import math

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import NewtonRaycastSensor, NewtonRaycastSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass

WAVE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=1.0,
    num_rows=1,
    num_cols=1,
    use_cache=False,
    sub_terrains={
        "waves": terrain_gen.HfWaveTerrainCfg(amplitude_range=(0.25, 0.25), num_waves=6),
    },
)


@configclass
class HeightfieldSceneCfg(InteractiveSceneCfg):
    """Wave heightfield with a floating sensor body."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=WAVE_TERRAIN_CFG
    )

    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.25, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
    )

    raycast = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        pattern_cfg=GridPatternCfg(resolution=0.25, size=(1.5, 1.0)),
        ray_alignment="base",
        global_world_only=True,
        max_distance=10.0,
        debug_vis=True,
    )


def _newton_gl_viewer(sim: sim_utils.SimulationContext):
    """Return the Newton GL viewer when the newton visualizer is active."""
    from isaaclab_visualizers.newton import NewtonVisualizer

    for viz in getattr(sim, "_visualizers", []):
        if isinstance(viz, NewtonVisualizer):
            return viz._viewer
    return None


def log_ray_lines(viewer, sensor: NewtonRaycastSensor, miss_length: float = 3.0):
    """Draw the sensor rays in the viewer: red to the hit point, gray for misses."""
    starts = sensor.ray_starts_w.torch.reshape(-1, 3)
    directions = sensor.ray_directions_w.torch.reshape(-1, 3)
    hits = sensor.data.ray_hits_w.torch.reshape(-1, 3)
    miss = torch.isinf(sensor.data.ray_distances.torch.reshape(-1, 1))
    ends = torch.where(miss, starts + directions * miss_length, hits)
    colors = torch.where(
        miss,
        torch.tensor([0.5, 0.5, 0.5], device=starts.device),
        torch.tensor([1.0, 0.15, 0.1], device=starts.device),
    )
    to_wp = lambda t: wp.array(t.cpu().numpy().astype(np.float32), dtype=wp.vec3)  # noqa: E731
    viewer.log_lines("/isaaclab/raycast/rays", to_wp(starts), to_wp(ends), to_wp(colors))


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Fly the sensor body over the terrain and visualize the rays."""
    body: RigidObject = scene["body"]
    sensor: NewtonRaycastSensor = scene["raycast"]
    viewer = _newton_gl_viewer(sim)

    sim_dt = sim.get_physics_dt()
    zero_vel = torch.zeros(1, 6, device=sim.device)
    t = 0.0
    while sim.is_headless_or_exist_active_visualizer():
        # Circle above the terrain while bobbing, pitching, rolling, and yawing.
        angle = 0.4 * t
        pos = torch.tensor(
            [[3.0 * math.cos(angle), 3.0 * math.sin(angle), 1.4 + 0.3 * math.sin(0.9 * t)]], device=sim.device
        )
        angles = torch.tensor([0.3 * math.sin(0.7 * t), 0.25 * math.sin(1.1 * t), angle + math.pi / 2.0])
        quat = math_utils.quat_from_euler_xyz(*(a.unsqueeze(0) for a in angles.to(sim.device)))
        body.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1))
        body.write_root_velocity_to_sim_index(root_velocity=zero_vel)
        scene.write_data_to_sim()

        sim.step()
        scene.update(sim_dt)
        if viewer is not None:
            log_ray_lines(viewer, sensor)
        t += sim_dt


def main():
    """Main function."""
    with launch_simulation(cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1 / 100, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[7.0, 7.0, 5.0], target=[0.0, 0.0, 0.0])
        scene = InteractiveScene(HeightfieldSceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
