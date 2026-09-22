# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise Newton BVH ray casting against static or moving geometry.

The ``heightfield`` scene scans a wave terrain. The ``moving-geometry`` scene
tracks falling boxes and a kinematic bar while Newton refits its BVH.

.. code-block:: bash

    uvx isaaclab example newton-raycast --scene moving-geometry
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton BVH ray-cast sensor example.")
parser.add_argument(
    "--scene",
    choices=("heightfield", "moving-geometry"),
    default="heightfield",
    help="Geometry scanned by the sensor.",
)
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

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
from isaaclab.utils import configclass

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

BOX_DROP_POSITIONS = ((1.0, 0.6, 3.0), (-0.8, -1.0, 3.5), (0.2, -1.2, 4.0))


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
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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


def _falling_box_cfg(index: int) -> RigidObjectCfg:
    """Create one falling box."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Box_{index}",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4 + 0.2 * index, 0.9 - 0.3 * index)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_DROP_POSITIONS[index]),
    )


@configclass
class MovingGeometrySceneCfg(InteractiveSceneCfg):
    """Ground plane, falling boxes, a sweeping bar, and a hovering sensor."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    box_0 = _falling_box_cfg(0)
    box_1 = _falling_box_cfg(1)
    box_2 = _falling_box_cfg(2)
    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.CuboidCfg(
            size=(3.5, 0.3, 0.3),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
    )
    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )
    raycast = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        offset=NewtonRaycastSensorCfg.OffsetCfg(pos=(0.0, 0.0, -0.1)),
        pattern_cfg=GridPatternCfg(resolution=0.25, size=(3.0, 3.0)),
        ray_alignment="yaw",
        max_distance=10.0,
        debug_vis=True,
    )


def _newton_gl_viewer(sim: sim_utils.SimulationContext) -> Any | None:
    """Return the active Newton GL viewer, if any."""
    from isaaclab_visualizers.newton import NewtonGLVisualizer

    return next(
        (
            visualizer._viewer
            for visualizer in getattr(sim, "_visualizers", [])
            if isinstance(visualizer, NewtonGLVisualizer)
        ),
        None,
    )


def _draw_ray_lines(viewer: Any, sensor: NewtonRaycastSensor, miss_length: float = 3.0) -> None:
    """Draw red hit rays and gray misses."""
    starts = sensor.ray_starts_w.torch.reshape(-1, 3)
    directions = sensor.ray_directions_w.torch.reshape(-1, 3)
    hits = sensor.data.ray_hits_w.torch.reshape(-1, 3)
    misses = torch.isinf(sensor.data.ray_distances.torch.reshape(-1, 1))
    ends = torch.where(misses, starts + directions * miss_length, hits)
    colors = torch.where(
        misses,
        torch.tensor([0.5, 0.5, 0.5], device=starts.device),
        torch.tensor([1.0, 0.15, 0.1], device=starts.device),
    )

    viewer.log_lines(
        "/isaaclab/raycast/rays",
        wp.from_torch(starts.contiguous(), dtype=wp.vec3f),
        wp.from_torch(ends.contiguous(), dtype=wp.vec3f),
        wp.from_torch(colors.contiguous(), dtype=wp.vec3f),
    )


def _animate_heightfield(body: RigidObject, time: float, zero_velocity: torch.Tensor) -> None:
    """Move the sensor body over the wave terrain."""
    angle = 0.4 * time
    position = torch.tensor(
        [[3.0 * math.cos(angle), 3.0 * math.sin(angle), 1.4 + 0.3 * math.sin(0.9 * time)]],
        device=body.device,
    )
    angles = torch.tensor(
        [0.3 * math.sin(0.7 * time), 0.25 * math.sin(1.1 * time), angle + math.pi / 2.0],
        device=body.device,
    )
    orientation = math_utils.quat_from_euler_xyz(*(value.unsqueeze(0) for value in angles))
    body.write_root_pose_to_sim_index(root_pose=torch.cat([position, orientation], dim=-1))
    body.write_root_velocity_to_sim_index(root_velocity=zero_velocity)


def _animate_moving_geometry(
    boxes: list[RigidObject],
    bar: RigidObject,
    body: RigidObject,
    step: int,
    sim_dt: float,
    zero_velocity: torch.Tensor,
    zero_angle: torch.Tensor,
) -> None:
    """Drop boxes while sweeping the bar and sensor."""
    if step % 400 == 0:
        for box, drop_position in zip(boxes, BOX_DROP_POSITIONS):
            pose = torch.tensor([[*drop_position, 0.3, 0.3, 0.0, 0.9]], device=body.device)
            pose[:, 3:] /= torch.linalg.norm(pose[:, 3:])
            box.write_root_pose_to_sim_index(root_pose=pose)
            box.write_root_velocity_to_sim_index(root_velocity=zero_velocity)
            box.reset()

    time = step * sim_dt
    bar_orientation = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, zero_angle + 0.8 * time)
    bar_position = torch.tensor([[0.0, 0.0, 0.8]], device=body.device)
    bar.write_root_pose_to_sim_index(root_pose=torch.cat([bar_position, bar_orientation], dim=-1))
    bar.write_root_velocity_to_sim_index(root_velocity=zero_velocity)

    body_orientation = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, zero_angle - 0.3 * time)
    body_position = torch.tensor(
        [[0.6 * math.cos(0.5 * time), 0.6 * math.sin(0.5 * time), 2.5]],
        device=body.device,
    )
    body.write_root_pose_to_sim_index(root_pose=torch.cat([body_position, body_orientation], dim=-1))
    body.write_root_velocity_to_sim_index(root_velocity=zero_velocity)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, scene_name: str, max_steps: int) -> None:
    """Animate the selected scene until it closes or reaches the step limit."""
    body: RigidObject = scene["body"]
    sensor: NewtonRaycastSensor = scene["raycast"]
    viewer = _newton_gl_viewer(sim)
    sim_dt = sim.get_physics_dt()
    zero_velocity = torch.zeros(1, 6, device=sim.device)
    zero_angle = torch.zeros(1, device=sim.device)
    boxes: list[RigidObject] = []
    bar: RigidObject | None = None
    if scene_name == "moving-geometry":
        boxes = [scene[f"box_{index}"] for index in range(len(BOX_DROP_POSITIONS))]
        bar = scene["bar"]

    step = 0
    while sim.is_headless_or_exist_active_visualizer() and (max_steps < 0 or step < max_steps):
        if bar is None:
            _animate_heightfield(body, step * sim_dt, zero_velocity)
        else:
            _animate_moving_geometry(boxes, bar, body, step, sim_dt, zero_velocity, zero_angle)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if viewer is not None:
            _draw_ray_lines(viewer, sensor)
        step += 1


def main() -> None:
    """Launch the selected Newton ray-cast scene."""
    with launch_simulation(cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1 / 100, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        if args_cli.scene == "heightfield":
            scene_cfg = HeightfieldSceneCfg(num_envs=1, env_spacing=1.0)
            sim.set_camera_view(eye=[7.0, 7.0, 5.0], target=[0.0, 0.0, 0.0])
        else:
            scene_cfg = MovingGeometrySceneCfg(num_envs=1, env_spacing=1.0)
            sim.set_camera_view(eye=[6.0, 6.0, 4.5], target=[0.0, 0.0, 1.0])
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene, args_cli.scene, args_cli.max_steps)


if __name__ == "__main__":
    main()
