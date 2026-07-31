# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a pile of cables that collide and settle on each other.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/cables.py
    ./isaaclab.sh -p scripts/demos/cables.py --num_cables 40 --num_segments 15

"""

from __future__ import annotations

import argparse
import math
import random

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Spawn a pile of cables with Newton VBD.")
parser.add_argument("--num_cables", type=int, default=25, help="Number of cables to spawn.")
parser.add_argument("--num_segments", type=int, default=20, help="Number of segments per cable.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

if args_cli.num_cables < 1:
    parser.error("--num_cables must be at least 1.")
if args_cli.num_segments < 2:
    parser.error("--num_segments must be at least 2.")

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObject, CableObjectCfg


def design_scene(num_cables: int, num_segments: int, colorize: bool) -> dict[str, CableObject]:
    """Spawn a ground plane, light, and randomly oriented cable pile.

    Args:
        num_cables: Number of cables to spawn.
        num_segments: Number of segments per cable.
        colorize: Whether to give each cable a random visual material.
    """
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    cable_length = 0.5
    segment_length = cable_length / num_segments
    thickness = 0.01
    radius = 0.5 * thickness
    target_stretch_stiffness = 5.0e5  # [N/m] per joint
    target_bend_stiffness = 20.0  # [N.m/rad] per joint
    stretch_modulus = target_stretch_stiffness * segment_length / (math.pi * radius**2)
    bend_modulus = target_bend_stiffness * segment_length / (0.25 * math.pi * radius**4)
    xy_jitter = 0.3
    z_spacing = 1.5 * thickness
    z_base = 0.8
    positions = [(index * segment_length, 0.0, 0.0) for index in range(num_segments + 1)]

    print(f"[INFO]: Spawning {num_cables} cables...")
    entities: dict[str, CableObject] = {}
    for index in range(num_cables):
        angle = random.uniform(0.0, 2.0 * math.pi)
        position = (
            random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.cos(angle),
            random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.sin(angle),
            z_base + index * z_spacing,
        )
        orientation = (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))
        visual_material = None
        if colorize:
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random())
            )
        cfg = CableObjectCfg(
            prim_path=f"/World/Env_0/Cable{index:03d}",
            spawn=sim_utils.CableCfg(
                positions=positions,
                visual_material=visual_material,
                physics_material=sim_utils.CableMaterialCfg(
                    thickness=thickness,
                    density=100.0,
                    stretch_stiffness=stretch_modulus,
                    bend_stiffness=bend_modulus,
                ),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=position, rot=orientation),
        )
        entities[f"cable_{index:03d}"] = CableObject(cfg=cfg)

    return entities


def reset_cables(entities: dict[str, CableObject]) -> None:
    """Restore every cable to its initial segment state."""
    for cable in entities.values():
        cable.write_segment_pose_to_sim_index(segment_pose=cable.data.default_segment_pose_w)
        cable.write_segment_velocity_to_sim_index(segment_velocity=cable.data.default_segment_velocity_w)


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, CableObject], max_steps: int = -1) -> None:
    """Run the simulation and periodically restore the cable pile."""
    sim_dt = sim.get_physics_dt()
    reset_steps = max(1, int(2.0 / sim_dt))
    count = 0

    while (max_steps < 0 or count < max_steps) and sim.is_headless_or_exist_active_visualizer():
        if count > 0 and count % reset_steps == 0:
            reset_cables(entities)
            print("[INFO]: Resetting cable state...")

        sim.step(render=False)
        for cable in entities.values():
            cable.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Launch and run the cable pile demo."""
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_contrib.deformable import VBDSolverCfg

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=20), num_substeps=8),
    )
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(2.0, 2.0, 1.0), target=(0.0, 0.0, 0.25))
        colorize = bool(args_cli.visualizer and "kit" in args_cli.visualizer)
        entities = design_scene(args_cli.num_cables, args_cli.num_segments, colorize)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, entities, args_cli.max_steps)


if __name__ == "__main__":
    main()
