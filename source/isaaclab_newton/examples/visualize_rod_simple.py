#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple Isaac Lab Visualization: Direct Position-Based Solver for Stiff Rods

This is a simpler visualization using debug drawing primitives that's faster to
set up and works well for debugging purposes.

Usage:
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_simple.py
"""

import argparse
import math

import torch

# Parse arguments before launching the app
parser = argparse.ArgumentParser(description="Simple Rod Solver Visualization")
parser.add_argument("--num-segments", type=int, default=20, help="Number of rod segments")
parser.add_argument("--stiffness", type=float, default=1e8, help="Young's modulus (Pa)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args_cli = parser.parse_args()

# Launch the simulation app
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

# Now import remaining modules
import numpy as np
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import prim_utils

# Import the rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


def create_rod_visuals(
    num_segments: int,
    segment_radius: float,
    segment_length: float,
    base_path: str = "/World/Rod",
) -> list:
    """Create visual capsule primitives for each rod segment.

    Args:
        num_segments: Number of rod segments.
        segment_radius: Radius of each segment.
        segment_length: Length of each segment.
        base_path: Base USD prim path.

    Returns:
        List of prim paths for each segment.
    """
    prim_paths = []

    # Create parent Xform for the rod
    prim_utils.create_prim(base_path, "Xform")

    for i in range(num_segments):
        prim_path = f"{base_path}/Segment_{i:02d}"

        # Determine color based on position
        if i == 0:
            # Fixed segment - red
            color = (0.9, 0.2, 0.2)
        elif i == num_segments - 1:
            # Tip segment - gold
            color = (1.0, 0.8, 0.2)
        else:
            # Regular segment - blue gradient
            t = i / (num_segments - 1)
            color = (0.2, 0.3 + 0.4 * t, 0.9 - 0.3 * t)

        # Create capsule using sim_utils
        cfg = sim_utils.CapsuleCfg(
            radius=segment_radius,
            height=segment_length,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.5,
                roughness=0.3,
            ),
        )
        cfg.func(prim_path, cfg)
        prim_paths.append(prim_path)

    return prim_paths


def update_segment_transforms(
    prim_paths: list,
    positions: torch.Tensor,
    orientations: torch.Tensor,
):
    """Update USD prim transforms from solver data.

    Args:
        prim_paths: List of USD prim paths.
        positions: Segment positions (N, 3).
        orientations: Segment orientations as quaternions (x, y, z, w) (N, 4).
    """
    stage = prim_utils.get_current_stage()

    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xformable = UsdGeom.Xformable(prim)

        # Get position
        pos = positions[i].cpu().numpy()
        translation = Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))

        # Get orientation (convert from x,y,z,w to USD quaternion w,x,y,z)
        quat = orientations[i].cpu().numpy()
        rotation = Gf.Quatd(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))

        # Clear existing ops and set new transform
        xformable.ClearXformOpOrder()

        # Add translate operation
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(translation)

        # Add orient operation
        orient_op = xformable.AddOrientOp()
        orient_op.Set(rotation)


def main():
    """Main function."""

    # Create simulation context
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
    )
    sim = SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=(2.5, 2.5, 2.0), target=(0.75, 0.0, 0.7))

    # Create rod configuration
    num_segments = args_cli.num_segments
    rod_length = 1.5
    segment_length = rod_length / num_segments
    segment_radius = 0.025

    rod_config = RodConfig(
        material=RodMaterialConfig(
            young_modulus=args_cli.stiffness,
            density=2700.0,
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=rod_length,
            radius=segment_radius,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 120.0,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, 0.0, -9.81),  # Z-up coordinate system
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # Add lighting
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/DomeLight", cfg_light)

    cfg_dist_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.95, 0.9))
    cfg_dist_light.func("/World/DistantLight", cfg_dist_light, translation=(10.0, 10.0, 20.0))

    # Create visual segments
    print("Creating rod visuals...")
    prim_paths = create_rod_visuals(
        num_segments=num_segments,
        segment_radius=segment_radius,
        segment_length=segment_length,
    )

    # Create rod solver
    print("Initializing rod solver...")
    solver = RodSolver(rod_config, num_envs=1)

    # Position the rod horizontally at height 1.0m
    initial_height = 1.0
    for i in range(num_segments):
        solver.data.positions[:, i, 0] = (i + 0.5) * segment_length
        solver.data.positions[:, i, 1] = 0.0
        solver.data.positions[:, i, 2] = initial_height

    # Fix the first segment (cantilever boundary condition)
    solver.data.fix_segment(slice(None), 0)
    solver.data.sync_to_warp()

    # Initial visualization update
    update_segment_transforms(
        prim_paths,
        solver.data.positions[0],
        solver.data.orientations[0],
    )

    # Reset simulation
    sim.reset()

    print("=" * 60)
    print("Rod Solver Visualization - Simple Version")
    print("=" * 60)
    print(f"Number of segments: {num_segments}")
    print(f"Young's modulus: {args_cli.stiffness:.2e} Pa")
    print()
    print("Press PLAY button to start the simulation")
    print("=" * 60)

    # Simulation loop
    sim_time = 0.0
    step_count = 0

    while simulation_app.is_running():
        if sim.is_playing():
            # Step the rod solver
            solver.step(dt=sim_cfg.dt)

            # Update visual transforms
            update_segment_transforms(
                prim_paths,
                solver.data.positions[0],
                solver.data.orientations[0],
            )

            sim_time += sim_cfg.dt
            step_count += 1

            # Print status every second
            if step_count % 120 == 0:
                tip_pos = solver.data.positions[0, -1].cpu().numpy()
                print(
                    f"Time: {sim_time:.2f}s | "
                    f"Tip: ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})"
                )

        # Step simulation (rendering)
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()

