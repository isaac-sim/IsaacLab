#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Isaac Lab Visualization: Direct Position-Based Solver for Stiff Rods

This script demonstrates visualizing the rod solver in Isaac Lab's simulation
environment. It creates a scene with capsule primitives representing rod segments
and updates their positions/orientations based on the solver output.

Based on: Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"

Usage:
    # Run with Isaac Lab's Python wrapper
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_isaaclab.py

    # With options
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_isaaclab.py --num-segments 20 --stiffness 1e8
"""

import argparse
import math

import torch

# Isaac Lab imports (must come after argparse due to Omniverse app bootstrap)
from isaacsim import SimulationApp

# Parse arguments before launching the app
parser = argparse.ArgumentParser(description="Visualize Rod Solver in Isaac Lab")
parser.add_argument("--num-segments", type=int, default=15, help="Number of rod segments")
parser.add_argument("--stiffness", type=float, default=1e8, help="Young's modulus (Pa)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
args_cli = parser.parse_args()

# Launch the simulation app
simulation_app = SimulationApp({"headless": args_cli.headless})

# Now import remaining modules
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz

# Import the rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


def create_rod_config(
    num_segments: int = 15,
    length: float = 1.5,
    radius: float = 0.03,
    young_modulus: float = 1e8,
) -> RodConfig:
    """Create rod configuration for visualization.

    Args:
        num_segments: Number of rod segments.
        length: Total rod length in meters.
        radius: Segment radius in meters.
        young_modulus: Material stiffness in Pascals.

    Returns:
        RodConfig instance.
    """
    return RodConfig(
        material=RodMaterialConfig(
            young_modulus=young_modulus,
            density=2700.0,  # Aluminum-like
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=length,
            radius=radius,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 120.0,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, 0.0, -9.81),  # Isaac Lab uses Z-up
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def design_scene(num_segments: int, segment_radius: float, segment_length: float):
    """Design the scene with ground plane, lighting, and rod markers.

    Args:
        num_segments: Number of rod segments.
        segment_radius: Radius of each segment.
        segment_length: Length of each segment.
    """
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # Dome light for nice ambient lighting
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_light.func("/World/DomeLight", cfg_light)

    # Distant light for shadows
    cfg_dist_light = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(1.0, 0.95, 0.9),
    )
    cfg_dist_light.func("/World/DistantLight", cfg_dist_light, translation=(10.0, 10.0, 20.0))

    # Create marker configuration for rod segments
    # Using capsules to represent cylindrical segments
    markers_cfg = VisualizationMarkersCfg(
        prim_path="/World/RodMarkers",
        markers={
            # Fixed segment (anchor) - red color
            "fixed_segment": sim_utils.CapsuleCfg(
                radius=segment_radius * 1.1,  # Slightly larger
                height=segment_length,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.2, 0.2),
                    metallic=0.3,
                    roughness=0.4,
                ),
            ),
            # Regular segment - metallic blue
            "segment": sim_utils.CapsuleCfg(
                radius=segment_radius,
                height=segment_length,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.4, 0.9),
                    metallic=0.6,
                    roughness=0.3,
                ),
            ),
            # Tip segment - gold color
            "tip_segment": sim_utils.CapsuleCfg(
                radius=segment_radius * 1.05,
                height=segment_length,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.8, 0.2),
                    metallic=0.8,
                    roughness=0.2,
                ),
            ),
        },
    )

    # Create the markers
    rod_markers = VisualizationMarkers(markers_cfg)

    return rod_markers


def rod_orientations_to_isaac_quats(orientations: torch.Tensor) -> torch.Tensor:
    """Convert rod solver orientations to Isaac Lab quaternion format.

    Rod solver uses (x, y, z, w) format, Isaac Lab uses (w, x, y, z) format.
    Also applies rotation to align rod local X-axis with visual capsule.

    Args:
        orientations: Quaternions in (x, y, z, w) format. Shape: (N, 4)

    Returns:
        Quaternions in (w, x, y, z) format. Shape: (N, 4)
    """
    # Reorder from (x, y, z, w) to (w, x, y, z)
    isaac_quats = torch.zeros_like(orientations)
    isaac_quats[:, 0] = orientations[:, 3]  # w
    isaac_quats[:, 1] = orientations[:, 0]  # x
    isaac_quats[:, 2] = orientations[:, 1]  # y
    isaac_quats[:, 3] = orientations[:, 2]  # z
    return isaac_quats


def main():
    """Main function."""

    # Create simulation context
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )
    sim = SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.75, 0.0, 0.5))

    # Create rod configuration
    rod_config = create_rod_config(
        num_segments=args_cli.num_segments,
        length=1.5,
        radius=0.03,
        young_modulus=args_cli.stiffness,
    )

    # Design the scene with markers
    segment_length = rod_config.geometry.segment_length
    segment_radius = rod_config.geometry.radius
    rod_markers = design_scene(args_cli.num_segments, segment_radius, segment_length)

    # Create rod solver
    solver = RodSolver(rod_config, num_envs=args_cli.num_envs)

    # Position the rod horizontally starting from origin
    # Shift to start at world origin with segments along X axis, elevated in Z
    initial_height = 1.0  # meters above ground
    for i in range(rod_config.geometry.num_segments):
        solver.data.positions[:, i, 0] = (i + 0.5) * segment_length  # X position
        solver.data.positions[:, i, 1] = 0.0  # Y position
        solver.data.positions[:, i, 2] = initial_height  # Z position (height)

    # Fix the first segment (cantilever boundary condition)
    solver.data.fix_segment(slice(None), 0)

    # Sync to warp
    solver.data.sync_to_warp()

    # Play the simulation (don't use reset which triggers Newton physics)
    sim.play()

    print("=" * 60)
    print("Rod Solver Visualization in Isaac Lab")
    print("=" * 60)
    print(f"Number of segments: {args_cli.num_segments}")
    print(f"Young's modulus: {args_cli.stiffness:.2e} Pa")
    print(f"Segment length: {segment_length:.4f} m")
    print(f"Segment radius: {segment_radius:.4f} m")
    print()
    print("Controls:")
    print("  - Press PLAY to start simulation")
    print("  - Use mouse to orbit camera")
    print("  - Close window to exit")
    print("=" * 60)

    # Prepare marker indices (0 = fixed, 1 = regular, 2 = tip)
    num_segments = args_cli.num_segments
    marker_indices = torch.ones(num_segments, dtype=torch.int32)
    marker_indices[0] = 0  # First segment is fixed (red)
    marker_indices[-1] = 2  # Last segment is tip (gold)

    # Simulation loop
    sim_time = 0.0
    step_count = 0

    while simulation_app.is_running():
        # Step physics if simulation is playing
        if sim.is_playing():
            # Step the rod solver
            solver.step(dt=sim_cfg.dt)

            # Get positions and orientations from solver
            positions = solver.data.positions[0].cpu()  # (num_segments, 3)
            orientations = solver.data.orientations[0].cpu()  # (num_segments, 4)

            # Convert orientations to Isaac Lab format
            isaac_quats = rod_orientations_to_isaac_quats(orientations)

            # Update marker visualization
            rod_markers.visualize(
                translations=positions,
                orientations=isaac_quats,
                marker_indices=marker_indices,
            )

            sim_time += sim_cfg.dt
            step_count += 1

            # Print status periodically
            if step_count % 120 == 0:
                tip_pos = positions[-1]
                print(
                    f"Time: {sim_time:.2f}s | "
                    f"Tip position: ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})"
                )

        # Render the scene
        sim.step()

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()

