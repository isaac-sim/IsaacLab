#!/usr/bin/env python3
"""
Rod Solver Visualization using Isaac Sim Debug Draw

This is the simplest way to visualize the rod solver in Isaac Sim.
It draws the rod segments as colored lines that update each frame.

Usage:
    cd /home/cdinea/Downloads/IsaacLab
    python source/isaaclab_newton/examples/rod_debug_draw.py
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--num-segments", type=int, default=20, help="Number of segments")
parser.add_argument("--stiffness", type=float, default=1e7, help="Young's modulus")
args = parser.parse_args()

# Launch Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import torch
from omni.isaac.debug_draw import _debug_draw

# Import rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


def main():
    # Get debug draw interface
    draw = _debug_draw.acquire_debug_draw_interface()
    
    # Rod configuration
    num_segments = args.num_segments
    rod_length = 1.5
    seg_length = rod_length / num_segments
    
    config = RodConfig(
        material=RodMaterialConfig(
            young_modulus=args.stiffness,
            density=2700.0,
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=rod_length,
            radius=0.02,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 60.0,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, 0.0, -9.81),
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create solver
    solver = RodSolver(config, num_envs=1)
    
    # Initialize rod horizontal at height 1.0m
    for i in range(num_segments):
        solver.data.positions[:, i, 0] = (i + 0.5) * seg_length
        solver.data.positions[:, i, 1] = 0.0
        solver.data.positions[:, i, 2] = 1.0
    
    # Fix the first segment (cantilever)
    solver.data.fix_segment(0, 0)
    solver.data.sync_to_warp()
    
    print("=" * 60)
    print("Rod Solver - Isaac Sim Debug Draw Visualization")
    print("=" * 60)
    print(f"Segments: {num_segments}")
    print(f"Stiffness: {args.stiffness:.2e} Pa")
    print("Red = fixed end, Blue = rod, Yellow = tip")
    print("=" * 60)
    
    dt = config.solver.dt
    frame = 0
    
    while simulation_app.is_running():
        # Step physics
        solver.step(dt)
        
        # Get positions
        pos = solver.data.positions[0].cpu().numpy()
        
        # Clear previous drawings
        draw.clear_lines()
        draw.clear_points()
        
        # Draw rod segments as lines
        for i in range(num_segments - 1):
            p1 = pos[i].tolist()
            p2 = pos[i + 1].tolist()
            
            # Color: red for fixed, blue for middle, yellow for tip
            if i == 0:
                color = [1.0, 0.2, 0.2, 1.0]  # Red
            elif i >= num_segments - 2:
                color = [1.0, 0.8, 0.2, 1.0]  # Yellow
            else:
                color = [0.2, 0.4, 0.9, 1.0]  # Blue
            
            draw.draw_lines([p1], [p2], [color], [3.0])
        
        # Draw segment centers as points
        point_colors = []
        for i in range(num_segments):
            if i == 0:
                point_colors.append([1.0, 0.0, 0.0, 1.0])
            elif i == num_segments - 1:
                point_colors.append([1.0, 1.0, 0.0, 1.0])
            else:
                point_colors.append([0.3, 0.5, 1.0, 1.0])
        
        draw.draw_points(pos.tolist(), point_colors, [8.0] * num_segments)
        
        # Update app
        simulation_app.update()
        
        frame += 1
        if frame % 60 == 0:
            tip = pos[-1]
            print(f"Frame {frame}: Tip position = ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})")
    
    simulation_app.close()


if __name__ == "__main__":
    main()

