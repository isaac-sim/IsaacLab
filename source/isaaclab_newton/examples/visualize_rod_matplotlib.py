#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Matplotlib Animation: Direct Position-Based Solver for Stiff Rods

This script creates an animated visualization of the rod solver using matplotlib.
It does NOT require Isaac Sim - works with just torch, warp, and matplotlib.

Based on: Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"

Usage:
    python visualize_rod_matplotlib.py
    python visualize_rod_matplotlib.py --num-segments 30 --stiffness 1e7
    python visualize_rod_matplotlib.py --save-gif  # Save as animation GIF
"""

import argparse
import time

import numpy as np
import torch

# Import the rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


def create_rod_config(
    num_segments: int = 20,
    length: float = 1.0,
    radius: float = 0.02,
    young_modulus: float = 1e8,
    device: str = "cuda",
) -> RodConfig:
    """Create rod configuration."""
    return RodConfig(
        material=RodMaterialConfig(
            young_modulus=young_modulus,
            density=2700.0,  # Aluminum
            damping=0.02,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=length,
            radius=radius,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 60.0,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, -9.81, 0.0),
        ),
        device=device,
    )


def run_animation(
    num_segments: int = 20,
    stiffness: float = 1e8,
    duration: float = 5.0,
    save_gif: bool = False,
    device: str = "cuda",
):
    """Run the animated visualization."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("ERROR: matplotlib is required for this visualization")
        print("Install with: pip install matplotlib")
        return

    # Create configuration and solver
    config = create_rod_config(
        num_segments=num_segments,
        length=1.0,
        radius=0.02,
        young_modulus=stiffness,
        device=device,
    )

    solver = RodSolver(config, num_envs=1)
    solver.data.fix_segment(0, 0)  # Fix first segment

    # Get initial positions
    positions = solver.data.positions[0].cpu().numpy()
    seg_len = config.geometry.segment_length

    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D View', fontsize=14, fontweight='bold')
    
    # Side view (XY plane)
    ax2 = fig.add_subplot(122)
    ax2.set_title('Side View (XY Plane)', fontsize=14, fontweight='bold')

    # Initialize plot elements
    line_3d, = ax1.plot([], [], [], 'b-', linewidth=3, label='Rod')
    points_3d = ax1.scatter([], [], [], c='blue', s=50)
    fixed_point = ax1.scatter([], [], [], c='red', s=150, marker='s', label='Fixed')
    tip_point = ax1.scatter([], [], [], c='gold', s=150, marker='o', label='Tip')

    line_2d, = ax2.plot([], [], 'b-', linewidth=3)
    points_2d, = ax2.plot([], [], 'bo', markersize=8)
    fixed_2d, = ax2.plot([], [], 'rs', markersize=12)
    tip_2d, = ax2.plot([], [], 'o', color='gold', markersize=12)

    # Set axis limits
    rod_length = config.geometry.rest_length
    margin = 0.3
    
    ax1.set_xlim(-margin, rod_length + margin)
    ax1.set_ylim(-rod_length - margin, margin)
    ax1.set_zlim(-margin, margin)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.legend(loc='upper right')

    ax2.set_xlim(-margin, rod_length + margin)
    ax2.set_ylim(-rod_length - margin, margin)
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='brown', linewidth=2, label='Ground')

    # Text annotations
    time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, 
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    info_text = ax2.text(0.02, 0.85, '', transform=ax2.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Simulation parameters
    dt = config.solver.dt
    steps_per_frame = 2  # Simulation steps per animation frame
    fps = 30
    total_frames = int(duration * fps)

    print("=" * 60)
    print("Rod Solver Animation")
    print("=" * 60)
    print(f"Segments: {num_segments}")
    print(f"Young's modulus: {stiffness:.2e} Pa")
    print(f"Duration: {duration}s")
    print(f"Device: {device}")
    print("=" * 60)
    print("Starting animation...")

    def init():
        """Initialize animation."""
        line_3d.set_data([], [])
        line_3d.set_3d_properties([])
        line_2d.set_data([], [])
        points_2d.set_data([], [])
        fixed_2d.set_data([], [])
        tip_2d.set_data([], [])
        time_text.set_text('')
        info_text.set_text('')
        return line_3d, line_2d, points_2d, fixed_2d, tip_2d, time_text, info_text

    def animate(frame):
        """Update animation frame."""
        # Run simulation steps
        for _ in range(steps_per_frame):
            solver.step()

        # Get current positions
        positions = solver.data.positions[0].cpu().numpy()
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Update 3D plot
        line_3d.set_data(x, y)
        line_3d.set_3d_properties(z)
        
        # Update scatter points (need to remove old ones first)
        while len(ax1.collections) > 0:
            ax1.collections[0].remove()
        ax1.scatter(x, y, z, c='blue', s=30, alpha=0.7)
        ax1.scatter([x[0]], [y[0]], [z[0]], c='red', s=150, marker='s')
        ax1.scatter([x[-1]], [y[-1]], [z[-1]], c='gold', s=150, marker='o')

        # Update 2D plot
        line_2d.set_data(x, y)
        points_2d.set_data(x, y)
        fixed_2d.set_data([x[0]], [y[0]])
        tip_2d.set_data([x[-1]], [y[-1]])

        # Update text
        sim_time = solver.time
        time_text.set_text(f'Time: {sim_time:.2f} s')
        
        tip_deflection = y[-1]
        info_text.set_text(
            f'Tip position:\n'
            f'  X: {x[-1]:.4f} m\n'
            f'  Y: {y[-1]:.4f} m\n'
            f'Deflection: {tip_deflection:.4f} m'
        )

        return line_3d, line_2d, points_2d, fixed_2d, tip_2d, time_text, info_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=False
    )

    if save_gif:
        print("Saving animation as 'rod_simulation.gif'...")
        anim.save('rod_simulation.gif', writer='pillow', fps=fps)
        print("Saved!")
    else:
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Rod Solver Matplotlib Animation"
    )
    parser.add_argument(
        "--num-segments", type=int, default=20,
        help="Number of rod segments (default: 20)"
    )
    parser.add_argument(
        "--stiffness", type=float, default=1e8,
        help="Young's modulus in Pa (default: 1e8)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Animation duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--save-gif", action="store_true",
        help="Save animation as GIF instead of displaying"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device (default: cuda)"
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    run_animation(
        num_segments=args.num_segments,
        stiffness=args.stiffness,
        duration=args.duration,
        save_gif=args.save_gif,
        device=device,
    )


if __name__ == "__main__":
    main()

