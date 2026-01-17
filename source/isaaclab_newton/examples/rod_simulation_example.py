#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Direct Position-Based Solver for Stiff Rods

This example demonstrates the rod solver implementation based on:
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"
Computer Graphics Forum, Vol. 37, No. 8

The solver uses XPBD (Extended Position-Based Dynamics) with a direct
solver that exploits the tree structure for linear-time complexity.

Usage:
    python rod_simulation_example.py [--num-segments N] [--duration T] [--visualize]
"""

import argparse
import time

import torch

# Import the rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodData,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


def create_cantilever_config(
    num_segments: int = 20,
    length: float = 1.0,
    radius: float = 0.02,
    young_modulus: float = 1e9,
    device: str = "cuda",
) -> RodConfig:
    """Create configuration for a cantilever beam simulation.

    Args:
        num_segments: Number of rigid segments.
        length: Total length of the rod [m].
        radius: Cross-section radius [m].
        young_modulus: Young's modulus [Pa].
        device: Computation device.

    Returns:
        Rod configuration.
    """
    return RodConfig(
        material=RodMaterialConfig(
            young_modulus=young_modulus,
            density=7800.0,  # Steel-like density
            damping=0.01,
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
            gravity=(0.0, -9.81, 0.0),
        ),
        device=device,
    )


def run_cantilever_simulation(config: RodConfig, duration: float = 5.0, verbose: bool = True):
    """Run a cantilever beam simulation.

    The first segment is fixed, and gravity pulls the beam down.

    Args:
        config: Rod configuration.
        duration: Simulation duration [s].
        verbose: Print progress information.

    Returns:
        Dictionary with simulation results.
    """
    # Create solver
    solver = RodSolver(config, num_envs=1)

    # Fix the first segment (cantilever boundary condition)
    solver.data.fix_segment(0, 0)

    # Simulation parameters
    dt = config.solver.dt
    num_steps = int(duration / dt)

    # Storage for results
    times = []
    tip_positions = []
    energies = []

    start_time = time.time()

    if verbose:
        print(f"Running cantilever simulation for {duration}s ({num_steps} steps)...")
        print(f"  Segments: {config.geometry.num_segments}")
        print(f"  Young's modulus: {config.material.young_modulus:.2e} Pa")
        print(f"  Solver: {'Direct' if config.solver.use_direct_solver else 'Gauss-Seidel'}")
        print()

    for step in range(num_steps):
        # Step simulation
        solver.step()

        # Record data every 10 steps
        if step % 10 == 0:
            times.append(solver.time)
            tip_pos = solver.data.positions[0, -1].clone()
            tip_positions.append(tip_pos.cpu().numpy())

            # Compute energy (for validation)
            ke, pe = solver.get_energy()
            energies.append((ke.item(), pe.item()))

        # Print progress
        if verbose and step % 100 == 0:
            tip_y = solver.data.positions[0, -1, 1].item()
            print(f"  Step {step}/{num_steps}: tip_y = {tip_y:.4f} m")

    wall_time = time.time() - start_time

    if verbose:
        print()
        print(f"Simulation completed in {wall_time:.2f}s")
        print(f"  Performance: {num_steps / wall_time:.1f} steps/s")
        print(f"  Final tip position: {solver.data.positions[0, -1].cpu().numpy()}")

    return {
        "times": times,
        "tip_positions": tip_positions,
        "energies": energies,
        "wall_time": wall_time,
        "final_state": solver.data,
    }


def run_stiffness_comparison(device: str = "cuda"):
    """Compare simulation results for different stiffness values.

    This demonstrates that the solver correctly handles a range of
    material stiffnesses from soft to very stiff.

    Args:
        device: Computation device.
    """
    print("=" * 60)
    print("Stiffness Comparison")
    print("=" * 60)

    stiffness_values = [1e6, 1e7, 1e8, 1e9, 1e10]

    results = []
    for E in stiffness_values:
        config = create_cantilever_config(
            num_segments=20, young_modulus=E, device=device
        )

        print(f"\nE = {E:.0e} Pa:")
        result = run_cantilever_simulation(config, duration=2.0, verbose=False)

        final_tip_y = result["final_state"].positions[0, -1, 1].item()
        print(f"  Final tip deflection: {final_tip_y:.4f} m")
        print(f"  Simulation time: {result['wall_time']:.2f} s")

        results.append({
            "young_modulus": E,
            "tip_deflection": final_tip_y,
            "wall_time": result["wall_time"],
        })

    print("\n" + "=" * 60)
    print("Summary: As stiffness increases, deflection decreases")
    print("=" * 60)

    for r in results:
        print(f"  E = {r['young_modulus']:.0e} Pa: δ = {r['tip_deflection']:.4f} m")


def run_solver_comparison(device: str = "cuda"):
    """Compare direct solver vs Gauss-Seidel performance.

    This demonstrates the speedup achieved by the direct solver
    for stiff rod simulations.

    Args:
        device: Computation device.
    """
    print("=" * 60)
    print("Solver Comparison: Direct vs Gauss-Seidel")
    print("=" * 60)

    num_segments = 50  # More segments to see the difference
    duration = 2.0

    # Direct solver
    print("\nDirect Solver:")
    direct_config = create_cantilever_config(
        num_segments=num_segments,
        young_modulus=1e9,
        device=device,
    )
    direct_config.solver.use_direct_solver = True
    direct_config.solver.newton_iterations = 4

    direct_result = run_cantilever_simulation(direct_config, duration=duration, verbose=False)
    direct_tip = direct_result["final_state"].positions[0, -1, 1].item()
    print(f"  Final tip deflection: {direct_tip:.4f} m")
    print(f"  Wall time: {direct_result['wall_time']:.2f} s")

    # Gauss-Seidel solver (needs more iterations for similar accuracy)
    print("\nGauss-Seidel Solver:")
    gs_config = create_cantilever_config(
        num_segments=num_segments,
        young_modulus=1e9,
        device=device,
    )
    gs_config.solver.use_direct_solver = False
    gs_config.solver.newton_iterations = 20  # More iterations needed

    gs_result = run_cantilever_simulation(gs_config, duration=duration, verbose=False)
    gs_tip = gs_result["final_state"].positions[0, -1, 1].item()
    print(f"  Final tip deflection: {gs_tip:.4f} m")
    print(f"  Wall time: {gs_result['wall_time']:.2f} s")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Direct solver: {direct_result['wall_time']:.2f}s, tip = {direct_tip:.4f}m")
    print(f"  Gauss-Seidel:  {gs_result['wall_time']:.2f}s, tip = {gs_tip:.4f}m")
    if gs_result['wall_time'] > 0:
        speedup = gs_result['wall_time'] / direct_result['wall_time']
        print(f"  Speedup: {speedup:.1f}x")
    print("=" * 60)


def visualize_rod(data: RodData, ax=None):
    """Visualize the rod configuration.

    Args:
        data: Rod data to visualize.
        ax: Matplotlib axes (optional).
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    positions = data.positions[0].cpu().numpy()

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Plot segments
    ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], 'b-o', linewidth=2, markersize=4)

    # Mark fixed segment
    ax.scatter([positions[0, 0]], [positions[0, 2]], [positions[0, 1]],
               c='r', s=100, marker='s', label='Fixed')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_zlabel('Y [m]')
    ax.set_title('Rod Configuration')
    ax.legend()

    # Set equal aspect ratio
    max_range = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min(),
    )
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.set_zlim(mid_y - max_range / 2, mid_y + max_range / 2)

    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Direct Position-Based Solver for Stiff Rods - Example"
    )
    parser.add_argument(
        "--num-segments", type=int, default=20,
        help="Number of rod segments (default: 20)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Simulation duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize the final rod configuration"
    )
    parser.add_argument(
        "--compare-stiffness", action="store_true",
        help="Run stiffness comparison"
    )
    parser.add_argument(
        "--compare-solvers", action="store_true",
        help="Compare direct vs Gauss-Seidel solvers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device (default: cuda)"
    )

    args = parser.parse_args()

    # Check device availability
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 60)
    print("Direct Position-Based Solver for Stiff Rods")
    print("Based on: Deul et al. 2018, Computer Graphics Forum")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    if args.compare_stiffness:
        run_stiffness_comparison(device)
    elif args.compare_solvers:
        run_solver_comparison(device)
    else:
        # Run standard cantilever simulation
        config = create_cantilever_config(
            num_segments=args.num_segments,
            device=device,
        )

        result = run_cantilever_simulation(config, duration=args.duration)

        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                ax = visualize_rod(result["final_state"])
                plt.savefig("rod_simulation.png", dpi=150)
                print("Visualization saved to rod_simulation.png")
                plt.show()
            except ImportError:
                print("Matplotlib not available for visualization")


if __name__ == "__main__":
    main()

