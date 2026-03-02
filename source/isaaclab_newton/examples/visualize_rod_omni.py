#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Omniverse Visualization: Direct Position-Based Solver for Stiff Rods

This script visualizes the rod solver using Omniverse Kit directly,
bypassing Isaac Lab's Newton physics integration.

Usage:
    conda activate isaaclab
    python source/isaaclab_newton/examples/visualize_rod_omni.py
"""

import argparse

# Parse arguments before launching the app
parser = argparse.ArgumentParser(description="Rod Solver Omniverse Visualization")
parser.add_argument("--num-segments", type=int, default=15, help="Number of rod segments")
parser.add_argument("--stiffness", type=float, default=1e6, help="Young's modulus (Pa) - try 1e5 for soft, 1e8 for stiff")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--show-algorithm", action="store_true", help="Show algorithm details (energy, constraints)")
parser.add_argument("--compare-solvers", action="store_true", help="Compare Direct vs Gauss-Seidel solvers")
args_cli = parser.parse_args()

# Launch the simulation app with window
from isaacsim import SimulationApp

# Configure for GUI mode with visible window
config = {
    "headless": args_cli.headless,
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "anti_aliasing": 0,
    "renderer": "RayTracedLighting",
}
simulation_app = SimulationApp(config)

# Now import remaining modules
import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import omni.kit.app
import omni.usd

# Import viewport for camera setup
try:
    from omni.kit.viewport.utility import get_active_viewport
    HAS_VIEWPORT = True
except ImportError:
    HAS_VIEWPORT = False

# Import the rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


def create_material(stage, path: str, color: tuple) -> UsdShade.Material:
    """Create a simple colored material."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_ground_plane(stage):
    """Create a ground plane."""
    plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    plane.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateNormalsAttr([(0, 0, 1)] * 4)
    
    # Gray material
    mat = create_material(stage, "/World/Materials/GroundMat", (0.3, 0.3, 0.3))
    UsdShade.MaterialBindingAPI(plane).Bind(mat)


def create_lighting(stage):
    """Create scene lighting."""
    # Dome light
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000)
    
    # Distant light
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr(3000)
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))


def create_rod_segments(stage, num_segments: int, seg_radius: float, seg_length: float):
    """Create capsule prims for rod segments."""
    # Create materials
    fixed_mat = create_material(stage, "/World/Materials/FixedMat", (0.9, 0.2, 0.2))
    segment_mat = create_material(stage, "/World/Materials/SegmentMat", (0.2, 0.4, 0.9))
    tip_mat = create_material(stage, "/World/Materials/TipMat", (1.0, 0.8, 0.2))
    
    # Create rod parent
    UsdGeom.Xform.Define(stage, "/World/Rod")
    
    capsules = []
    for i in range(num_segments):
        path = f"/World/Rod/Segment_{i:02d}"
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateRadiusAttr(seg_radius)
        capsule.CreateHeightAttr(seg_length)
        capsule.CreateAxisAttr("X")
        
        # Bind material
        binding = UsdShade.MaterialBindingAPI(capsule)
        if i == 0:
            binding.Bind(fixed_mat)
        elif i == num_segments - 1:
            binding.Bind(tip_mat)
        else:
            binding.Bind(segment_mat)
        
        capsules.append(capsule)
    
    return capsules


def setup_segment_transforms(capsules):
    """Initialize transform ops for segments. Returns (translate_ops, orient_ops)."""
    translate_ops = []
    orient_ops = []
    
    for capsule in capsules:
        xformable = UsdGeom.Xformable(capsule.GetPrim())
        xformable.ClearXformOpOrder()
        
        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp()
        
        translate_ops.append(translate_op)
        orient_ops.append(orient_op)
    
    return translate_ops, orient_ops


def update_segment_transforms(translate_ops, orient_ops, positions: torch.Tensor, orientations: torch.Tensor):
    """Update segment transforms from solver data."""
    for i, (trans_op, orient_op) in enumerate(zip(translate_ops, orient_ops)):
        pos = positions[i].cpu().numpy()
        quat = orientations[i].cpu().numpy()  # x, y, z, w
        
        # Translation
        trans_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        
        # Orientation (convert x,y,z,w to w,x,y,z for USD)
        orient_op.Set(Gf.Quatf(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])))


def main():
    """Main function."""
    # Get USD stage
    stage = omni.usd.get_context().get_stage()
    
    # Create scene
    create_ground_plane(stage)
    create_lighting(stage)
    
    # Rod configuration
    num_segments = args_cli.num_segments
    rod_length = 1.5
    seg_length = rod_length / num_segments
    seg_radius = 0.03
    
    config = RodConfig(
        material=RodMaterialConfig(
            young_modulus=args_cli.stiffness,
            density=2700.0,
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=rod_length,
            radius=seg_radius,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 60.0,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, 0.0, -9.81),  # Z-up
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create visual segments
    capsules = create_rod_segments(stage, num_segments, seg_radius, seg_length)
    
    # Setup transform ops (do this once)
    translate_ops, orient_ops = setup_segment_transforms(capsules)
    
    # Create solver
    solver = RodSolver(config, num_envs=1)
    
    # Position rod horizontally at height 1.0m
    for i in range(num_segments):
        solver.data.positions[:, i, 0] = (i + 0.5) * seg_length
        solver.data.positions[:, i, 1] = 0.0
        solver.data.positions[:, i, 2] = 1.0
    
    solver.data.fix_segment(0, 0)
    solver.data.sync_to_warp()
    
    # Initial update
    update_segment_transforms(translate_ops, orient_ops, solver.data.positions[0], solver.data.orientations[0])
    
    # Setup camera to view the rod
    camera_path = "/World/Camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.75, -2.5, 1.0))  # Position camera to see rod
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(80, 0, 0))  # Angle down to see rod
    
    # Set viewport camera if available
    if HAS_VIEWPORT and not args_cli.headless:
        try:
            viewport = get_active_viewport()
            if viewport:
                viewport.set_active_camera(camera_path)
                print("Camera set to view rod")
        except Exception as e:
            print(f"Could not set viewport camera: {e}")
    
    # Print solver configuration
    print("=" * 60)
    print("DIRECT POSITION-BASED SOLVER FOR STIFF RODS")
    print("Algorithm from Deul et al. 2018 (CGF)")
    print("=" * 60)
    print(f"SOLVER TYPE: {'Direct Tree Solver (O(n) - Paper Algorithm)' if config.solver.use_direct_solver else 'Gauss-Seidel (Standard XPBD)'}")
    print(f"Newton Iterations: {config.solver.newton_iterations}")
    print(f"Substeps per frame: {config.solver.num_substeps}")
    print(f"Time step: {config.solver.dt:.4f}s")
    print("-" * 60)
    print(f"Segments: {num_segments}")
    print(f"Young's Modulus: {args_cli.stiffness:.2e} Pa")
    print(f"Density: {config.material.density} kg/m³")
    print(f"Rod Length: {rod_length} m")
    print("-" * 60)
    print("CONSTRAINT TYPES:")
    print("  • Stretch (inextensibility) - keeps segments at rest length")
    print("  • Bend/Twist (Cosserat model) - resists bending and torsion")
    print("=" * 60)
    
    if args_cli.show_algorithm:
        print("\nALGORITHM STEPS PER SUBSTEP:")
        print("  1. Predict positions: x* = x + v*dt + gravity*dt²")
        print("  2. Predict orientations: q* = q + ω*dt")
        print("  3. Newton iterations (direct solve):")
        print("     - Compute constraint Jacobians")
        print("     - Build block-tridiagonal system")
        print("     - Solve in O(n) using tree structure")
        print("     - Update positions and Lagrange multipliers")
        print("  4. Update velocities: v = (x - x_old) / dt")
        print("=" * 60)
    
    # Main loop
    frame = 0
    sim_time = 0.0
    dt = config.solver.dt
    
    # Track algorithm metrics
    energy_history = []
    stretch_error_history = []
    
    app = omni.kit.app.get_app()
    
    while simulation_app.is_running():
        # Step the rod solver
        solver.step(dt)
        sim_time += dt
        
        # Update visuals
        update_segment_transforms(translate_ops, orient_ops, solver.data.positions[0], solver.data.orientations[0])
        
        # Compute algorithm metrics
        avg_stretch_error = 0.0
        energy = 0.0
        if args_cli.show_algorithm and frame % 10 == 0:
            positions = solver.data.positions[0].cpu().numpy()
            velocities = solver.data.velocities[0].cpu().numpy()
            
            # Compute kinetic energy: 0.5 * m * v^2
            segment_mass = config.material.density * np.pi * seg_radius**2 * seg_length
            kinetic_energy = 0.5 * segment_mass * np.sum(velocities**2)
            
            # Compute gravitational potential energy: m * g * h
            potential_energy = segment_mass * 9.81 * np.sum(positions[:, 2])
            
            energy = kinetic_energy + potential_energy
            energy_history.append(energy)
            
            # Compute stretch constraint error (how well inextensibility is maintained)
            stretch_errors = []
            for i in range(num_segments - 1):
                actual_dist = np.linalg.norm(positions[i+1] - positions[i])
                rest_dist = seg_length
                stretch_errors.append(abs(actual_dist - rest_dist) / rest_dist * 100)
            avg_stretch_error = np.mean(stretch_errors)
            stretch_error_history.append(avg_stretch_error)
        
        # Render
        app.update()
        
        frame += 1
        if frame % 60 == 0:
            tip = solver.data.positions[0, -1].cpu().numpy()
            
            if args_cli.show_algorithm:
                print(f"Time: {sim_time:.2f}s | Tip Z: {tip[2]:.3f}m | Energy: {energy:.4f}J | Stretch Error: {avg_stretch_error:.2f}%")
            else:
                print(f"Time: {sim_time:.2f}s | Tip: ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})")
    
    simulation_app.close()


if __name__ == "__main__":
    main()

