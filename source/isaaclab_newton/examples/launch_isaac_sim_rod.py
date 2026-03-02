#!/usr/bin/env python3
"""
Launch Isaac Sim with Rod Visualization - WITH VISIBLE WINDOW

This script explicitly creates a viewport window to visualize the rod.
"""

import argparse
parser = argparse.ArgumentParser(description="Interactive Catheter Simulation in Isaac Sim")

# Geometry
parser.add_argument("--num-segments", type=int, default=50, help="Number of segments")
parser.add_argument("--length", type=float, default=0.5, help="Guidewire length in meters (50cm)")
parser.add_argument("--radius", type=float, default=0.0004, help="Guidewire radius (0.035 inch = 0.89mm diameter)")
parser.add_argument("--tip-segments", type=int, default=15, help="Number of tip segments for shaping")

# Stiffness (normalized 0-1, like Newton Viewer)
parser.add_argument("--stretch-stiffness", type=float, default=1.0, help="Stretch stiffness (0-1)")
parser.add_argument("--shear-stiffness", type=float, default=1.0, help="Shear stiffness (0-1)")
parser.add_argument("--bend-stiffness", type=float, default=0.1, help="Bend stiffness (0-1)")
parser.add_argument("--twist-stiffness", type=float, default=0.4, help="Twist stiffness (0-1)")
parser.add_argument("--young-modulus", type=float, default=1e7, help="Base Young's modulus [Pa]")

# Tip shape (rest curvature)
parser.add_argument("--tip-bend", type=float, default=0.0, help="Tip rest curvature [rad/m]")

# Simulation
parser.add_argument("--gravity", action="store_true", help="Enable gravity")
parser.add_argument("--damping", type=float, default=0.05, help="Velocity damping")

args = parser.parse_args()

# Launch with explicit window settings
from isaacsim import SimulationApp

launch_config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "window_width": 1920, 
    "window_height": 1080,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Enable viewport display
    "anti_aliasing": 0,
}

print("Launching Isaac Sim...")
simulation_app = SimulationApp(launch_config)

# Now import everything else
import numpy as np
import torch
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

import omni.kit.app
import omni.usd
import omni.kit.viewport.utility as viewport_utils

from isaaclab_newton.solvers import (
    RodConfig, RodGeometryConfig, RodMaterialConfig, 
    RodSolver, RodSolverConfig, RodTipConfig,
)

def create_scene(stage):
    """Create the visual scene with rod."""
    
    # Ground
    plane = UsdGeom.Mesh.Define(stage, "/World/Ground")
    plane.CreatePointsAttr([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    
    # Lights
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000)
    
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight") 
    distant.CreateIntensityAttr(3000)
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    
    # Materials - Guidewire metallic appearance
    def make_mat(path, color, metallic=0.9, roughness=0.2):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat
    
    # Stainless steel / Nitinol guidewire colors
    guidewire_mat = make_mat("/World/Materials/Guidewire", (0.75, 0.75, 0.78), metallic=0.95, roughness=0.15)
    hub_mat = make_mat("/World/Materials/Hub", (0.2, 0.2, 0.25), metallic=0.8, roughness=0.3)  # Dark hub
    tip_mat = make_mat("/World/Materials/Tip", (0.9, 0.85, 0.7), metallic=0.7, roughness=0.25)  # Soft gold tip
    
    # Aliases for compatibility
    segment_mat = guidewire_mat
    fixed_mat = hub_mat
    
    return segment_mat, fixed_mat, tip_mat


def main():
    # Guidewire parameters (realistic medical guidewire dimensions)
    num_segments = args.num_segments
    rod_length = args.length  # 50cm typical guidewire
    seg_length = rod_length / num_segments
    seg_radius = args.radius  # 0.035" = 0.89mm diameter (standard guidewire)
    
    print("=" * 60)
    print("INTERACTIVE CATHETER SIMULATION IN ISAAC SIM")
    print("Direct Position-Based Solver for Stiff Rods (Deul et al. 2018)")
    print("=" * 60)
    print(f"Length: {rod_length*100:.1f} cm")
    print(f"Diameter: {seg_radius*2*1000:.2f} mm")
    print(f"Segments: {num_segments}")
    print(f"Stiffness: stretch={args.stretch_stiffness}, shear={args.shear_stiffness}, "
          f"bend={args.bend_stiffness}, twist={args.twist_stiffness}")
    print(f"Tip curvature: {args.tip_bend} rad/m ({args.tip_segments} tip segments)")
    print("=" * 60)
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # Create scene
    segment_mat, fixed_mat, tip_mat = create_scene(stage)
    
    # Create smooth rod curve
    UsdGeom.Xform.Define(stage, "/World/Rod")
    rod_curve = UsdGeom.BasisCurves.Define(stage, "/World/Rod/Curve")
    rod_curve.CreateTypeAttr("cubic")
    rod_curve.CreateBasisAttr("catmullRom")
    rod_curve.CreateWrapAttr("nonperiodic")
    
    # Initial points - duplicate first/last for Catmull-Rom endpoint interpolation
    points = []
    first_pt = Gf.Vec3f(0.5 * seg_length, 0, 1)
    last_pt = Gf.Vec3f((num_segments - 0.5) * seg_length, 0, 1)
    points.append(first_pt)  # Duplicate first
    for i in range(num_segments):
        points.append(Gf.Vec3f((i + 0.5) * seg_length, 0, 1))
    points.append(last_pt)  # Duplicate last
    
    rod_curve.CreatePointsAttr(points)
    rod_curve.CreateCurveVertexCountsAttr([len(points)])
    rod_curve.CreateWidthsAttr([seg_radius * 2] * len(points))
    UsdShade.MaterialBindingAPI(rod_curve).Bind(segment_mat)
    
    # Hub (connector at base) - larger cylinder representing the Luer lock hub
    hub = UsdGeom.Cylinder.Define(stage, "/World/Rod/Hub")
    hub.CreateRadiusAttr(seg_radius * 8)  # Hub is wider than wire
    hub.CreateHeightAttr(seg_radius * 20)
    hub.CreateAxisAttr("X")
    UsdShade.MaterialBindingAPI(hub).Bind(fixed_mat)
    hub_xform = UsdGeom.Xformable(hub.GetPrim())
    hub_xform.ClearXformOpOrder()
    hub_translate_op = hub_xform.AddTranslateOp()
    hub_translate_op.Set(Gf.Vec3d(-seg_radius * 10, 0, 0.1))  # Positioned at origin
    
    # Tip marker - small rounded end (atraumatic tip)
    tip_sphere = UsdGeom.Sphere.Define(stage, "/World/Rod/Tip")
    tip_sphere.CreateRadiusAttr(seg_radius * 1.2)  # Slightly larger than wire
    UsdShade.MaterialBindingAPI(tip_sphere).Bind(tip_mat)
    tip_xform = UsdGeom.Xformable(tip_sphere.GetPrim())
    tip_xform.ClearXformOpOrder()
    tip_translate_op = tip_xform.AddTranslateOp()
    tip_translate_op.Set(Gf.Vec3d((num_segments - 0.5) * seg_length, 0, 0.1))
    
    # Keep fixed_translate_op for compatibility (points to hub)
    fixed_translate_op = hub_translate_op
    
    # Tip configuration for catheter-like behavior
    tip_config = RodTipConfig(
        num_tip_segments=args.tip_segments,
        rest_bend_omega1=args.tip_bend,
        rest_bend_omega2=0.0,
        rest_twist=0.0,
    )
    
    # Create solver with configurable stiffness parameters
    config = RodConfig(
        material=RodMaterialConfig(
            young_modulus=args.young_modulus,
            density=6450.0,  # Nitinol density
            damping=args.damping,
            # Normalized stiffness controls (Newton Viewer style)
            stretch_stiffness=args.stretch_stiffness,
            shear_stiffness=args.shear_stiffness,
            bend_stiffness=args.bend_stiffness,
            twist_stiffness=args.twist_stiffness,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments, 
            rest_length=rod_length, 
            radius=seg_radius,
            tip=tip_config,
        ),
        solver=RodSolverConfig(
            dt=1/120,  # Smaller timestep for thin wire stability
            num_substeps=4, 
            newton_iterations=6, 
            use_direct_solver=True, 
            gravity=(0, 0, -9.81) if args.gravity else (0, 0, 0),
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    solver = RodSolver(config, num_envs=1)
    
    # Initialize positions - guidewire extending horizontally
    for i in range(num_segments):
        solver.data.positions[:, i, 0] = (i + 0.5) * seg_length
        solver.data.positions[:, i, 1] = 0
        solver.data.positions[:, i, 2] = 0.1  # Low height, like on a table
    solver.data.fix_segment(0, 0)
    solver.data.sync_to_warp()
    
    print("Solver initialized. Starting simulation...")
    print("=" * 60)
    
    # Try to get viewport and set camera
    try:
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            print("Viewport found!")
    except:
        print("No viewport available")
    
    # Main loop
    app = omni.kit.app.get_app()
    frame = 0
    dt = 1/60
    
    while simulation_app.is_running():
        # Step physics
        solver.step(dt)
        
        # Periodic perturbation - gentle manipulation of guidewire tip
        if frame % 240 == 120:
            impulse = torch.tensor([0, np.sin(frame * 0.01) * 0.1, 0.05], dtype=torch.float32)
            solver.data.velocities[0, -1, :] += impulse.to(solver.data.velocities.device)
        
        # Update visuals
        positions = solver.data.positions[0].cpu().numpy()
        
        # Update curve - duplicate first and last points for Catmull-Rom to pass through endpoints
        new_points = []
        # Duplicate first point
        new_points.append(Gf.Vec3f(float(positions[0][0]), float(positions[0][1]), float(positions[0][2])))
        # All points
        for p in positions:
            new_points.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
        # Duplicate last point
        new_points.append(Gf.Vec3f(float(positions[-1][0]), float(positions[-1][1]), float(positions[-1][2])))
        rod_curve.GetPointsAttr().Set(new_points)
        rod_curve.GetCurveVertexCountsAttr().Set([len(new_points)])
        rod_curve.GetWidthsAttr().Set([seg_radius * 2] * len(new_points))
        
        # Update sphere positions (use the pre-created translate ops)
        fixed_translate_op.Set(Gf.Vec3d(float(positions[0][0]), float(positions[0][1]), float(positions[0][2])))
        tip_translate_op.Set(Gf.Vec3d(float(positions[-1][0]), float(positions[-1][1]), float(positions[-1][2])))
        
        # Render
        app.update()
        
        frame += 1
        if frame % 60 == 0:
            print(f"Time: {frame/60:.1f}s | Tip: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f})")
    
    simulation_app.close()


if __name__ == "__main__":
    main()

