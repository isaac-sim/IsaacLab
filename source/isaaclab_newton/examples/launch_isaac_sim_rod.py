#!/usr/bin/env python3
"""
Launch Isaac Sim with Rod Visualization - WITH VISIBLE WINDOW

This script explicitly creates a viewport window to visualize the rod.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--stiffness", type=float, default=1e5)
parser.add_argument("--num-segments", type=int, default=15)
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
    RodSolver, RodSolverConfig,
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
    
    # Materials
    def make_mat(path, color):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.3)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat
    
    segment_mat = make_mat("/World/Materials/Segment", (0.2, 0.4, 0.9))
    fixed_mat = make_mat("/World/Materials/Fixed", (0.9, 0.2, 0.2))
    tip_mat = make_mat("/World/Materials/Tip", (1.0, 0.8, 0.2))
    
    return segment_mat, fixed_mat, tip_mat


def main():
    print("=" * 60)
    print("ISAAC SIM ROD VISUALIZATION")
    print("Direct Position-Based Solver for Stiff Rods")
    print("=" * 60)
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # Create scene
    segment_mat, fixed_mat, tip_mat = create_scene(stage)
    
    # Rod parameters
    num_segments = args.num_segments
    rod_length = 1.5
    seg_length = rod_length / num_segments
    seg_radius = 0.03
    
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
    
    # End spheres (positioned at rod endpoints)
    fixed_sphere = UsdGeom.Sphere.Define(stage, "/World/Rod/Fixed")
    fixed_sphere.CreateRadiusAttr(seg_radius * 1.5)
    UsdShade.MaterialBindingAPI(fixed_sphere).Bind(fixed_mat)
    fixed_xform = UsdGeom.Xformable(fixed_sphere.GetPrim())
    fixed_xform.ClearXformOpOrder()
    fixed_translate_op = fixed_xform.AddTranslateOp()
    fixed_translate_op.Set(Gf.Vec3d(0.5 * seg_length, 0, 1))
    
    tip_sphere = UsdGeom.Sphere.Define(stage, "/World/Rod/Tip")
    tip_sphere.CreateRadiusAttr(seg_radius * 1.5)
    UsdShade.MaterialBindingAPI(tip_sphere).Bind(tip_mat)
    tip_xform = UsdGeom.Xformable(tip_sphere.GetPrim())
    tip_xform.ClearXformOpOrder()
    tip_translate_op = tip_xform.AddTranslateOp()
    tip_translate_op.Set(Gf.Vec3d((num_segments - 0.5) * seg_length, 0, 1))
    
    # Create solver
    config = RodConfig(
        material=RodMaterialConfig(young_modulus=args.stiffness, density=2700, damping=0.05),
        geometry=RodGeometryConfig(num_segments=num_segments, rest_length=rod_length, radius=seg_radius),
        solver=RodSolverConfig(dt=1/60, num_substeps=2, newton_iterations=4, use_direct_solver=True, gravity=(0, 0, -9.81)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    solver = RodSolver(config, num_envs=1)
    
    # Initialize positions
    for i in range(num_segments):
        solver.data.positions[:, i, 0] = (i + 0.5) * seg_length
        solver.data.positions[:, i, 1] = 0
        solver.data.positions[:, i, 2] = 1
    solver.data.fix_segment(0, 0)
    solver.data.sync_to_warp()
    
    print(f"Stiffness: {args.stiffness:.2e} Pa")
    print(f"Segments: {num_segments}")
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
        
        # Periodic perturbation
        if frame % 180 == 90:
            impulse = torch.tensor([0, np.sin(frame * 0.01) * 0.5, 0.3], dtype=torch.float32)
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

