#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive Rod Solver Visualization

User Interaction:
  - Click on rod segments to select them
  - Drag selected segment with mouse
  - Keyboard controls for forces and parameters

Controls:
  SPACE  - Toggle pause/play
  R      - Reset rod to initial position
  G      - Toggle gravity on/off
  W/S    - Apply upward/downward force to tip
  A/D    - Apply left/right force to tip
  Q/E    - Apply forward/backward force to tip
  +/-    - Increase/decrease stiffness
  1-5    - Set number of Newton iterations
  ESC    - Exit

Usage:
    python source/isaaclab_newton/examples/visualize_rod_interactive.py
"""

import argparse

parser = argparse.ArgumentParser(description="Interactive Rod Solver")
parser.add_argument("--num-segments", type=int, default=15, help="Number of segments")
parser.add_argument("--stiffness", type=float, default=1e6, help="Young's modulus (Pa)")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--no-auto-perturb", action="store_true", help="Disable automatic perturbations")
parser.add_argument("--save-usd", type=str, default=None, help="Save USD file to this path")
args_cli = parser.parse_args()

# Launch Isaac Sim
from isaacsim import SimulationApp

config = {
    "headless": args_cli.headless,
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "renderer": "RayTracedLighting",
}
simulation_app = SimulationApp(config)

import numpy as np
import torch
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

import carb.input
import omni.appwindow
import omni.kit.app
import omni.usd

from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


class InteractiveRodSimulation:
    """Interactive rod simulation with mouse and keyboard controls."""

    def __init__(self, num_segments: int, stiffness: float):
        self.num_segments = num_segments
        self.stiffness = stiffness
        self.rod_length = 1.5
        self.seg_length = self.rod_length / num_segments
        self.seg_radius = 0.03

        # State
        self.paused = False
        self.gravity_enabled = True
        self.applied_force = torch.zeros(3)
        self.selected_segment = -1
        self.dragging = False

        # Setup
        self.stage = omni.usd.get_context().get_stage()
        self._setup_scene()
        self._setup_solver()
        self._setup_input()

        self.frame = 0
        self.sim_time = 0.0

    def _setup_scene(self):
        """Create the visual scene."""
        # Ground plane
        plane = UsdGeom.Mesh.Define(self.stage, "/World/Ground")
        plane.CreatePointsAttr([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)])
        plane.CreateFaceVertexCountsAttr([4])
        plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

        # Lighting
        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1000)
        distant = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant.CreateIntensityAttr(3000)
        distant.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

        # Materials
        self._create_materials()

        # Rod as a smooth curve (BasisCurves for smooth tube rendering)
        UsdGeom.Xform.Define(self.stage, "/World/Rod")
        
        # Create smooth tube using BasisCurves
        self.rod_curve = UsdGeom.BasisCurves.Define(self.stage, "/World/Rod/SmoothTube")
        self.rod_curve.CreateTypeAttr("cubic")
        self.rod_curve.CreateBasisAttr("catmullRom")  # Smooth Catmull-Rom spline
        self.rod_curve.CreateWrapAttr("nonperiodic")
        
        # Initialize with segment positions - duplicate endpoints for Catmull-Rom
        initial_points = []
        first_pt = Gf.Vec3f(0.5 * self.seg_length, 0.0, 1.0)
        last_pt = Gf.Vec3f((self.num_segments - 0.5) * self.seg_length, 0.0, 1.0)
        initial_points.append(first_pt)  # Duplicate first
        for i in range(self.num_segments):
            initial_points.append(Gf.Vec3f((i + 0.5) * self.seg_length, 0.0, 1.0))
        initial_points.append(last_pt)  # Duplicate last
        
        self.rod_curve.CreatePointsAttr(initial_points)
        self.rod_curve.CreateCurveVertexCountsAttr([len(initial_points)])
        
        # Tube width (radius at each point)
        widths = [self.seg_radius * 2.0] * len(initial_points)
        self.rod_curve.CreateWidthsAttr(widths)
        
        # Bind material to curve
        UsdShade.MaterialBindingAPI(self.rod_curve).Bind(self.segment_mat)
        
        # Also create small spheres at fixed end and tip for visibility
        self.fixed_sphere = UsdGeom.Sphere.Define(self.stage, "/World/Rod/FixedEnd")
        self.fixed_sphere.CreateRadiusAttr(self.seg_radius * 1.5)
        UsdShade.MaterialBindingAPI(self.fixed_sphere).Bind(self.fixed_mat)
        
        self.tip_sphere = UsdGeom.Sphere.Define(self.stage, "/World/Rod/Tip")
        self.tip_sphere.CreateRadiusAttr(self.seg_radius * 1.5)
        UsdShade.MaterialBindingAPI(self.tip_sphere).Bind(self.tip_mat)
        
        # Setup transforms for spheres
        self.fixed_xform = UsdGeom.Xformable(self.fixed_sphere.GetPrim())
        self.fixed_xform.ClearXformOpOrder()
        self.fixed_translate = self.fixed_xform.AddTranslateOp()
        
        self.tip_xform = UsdGeom.Xformable(self.tip_sphere.GetPrim())
        self.tip_xform.ClearXformOpOrder()
        self.tip_translate = self.tip_xform.AddTranslateOp()
        
        # Keep references for compatibility (empty lists since we use curve now)
        self.capsules = []
        self.translate_ops = []
        self.orient_ops = []

        # Force indicator (arrow showing applied force)
        self.force_arrow = UsdGeom.Cylinder.Define(self.stage, "/World/ForceArrow")
        self.force_arrow.CreateRadiusAttr(0.01)
        self.force_arrow.CreateHeightAttr(0.3)
        self.force_arrow.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI(self.force_arrow).Bind(self.force_mat)

        # Hide force arrow initially
        self.force_arrow.GetPrim().GetAttribute("visibility").Set("invisible")

    def _create_materials(self):
        """Create materials for visualization."""
        def make_mat(path, color):
            mat = UsdShade.Material.Define(self.stage, path)
            shader = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.3)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        self.fixed_mat = make_mat("/World/Materials/Fixed", (0.9, 0.2, 0.2))
        self.segment_mat = make_mat("/World/Materials/Segment", (0.2, 0.4, 0.9))
        self.tip_mat = make_mat("/World/Materials/Tip", (1.0, 0.8, 0.2))
        self.selected_mat = make_mat("/World/Materials/Selected", (0.2, 0.9, 0.2))
        self.force_mat = make_mat("/World/Materials/Force", (1.0, 0.3, 0.8))

    def _setup_solver(self):
        """Initialize the rod solver."""
        gravity = (0.0, 0.0, -9.81) if self.gravity_enabled else (0.0, 0.0, 0.0)

        self.config = RodConfig(
            material=RodMaterialConfig(
                young_modulus=self.stiffness,
                density=2700.0,
                damping=0.05,
            ),
            geometry=RodGeometryConfig(
                num_segments=self.num_segments,
                rest_length=self.rod_length,
                radius=self.seg_radius,
            ),
            solver=RodSolverConfig(
                dt=1.0 / 60.0,
                num_substeps=2,
                newton_iterations=4,
                use_direct_solver=True,
                gravity=gravity,
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.solver = RodSolver(self.config, num_envs=1)
        self._reset_rod()

    def _reset_rod(self):
        """Reset rod to horizontal position."""
        for i in range(self.num_segments):
            self.solver.data.positions[:, i, 0] = (i + 0.5) * self.seg_length
            self.solver.data.positions[:, i, 1] = 0.0
            self.solver.data.positions[:, i, 2] = 1.0
            self.solver.data.velocities[:, i, :] = 0.0

        # Reset orientations to identity
        self.solver.data.orientations[:, :, :] = 0.0
        self.solver.data.orientations[:, :, 3] = 1.0  # w component
        self.solver.data.angular_velocities[:, :, :] = 0.0

        self.solver.data.fix_segment(0, 0)
        self.solver.data.sync_to_warp()

        self.sim_time = 0.0
        print("Rod reset to initial position")

    def _setup_input(self):
        """Setup keyboard input handlers."""
        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard_sub = None
        
        # Try to get app window and keyboard device
        try:
            self._app_window = omni.appwindow.get_default_app_window()
            if self._app_window is not None:
                self._keyboard = self._app_window.get_keyboard()
                # Subscribe to keyboard events
                self._keyboard_sub = self._input_interface.subscribe_to_keyboard_events(
                    self._keyboard, self._on_keyboard_event
                )
                print("\n[Keyboard controls enabled]")
            else:
                print("\n[No window - keyboard controls disabled]")
                print("[Simulation will run automatically]")
        except Exception as e:
            print(f"\n[Could not setup keyboard: {e}]")
            print("[Simulation will run automatically]")

        print("\n" + "=" * 60)
        print("INTERACTIVE CONTROLS (if window available)")
        print("=" * 60)
        print("SPACE  - Pause/Resume simulation")
        print("R      - Reset rod")
        print("G      - Toggle gravity")
        print("W/S    - Apply up/down force to tip")
        print("A/D    - Apply left/right force to tip")
        print("Q/E    - Apply forward/backward force to tip")
        print("+/-    - Increase/decrease stiffness")
        print("ESC    - Exit")
        print("=" * 60 + "\n")

    def _on_keyboard_event(self, event):
        """Handle keyboard input."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input

            # Pause/Resume
            if key == carb.input.KeyboardInput.SPACE:
                self.paused = not self.paused
                print(f"Simulation {'PAUSED' if self.paused else 'RUNNING'}")

            # Reset
            elif key == carb.input.KeyboardInput.R:
                self._reset_rod()

            # Toggle gravity
            elif key == carb.input.KeyboardInput.G:
                self.gravity_enabled = not self.gravity_enabled
                gravity = (0.0, 0.0, -9.81) if self.gravity_enabled else (0.0, 0.0, 0.0)
                self.config.solver.gravity = gravity
                # Update solver gravity
                self.solver.data.wp_gravity.numpy()[:] = np.array(gravity)
                print(f"Gravity {'ON' if self.gravity_enabled else 'OFF'}")

            # Apply forces (W/S = up/down, A/D = left/right, Q/E = forward/back)
            force_magnitude = 50.0
            if key == carb.input.KeyboardInput.W:
                self.applied_force[2] = force_magnitude  # Up
            elif key == carb.input.KeyboardInput.S:
                self.applied_force[2] = -force_magnitude  # Down
            elif key == carb.input.KeyboardInput.A:
                self.applied_force[1] = -force_magnitude  # Left
            elif key == carb.input.KeyboardInput.D:
                self.applied_force[1] = force_magnitude  # Right
            elif key == carb.input.KeyboardInput.Q:
                self.applied_force[0] = -force_magnitude  # Back
            elif key == carb.input.KeyboardInput.E:
                self.applied_force[0] = force_magnitude  # Forward

            # Stiffness adjustment
            if key == carb.input.KeyboardInput.EQUAL:  # + key
                self.stiffness *= 2.0
                print(f"Stiffness: {self.stiffness:.2e} Pa")
            elif key == carb.input.KeyboardInput.MINUS:  # - key
                self.stiffness /= 2.0
                print(f"Stiffness: {self.stiffness:.2e} Pa")

            # Exit
            elif key == carb.input.KeyboardInput.ESCAPE:
                simulation_app.close()

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input
            # Release forces
            if key in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]:
                self.applied_force[2] = 0.0
            elif key in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]:
                self.applied_force[1] = 0.0
            elif key in [carb.input.KeyboardInput.Q, carb.input.KeyboardInput.E]:
                self.applied_force[0] = 0.0

        return True

    def _update_visuals(self):
        """Update rod curve and markers from solver data."""
        positions = self.solver.data.positions[0].cpu().numpy()

        # Update the smooth curve points - duplicate endpoints for Catmull-Rom
        curve_points = []
        # Duplicate first point
        curve_points.append(Gf.Vec3f(float(positions[0][0]), float(positions[0][1]), float(positions[0][2])))
        # All segment centers
        for i in range(self.num_segments):
            pos = positions[i]
            curve_points.append(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
        # Duplicate last point  
        curve_points.append(Gf.Vec3f(float(positions[-1][0]), float(positions[-1][1]), float(positions[-1][2])))
        
        self.rod_curve.GetPointsAttr().Set(curve_points)
        self.rod_curve.GetCurveVertexCountsAttr().Set([len(curve_points)])
        self.rod_curve.GetWidthsAttr().Set([self.seg_radius * 2] * len(curve_points))
        
        # Update fixed end sphere position (first segment center)
        fixed_pos = positions[0]
        self.fixed_translate.Set(Gf.Vec3d(float(fixed_pos[0]), float(fixed_pos[1]), float(fixed_pos[2])))
        
        # Update tip sphere position (last segment center)
        tip_pos = positions[-1]
        self.tip_translate.Set(Gf.Vec3d(float(tip_pos[0]), float(tip_pos[1]), float(tip_pos[2])))
        
        # Note: The spheres mark segment CENTERS, the curve passes through them
        # This creates a smooth visualization where spheres are at the endpoints

        # Update force arrow visibility and position
        if torch.norm(self.applied_force) > 0.1:
            force_dir = self.applied_force.numpy()
            force_mag = np.linalg.norm(force_dir)

            # Position arrow at tip
            xformable = UsdGeom.Xformable(self.force_arrow.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(float(tip_pos[0]), float(tip_pos[1]), float(tip_pos[2])))

            # Scale arrow by force magnitude
            self.force_arrow.CreateHeightAttr(force_mag / 50.0 * 0.5)
            self.force_arrow.GetPrim().GetAttribute("visibility").Set("inherited")
        else:
            self.force_arrow.GetPrim().GetAttribute("visibility").Set("invisible")

    def step(self):
        """Run one simulation step."""
        if self.paused:
            return

        dt = self.config.solver.dt

        # Apply external force to tip segment
        if torch.norm(self.applied_force) > 0.1:
            # Add force as velocity impulse
            force_impulse = self.applied_force * dt / (
                self.config.material.density * 
                np.pi * self.seg_radius**2 * self.seg_length
            )
            self.solver.data.velocities[0, -1, :] += force_impulse.to(self.solver.data.velocities.device)

        # Auto-perturbation: apply periodic forces to make it interesting
        if not args_cli.no_auto_perturb and self.frame % 180 == 90:  # Every 3 seconds, give a push
            push_force = 20.0
            direction = np.array([0.0, np.sin(self.sim_time), np.cos(self.sim_time * 0.5)])
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            impulse = torch.tensor(direction * push_force * dt, dtype=torch.float32)
            self.solver.data.velocities[0, -1, :] += impulse.to(self.solver.data.velocities.device)

        # Step the solver
        self.solver.step(dt)
        self.sim_time += dt

        # Update visuals
        self._update_visuals()

        self.frame += 1
        if self.frame % 60 == 0:
            tip = self.solver.data.positions[0, -1].cpu().numpy()
            status = "PAUSED" if self.paused else "RUNNING"
            gravity = "ON" if self.gravity_enabled else "OFF"
            print(f"[{status}] Time: {self.sim_time:.1f}s | Tip Z: {tip[2]:.3f}m | Gravity: {gravity} | Stiffness: {self.stiffness:.1e}")


def main():
    """Main function."""
    print("=" * 60)
    print("INTERACTIVE ROD SOLVER")
    print("Direct Position-Based Solver for Stiff Rods (Deul et al. 2018)")
    print("=" * 60)

    sim = InteractiveRodSimulation(
        num_segments=args_cli.num_segments,
        stiffness=args_cli.stiffness,
    )

    # Save USD if requested
    if args_cli.save_usd:
        stage = omni.usd.get_context().get_stage()
        stage.Export(args_cli.save_usd)
        print(f"\nUSD saved to: {args_cli.save_usd}")
        print("You can open this in Omniverse with:")
        print(f"  omniverse-launcher://open?path={args_cli.save_usd}")

    app = omni.kit.app.get_app()
    
    frame_count = 0
    while simulation_app.is_running():
        sim.step()
        app.update()
        
        # Auto-save USD every 5 seconds if save path provided
        frame_count += 1
        if args_cli.save_usd and frame_count % 300 == 0:
            stage = omni.usd.get_context().get_stage()
            stage.Export(args_cli.save_usd)

    simulation_app.close()


if __name__ == "__main__":
    main()

