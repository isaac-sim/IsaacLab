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

    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_simple.py --use-xpbd

    # Or with external Newton package (requires PR #1981 branch):
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_simple.py --use-newton-xpbd
"""

import argparse
import math

import torch

# Parse arguments before launching the app
parser = argparse.ArgumentParser(description="Simple Rod Solver Visualization")
parser.add_argument("--num-segments", type=int, default=20, help="Number of rod segments")
parser.add_argument("--stiffness", type=float, default=1e8, help="Young's modulus (Pa)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument(
    "--use-newton-xpbd",
    action="store_true",
    help="Use Newton SolverXPBDRod (requires Newton PR #1981 branch)",
)
parser.add_argument(
    "--use-xpbd",
    action="store_true",
    help="Use self-contained XPBD rod solver (no external newton dependency)",
)
parser.add_argument(
    "--newton-backend",
    type=str,
    default="block_thomas",
    help="Newton backend when --use-newton-xpbd",
)
args_cli = parser.parse_args()

# Launch the simulation app
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

# Now import remaining modules
import numpy as np
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.utils import prims as prim_utils
from isaaclab.sim.utils import stage as stage_utils

# Import the rod solver
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)


class RodVisuals:
    """Manages USD capsule prims for rod visualization with cached xform ops."""

    def __init__(
        self,
        num_segments: int,
        segment_radius: float,
        segment_length: float,
        base_path: str = "/World/Rod",
    ):
        self.prim_paths: list[str] = []
        self._translate_ops = []
        self._orient_ops = []

        prim_utils.create_prim(base_path, "Xform")

        for i in range(num_segments):
            prim_path = f"{base_path}/Segment_{i:02d}"

            if i == 0:
                color = (0.9, 0.2, 0.2)
            elif i == num_segments - 1:
                color = (1.0, 0.8, 0.2)
            else:
                t = i / (num_segments - 1)
                color = (0.2, 0.3 + 0.4 * t, 0.9 - 0.3 * t)

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
            self.prim_paths.append(prim_path)

        self._cache_xform_ops()

    def _cache_xform_ops(self):
        """Create translate + orient ops once and cache the handles."""
        stage = stage_utils.get_current_stage()
        for prim_path in self.prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            self._translate_ops.append(xformable.AddTranslateOp())
            self._orient_ops.append(xformable.AddOrientOp())

    def update(self, positions: torch.Tensor, orientations: torch.Tensor):
        """Update cached xform ops in-place (no ClearXformOpOrder per frame).

        Args:
            positions: (N, 3) segment centres.
            orientations: (N, 4) quaternions in (x, y, z, w) layout.
        """
        for i in range(len(self.prim_paths)):
            pos = positions[i].cpu().numpy()
            self._translate_ops[i].Set(
                Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
            )
            q = orientations[i].cpu().numpy()
            self._orient_ops[i].Set(
                Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            )


def main():
    """Main function."""

    # Create simulation context
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
    )
    sim = SimulationContext(sim_cfg)

    # Camera: isometric view looking at the rod centre
    sim.set_camera_view(eye=(1.0, 2.0, 1.8), target=(0.75, 0.0, 0.5))

    # Rod configuration
    num_segments = args_cli.num_segments
    rod_length = 1.5
    segment_length = rod_length / num_segments
    segment_radius = 0.04  # big enough to be clearly visible

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
            dt=sim_cfg.dt,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, 0.0, -9.81),
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Ground plane + lighting
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/DomeLight", cfg_light)

    cfg_dist_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.95, 0.9))
    cfg_dist_light.func("/World/DistantLight", cfg_dist_light, translation=(10.0, 10.0, 20.0))

    # Create visual capsules (xform ops are cached internally)
    print("Creating rod visuals...")
    rod_vis = RodVisuals(
        num_segments=num_segments,
        segment_radius=segment_radius,
        segment_length=segment_length,
    )

    initial_height = 1.0

    # Solver instantiation
    print("Initializing rod solver...")
    if args_cli.use_xpbd:
        from isaaclab_newton.solvers import XPBDRodSolver, orientations_xyzw_along_polyline

        solver = XPBDRodSolver(
            rod_config,
            num_envs=1,
            floor_z=None,
            initial_height=initial_height,
        )
        use_newton = True
    elif args_cli.use_newton_xpbd:
        from isaaclab_newton.solvers import NewtonXPBDRodSolver, orientations_xyzw_along_polyline

        solver = NewtonXPBDRodSolver(
            rod_config,
            num_envs=1,
            solver_backend=args_cli.newton_backend,
            floor_z=None,
            initial_z=initial_height,
        )
        use_newton = True
    else:
        solver = RodSolver(rod_config, num_envs=1)
        for i in range(num_segments):
            solver.data.positions[:, i, 0] = (i + 0.5) * segment_length
            solver.data.positions[:, i, 1] = 0.0
            solver.data.positions[:, i, 2] = initial_height
        solver.data.fix_segment(slice(None), 0)
        solver.data.sync_to_warp()
        use_newton = False

    # Helper to fetch solver state
    def _get_pos_ori():
        if args_cli.use_xpbd:
            p = solver.positions
            o = orientations_xyzw_along_polyline(p)
        elif use_newton:
            p = solver.positions[0]
            o = orientations_xyzw_along_polyline(p)
        else:
            p = solver.data.positions[0]
            o = solver.data.orientations[0]
        return p, o

    # Reset → set initial transforms → auto-play
    sim.reset()
    pos0, ori0 = _get_pos_ori()
    rod_vis.update(pos0, ori0)
    sim.step()  # render one frame so capsules appear
    sim.play()

    if args_cli.use_xpbd:
        backend_name = "XPBDRodSolver (self-contained)"
    elif args_cli.use_newton_xpbd:
        backend_name = "Newton SolverXPBDRod"
    else:
        backend_name = "RodSolver"

    print("=" * 60)
    print("  Rod Solver Visualization")
    print(f"  Backend:  {backend_name}")
    print(f"  Segments: {num_segments}   Radius: {segment_radius}")
    print(f"  Stiffness: {args_cli.stiffness:.2e} Pa")
    print(f"  Initial pos[0]:  {pos0[0].tolist()}")
    print(f"  Initial pos[-1]: {pos0[-1].tolist()}")
    print("=" * 60)

    sim_time = 0.0
    step_count = 0

    while simulation_app.is_running():
        if sim.is_playing():
            solver.step(dt=sim_cfg.dt)
            pos, ori = _get_pos_ori()
            rod_vis.update(pos, ori)

            sim_time += sim_cfg.dt
            step_count += 1

            if step_count % 120 == 0:
                tip_pos = pos[-1].cpu().numpy()
                print(
                    f"Time: {sim_time:.2f}s | "
                    f"Tip: ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})"
                )

        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()

