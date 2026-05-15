#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Isaac Lab Visualization: Rod Solver with Fluoroscopy DRR Backdrop

Runs the direct rod solver (Deul et al. 2018) inside Isaac Lab and renders a
DRR fluoroscopy image (from DiffDRR+Slang) as a textured backdrop plane in the
Omniverse viewport.  The catheter/guidewire is visualised as metallic capsule
segments overlaid on the fluoroscopy image.

DRR images are loaded from:
    /home/cdinea/Downloads/new_CTdata/drr_output_diffdrr_slang/

Usage:
    # Run with Isaac Lab's Python wrapper
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_fluoro_isaaclab.py

    # With options
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_fluoro_isaaclab.py \\
        --num-segments 30 --stiffness 1e8 --view AP

    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_fluoro_isaaclab.py --use-xpbd

    # Or with external Newton package (requires PR #1981 branch):
    ./isaaclab.sh -p source/isaaclab_newton/examples/visualize_rod_fluoro_isaaclab.py --use-newton-xpbd
"""

import argparse
import math
import os

import torch

from isaacsim import SimulationApp

# ── CLI ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Rod Solver + Fluoroscopy in Isaac Lab")
parser.add_argument("--num-segments", type=int, default=20, help="Number of rod segments")
parser.add_argument("--stiffness", type=float, default=1e8, help="Young's modulus (Pa)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument(
    "--view",
    type=str,
    default="AP",
    choices=["AP", "LAO_30", "Lateral", "RAO_30"],
    help="Which DRR view to show as backdrop",
)
parser.add_argument(
    "--drr-dir",
    type=str,
    default="/home/cdinea/Downloads/new_CTdata/drr_output_diffdrr_slang",
    help="Directory containing DiffDRR output images",
)
parser.add_argument(
    "--use-newton-xpbd",
    action="store_true",
    help="Use Newton SolverXPBDRod (requires Newton PR #1981)",
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
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Sim duration in seconds; 0 = run forever (default)",
)
parser.add_argument(
    "--screenshot-dir",
    type=str,
    default="",
    help="Save viewport screenshots to this directory (headless)",
)
args_cli = parser.parse_args()

# Launch Omniverse
simulation_app = SimulationApp({"headless": args_cli.headless})

# ── Post-app imports ────────────────────────────────────────────────────
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)

# ── DRR view map ────────────────────────────────────────────────────────
DRR_VIEWS = {
    "AP":      "xray_diffdrr_slang_AP.png",
    "LAO_30":  "xray_diffdrr_slang_LAO_30.png",
    "Lateral": "xray_diffdrr_slang_Lateral.png",
    "RAO_30":  "xray_diffdrr_slang_RAO_30.png",
}


# ═══════════════════════════════════════════════════════════════════════
# Scene construction
# ═══════════════════════════════════════════════════════════════════════

def create_fluoroscopy_backdrop(stage, drr_path: str, size: float = 2.0):
    """Create a vertical textured quad showing the DRR fluoroscopy image.

    The quad is placed at Y = -0.5 (behind the rod) facing +Y so the
    camera looking from +Y sees it as a backdrop.

    Args:
        stage: USD stage.
        drr_path: Absolute path to the DRR PNG file.
        size: Half-extent of the quad in metres.
    """
    half = size / 2.0

    # ── Quad geometry (vertical, in XZ plane) ─────────────────────────
    quad = UsdGeom.Mesh.Define(stage, "/World/Fluoroscopy/Backdrop")
    quad.CreatePointsAttr([
        Gf.Vec3f(-half, -0.5, 0.0),
        Gf.Vec3f( half, -0.5, 0.0),
        Gf.Vec3f( half, -0.5, size),
        Gf.Vec3f(-half, -0.5, size),
    ])
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    quad.CreateNormalsAttr([Gf.Vec3f(0, 1, 0)] * 4)

    # UV coordinates to map the full texture
    st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ])

    # ── Textured material ─────────────────────────────────────────────
    mat = UsdShade.Material.Define(stage, "/World/Fluoroscopy/FluoroMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Fluoroscopy/FluoroMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    # Emissive so the fluoroscopy image is visible without scene lighting
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))

    # Texture reader for the DRR image
    tex_reader = UsdShade.Shader.Define(stage, "/World/Fluoroscopy/FluoroMaterial/DRRTexture")
    tex_reader.CreateIdAttr("UsdUVTexture")
    tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(drr_path)
    tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # ST coordinate reader
    st_reader = UsdShade.Shader.Define(stage, "/World/Fluoroscopy/FluoroMaterial/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # Wire: ST reader → texture → shader
    tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_reader.ConnectableAPI(), "rgb"
    )
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Bind material to quad
    UsdShade.MaterialBindingAPI(quad).Bind(mat)

    print(f"  Fluoroscopy backdrop: {drr_path}")
    print(f"  Quad size: {size:.1f} m  (placed at Y = -0.5)")


def design_scene(num_segments: int, segment_radius: float, segment_length: float, drr_path: str):
    """Create ground, lighting, fluoroscopy backdrop, and rod markers."""
    stage = omni.usd.get_context().get_stage()

    # Ground plane disabled — the catheter droops below Z=0 and would be occluded
    # cfg_ground = sim_utils.GroundPlaneCfg()
    # cfg_ground.func("/World/GroundPlane", cfg_ground)

    # ── Lighting (dim, to emphasise fluoroscopy backdrop) ─────────────
    cfg_light = sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.95, 1.0))
    cfg_light.func("/World/DomeLight", cfg_light)

    cfg_dist_light = sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.95, 0.9))
    cfg_dist_light.func("/World/DistantLight", cfg_dist_light, translation=(5.0, 5.0, 10.0))

    # ── Fluoroscopy DRR backdrop ──────────────────────────────────────
    rod_span = num_segments * segment_length
    backdrop_size = max(rod_span * 1.5, 2.0)
    create_fluoroscopy_backdrop(stage, drr_path, size=backdrop_size)

    # ── Rod markers (radio-opaque catheter look) ──────────────────────
    markers_cfg = VisualizationMarkersCfg(
        prim_path="/World/RodMarkers",
        markers={
            # Fixed (proximal hub) — dark steel
            "fixed_segment": sim_utils.CapsuleCfg(
                radius=segment_radius * 1.2,
                height=segment_length,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.25, 0.30),
                    metallic=0.9,
                    roughness=0.3,
                ),
            ),
            # Regular segment — bright metallic (Nitinol)
            "segment": sim_utils.CapsuleCfg(
                radius=segment_radius,
                height=segment_length,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.82, 0.82, 0.85),
                    metallic=0.95,
                    roughness=0.15,
                ),
            ),
            # Tip (distal marker band) — gold
            "tip_segment": sim_utils.CapsuleCfg(
                radius=segment_radius * 1.1,
                height=segment_length,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.3),
                    metallic=0.85,
                    roughness=0.2,
                ),
            ),
        },
    )
    rod_markers = VisualizationMarkers(markers_cfg)
    return rod_markers


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def rod_orientations_to_isaac_quats(orientations: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, z, w) → (w, x, y, z) quaternion layout for Isaac Lab."""
    q = torch.zeros_like(orientations)
    q[:, 0] = orientations[:, 3]  # w
    q[:, 1] = orientations[:, 0]  # x
    q[:, 2] = orientations[:, 1]  # y
    q[:, 3] = orientations[:, 2]  # z
    return q


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Resolve DRR path ──────────────────────────────────────────────
    drr_filename = DRR_VIEWS.get(args_cli.view, DRR_VIEWS["AP"])
    drr_path = os.path.join(args_cli.drr_dir, drr_filename)
    if not os.path.isfile(drr_path):
        print(f"WARNING: DRR image not found at {drr_path}")
        print("         The backdrop will appear blank.")

    # ── Simulation context ────────────────────────────────────────────
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

    rod_length = 1.5
    # Camera: front view looking along +Y toward the backdrop (Y=-0.5).
    # Place camera at Y=4 and look at the rod center (X=0.75, Z=0.5).
    sim.set_camera_view(
        eye=(rod_length / 2.0, 4.0, 0.8),
        target=(rod_length / 2.0, -0.5, 0.5),
    )

    # ── Rod solver ────────────────────────────────────────────────────
    num_segments = args_cli.num_segments
    rod_config = RodConfig(
        material=RodMaterialConfig(
            young_modulus=args_cli.stiffness,
            density=6450.0,  # Nitinol
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=rod_length,
            radius=0.03,  # visible catheter
        ),
        solver=RodSolverConfig(
            dt=sim_cfg.dt,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, 0.0, -9.81),  # Z-up
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    segment_length = rod_config.geometry.segment_length
    segment_radius = rod_config.geometry.radius

    # ── Scene ─────────────────────────────────────────────────────────
    rod_markers = design_scene(num_segments, segment_radius, segment_length, drr_path)

    initial_height = 0.8
    if args_cli.use_xpbd:
        if args_cli.num_envs != 1:
            raise SystemExit("--use-xpbd requires --num-envs 1.")
        from isaaclab_newton.solvers import XPBDRodSolver, orientations_xyzw_along_polyline

        solver = XPBDRodSolver(
            rod_config,
            floor_z=None,
            initial_height=initial_height,
        )
        use_newton = False
        use_xpbd_solver = True
    elif args_cli.use_newton_xpbd:
        if args_cli.num_envs != 1:
            raise SystemExit("--use-newton-xpbd requires --num-envs 1.")
        from isaaclab_newton.solvers import NewtonXPBDRodSolver, orientations_xyzw_along_polyline

        solver = NewtonXPBDRodSolver(
            rod_config,
            num_envs=1,
            solver_backend=args_cli.newton_backend,
            floor_z=None,
            initial_z=initial_height,
        )
        use_newton = True
        use_xpbd_solver = False
    else:
        solver = RodSolver(rod_config, num_envs=args_cli.num_envs)
        for i in range(num_segments):
            solver.data.positions[:, i, 0] = (i + 0.5) * segment_length
            solver.data.positions[:, i, 1] = 0.0
            solver.data.positions[:, i, 2] = initial_height
        solver.data.fix_segment(slice(None), 0)
        solver.data.sync_to_warp()
        use_newton = False
        use_xpbd_solver = False

    # ── Start simulation ──────────────────────────────────────────────
    sim.play()

    print("=" * 60)
    print("  ROD SOLVER + FLUOROSCOPY BACKDROP — Isaac Lab")
    print("=" * 60)
    if args_cli.use_xpbd:
        backend_name = "XPBDRodSolver (self-contained)"
    elif args_cli.use_newton_xpbd:
        backend_name = "Newton SolverXPBDRod"
    else:
        backend_name = "RodSolver"
    print(f"  Backend:      {backend_name}")
    print(f"  DRR view:     {args_cli.view} ({drr_filename})")
    print(f"  DRR path:     {drr_path}")
    print(f"  Segments:     {num_segments}")
    print(f"  Stiffness:    {args_cli.stiffness:.2e} Pa")
    print(f"  Seg length:   {segment_length:.4f} m")
    print(f"  Seg radius:   {segment_radius:.4f} m")
    print("=" * 60)

    # Marker type indices: 0=fixed, 1=regular, 2=tip
    marker_indices = torch.ones(num_segments, dtype=torch.int32)
    marker_indices[0] = 0
    marker_indices[-1] = 2

    sim_time = 0.0
    step_count = 0
    print(f"  sim.is_playing() = {sim.is_playing()}")
    print(f"  marker prim path = {rod_markers.prim_path}")
    print(f"  num markers = {rod_markers.num_prototypes}")

    while simulation_app.is_running():
        if sim.is_playing():
            solver.step(dt=sim_cfg.dt)

            if use_xpbd_solver:
                positions = solver.positions.cpu()
                orientations = orientations_xyzw_along_polyline(solver.positions).cpu()
            elif use_newton:
                positions = solver.positions[0].cpu()
                orientations = orientations_xyzw_along_polyline(solver.positions[0]).cpu()
            else:
                positions = solver.data.positions[0].cpu()
                orientations = solver.data.orientations[0].cpu()
            isaac_quats = rod_orientations_to_isaac_quats(orientations)

            rod_markers.visualize(
                translations=positions,
                orientations=isaac_quats,
                marker_indices=marker_indices,
            )

            sim_time += sim_cfg.dt
            step_count += 1

            if step_count == 1:
                print(f"  [DEBUG] First frame positions shape: {positions.shape}")
                print(f"  [DEBUG] First frame orientations shape: {isaac_quats.shape}")
                print(f"  [DEBUG] pos[0] = {positions[0].tolist()}")
                print(f"  [DEBUG] pos[-1] = {positions[-1].tolist()}")
                print(f"  [DEBUG] quat[0] = {isaac_quats[0].tolist()}")
                has_nan = torch.isnan(positions).any() or torch.isnan(isaac_quats).any()
                print(f"  [DEBUG] any NaN: {has_nan}")

            if step_count % 120 == 0:
                tip = positions[-1]
                print(
                    f"t={sim_time:.2f}s  tip=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})"
                )

        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()


