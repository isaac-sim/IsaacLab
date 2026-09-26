# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tune Newton surface reconstruction on a falling MPM water blob.

This scene intentionally uses a broad basin instead of the teapot demo's small
tabletop composition. It keeps the particle simulation identical across surface
presets so each run isolates reconstruction choices rather than material motion.

.. code-block:: bash

    uv run --extra ovrtx isaaclab example mpm-surface-reconstruction \
      --device cuda:0 --visualizer newton_rtx

    uv run isaaclab example mpm-surface-reconstruction \
      --device cuda:0 --visualizer newton_gl --surface_preset coarse_grid
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

if TYPE_CHECKING:
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer


class SurfacePreset(NamedTuple):
    """One controlled surface-reconstruction configuration."""

    title: str
    surface_voxel_size: float
    kernel_radius: float
    threshold: float
    anisotropic: bool
    field_smooth_iterations: int
    mesh_smooth_iterations: int


SURFACE_PRESETS = {
    "balanced": SurfacePreset("Balanced detail", 0.025, 0.075, 0.40, True, 0, 1),
    "coarse_grid": SurfacePreset("Coarse 50 mm grid", 0.050, 0.075, 0.40, True, 0, 1),
    "isotropic": SurfacePreset("Isotropic kernel", 0.025, 0.075, 0.40, False, 0, 1),
    "heavy_smoothing": SurfacePreset("Six smoothing passes", 0.025, 0.075, 0.40, True, 0, 6),
}

SIMULATION_HZ = 480
MPM_VOXEL_SIZE = 0.025
PARTICLES_PER_VOXEL_AXIS = 2.0
PARTICLE_SPACING = MPM_VOXEL_SIZE / PARTICLES_PER_VOXEL_AXIS
PARTICLE_RADIUS = 0.5 * PARTICLE_SPACING
PARTICLE_DENSITY = 1000.0
PARTICLE_MASS = PARTICLE_SPACING**3 * PARTICLE_DENSITY
DEFAULT_PARTICLE_JITTER_FRACTION = 0.18
DEFAULT_PARTICLE_SEED = 42

BASIN_INNER_HALF_EXTENTS = (1.65, 1.15)
BASIN_BOTTOM_TOP_Z = 0.10
BASIN_WALL_HEIGHT = 1.05
BASIN_WALL_THICKNESS = 0.10
BASIN_COLOR = (0.12, 0.17, 0.22)
BASIN_RIM_COLOR = (0.24, 0.34, 0.42)
WATER_COLOR = (0.04, 0.32, 0.82)
SURFACE_PATH = "/splash_surface"
# The 3.5 m by 2.5 m basin occupies fewer than 600,000 reconstruction
# cells at the finest preset, including its full wall height.
MAX_SURFACE_GRID_CELLS = 1_000_000

CAMERA_EYE = (3.60, -5.40, 3.30)
CAMERA_TARGET = (0.0, 0.0, 0.48)


def _positive_finite_float(value: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite value, got {value!r}")
    return resolved


def _nonnegative_int(value: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value!r}")
    return resolved


parser = argparse.ArgumentParser(description="Newton MPM falling-blob surface-reconstruction study.")
parser.add_argument(
    "--surface_preset",
    choices=tuple(SURFACE_PRESETS),
    default="balanced",
    help="Controlled reconstruction preset; explicit surface options override its fields.",
)
parser.add_argument(
    "--fluid_render_mode",
    choices=("particles", "surface", "both"),
    default="surface",
    help="Render the source particles, reconstructed surface, or both.",
)
parser.add_argument("--surface_voxel_size", type=_positive_finite_float, default=None, help="Surface voxel size [m].")
parser.add_argument("--surface_kernel_radius", type=_positive_finite_float, default=None, help="Kernel radius [m].")
parser.add_argument(
    "--surface_threshold",
    type=_positive_finite_float,
    default=None,
    help="Density isosurface threshold.",
)
parser.add_argument(
    "--surface_anisotropic",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable anisotropic particle kernels.",
)
parser.add_argument(
    "--surface_field_smooth_iterations",
    type=_nonnegative_int,
    default=None,
    help="Scalar-field smoothing passes.",
)
parser.add_argument(
    "--surface_mesh_smooth_iterations",
    type=_nonnegative_int,
    default=None,
    help="Extracted-mesh smoothing passes.",
)
parser.add_argument(
    "--surface_opacity",
    type=float,
    default=0.65,
    help="Reconstructed water opacity in [0, 1]. Defaults to 0.65.",
)
parser.add_argument(
    "--rtx_quality",
    type=_nonnegative_int,
    default=16,
    help="Newton RTX convergence quality. Use 0 for interactive one-sample rendering.",
)
parser.add_argument(
    "--particle_jitter_fraction",
    type=float,
    default=DEFAULT_PARTICLE_JITTER_FRACTION,
    help="Maximum per-axis particle jitter as a fraction of particle spacing in [0, 0.5].",
)
parser.add_argument(
    "--particle_seed",
    type=int,
    default=DEFAULT_PARTICLE_SEED,
    help=f"Seed for deterministic particle jitter. Defaults to {DEFAULT_PARTICLE_SEED}.",
)
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable CUDA graph capture for debugging.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_rtx"])
args_cli = parser.parse_args()

if not 0.0 <= args_cli.surface_opacity <= 1.0:
    parser.error("--surface_opacity must be in [0, 1].")
if not 0.0 <= args_cli.particle_jitter_fraction <= 0.5:
    parser.error("--particle_jitter_fraction must be in [0, 0.5].")
if args_cli.fluid_render_mode != "particles":
    requested_visualizers = set(args_cli.visualizer or [])
    if "kit" in requested_visualizers:
        parser.error("Surface reconstruction is available in Newton GL and Newton RTX, not Kit.")
    if not {"newton", "newton_gl", "newton_rtx"}.intersection(requested_visualizers):
        parser.error("Surface reconstruction requires the Newton GL or Newton RTX visualizer.")

_preset = SURFACE_PRESETS[args_cli.surface_preset]
SURFACE_CFG = SurfacePreset(
    title=_preset.title,
    surface_voxel_size=args_cli.surface_voxel_size or _preset.surface_voxel_size,
    kernel_radius=args_cli.surface_kernel_radius or _preset.kernel_radius,
    threshold=args_cli.surface_threshold or _preset.threshold,
    anisotropic=_preset.anisotropic if args_cli.surface_anisotropic is None else args_cli.surface_anisotropic,
    field_smooth_iterations=(
        _preset.field_smooth_iterations
        if args_cli.surface_field_smooth_iterations is None
        else args_cli.surface_field_smooth_iterations
    ),
    mesh_smooth_iterations=(
        _preset.mesh_smooth_iterations
        if args_cli.surface_mesh_smooth_iterations is None
        else args_cli.surface_mesh_smooth_iterations
    ),
)
SHOW_PARTICLES = args_cli.fluid_render_mode in ("particles", "both")
SHOW_SURFACE = args_cli.fluid_render_mode in ("surface", "both")


def create_fluid_points() -> tuple[np.ndarray, np.ndarray, int]:
    """Create a shallow pool and an offset airborne sphere with deterministic spacing."""

    def box_points(lower: tuple[float, float, float], upper: tuple[float, float, float]) -> np.ndarray:
        axes = [np.arange(lo + 0.5 * PARTICLE_SPACING, hi, PARTICLE_SPACING) for lo, hi in zip(lower, upper)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    pool = box_points((-1.04, -0.54, 0.14), (1.04, 0.54, 0.32))
    blob_radius = 0.26
    blob_center = np.array((-0.34, 0.02, 0.95), dtype=np.float32)
    blob_local = box_points((-blob_radius,) * 3, (blob_radius,) * 3)
    blob_local = blob_local[np.linalg.norm(blob_local, axis=1) <= blob_radius - 0.2 * PARTICLE_SPACING]
    blob = blob_local + blob_center

    points = np.concatenate((pool, blob)).astype(np.float32, copy=False)
    jitter = args_cli.particle_jitter_fraction * PARTICLE_SPACING
    points += np.random.default_rng(args_cli.particle_seed).uniform(-jitter, jitter, points.shape)
    velocities = np.zeros_like(points)
    return points, velocities, blob.shape[0]


def create_visualizer_cfgs():
    """Create a visualizer for the selected renderer."""
    common = {
        "eye": CAMERA_EYE,
        "lookat": CAMERA_TARGET,
        "focal_length": 30.0,
        "show_particles": SHOW_PARTICLES,
        "particle_color": WATER_COLOR,
        "enable_live_plots": False,
    }
    cfgs = []
    if "kit" in (args_cli.visualizer or []):
        from isaaclab_visualizers.kit import KitVisualizerCfg

        cfgs.append(KitVisualizerCfg(eye=CAMERA_EYE, lookat=CAMERA_TARGET, focal_length=30.0))
    if not {"newton", "newton_gl", "newton_rtx"}.intersection(args_cli.visualizer or []):
        return cfgs

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    for name in args_cli.visualizer or []:
        if name in ("newton", "newton_gl"):
            cfgs.append(NewtonGLVisualizerCfg(streaming_view=False, background_color=None, **common))
        elif name == "newton_rtx":
            cfgs.append(
                NewtonRTXVisualizerCfg(
                    rtx_environment="studio",
                    render_settings={
                        "omni:rtx:quality": ("Int", args_cli.rtx_quality),
                        "omni:rtx:rt:reflections:enabled": ("Bool", True),
                        "omni:rtx:rt:reflections:maxBounces": ("Int", 2),
                    },
                    **common,
                )
            )
    return cfgs


def create_sim_cfg():
    """Create the sparse implicit-MPM simulation configuration."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=1.0 / SIMULATION_HZ,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=MPM_VOXEL_SIZE,
                grid_type="sparse",
                grid_padding=0,
                max_active_cell_count=1 << 18,
                max_leaf_node_count=1 << 16,
                max_lower_node_count=1 << 12,
                max_upper_node_count=1 << 9,
                max_iterations=100,
                tolerance=1.0e-4,
                collider_basis="S2",
                strain_basis="P0",
                transfer_scheme="apic",
                integration_scheme="pic",
                air_drag=1.0e-3,
            ),
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )


def create_scene_cfg(points: np.ndarray, velocities: np.ndarray):
    """Create the basin, shallow pool, and airborne MPM blob."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    def basin_part(
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        color: tuple[float, float, float],
        *,
        visible: bool = True,
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=f"/World/Basin/{name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=0.5 * MPM_VOXEL_SIZE),
                ],
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.18,
                    dynamic_friction=0.18,
                ),
                visible=visible,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color,
                        roughness=0.18,
                        metallic=0.25,
                    )
                    if visible
                    else None
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        )

    def basin_visual(
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=f"/World/Basin/{name}",
            spawn=sim_utils.MeshCuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    roughness=0.18,
                    metallic=0.25,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        )

    outer_x = BASIN_INNER_HALF_EXTENTS[0] + BASIN_WALL_THICKNESS
    outer_y = BASIN_INNER_HALF_EXTENTS[1] + BASIN_WALL_THICKNESS
    half_wall_z = 0.5 * BASIN_WALL_HEIGHT

    @configclass
    class SurfaceSplashSceneCfg(InteractiveSceneCfg):
        """One large catch basin containing a pool and falling water blob."""

        basin_bottom_collider = basin_part(
            "BottomCollider",
            (2.0 * outer_x, 2.0 * outer_y, 2.0 * BASIN_BOTTOM_TOP_Z),
            (0.0, 0.0, 0.0),
            BASIN_COLOR,
            visible=False,
        )
        basin_bottom_visual = basin_visual(
            "BottomVisual",
            (2.0 * outer_x, 2.0 * outer_y, 2.0 * BASIN_BOTTOM_TOP_Z),
            (0.0, 0.0, 0.0),
            BASIN_COLOR,
        )
        basin_left_collider = basin_part(
            "LeftWallCollider",
            (BASIN_WALL_THICKNESS, 2.0 * outer_y, BASIN_WALL_HEIGHT),
            (-outer_x + 0.5 * BASIN_WALL_THICKNESS, 0.0, half_wall_z),
            BASIN_RIM_COLOR,
            visible=False,
        )
        basin_left_visual = basin_visual(
            "LeftWallVisual",
            (BASIN_WALL_THICKNESS, 2.0 * outer_y, BASIN_WALL_HEIGHT),
            (-outer_x + 0.5 * BASIN_WALL_THICKNESS, 0.0, half_wall_z),
            BASIN_RIM_COLOR,
        )
        basin_right_collider = basin_part(
            "RightWallCollider",
            (BASIN_WALL_THICKNESS, 2.0 * outer_y, BASIN_WALL_HEIGHT),
            (outer_x - 0.5 * BASIN_WALL_THICKNESS, 0.0, half_wall_z),
            BASIN_RIM_COLOR,
            visible=False,
        )
        basin_right_visual = basin_visual(
            "RightWallVisual",
            (BASIN_WALL_THICKNESS, 2.0 * outer_y, BASIN_WALL_HEIGHT),
            (outer_x - 0.5 * BASIN_WALL_THICKNESS, 0.0, half_wall_z),
            BASIN_RIM_COLOR,
        )
        basin_back_collider = basin_part(
            "BackWallCollider",
            (2.0 * outer_x, BASIN_WALL_THICKNESS, BASIN_WALL_HEIGHT),
            (0.0, outer_y - 0.5 * BASIN_WALL_THICKNESS, half_wall_z),
            BASIN_RIM_COLOR,
            visible=False,
        )
        basin_back_visual = basin_visual(
            "BackWallVisual",
            (2.0 * outer_x, BASIN_WALL_THICKNESS, BASIN_WALL_HEIGHT),
            (0.0, outer_y - 0.5 * BASIN_WALL_THICKNESS, half_wall_z),
            BASIN_RIM_COLOR,
        )
        basin_front_collider = basin_part(
            "FrontWallCollider",
            (2.0 * outer_x, BASIN_WALL_THICKNESS, BASIN_WALL_HEIGHT),
            (0.0, -outer_y + 0.5 * BASIN_WALL_THICKNESS, half_wall_z),
            BASIN_RIM_COLOR,
            visible=False,
        )
        basin_front_rim = basin_visual(
            "FrontRim",
            (2.0 * outer_x, BASIN_WALL_THICKNESS, 0.34),
            (0.0, -outer_y + 0.5 * BASIN_WALL_THICKNESS, 0.17),
            BASIN_RIM_COLOR,
        )

        fluid = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fluid",
            spawn=MPMPointsCfg(
                positions=points.tolist(),
                velocities=velocities.tolist(),
                mass=PARTICLE_MASS,
                radius=PARTICLE_RADIUS,
                material=MPMParticleMaterialCfg(
                    density=PARTICLE_DENSITY,
                    young_modulus=1.0e7,
                    poisson_ratio=0.3,
                    viscosity=5.0e-2,
                    friction=0.0,
                    damping=1.0e-3,
                    yield_pressure=1.0e15,
                    tensile_yield_ratio=1.0,
                ),
                visual_color=WATER_COLOR,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=WATER_COLOR,
                    roughness=0.08,
                    opacity=args_cli.surface_opacity,
                ),
            ),
        )

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.11)),
        )

    return SurfaceSplashSceneCfg(num_envs=1, env_spacing=0.0)


def create_surface_renderer(sim) -> ParticleSurfaceRenderer:
    """Configure splash-specific surface extraction after Newton initializes."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer
    from newton.geometry import ParticleSurface

    model = NewtonManager.get_model()
    surface = ParticleSurface(
        voxel_size=SURFACE_CFG.surface_voxel_size,
        max_grid_cells=MAX_SURFACE_GRID_CELLS,
        world_count=max(model.world_count, 1),
        kernel_radius=SURFACE_CFG.kernel_radius,
        threshold=SURFACE_CFG.threshold,
        smooth_lambda=0.0,
        anisotropic=SURFACE_CFG.anisotropic,
        kernel_scale=0.5,
        anisotropy_ratio=16.0,
        anisotropy_scale=1.0,
        anisotropy_min_neighbors=4,
        anisotropy_binning=True,
        anisotropy_strength=0.95,
        field_smooth_iterations=SURFACE_CFG.field_smooth_iterations,
        mesh_smooth_iterations=SURFACE_CFG.mesh_smooth_iterations,
        device=model.device,
    )
    return ParticleSurfaceRenderer(
        sim.visualizers,
        surface,
        path=SURFACE_PATH,
        color=WATER_COLOR,
        opacity=args_cli.surface_opacity,
        roughness=0.06,
        use_cuda_graph=not args_cli.disable_cuda_graph,
    )


def fluid_metrics(scene, triangle_count: int) -> dict[str, float | int]:
    """Return finite motion and geometry diagnostics for the encoded frame."""
    points = scene["fluid"].data.particle_pos_w.torch[0]
    lower = points.amin(dim=0).detach().cpu().numpy()
    upper = points.amax(dim=0).detach().cpu().numpy()
    center = points.mean(dim=0).detach().cpu().numpy()
    return {
        "surface_triangle_count": triangle_count,
        "fluid_com_x_m": float(center[0]),
        "fluid_com_y_m": float(center[1]),
        "fluid_com_z_m": float(center[2]),
        "fluid_aabb_x_m": float(upper[0] - lower[0]),
        "fluid_aabb_y_m": float(upper[1] - lower[1]),
        "fluid_aabb_z_m": float(upper[2] - lower[2]),
    }


def run_simulator(sim, scene, surface_renderer: ParticleSurfaceRenderer | None) -> dict[str, float | int]:
    """Advance the splash and return its final geometry diagnostics."""
    sim_dt = sim.get_physics_dt()
    count = 0
    triangle_count = surface_renderer.update() if surface_renderer is not None else 0

    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or count < args_cli.max_steps):
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering and count % 2 == 0:
            if surface_renderer is not None:
                triangle_count = surface_renderer.update()
            sim.render()
        count += 1
    return fluid_metrics(scene, triangle_count)


def main() -> None:
    """Launch the surface-reconstruction study."""
    points, velocities, blob_particle_count = create_fluid_points()
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg(points, velocities))
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
        surface_renderer = create_surface_renderer(sim) if SHOW_SURFACE else None
        print(
            f"[INFO]: Surface splash ready: {points.shape[0]} particles ({blob_particle_count} airborne), "
            f"{args_cli.fluid_render_mode} mode, {args_cli.surface_preset} preset.",
            flush=True,
        )
        if SHOW_SURFACE:
            print(f"[INFO]: Surface settings: {SURFACE_CFG._asdict()}.", flush=True)
        final_metrics = run_simulator(sim, scene, surface_renderer)
        print(
            "[INFO]: Surface splash final bounds: "
            f"({float(final_metrics['fluid_aabb_x_m']):.3f}, "
            f"{float(final_metrics['fluid_aabb_y_m']):.3f}, "
            f"{float(final_metrics['fluid_aabb_z_m']):.3f}) m; "
            f"triangles={int(final_metrics['surface_triangle_count'])}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
