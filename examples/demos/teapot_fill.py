# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fill and pour water from the Utah teapot with Newton implicit MPM.

The teapot is a hollow, double-walled shell, so the fluid is seeded in its enclosed air cavity with
:func:`~isaaclab.utils.warp.sample_particles_in_cavity` instead of using a winding-number volume fill.

.. code-block:: bash

    # Fast Newton visualizer (the default):
    uvx isaaclab demo teapot-fill --device cuda:0 --visualizer newton_gl
    # Display both the raw particles and reconstructed surface:
    uvx isaaclab demo teapot-fill --visualizer newton_gl --fluid_render_mode both
    # Newton RTX path-traced visualizer:
    uvx --from 'isaaclab[ovrtx]' isaaclab demo teapot-fill --device cuda:0 --visualizer newton_rtx
    # Isaac Sim Kit visualizer (particles only):
    uvx --from 'isaaclab[isaacsim]' isaaclab demo teapot-fill --device cuda:0 --visualizer kit
    # Fuller / coarser (faster) fill:
    uvx isaaclab demo teapot-fill --fill_level 1.0 --fill_spacing 0.003
    # Rizon4s with the Sharpa hand and the existing licensed RJ45 asset bundle:
    uv run isaaclab demo teapot-fill --robot rizon_sharpa
    # Use an index finger hook through the same handle:
    uv run isaaclab demo teapot-fill --robot rizon_sharpa --grasp_finger index
    # Record the full pickup and pour with particles and a smooth camera move:
    uv run --extra ovrtx --with imageio-ffmpeg isaaclab demo teapot-fill \\
        --robot rizon_sharpa --visualizer newton_rtx --fluid_render_mode particles \\
        --video logs/teapot/rizon_sharpa_teapot_pickup_particles.mp4 --motion_report logs/teapot/motion-pickup.json
    # Compare with --robot none --sequence pickup_pour, keeping resolution and steps identical:
    uv run isaaclab demo teapot-fill --robot rizon_sharpa --visualizer none \\
        --sequence pickup_pour --benchmark logs/teapot/robot.json

The robot approaches the tabletop teapot with a cleared hook-shaped hand and closes
against its handle. MJWarp resolves hand--pot and hand self-collisions on convex geometry; MPM
uses the exact hollow shell with the measured pot motion synchronized at 800 Hz.
This default one-way boundary avoids the cost of solving a dynamic MPM proxy;
liquid forces are not returned to the robot. ``--fluid_coupling two_way`` enables
proxy coupling that exchanges liquid reaction forces. One finger passes through
the original handle opening, opposed by the thumb on its upper connection.
``--grasp_finger`` selects the default middle finger or the index finger; each
uses its own calibrated palm pose and smooth closure timing. The dynamic pot is held entirely by physical
contacts; no weld or attachment constraint is created.
The optional supported-pot proxy uses the coupled Franka pouring task's mass scale of 1000;
this scales its mass and inertia inside MPM. MJWarp retains the configured physical
pot mass and proportionally scaled inertia.
Robot playback uses ``pickup_pour`` to establish contacts before lifting.
The pedestal height increases with the requested rise to preserve arm reach.
The pickup trajectory lowers the spout over the bowl as it tilts, rises 70 cm
over six seconds while pouring, then smoothly recovers. The spout retreats
as it rises to compensate for the longer falling stream. The robot variant applies
a native particle collision correction against the thin teapot and bowl shells.
``--motion_report`` also records escaped particle identities at
10 Hz and checks containment after a complete sequence.
Startup IK and a cubic spline supply the approach and finger closure. After grasp,
a 100 Hz object-pose controller uses public articulation data and bounded differential
IK to drive the arm within joint position, speed, and acceleration limits. Hand
contacts remain active; the palm may pivot naturally around the hooked finger.
The grasp and spout trajectory are calibrated to the default Utah teapot; custom
containers need a new palm offset, finger pose, and outlet position.

The Rizon–Sharpa USD and textures are not redistributed. Use the pinned asset cache
shared with the coffee and RJ45 demos, or set ``ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT``
to the complete licensed bundle. See ``rizon_sharpa_teapot.py`` for its hash and resolver.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

if TYPE_CHECKING:
    from rizon_sharpa_teapot import TeapotPourMotion
    from teapot_presentation import PourCameraTracker

    from isaaclab.assets import RigidObject
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

logger = logging.getLogger(__name__)

DEFAULT_VOXEL_SIZE = 0.003
DEFAULT_PARTICLES_PER_VOXEL_AXIS = 2.0
DEFAULT_FILL_LEVEL = 0.70
DEFAULT_MIN_RAY_HITS = 5
DEFAULT_ISLAND_USD = (
    f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Counters/Island_A01/sm_fixture_island_a01_01.usd"
)
DEFAULT_BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Dishware/Bowl_G01/sm_kitchenware_bowl_g01_01.usd"


def _positive_finite_float(value: str) -> float:
    """Parse a positive finite floating-point argument."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite value, got {value!r}")
    return resolved


def _unit_interval_float(value: str) -> float:
    """Parse a finite floating-point argument in the closed unit interval."""
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a value in [0, 1], got {value!r}")
    return resolved


parser = argparse.ArgumentParser(description="Newton implicit MPM teapot-fill demo.")
parser.add_argument(
    "--robot", choices=["none", "rizon_sharpa"], default="none", help="Pour with the Rizon–Sharpa hand."
)
parser.add_argument(
    "--grasp_finger",
    choices=["middle", "index"],
    default="middle",
    help="Finger inserted through the robot's teapot handle.",
)
parser.add_argument(
    "--fluid_coupling",
    choices=["one_way", "two_way"],
    default="one_way",
    help="Use measured teapot motion in MPM, or experimental two-way fluid reactions through a dynamic proxy.",
)
parser.add_argument(
    "--sequence",
    choices=["pour", "pickup_pour"],
    default=None,
    help="Motion sequence; defaults to pickup_pour with the robot and pour without it.",
)
parser.add_argument(
    "--video", type=str, default=None, help="Record the Newton viewer to an MP4 (requires imageio-ffmpeg)."
)
parser.add_argument("--video_fps", type=int, default=60, help="Recorded frames per simulated second (1–120).")
parser.add_argument(
    "--benchmark", type=str, default=None, help="Write steady-state timing to a JSON file; use --visualizer none."
)
parser.add_argument("--benchmark_warmup", type=int, default=400, help="Exclude this many initial steps from timing.")
parser.add_argument(
    "--motion_report",
    type=str,
    default=None,
    help="Validate measured grasp, tracking and fluid containment; write metrics to JSON.",
)
parser.add_argument(
    "--pour_rise_height", type=_positive_finite_float, default=0.70, help="Pickup/pour vertical rise [m]."
)
parser.add_argument("--pour_rise_time", type=_positive_finite_float, default=6.0, help="Pickup/pour rise duration [s].")
parser.add_argument(
    "--teapot_mass",
    type=_positive_finite_float,
    default=0.35,
    help="Robot-held teapot mass [kg]; authored inertia scales consistently with the mass.",
)
parser.add_argument(
    "--pour_angle",
    type=_positive_finite_float,
    default=None,
    help="Maximum tilt [degrees, up to 90]; defaults to 50 for pickup sequences and 65 otherwise.",
)
parser.add_argument(
    "--pour_tilt_time", type=_positive_finite_float, default=None, help="Time to reach the pouring tilt [s]."
)
parser.add_argument(
    "--pour_upper_angle",
    type=_positive_finite_float,
    default=None,
    help="Ease to this tilt early in the rise [degrees]; defaults to retaining the maximum tilt.",
)
parser.add_argument(
    "--pour_aim_offset_x",
    type=float,
    default=0.0,
    help="Adjust the spout's pouring target along world X [m]; negative aims upstream.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many simulation steps; defaults to the complete sequence; negative runs forever.",
)
parser.add_argument(
    "--voxel_size",
    type=_positive_finite_float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size in meters. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument(
    "--grid_type",
    type=str,
    default="sparse",
    choices=["fixed", "sparse"],
    help="MPM grid topology. Sparse uses a bounded rebuildable grid over active voxels and is "
    "CUDA-graph-capturable (fastest); fixed pre-allocates a frozen padded grid.",
)
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graph capture.")
parser.add_argument("--physics_hz", type=int, default=800, help="Outer simulation frequency [Hz].")
parser.add_argument("--physics_substeps", type=int, default=1, help="Fluid/rigid coupled substeps per outer step.")
parser.add_argument("--rigid_substeps", type=int, default=1, help="Robot contact substeps per coupled substep.")
parser.add_argument(
    "--controller_hz", type=_positive_finite_float, default=100.0, help="Robot controller frequency [Hz]."
)
parser.add_argument(
    "--fill_spacing",
    type=_positive_finite_float,
    default=None,
    help=(
        "Particle spacing used to sample the cavity volume [m]."
        f" Defaults to voxel_size / {DEFAULT_PARTICLES_PER_VOXEL_AXIS:g}."
    ),
)
parser.add_argument(
    "--fill_level",
    type=_unit_interval_float,
    default=DEFAULT_FILL_LEVEL,
    help=(
        "Water line as a fraction [0, 1] of the teapot height to fill the cavity up to."
        f" Defaults to {DEFAULT_FILL_LEVEL:g}; use 1.0 to fill to the brim."
    ),
)
parser.add_argument(
    "--min_ray_hits",
    type=int,
    default=DEFAULT_MIN_RAY_HITS,
    choices=range(1, 7),
    help=(
        "Enclosure strictness for cavity detection: how many of the 6 axis rays must hit the shell"
        f" (1-6). Higher drops thin features like the spout/handle. Defaults to {DEFAULT_MIN_RAY_HITS}."
    ),
)
parser.add_argument(
    "--fluid_render_mode",
    type=str,
    default="surface",
    choices=["particles", "surface", "both"],
    help="Fluid visualization: raw MPM particles, reconstructed surface mesh, or both.",
)
parser.add_argument(
    "--presentation",
    choices=["default", "pour_closeups"],
    default="default",
    help="Use the original view or closeups with a two-second reconstructed-water segment.",
)
parser.add_argument(
    "--container_usd",
    type=str,
    default=f"{ISAAC_NUCLEUS_DIR}/Props/Teapot/utah_teapot.usdc",
    help="USD asset used as the pouring container (rigid collider).",
)
parser.add_argument(
    "--island_usd",
    type=str,
    default=DEFAULT_ISLAND_USD,
    help="Optional RTX kitchen-island visual. An empty or unavailable path uses the procedural table.",
)
parser.add_argument(
    "--bowl_usd",
    type=str,
    default=DEFAULT_BOWL_USD,
    help="Optional RTX catch-bowl visual. An empty or unavailable path uses the procedural bowl.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if min(args_cli.physics_hz, args_cli.physics_substeps, args_cli.rigid_substeps) < 1:
    parser.error("--physics_hz, --physics_substeps and --rigid_substeps must be positive.")
if args_cli.robot == "none" and args_cli.physics_substeps > 1:
    parser.error("--physics_substeps requires a robot; the scripted teapot updates once per outer step.")
if args_cli.robot != "none":
    controller_decimation = args_cli.physics_hz / args_cli.controller_hz
    if controller_decimation < 1 or not math.isclose(controller_decimation, round(controller_decimation)):
        parser.error("--physics_hz must be an integer multiple of --controller_hz.")
if args_cli.sequence is None:
    args_cli.sequence = "pickup_pour" if args_cli.robot != "none" else "pour"
if args_cli.robot != "none" and args_cli.sequence != "pickup_pour":
    parser.error("The robot requires --sequence pickup_pour to acquire its physical grasp.")
if not 1 <= args_cli.video_fps <= 120:
    parser.error("--video_fps must be between 1 and 120.")
if args_cli.video and args_cli.video_fps > args_cli.physics_hz:
    parser.error("--video_fps must not exceed --physics_hz.")
if args_cli.pour_angle is not None and args_cli.pour_angle > 90.0:
    parser.error("--pour_angle must be at most 90 degrees.")
if args_cli.pour_upper_angle is not None:
    peak_angle = (
        args_cli.pour_angle
        if args_cli.pour_angle is not None
        else (50.0 if args_cli.sequence == "pickup_pour" else 65.0)
    )
    if args_cli.pour_upper_angle > peak_angle:
        parser.error("--pour_upper_angle must not exceed the maximum pouring tilt.")
if not math.isfinite(args_cli.pour_aim_offset_x):
    parser.error("--pour_aim_offset_x must be finite.")
if args_cli.video and not {"newton_gl", "newton_rtx"}.intersection(args_cli.visualizer or []):
    parser.error("--video requires a Newton GL or RTX visualizer.")
if args_cli.benchmark and args_cli.visualizer:
    parser.error("--benchmark requires --visualizer none to exclude rendering.")
if args_cli.motion_report and (args_cli.robot == "none" or args_cli.benchmark):
    parser.error("--motion_report requires --robot rizon_sharpa and a separate run from --benchmark.")
if args_cli.presentation == "pour_closeups" and args_cli.robot == "none":
    parser.error("--presentation pour_closeups requires --robot rizon_sharpa.")
if args_cli.presentation == "pour_closeups" and args_cli.visualizer:
    if not {"newton_gl", "newton_rtx"}.intersection(args_cli.visualizer):
        parser.error("--presentation pour_closeups requires a Newton GL or RTX visualizer.")


# Keep the default 1.25 ms step and cap interactive rendering at approximately 400 Hz.
SIMULATION_HZ = args_cli.physics_hz
RENDER_INTERVAL = max(1, round(SIMULATION_HZ / 400))
VOXEL_SIZE = args_cli.voxel_size
FILL_SPACING = (
    args_cli.fill_spacing if args_cli.fill_spacing is not None else VOXEL_SIZE / DEFAULT_PARTICLES_PER_VOXEL_AXIS
)
FILL_LEVEL = args_cli.fill_level
MIN_RAY_HITS = args_cli.min_ray_hits
SHOW_FLUID_PARTICLES = args_cli.fluid_render_mode in ("particles", "both") or args_cli.presentation == "pour_closeups"
SHOW_FLUID_SURFACE = args_cli.fluid_render_mode in ("surface", "both") or args_cli.presentation == "pour_closeups"

# Sparse grids reserve capture-stable storage; fixed grids need padding for the full pour trajectory.
GRID_TYPE = args_cli.grid_type
GRID_PADDING = 0 if GRID_TYPE == "sparse" else 64
MAX_ACTIVE_CELL_COUNT = (1 << 16) if GRID_TYPE == "sparse" else (1 << 18)
MPM_SUBSTEPS = args_cli.physics_substeps

PARTICLE_DENSITY = 1000.0
FILL_JITTER = 0.2
FILL_SEED = 7
PARTICLE_RADIUS = 0.5 * FILL_SPACING
PARTICLE_MASS = FILL_SPACING**3 * PARTICLE_DENSITY

COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
PARTICLE_SURFACE_CLEARANCE = COLLIDER_MARGIN + PARTICLE_RADIUS
CONTAINER_FRICTION = 0.0
BOWL_FRICTION = 0.20
TABLE_FRICTION = 0.5

HOLD_TIME = 0.55
TILT_TIME = (
    args_cli.pour_tilt_time
    if args_cli.pour_tilt_time is not None
    else (2.8 if args_cli.robot != "none" or args_cli.sequence == "pickup_pour" else 2.0)
)
PICKUP_ENABLED = args_cli.sequence == "pickup_pour"
# Begin pouring close to the bowl, then raise the tilted pot gently.
POUR_ANGLE = math.radians(
    args_cli.pour_angle if args_cli.pour_angle is not None else (50.0 if PICKUP_ENABLED else 65.0)
)
UPPER_POUR_ANGLE = math.radians(args_cli.pour_upper_angle) if args_cli.pour_upper_angle is not None else POUR_ANGLE
# Reduce flow while the outlet is still close to the bowl, before the high drop.
POUR_TAPER_START_FRACTION = 0.15
POUR_TAPER_END_FRACTION = 0.35
CONTAINER_LIFT_HEIGHT = args_cli.pour_rise_height if PICKUP_ENABLED else 0.24
CONTAINER_LIFT_TIME = args_cli.pour_rise_time if PICKUP_ENABLED else 3.0
APPROACH_HOLD_TIME = 0.3
PREGRASP_TIME = 1.8
ALIGNMENT_TIME = 2.6
GRASP_CLOSE_START = 3.4
GRASP_CLOSE_END = 4.2
PICKUP_TIME = 5.6
INITIAL_LIFT_TIME = 5.0
RECOVERY_TIME = 5.0 if PICKUP_ENABLED else TILT_TIME
POUR_PREFIX_TIME = PICKUP_TIME + INITIAL_LIFT_TIME if PICKUP_ENABLED else 0.0
SEQUENCE_DURATION = POUR_PREFIX_TIME + HOLD_TIME + TILT_TIME + CONTAINER_LIFT_TIME + RECOVERY_TIME + 0.45
if args_cli.max_steps is None:
    args_cli.max_steps = round(SIMULATION_HZ * (SEQUENCE_DURATION if PICKUP_ENABLED else 7.5))
if args_cli.benchmark and not 0 <= args_cli.benchmark_warmup < args_cli.max_steps:
    parser.error("--benchmark_warmup must be nonnegative and smaller than --max_steps.")

TABLE_TOP_Z = 0.90041
TABLE_HALF_EXTENTS = (0.31565, 0.62045, 0.009)
TABLE_ORIENTATION = (0.0, 0.0, -math.sin(0.25 * math.pi), math.cos(0.25 * math.pi))
# Translate the manipulation together, leaving the kitchen island fixed.
MANIPULATION_OFFSET_X = -0.25 if PICKUP_ENABLED else 0.0
BOWL_SCALE = 1.0
# Proxy dimensions measured from the default Bowl_G01 visual asset.
BOWL_LOCAL_Z_MIN = 0.000001783
BOWL_LOCAL_Z_MAX = 0.043571252
BOWL_INNER_BOTTOM_RADIUS = 0.0254
BOWL_INNER_TOP_RADIUS = 0.0508
BOWL_OUTER_BOTTOM_RADIUS = 0.0301
BOWL_OUTER_TOP_RADIUS = 0.0526
BOWL_BOTTOM_THICKNESS = 0.0049
BOWL_BASE_POS = (0.106 + MANIPULATION_OFFSET_X, 0.0, TABLE_TOP_Z - BOWL_SCALE * BOWL_LOCAL_Z_MIN)
BOWL_WORLD_TOP_Z = BOWL_BASE_POS[2] + BOWL_SCALE * BOWL_LOCAL_Z_MAX
TABLETOP_CONTAINER_POS = (-0.105 + MANIPULATION_OFFSET_X, 0.0, TABLE_TOP_Z)
CONTAINER_BASE_POS = (
    (-0.090 + MANIPULATION_OFFSET_X, 0.0, 1.020) if PICKUP_ENABLED else (-0.105, 0.0, BOWL_WORLD_TOP_Z + 0.115)
)
# The default mesh's outlet center. Lower it before flow starts, then rotate
# around this target. Allow for forward flow and clear the bowl's curved lip.
SPOUT_LOCAL_POS = (0.111, 0.0, 0.0805)
SPOUT_POUR_POS = (
    BOWL_BASE_POS[0] - 0.025 + args_cli.pour_aim_offset_x,
    BOWL_BASE_POS[1],
    BOWL_WORLD_TOP_Z + 0.028,
)
SPOUT_LOWERING_ANGLE = math.radians(35.0)
# Measured Utah-teapot jet estimates; the forward component eases as the pot empties.
STREAM_FORWARD_SPEED = 0.30
STREAM_FORWARD_SPEED_DROP = 0.04
STREAM_DOWNWARD_SPEED = 0.45
STREAM_GRAVITY = 9.81
# Keep the workspace centered between tabletop pickup and the high pour.
ROBOT_BASE_POS = (-0.68 + MANIPULATION_OFFSET_X, 0.0, TABLE_TOP_Z + max(0.0, CONTAINER_LIFT_HEIGHT - 0.45))

CONTAINER_COLOR = (0.70, 0.35, 0.16)
TABLE_COLOR = (0.48, 0.38, 0.26)
BOWL_COLOR = (1.0, 1.0, 1.0)
WATER_COLOR = (0.12, 0.35, 0.78)
WATER_OPACITY = 0.65

CAMERA_EYE = (MANIPULATION_OFFSET_X, -0.72, 1.18)
CAMERA_TARGET = (MANIPULATION_OFFSET_X, 0.0, TABLE_TOP_Z + 0.10)

# Use tighter kernels than the open-tank dam-break preset so the reconstructed
# surface stays inside narrow features such as the teapot spout.
SURFACE_VOXEL_SIZE = 0.5 * VOXEL_SIZE
SURFACE_KERNEL_RADIUS = max(3.0 * FILL_SPACING, 1.5 * SURFACE_VOXEL_SIZE)
SURFACE_MAX_GRID_CELLS = 4_000_000
SURFACE_PATH = "/fluid_surface"


def create_visualizer_cfgs():
    """Create demo-specific visualizer configs for requested backends."""
    requested = args_cli.visualizer or []
    if not any(name in requested for name in ("newton_gl", "newton_rtx")):
        return []

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    cfg_types = {"newton_gl": NewtonGLVisualizerCfg, "newton_rtx": NewtonRTXVisualizerCfg}
    configs = [
        cfg_types[name](
            show_particles=SHOW_FLUID_PARTICLES,
            particle_color=WATER_COLOR,
        )
        for name in requested
        if name in cfg_types
    ]
    for cfg in configs:
        if args_cli.presentation == "pour_closeups":
            cfg.focal_length = 18.0
    if args_cli.video:
        for cfg in configs:
            cfg.headless = True
            if cfg.visualizer_type == "newton_rtx":
                cfg.rtx_environment = "studio"
                cfg.background_color = (0.08, 0.10, 0.14)
                cfg.render_settings = {
                    "omni:rtx:quality": ("Int", 64),
                    "omni:rtx:dlss:frameGeneration": ("Bool", False),
                }
    return configs


class FluidSurfaceRenderer:
    """Extract and display a dynamic water surface in Newton visualizers."""

    def __init__(self, sim) -> None:
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
        from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonRTXVisualizer
        from newton.geometry import ParticleSurface

        self._wp = wp
        self._visualizers = tuple(
            visualizer
            for visualizer in sim.visualizers
            if isinstance(visualizer, (NewtonGLVisualizer, NewtonRTXVisualizer))
        )
        if not self._visualizers:
            raise RuntimeError("Particle surface rendering requires a Newton GL or RTX visualizer.")

        self._model = NewtonManager.get_model()
        self._state = NewtonManager.get_state_0()
        self._surface = ParticleSurface(
            voxel_size=SURFACE_VOXEL_SIZE,
            max_grid_cells=SURFACE_MAX_GRID_CELLS,
            world_count=max(self._model.world_count, 1),
            kernel_radius=SURFACE_KERNEL_RADIUS,
            threshold=0.4,
            smooth_lambda=0.0,
            anisotropic=True,
            kernel_scale=0.5,
            anisotropy_ratio=16.0,
            anisotropy_scale=1.0,
            anisotropy_min_neighbors=4,
            anisotropy_binning=True,
            anisotropy_strength=0.95,
            field_smooth_iterations=0,
            mesh_smooth_iterations=1,
            device=self._model.device,
        )
        self._empty_points = wp.empty(0, dtype=wp.vec3, device=self._model.device)
        self._empty_indices = wp.empty(0, dtype=wp.int32, device=self._model.device)
        self._empty_normals = wp.empty(0, dtype=wp.vec3, device=self._model.device)
        self._surface_mesh = None
        self._surface_graph = None
        self._visible = False
        self._capture_surface_extraction()

    def _extract_surface(self):
        """Extract the water surface from the current Newton particle state."""
        return self._surface.extract(
            self._state.particle_q,
            self._model.particle_radius,
            particle_flags=self._model.particle_flags,
            particle_world=self._model.particle_world if self._surface.world_count > 1 else None,
        )

    def _capture_surface_extraction(self) -> None:
        """Capture reconstruction separately from the MPM physics graph."""
        if not self._model.device.is_cuda or args_cli.disable_cuda_graph:
            return
        self._surface_mesh = self._extract_surface()
        with self._wp.ScopedCapture(device=self._model.device) as capture:
            self._surface_mesh = self._extract_surface()
        self._surface_graph = capture.graph

    def update(self, *, visible: bool = True) -> int:
        """Reconstruct and publish the current water surface, returning its triangle count."""
        if not visible:
            if self._visible:
                # RTX requires nonempty point buffers even when a mesh is hidden.
                vertices, indices, normals = self._surface_mesh.to_arrays()
                for visualizer in self._visualizers:
                    visualizer.log_mesh(SURFACE_PATH, vertices, indices, normals=normals, hidden=True, dynamic=True)
            self._visible = False
            return 0

        if self._surface_graph is None:
            self._surface_mesh = self._extract_surface()
        else:
            self._wp.capture_launch(self._surface_graph)

        vertices, indices, normals = self._surface_mesh.to_arrays()
        if vertices is None:
            vertices = self._empty_points
            indices = self._empty_indices
            normals = self._empty_normals
            hidden = True
            triangle_count = 0
        else:
            hidden = False
            triangle_count = indices.shape[0] // 3

        for visualizer in self._visualizers:
            visualizer.log_mesh(
                SURFACE_PATH,
                vertices,
                indices,
                normals=normals,
                hidden=hidden,
                backface_culling=False,
                color=WATER_COLOR,
                roughness=0.1,
                metallic=0.0,
                dynamic=True,
                opacity=WATER_OPACITY,
            )
        self._visible = not hidden
        return triangle_count


def quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion for a rotation about +Y."""
    half = 0.5 * angle_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def spout_aligned_translation(
    angle: float, angular_speed: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Lower the outlet smoothly, then keep it over the bowl while rotating."""
    progress = max(0.0, min(1.0, angle / SPOUT_LOWERING_ANGLE))
    blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    blend_speed = (
        30.0 * progress**2 * (1.0 - progress) ** 2 * angular_speed / SPOUT_LOWERING_ANGLE
        if 0.0 < progress < 1.0
        else 0.0
    )
    x, y, z = SPOUT_LOCAL_POS
    sine, cosine = math.sin(angle), math.cos(angle)
    rotated_spout = (x * cosine + z * sine, y, -x * sine + z * cosine)
    target_root = tuple(a - b for a, b in zip(SPOUT_POUR_POS, rotated_spout, strict=True))
    target_velocity = (angular_speed * (x * sine - z * cosine), 0.0, angular_speed * (x * cosine + z * sine))
    displacement = tuple(a - b for a, b in zip(target_root, CONTAINER_BASE_POS, strict=True))
    position = tuple(a + blend * delta for a, delta in zip(CONTAINER_BASE_POS, displacement, strict=True))
    velocity = tuple(
        blend_speed * delta + blend * speed for delta, speed in zip(displacement, target_velocity, strict=True)
    )
    return position, velocity


def container_pose_at_time(
    sim_time: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    """Return teapot ``(position, orientation, twist)`` for the scripted pour."""
    if not PICKUP_ENABLED:
        raw = (sim_time - HOLD_TIME) / TILT_TIME
        clamped = max(0.0, min(1.0, raw))
        alpha = clamped * clamped * (3.0 - 2.0 * clamped)
        alpha_dot = (6.0 * clamped * (1.0 - clamped)) / TILT_TIME if 0.0 < raw < 1.0 else 0.0

        lift_raw = (sim_time - HOLD_TIME - TILT_TIME) / CONTAINER_LIFT_TIME
        lift_alpha = max(0.0, min(1.0, lift_raw))
        lift_speed = CONTAINER_LIFT_HEIGHT / CONTAINER_LIFT_TIME if 0.0 < lift_raw < 1.0 else 0.0

        angle = POUR_ANGLE * alpha
        angular_speed = POUR_ANGLE * alpha_dot
        pos = (
            CONTAINER_BASE_POS[0],
            CONTAINER_BASE_POS[1],
            CONTAINER_BASE_POS[2] + CONTAINER_LIFT_HEIGHT * lift_alpha,
        )
        # Newton spatial vectors are (linear, angular).
        twist = (0.0, 0.0, lift_speed, 0.0, angular_speed, 0.0)
        return pos, quat_y(angle), twist
    if PICKUP_ENABLED and sim_time < POUR_PREFIX_TIME:
        raw = (sim_time - PICKUP_TIME) / INITIAL_LIFT_TIME
        progress = max(0.0, min(1.0, raw))
        alpha = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
        alpha_dot = 30.0 * progress**2 * (1.0 - progress) ** 2 / INITIAL_LIFT_TIME if 0.0 < raw < 1.0 else 0.0
        displacement = tuple(b - a for a, b in zip(TABLETOP_CONTAINER_POS, CONTAINER_BASE_POS, strict=True))
        return (
            tuple(a + alpha * delta for a, delta in zip(TABLETOP_CONTAINER_POS, displacement, strict=True)),
            quat_y(0.0),
            (*tuple(alpha_dot * delta for delta in displacement), 0.0, 0.0, 0.0),
        )
    sim_time -= POUR_PREFIX_TIME
    raw = (sim_time - HOLD_TIME) / TILT_TIME
    clamped = max(0.0, min(1.0, raw))
    alpha = clamped**3 * (10.0 + clamped * (-15.0 + 6.0 * clamped))
    alpha_dot = 30.0 * clamped**2 * (1.0 - clamped) ** 2 / TILT_TIME if 0.0 < raw < 1.0 else 0.0

    lift_raw = (sim_time - HOLD_TIME - TILT_TIME) / CONTAINER_LIFT_TIME
    lift_progress = max(0.0, min(1.0, lift_raw))
    lift_alpha = lift_progress**3 * (10.0 + lift_progress * (-15.0 + 6.0 * lift_progress))
    lift_speed = (
        CONTAINER_LIFT_HEIGHT * 30.0 * lift_progress**2 * (1.0 - lift_progress) ** 2 / CONTAINER_LIFT_TIME
        if 0.0 < lift_raw < 1.0
        else 0.0
    )

    angle = POUR_ANGLE * alpha
    angular_speed = POUR_ANGLE * alpha_dot
    taper_span = POUR_TAPER_END_FRACTION - POUR_TAPER_START_FRACTION
    taper_raw = (lift_raw - POUR_TAPER_START_FRACTION) / taper_span
    taper = max(0.0, min(1.0, taper_raw))
    taper_alpha = taper**3 * (10.0 + taper * (-15.0 + 6.0 * taper))
    angle -= (POUR_ANGLE - UPPER_POUR_ANGLE) * taper_alpha
    if 0.0 < taper_raw < 1.0:
        angular_speed -= (
            (POUR_ANGLE - UPPER_POUR_ANGLE) * 30.0 * taper**2 * (1.0 - taper) ** 2 / (taper_span * CONTAINER_LIFT_TIME)
        )
    recover_raw = (sim_time - HOLD_TIME - TILT_TIME - CONTAINER_LIFT_TIME) / RECOVERY_TIME
    recovery = max(0.0, min(1.0, recover_raw))
    recover_alpha = recovery**3 * (10.0 + recovery * (-15.0 + 6.0 * recovery))
    angle -= UPPER_POUR_ANGLE * recover_alpha
    if 0.0 < recover_raw < 1.0:
        angular_speed -= UPPER_POUR_ANGLE * 30.0 * recovery**2 * (1.0 - recovery) ** 2 / RECOVERY_TIME
    if PICKUP_ENABLED:
        pos, linear_velocity = spout_aligned_translation(angle, angular_speed)
    else:
        pos = CONTAINER_BASE_POS
        linear_velocity = (0.0, 0.0, 0.0)
    pos = (pos[0], pos[1], pos[2] + CONTAINER_LIFT_HEIGHT * lift_alpha)
    linear_velocity = (linear_velocity[0], linear_velocity[1], linear_velocity[2] + lift_speed)
    if PICKUP_ENABLED:
        gap = SPOUT_POUR_POS[2] - BOWL_WORLD_TOP_Z
        fall_speed = math.sqrt(STREAM_DOWNWARD_SPEED**2 + 2.0 * STREAM_GRAVITY * gap)
        raised_fall_speed = math.sqrt(
            STREAM_DOWNWARD_SPEED**2 + 2.0 * STREAM_GRAVITY * (gap + CONTAINER_LIFT_HEIGHT * lift_alpha)
        )
        forward_speed = STREAM_FORWARD_SPEED - STREAM_FORWARD_SPEED_DROP * lift_alpha
        forward_acceleration = -STREAM_FORWARD_SPEED_DROP * lift_speed / CONTAINER_LIFT_HEIGHT
        extra_fall_time = (raised_fall_speed - fall_speed) / STREAM_GRAVITY
        retreat = forward_speed * extra_fall_time
        retreat_speed = forward_acceleration * extra_fall_time + forward_speed * lift_speed / raised_fall_speed
        pos = (pos[0] - retreat, pos[1], pos[2])
        linear_velocity = (linear_velocity[0] - retreat_speed, linear_velocity[1], linear_velocity[2])
        if args_cli.grasp_finger == "index":
            # Slow final drips need less ballistic retreat than the steady jet.
            drip_raw = recover_raw * RECOVERY_TIME / 0.8
            drip_progress = max(0.0, min(1.0, drip_raw))
            drip_shift = 0.01 * drip_progress**3 * (10.0 + drip_progress * (-15.0 + 6.0 * drip_progress))
            drip_speed = (
                0.01 * 30.0 * drip_progress**2 * (1.0 - drip_progress) ** 2 / 0.8 if 0.0 < drip_raw < 1.0 else 0.0
            )
            pos = (pos[0] + drip_shift, pos[1], pos[2])
            linear_velocity = (linear_velocity[0] + drip_speed, linear_velocity[1], linear_velocity[2])
    # Newton spatial vectors are (linear, angular).
    twist = (*linear_velocity, 0.0, angular_speed, 0.0)
    return pos, quat_y(angle), twist


def approach_offset_at_time(sim_time: float) -> tuple[float, float, float]:
    """Descend in clear space, align at handle height, then insert the selected hook finger."""
    away = (-0.09, -0.16, 0.20)
    pregrasp = (0.0, -0.10, 0.065)
    aligned = (0.0, -0.08, 0.0)
    if sim_time < PREGRASP_TIME:
        progress = (sim_time - APPROACH_HOLD_TIME) / (PREGRASP_TIME - APPROACH_HOLD_TIME)
        first, second = away, pregrasp
    elif sim_time < ALIGNMENT_TIME:
        progress = (sim_time - PREGRASP_TIME) / (ALIGNMENT_TIME - PREGRASP_TIME)
        first, second = pregrasp, aligned
    else:
        progress = (sim_time - ALIGNMENT_TIME) / (GRASP_CLOSE_START - ALIGNMENT_TIME)
        first, second = aligned, (0.0, 0.0, 0.0)
    progress = max(0.0, min(1.0, progress))
    alpha = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    return tuple(a + alpha * (b - a) for a, b in zip(first, second, strict=True))


def create_bowl_collider_mesh(num_segments: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Build a lightweight local-space collision proxy matching the visual bowl."""
    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(num_segments, z)])

    vertices = np.vstack(
        [
            ring(BOWL_INNER_BOTTOM_RADIUS, BOWL_BOTTOM_THICKNESS),
            ring(BOWL_INNER_TOP_RADIUS, BOWL_LOCAL_Z_MAX),
            ring(BOWL_OUTER_TOP_RADIUS, BOWL_LOCAL_Z_MAX),
            ring(BOWL_OUTER_BOTTOM_RADIUS, BOWL_LOCAL_Z_MIN),
            np.array([[0.0, 0.0, BOWL_BOTTOM_THICKNESS], [0.0, 0.0, BOWL_LOCAL_Z_MIN]], dtype=np.float32),
        ]
    ).astype(np.float32)

    inner_center_id = 4 * num_segments
    outer_center_id = inner_center_id + 1
    indices: list[int] = []
    for i in range(num_segments):
        j = (i + 1) % num_segments
        ib_i, ib_j = i, j
        it_i, it_j = i + num_segments, j + num_segments
        ot_i, ot_j = i + 2 * num_segments, j + 2 * num_segments
        ob_i, ob_j = i + 3 * num_segments, j + 3 * num_segments
        indices.extend([ib_i, it_i, ib_j, ib_j, it_i, it_j])
        indices.extend([ob_i, ob_j, ot_i, ot_i, ob_j, ot_j])
        indices.extend([it_i, ot_i, it_j, it_j, ot_i, ot_j])
        indices.extend([inner_center_id, ib_i, ib_j, outer_center_id, ob_j, ob_i])

    return vertices, np.asarray(indices, dtype=np.int32).reshape(-1, 3)


def spawn_demo_mesh(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn an exact triangle mesh with standard Isaac Lab rigid/collision schemas."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg, spawn_physics_material
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material, create_prim, get_current_stage

    stage = get_current_stage()
    vertices = np.asarray(cfg.vertices, dtype=np.float32)
    faces = np.asarray(cfg.faces, dtype=np.int32)

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    create_prim(geom_prim_path, prim_type="Xform", stage=stage)
    create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices,
            "faceVertexIndices": faces.reshape(-1),
            "faceVertexCounts": np.full(faces.shape[0], 3, dtype=np.int32),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )
    if not cfg.visible:
        UsdGeom.Imageable(stage.GetPrimAtPath(mesh_prim_path)).MakeInvisible()

    def as_fragments(value) -> list:
        # the schema slots accept a bare fragment or a list of fragments
        return list(value) if isinstance(value, (list, tuple)) else [value]

    if cfg.collision_props is not None:
        schemas.apply_collision_properties(
            mesh_prim_path, as_fragments(cfg.collision_props), create_if_missing=True, stage=stage
        )
    if cfg.mesh_collision_props is not None:
        schemas.apply_mesh_collision_properties(mesh_prim_path, as_fragments(cfg.mesh_collision_props), stage=stage)
    if cfg.rigid_props is not None:
        schemas.apply_rigid_body_properties(
            prim_path, as_fragments(cfg.rigid_props), create_if_missing=True, stage=stage
        )
    if cfg.rigid_contact_usd is not None:
        from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg

        import isaaclab.sim as sim_utils

        # Keep the authored convex decomposition, including the handle opening,
        # on the same body as the exact fluid shell. Only the shell is rendered.
        contact_cfg = sim_utils.UsdFileCfg(
            usd_path=cfg.rigid_contact_usd,
            rigid_props={"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]},
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0)),
                NewtonCollisionCfg(contact_gap=0.001),
            ],
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visible=False,
        )
        # The public spawner applies visible=False to the referenced hierarchy;
        # its low-level file helper would leave a duplicate opaque teapot visible.
        contact_root = sim_utils.spawn_from_usd(f"{prim_path}/RigidCollider", contact_cfg)
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(prim_path))
        for prim in Usd.PrimRange(contact_root):
            if prim.HasAPI(UsdPhysics.MassAPI):
                original_mass = UsdPhysics.MassAPI(prim)
                for field in ("Mass", "CenterOfMass", "DiagonalInertia", "PrincipalAxes"):
                    value = getattr(original_mass, f"Get{field}Attr")().Get()
                    if value is not None:
                        getattr(mass, f"Create{field}Attr")(value)
                prim.RemoveAPI(UsdPhysics.MassAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

    if cfg.mass_props is not None:
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(prim_path))
        authored_mass = mass.GetMassAttr().Get()
        authored_inertia = mass.GetDiagonalInertiaAttr().Get()
        schemas.apply_mass_properties(prim_path, as_fragments(cfg.mass_props), create_if_missing=True, stage=stage)
        configured_mass = mass.GetMassAttr().Get()
        if (
            authored_mass is not None
            and authored_mass > 0.0
            and authored_inertia is not None
            and configured_mass is not None
            and configured_mass > 0.0
        ):
            # Keep the stock COM and principal axes; a uniform mass change scales
            # its authored inertia rather than leaving mismatched mechanics.
            scale = configured_mass / authored_mass
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(value * scale for value in authored_inertia)))

    if cfg.visual_material is not None:
        material_path = cfg.visual_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    if cfg.physics_material is not None:
        material_path = cfg.physics_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{material_path}"
        spawn_physics_material(material_path, cfg.physics_material, stage=stage)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


def load_asset_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a USD asset and return its triangulated geometry in the asset's local frame.

    Meshes, including instance proxies, are concatenated, transformed into the asset frame, and
    fan-triangulated so visual assets and their exact MPM colliders align.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD asset: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices_list: list[np.ndarray] = []
    triangles_list: list[np.ndarray] = []
    vertex_offset = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_vertex_counts or not face_vertex_indices:
            continue

        points = np.asarray(points, dtype=np.float64)
        face_vertex_counts = np.asarray(face_vertex_counts, dtype=np.int64)
        face_vertex_indices = np.asarray(face_vertex_indices, dtype=np.int64)
        # Bake the local-to-root transform (USD uses row-vector convention: v' = v * M).
        matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64).reshape(4, 4)
        homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (homogeneous @ matrix)[:, :3]

        triangles: list[tuple[int, int, int]] = []
        cursor = 0
        for count in face_vertex_counts:
            for k in range(1, count - 1):
                triangles.append(
                    (
                        face_vertex_indices[cursor],
                        face_vertex_indices[cursor + k],
                        face_vertex_indices[cursor + k + 1],
                    )
                )
            cursor += count
        if not triangles:
            continue
        vertices_list.append(transformed_points.astype(np.float32))
        triangles_list.append(np.asarray(triangles, dtype=np.int64) + vertex_offset)
        vertex_offset += points.shape[0]

    if not vertices_list:
        raise RuntimeError(f"No meshes found in USD asset: {usd_path}")
    return np.concatenate(vertices_list), np.concatenate(triangles_list).astype(np.int32)


def create_fluid_particles(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Sample local-space MPM particles in the teapot's enclosed cavity."""
    from isaaclab.utils.warp import sample_particles_in_cavity

    water_level = float(vertices[:, 2].min() + FILL_LEVEL * (vertices[:, 2].max() - vertices[:, 2].min()))
    points = sample_particles_in_cavity(
        vertices,
        faces,
        spacing=FILL_SPACING,
        device=args_cli.device,
        jitter=FILL_JITTER,
        seed=FILL_SEED,
        surface_margin=PARTICLE_SURFACE_CLEARANCE,
        min_ray_hits=MIN_RAY_HITS,
        water_level=water_level,
    )
    if points.shape[0] == 0:
        raise RuntimeError("Teapot cavity sampling produced no particles; reduce --fill_spacing or --min_ray_hits.")

    return points.astype(np.float32, copy=False), PARTICLE_RADIUS, PARTICLE_MASS


def retrieve_optional_visual_asset(path: str, label: str) -> str | None:
    """Resolve an optional presentation asset, falling back to procedural geometry."""
    if not path:
        return None
    try:
        return retrieve_file_path(path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("Could not load optional %s visual (%s); using procedural geometry.", label, exc)
        return None


def create_sim_cfg() -> SimulationCfg:
    """Create the Isaac Lab simulation config using the MPM manager."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    solver_cfg = MPMSolverCfg(
        voxel_size=VOXEL_SIZE,
        grid_type=GRID_TYPE,
        grid_padding=GRID_PADDING,
        max_active_cell_count=MAX_ACTIVE_CELL_COUNT,
        max_iterations=100,
        tolerance=1.0e-4,
        collider_basis="S2",
        strain_basis="P0",
        transfer_scheme="apic",
        integration_scheme="pic",
        # Keep nearly empty grid nodes conditioned during affine velocity transfer.
        air_drag=1.0 if args_cli.robot != "none" else 1.0e-3,
        collider_velocity_mode="forward",
        # Coupled playback applies the same native correction inside its
        # fluid entry, since this manager option only handles standalone MPM.
        project_outside_colliders=False,
    )
    if args_cli.robot == "rizon_sharpa":
        from rizon_sharpa_teapot_cfg import make_solver_cfg

        solver_cfg = make_solver_cfg(
            solver_cfg, fluid_coupling=args_cli.fluid_coupling, rigid_substeps=args_cli.rigid_substeps
        )
    return sim_utils.SimulationCfg(
        dt=1.0 / SIMULATION_HZ,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            # Refresh the measured collider boundary before every fluid substep.
            num_substeps=MPM_SUBSTEPS,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )


def create_scene_cfg(container_usd: str, island_usd: str | None, bowl_usd: str | None) -> InteractiveSceneCfg:
    """Create the teapot-fill scene using declarative Isaac Lab assets."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    container_pos, container_rot, _ = container_pose_at_time(0.0)
    container_vertices, container_faces = load_asset_mesh(container_usd)
    fluid_points, particle_radius, particle_mass = create_fluid_particles(container_vertices, container_faces)
    bowl_vertices, bowl_faces = create_bowl_collider_mesh()

    @configclass
    class DemoMeshCfg(sim_utils.MeshCfg):
        """Demo-local exact triangle-mesh asset config."""

        func: Callable | str = clone(spawn_demo_mesh)
        vertices: list[list[float]] = MISSING
        faces: list[list[int]] = MISSING
        mesh_collision_props: sim_utils.UsdPhysicsMeshCollisionCfg | None = None
        rigid_contact_usd: str | None = None

    # The visual assets only disable rigid bodies and colliders they already carry; the explicit
    # target mappings keep an asset without physics from gaining a body on its spawn prim.
    disable_asset_rigid_bodies = {"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]}
    disable_asset_colliders = {"(/.*)?": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)]}

    island_cfg = None
    if island_usd is not None:
        island_cfg = AssetBaseCfg(
            prim_path="/World/Island",
            spawn=sim_utils.UsdFileCfg(
                usd_path=island_usd,
                variants={"Physics": "none"},
                make_uninstanceable=True,
                rigid_props=disable_asset_rigid_bodies,
                collision_props=disable_asset_colliders,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(rot=TABLE_ORIENTATION),
        )

    bowl_visual_cfg = None
    if bowl_usd is not None:
        bowl_visual_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CatchBowlVisual",
            spawn=sim_utils.UsdFileCfg(
                usd_path=bowl_usd,
                scale=(BOWL_SCALE,) * 3,
                variants={"Physics": "none"},
                make_uninstanceable=True,
                rigid_props=disable_asset_rigid_bodies,
                collision_props=disable_asset_colliders,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=BOWL_BASE_POS),
        )

    @configclass
    class TeapotFillSceneCfg(InteractiveSceneCfg):
        """Scene containing MPM colliders and one MPM fluid object sampled inside the teapot."""

        island: AssetBaseCfg | None = island_cfg

        tabletop_collider = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TabletopCollider",
            spawn=sim_utils.CuboidCfg(
                size=(2.0 * TABLE_HALF_EXTENTS[0], 2.0 * TABLE_HALF_EXTENTS[1], 2.0 * TABLE_HALF_EXTENTS[2]),
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(
                        contact_margin=COLLIDER_MARGIN, contact_gap=0.001 if args_cli.robot != "none" else None
                    ),
                    *([MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0))] if args_cli.robot != "none" else []),
                ],
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=TABLE_FRICTION,
                    dynamic_friction=TABLE_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visible=island_usd is None,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR) if island_usd is None else None
                ),
                visual_material_path="visualMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]),
                rot=TABLE_ORIENTATION,
            ),
        )

        catch_bowl_visual: AssetBaseCfg | None = bowl_visual_cfg

        catch_bowl_collider = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CatchBowlCollider",
            spawn=DemoMeshCfg(
                vertices=(BOWL_SCALE * bowl_vertices).tolist(),
                faces=bowl_faces.tolist(),
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=COLLIDER_MARGIN),
                ],
                mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=BOWL_FRICTION,
                    dynamic_friction=BOWL_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visible=bowl_usd is None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOWL_COLOR) if bowl_usd is None else None,
                visual_material_path="visualMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=BOWL_BASE_POS),
        )

        # Utah Teapot: free for any use; credit as the (Modified) Utah Teapot
        # (Univ. of Utah); provided "as is", no warranty.
        container = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PourContainer",
            # Re-spawn the source geometry as one exact triangle mesh. The USD
            # asset's authored convex decomposition is unsuitable for a hollow
            # MPM collider and SolverImplicitMPM does not accept convex meshes.
            spawn=DemoMeshCfg(
                vertices=container_vertices.tolist(),
                faces=container_faces.tolist(),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    PhysxRigidBodyCfg(disable_gravity=True),
                ],
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=COLLIDER_MARGIN),
                ],
                mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=CONTAINER_FRICTION,
                    dynamic_friction=CONTAINER_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CONTAINER_COLOR),
                visual_material_path="visualMaterial",
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=container_pos, rot=container_rot),
        )

        fluid = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fluid",
            spawn=MPMPointsCfg(
                positions=fluid_points.tolist(),
                mass=particle_mass,
                radius=particle_radius,
                material=MPMParticleMaterialCfg(
                    viscosity=1.0e-3,
                    friction=0.0,
                    damping=1.0e-3,
                    yield_pressure=1.0e15,
                    tensile_yield_ratio=1.0,
                ),
                visual_color=WATER_COLOR,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=WATER_COLOR,
                    roughness=0.1,
                    opacity=0.7,
                ),
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=container_pos),
        )

        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

    cfg = TeapotFillSceneCfg(num_envs=1, env_spacing=0.0)
    if args_cli.robot == "rizon_sharpa":
        from rizon_sharpa_teapot import get_robot_cfg, prepared_contact_path

        cfg.robot = get_robot_cfg(base_position=ROBOT_BASE_POS, grasp_finger=args_cli.grasp_finger)
        cfg.container.spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False)
        cfg.container.spawn.mass_props = sim_utils.MassCfg(mass=args_cli.teapot_mass)
        cfg.container.spawn.rigid_contact_usd = str(prepared_contact_path(Path(container_usd)))
        cfg.container.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        cfg.robot_pedestal = AssetBaseCfg(
            prim_path="/World/RobotPedestal",
            spawn=sim_utils.CylinderCfg(
                radius=0.10,
                height=ROBOT_BASE_POS[2],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.12, 0.15), metallic=0.65, roughness=0.3
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(*ROBOT_BASE_POS[:2], 0.5 * ROBOT_BASE_POS[2])),
        )
    return cfg


def particle_count(scene: InteractiveScene) -> int:
    """Return the number of MPM particles in the scene."""
    fluid = scene["fluid"]
    return fluid.num_instances * fluid.particles_per_object


def keep_running(sim: SimulationContext, count: int) -> bool:
    """Return whether the demo loop should continue this frame."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    return sim.is_running()


def write_container_state(container: RigidObject, sim_time: float) -> None:
    """Write the scripted container pose and velocity through the rigid-object API."""
    pos, quat, twist = container_pose_at_time(sim_time)
    pose = torch.tensor([pos + quat], dtype=torch.float32, device=container.device)
    velocity = torch.tensor([twist], dtype=torch.float32, device=container.device)
    container.write_root_link_pose_to_sim_index(root_pose=pose)
    container.write_root_link_velocity_to_sim_index(root_velocity=velocity)


class FluidContainmentMonitor:
    """Track escaped particles against the bowl and measured teapot bounds.

    This validation-only monitor detects particles below the bowl rim that are
    outside both the bowl footprint and the teapot's moving local mesh bounds.
    Remembering particle identities also catches splash that later re-enters.
    Bowl volume sums the captured particles' initial represented volumes.
    """

    def __init__(self, fluid, container) -> None:
        self._fluid = fluid
        self._container = container
        vertices = np.asarray(container.cfg.spawn.vertices)
        self._lower = vertices.min(axis=0) - PARTICLE_SURFACE_CLEARANCE
        self._upper = vertices.max(axis=0) + PARTICLE_SURFACE_CLEARANCE
        self._spilled = np.zeros(fluid.particles_per_object, dtype=bool)
        self._next_sample = 0.0
        self.metrics = {
            "initial_particles": int(self._spilled.size),
            "observation_hz": SIMULATION_HZ / max(1, round(SIMULATION_HZ / 10)),
            "spilled_particles": 0,
            "first_spill_time_s": None,
            "samples": [],
        }

    def check(self, sim_time: float) -> None:
        """Accumulate escaped identities and measure water inside the bowl."""
        from scipy.spatial.transform import Rotation

        points = self._fluid.data.particle_pos_w.warp.numpy()[0]
        if not np.all(np.isfinite(points)):
            raise RuntimeError("Fluid containment validation found nonfinite particle positions.")
        pose = self._container.data.root_link_pose_w.warp.numpy()[0]
        local = (points - pose[:3]) @ Rotation.from_quat(pose[3:]).as_matrix()
        in_pot_bounds = np.all((local >= self._lower) & (local <= self._upper), axis=1)
        radius = np.linalg.norm(points[:, :2] - BOWL_BASE_POS[:2], axis=1)
        escaped = (
            (points[:, 2] < BOWL_WORLD_TOP_Z - PARTICLE_RADIUS)
            & (radius > BOWL_SCALE * BOWL_OUTER_TOP_RADIUS + 2.0 * PARTICLE_RADIUS)
            & ~in_pot_bounds
        )
        self._spilled |= escaped
        spilled = int(np.count_nonzero(self._spilled))
        if spilled and self.metrics["first_spill_time_s"] is None:
            self.metrics["first_spill_time_s"] = sim_time
        self.metrics["spilled_particles"] = spilled
        height = points[:, 2] - BOWL_BASE_POS[2]
        fraction = np.clip(
            (height / BOWL_SCALE - BOWL_BOTTOM_THICKNESS) / (BOWL_LOCAL_Z_MAX - BOWL_BOTTOM_THICKNESS), 0.0, 1.0
        )
        inner_radius = BOWL_SCALE * (
            BOWL_INNER_BOTTOM_RADIUS + fraction * (BOWL_INNER_TOP_RADIUS - BOWL_INNER_BOTTOM_RADIUS)
        )
        in_bowl = (
            (height >= BOWL_SCALE * BOWL_BOTTOM_THICKNESS - PARTICLE_SURFACE_CLEARANCE)
            & (height <= BOWL_SCALE * BOWL_LOCAL_Z_MAX + PARTICLE_SURFACE_CLEARANCE)
            & (radius <= inner_radius + PARTICLE_SURFACE_CLEARANCE)
        )
        self.metrics["bowl_particles"] = int(np.count_nonzero(in_bowl))
        self.metrics["bowl_volume_ml"] = self.metrics["bowl_particles"] * FILL_SPACING**3 * 1.0e6
        if sim_time >= self._next_sample:
            self.metrics["samples"].append(
                {
                    "time_s": sim_time,
                    "bowl_volume_ml": self.metrics["bowl_volume_ml"],
                    "spilled_particles": spilled,
                    "teapot_position_w_m": pose[:3].tolist(),
                }
            )
            self._next_sample += 1.0
            logger.info(
                "Water at %.1fs: %.1f ml in bowl; %d escaped particles.",
                sim_time,
                self.metrics["bowl_volume_ml"],
                spilled,
            )

    def validate_complete(self) -> None:
        """Require no escaped particles and a visible dose after complete playback."""
        if self.metrics["spilled_particles"]:
            raise RuntimeError(f"Pour spilled {self.metrics['spilled_particles']} particles outside the bowl and pot.")
        if self.metrics["bowl_particles"] < math.ceil(0.01 * self._spilled.size):
            raise RuntimeError("Pour delivered less than 1% of the initial particle payload into the bowl.")


def update_pour_presentation(
    sim: SimulationContext,
    container_pose_w: np.ndarray,
    surface_renderer: FluidSurfaceRenderer,
    sim_time: float,
    *,
    camera_tracker: PourCameraTracker,
) -> int:
    """Frame measured pot motion and display either particles or reconstructed water."""
    from teapot_presentation import pour_closeup_view

    eye, target, show_surface = pour_closeup_view(
        sim_time,
        (container_pose_w[:3], container_pose_w[3:]),
        spout_local_pos=SPOUT_LOCAL_POS,
        bowl_top_pos_w=(*BOWL_BASE_POS[:2], BOWL_WORLD_TOP_Z),
        robot_base_pos_w=ROBOT_BASE_POS,
        rise_start_time=POUR_PREFIX_TIME + HOLD_TIME + TILT_TIME,
    )
    eye, target = camera_tracker.update(sim_time, eye, target)
    sim.set_camera_view(eye=eye, target=target)
    for visualizer in sim.visualizers:
        if visualizer.cfg.visualizer_type in ("newton_gl", "newton_rtx"):
            visualizer.set_particle_visibility(not show_surface)
    return surface_renderer.update(visible=show_surface)


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    surface_renderer: FluidSurfaceRenderer | None,
    robot_motion: TeapotPourMotion | None = None,
) -> None:
    """Run the scripted teapot-fill MPM loop."""
    from teapot_presentation import PourCameraTracker

    sim_dt = sim.get_physics_dt()
    camera_tracker = PourCameraTracker()
    container = scene["container"]
    count = 0
    writer = None
    video_visualizer = next((v for v in sim.visualizers if v.cfg.visualizer_type in ("newton_gl", "newton_rtx")), None)
    last_render_frame = -1
    started = None
    fluid_monitor = FluidContainmentMonitor(scene["fluid"], container) if args_cli.motion_report else None
    try:
        while keep_running(sim, count):
            if args_cli.video and count % SIMULATION_HZ == 0:
                logger.info("Recording %.0f/%.1f simulated seconds.", count * sim_dt, SEQUENCE_DURATION)
            if args_cli.benchmark and count == args_cli.benchmark_warmup:
                if torch.device(sim.device).type == "cuda":
                    torch.cuda.synchronize(sim.device)
                started = time.perf_counter()
            if robot_motion is None:
                write_container_state(container, count / SIMULATION_HZ)
            else:
                robot_motion.update(count / SIMULATION_HZ)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            if robot_motion is not None and args_cli.motion_report:
                robot_motion.observe_contacts()
            if args_cli.motion_report and count % max(1, round(SIMULATION_HZ / 10)) == 0:
                robot_motion.check((count + 1) / SIMULATION_HZ, container_pose_at_time)
                fluid_monitor.check((count + 1) / SIMULATION_HZ)
            render_frame = count * (args_cli.video_fps if args_cli.video else 60) // SIMULATION_HZ
            render_due = (
                render_frame > last_render_frame
                if args_cli.video or args_cli.presentation == "pour_closeups"
                else count % RENDER_INTERVAL == 0
            )
            if (sim.is_rendering or args_cli.video) and render_due:
                if args_cli.presentation == "pour_closeups":
                    update_pour_presentation(
                        sim,
                        container.data.root_link_pose_w.warp.numpy()[0],
                        surface_renderer,
                        count / SIMULATION_HZ,
                        camera_tracker=camera_tracker,
                    )
                elif args_cli.video and PICKUP_ENABLED:
                    from teapot_presentation import pickup_video_view

                    sim_time = count / SIMULATION_HZ
                    eye, target = pickup_video_view(
                        sim_time,
                        container_pose_at_time(sim_time)[0][2],
                        container_base_height_w=CONTAINER_BASE_POS[2],
                        manipulation_offset_x=MANIPULATION_OFFSET_X,
                        recovery_start_time=POUR_PREFIX_TIME + HOLD_TIME + TILT_TIME + CONTAINER_LIFT_TIME,
                        recovery_time=RECOVERY_TIME,
                    )
                    sim.set_camera_view(eye=eye, target=target)
                if surface_renderer is not None and args_cli.presentation != "pour_closeups":
                    surface_renderer.update()
                sim.render()
                if args_cli.video:
                    frame = video_visualizer.render_rgb_array()
                    if frame is None:
                        raise RuntimeError("Newton viewer did not produce a video frame.")
                    if writer is None:
                        import imageio_ffmpeg

                        path = Path(args_cli.video).expanduser()
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writer = imageio_ffmpeg.write_frames(
                            str(path),
                            (frame.shape[1], frame.shape[0]),
                            fps=args_cli.video_fps,
                            codec="libx264",
                            quality=8,
                            macro_block_size=2,
                            output_params=["-movflags", "+faststart"],
                        )
                        writer.send(None)
                    writer.send(np.ascontiguousarray(frame))
                last_render_frame = render_frame
            count += 1
        if started is not None:
            if torch.device(sim.device).type == "cuda":
                torch.cuda.synchronize(sim.device)
            elapsed = time.perf_counter() - started
            measured = count - args_cli.benchmark_warmup
            report = {
                "robot": args_cli.robot,
                "fluid_coupling": args_cli.fluid_coupling if args_cli.robot != "none" else "scripted",
                "sequence": args_cli.sequence,
                "sequence_duration_s": SEQUENCE_DURATION,
                "pour_rise_height_m": CONTAINER_LIFT_HEIGHT,
                "pour_rise_time_s": CONTAINER_LIFT_TIME,
                "device": torch.cuda.get_device_name(sim.device) if torch.device(sim.device).type == "cuda" else "cpu",
                "particles": particle_count(scene),
                "voxel_size": VOXEL_SIZE,
                "physics_hz": SIMULATION_HZ,
                "physics_substeps": MPM_SUBSTEPS,
                "fluid_hz": SIMULATION_HZ * MPM_SUBSTEPS,
                "rigid_hz": (
                    SIMULATION_HZ * MPM_SUBSTEPS * args_cli.rigid_substeps if args_cli.robot != "none" else None
                ),
                "controller_hz": args_cli.controller_hz if args_cli.robot != "none" else None,
                "warmup_steps": args_cli.benchmark_warmup,
                "measured_steps": measured,
                "elapsed_seconds": elapsed,
                "mean_step_ms": 1000.0 * elapsed / measured,
                "steps_per_second": measured / elapsed,
                "realtime_factor": measured * sim_dt / elapsed,
            }
            path = Path(args_cli.benchmark).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            logger.info("Benchmark: %.3f ms/step (%s).", report["mean_step_ms"], path)
        if args_cli.motion_report:
            robot_motion.check(count / SIMULATION_HZ, container_pose_at_time)
            fluid_monitor.check(count / SIMULATION_HZ)
            robot_motion.metrics.update(
                commanded_pour_angle_deg=math.degrees(POUR_ANGLE),
                commanded_teapot_mass_kg=args_cli.teapot_mass,
                commanded_upper_pour_angle_deg=math.degrees(UPPER_POUR_ANGLE),
                commanded_pour_taper_start_fraction=POUR_TAPER_START_FRACTION,
                commanded_pour_taper_end_fraction=POUR_TAPER_END_FRACTION,
                commanded_pour_tilt_time_s=TILT_TIME,
                commanded_pour_aim_offset_x_m=args_cli.pour_aim_offset_x,
                commanded_pour_rise_height_m=CONTAINER_LIFT_HEIGHT,
                commanded_pour_rise_time_s=CONTAINER_LIFT_TIME,
                robot_base_position_w_m=ROBOT_BASE_POS,
                sequence_duration_s=SEQUENCE_DURATION,
                physics_hz=SIMULATION_HZ,
                physics_substeps=MPM_SUBSTEPS,
                rigid_substeps=args_cli.rigid_substeps,
                controller_hz=args_cli.controller_hz,
                interpolate_arm_targets=(
                    args_cli.fluid_coupling == "one_way" and (args_cli.rigid_substeps > 1 or MPM_SUBSTEPS > 1)
                ),
                manipulation_offset_x_m=MANIPULATION_OFFSET_X,
                ground_type="isaaclab_default_ground_plane",
            )
            robot_motion.metrics["fluid_containment"] = fluid_monitor.metrics
            path = Path(args_cli.motion_report).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(robot_motion.metrics, indent=2) + "\n")
            if count / SIMULATION_HZ >= SEQUENCE_DURATION - sim_dt:
                fluid_monitor.validate_complete()
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    """Set up and run the Isaac Lab Newton MPM teapot-fill demo."""
    logging.basicConfig(format="[%(levelname)s]: %(message)s")
    if args_cli.robot == "rizon_sharpa":
        from rizon_sharpa_teapot import configured_asset_path

        configured_asset_path()
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        if "kit" in (args_cli.visualizer or []):
            from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

            apply_isaac_rtx_global_settings(
                IsaacRtxRendererGlobalSettingsCfg(enable_translucency=True),
            )

        # Resolve after launching so Kit runs never import USD modules before
        # Kit starts; Newton-only runs still use standalone omni.client.
        container_usd = retrieve_file_path(args_cli.container_usd)
        if {"kit", "newton_rtx"}.intersection(args_cli.visualizer or []):
            island_usd = retrieve_optional_visual_asset(args_cli.island_usd, "kitchen island")
            bowl_usd = retrieve_optional_visual_asset(args_cli.bowl_usd, "catch bowl")
        else:
            island_usd = bowl_usd = None

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        if args_cli.video:
            sim.require_visual_shapes()
        scene = InteractiveScene(create_scene_cfg(container_usd, island_usd, bowl_usd))
        sim.reset()
        robot_motion = None
        if args_cli.robot == "rizon_sharpa":
            from rizon_sharpa_teapot import TeapotPourMotion

            robot = scene["robot"]
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.warp)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.warp)
            sim.forward()
            robot_motion = TeapotPourMotion(
                scene["robot"],
                scene["container"],
                container_pose_at_time,
                sim_dt=sim.get_physics_dt(),
                controller_hz=args_cli.controller_hz,
                duration=SEQUENCE_DURATION,
                interpolate_arm_targets=(
                    args_cli.fluid_coupling == "one_way" and (args_cli.rigid_substeps > 1 or MPM_SUBSTEPS > 1)
                ),
                approach_offset_at_time=approach_offset_at_time if PICKUP_ENABLED else None,
                grasp_time=PICKUP_TIME,
                pickup_end_time=POUR_PREFIX_TIME,
                close_interval=(GRASP_CLOSE_START, GRASP_CLOSE_END),
                fluid_coupling=args_cli.fluid_coupling,
                grasp_finger=args_cli.grasp_finger,
            )
            robot.write_joint_position_to_sim_index(position=robot_motion.initial_joint_positions)
            sim.forward()
        sim.set_camera_view(
            eye=(0.55, -1.5, 1.72) if robot_motion is not None else CAMERA_EYE,
            target=(-0.28, 0.0, 1.22) if robot_motion is not None else CAMERA_TARGET,
        )
        surface_renderer = (
            FluidSurfaceRenderer(sim)
            if SHOW_FLUID_SURFACE and any(v in (args_cli.visualizer or []) for v in ("newton_gl", "newton_rtx"))
            else None
        )
        surface_triangle_count = (
            surface_renderer.update(visible=args_cli.presentation != "pour_closeups")
            if surface_renderer is not None
            else 0
        )

        logger.info(
            "Isaac Lab Newton teapot-fill MPM demo ready."
            " Sampled %d MPM particles inside the teapot;"
            " extracted %d surface triangles; rendering %s; fill spacing %.4g m;"
            " the teapot will tilt after %.2fs.",
            particle_count(scene),
            surface_triangle_count,
            args_cli.fluid_render_mode,
            FILL_SPACING,
            POUR_PREFIX_TIME + HOLD_TIME,
        )
        run_simulator(sim, scene, surface_renderer, robot_motion)


if __name__ == "__main__":
    main()
