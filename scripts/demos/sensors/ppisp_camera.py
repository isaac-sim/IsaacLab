# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates USD-authored PPISP on a Gaussian scene through the Isaac Lab camera sensor with
the Newton Warp or Isaac RTX renderer.

.. code-block:: bash

    # Run a finite smoke with the default Newton Warp renderer and save comparison images.
    ./isaaclab.sh -p scripts/demos/sensors/ppisp_camera.py \
        --input_scene /path/to/scene.usd --renderer newton_renderer --visualizer none --max_steps 60

    # Run the same saved-image workflow with Isaac RTX.
    ./isaaclab.sh -p scripts/demos/sensors/ppisp_camera.py \
        --input_scene /path/to/scene.usd --renderer isaac_rtx --visualizer none --max_steps 60

    # Follow xform time samples on the selected camera or its parent rig at 30 simulation/render FPS.
    ./isaaclab.sh -p scripts/demos/sensors/ppisp_camera.py \
        --input_scene /path/to/scene.usd --renderer isaac_rtx \
        --fps 30 --write_fps 10 --visualizer none

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import time
from typing import Any

from isaaclab.app import AppLauncher
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# add argparse arguments
DEFAULT_INPUT_SCENE = f"{ISAAC_NUCLEUS_DIR}/Samples/Scene_ParticleField/valiant_auto.usdz"

parser = argparse.ArgumentParser(description="Example of a USD-authored PPISP effect on camera RGB output.")
parser.add_argument(
    "--input_scene",
    type=str,
    default=DEFAULT_INPUT_SCENE,
    help="USD or USDZ scene containing the Gaussian PPISP setup.",
)
parser.add_argument(
    "--camera_prim_path",
    type=str,
    default=None,
    help="Optional camera prim path override. Omit to auto-select the first camera with PPISP attributes.",
)
parser.add_argument(
    "--camera_time_code",
    type=float,
    default=0.0,
    help="USD time code used for a static camera when no xform trajectory is authored.",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="Simulation and camera-render FPS; also used to resample the USD camera trajectory.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of duplicated input-scene envs to render in the tiled camera batch.",
)
parser.add_argument("--env_spacing", type=float, default=20.0, help="Spacing between duplicated input-scene envs.")
parser.add_argument("--image_width", type=int, default=320, help="Output image width.")
parser.add_argument(
    "--image_height",
    type=int,
    default=None,
    help="Output image height. Defaults to preserving the selected USD RenderProduct aspect ratio.",
)
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument(
    "--renderer",
    type=str,
    choices=["newton_renderer", "isaac_rtx"],
    default="newton_renderer",
    help="Camera renderer backend to use. Newton Warp is the default for this PPISP smoke.",
)
parser.add_argument(
    "--render_only",
    action="store_true",
    help="Render and save only the PPISP camera output; skip baseline, comparison, and diff outputs.",
)
isaacrtx_settings_group = parser.add_argument_group("Isaac RTX Gaussian settings")
isaacrtx_settings_group.add_argument(
    "--isaacrtx_gaussian_max_intersections",
    type=int,
    default=None,
    help=(
        "Maximum number of Gaussian intersections evaluated along each ray before traversal stops; lower values "
        "reduce RTX work but may omit farther Gaussian contributions, while -1 means unlimited."
    ),
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_gaussian_self_shadow_distance",
    type=float,
    default=None,
    help="RTX Gaussian self-shadow distance in Gaussian-radius units; 0 disables the filter.",
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_enable_accumulation",
    action="store_true",
    help="Enable RTX ray-tracing accumulation.",
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_accumulation_limit",
    type=int,
    default=None,
    help="Maximum RTX ray-tracing accumulation iterations/frames.",
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_gaussian_depth_all_hits",
    action="store_true",
    help=(
        "Use all hits when accumulating RTPT Gaussian depth only; this does not affect Gaussian color/albedo "
        "or general RTX accumulation."
    ),
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_gaussian_accumulated_albedo",
    action="store_true",
    help="Accumulate Gaussian SH0 color as albedo in RTPT.",
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_gaussian_skip_tonemapping",
    action="store_true",
    help="Skip tonemapping for Gaussian pixels in RTPT.",
)
isaacrtx_settings_group.add_argument(
    "--isaacrtx_render_mode",
    choices=["RayTracedLighting", "RealTimePathTracing", "PathTracing"],
    default=None,
    help="Isaac RTX render mode; RTPT-only Gaussian options require RealTimePathTracing.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=None,
    help="Simulation/render steps to run before saving images. Defaults to 32 for Isaac RTX and 0 for Newton.",
)
parser.add_argument(
    "--write_fps",
    type=float,
    default=None,
    help="Output image writing rate in FPS. Defaults to --fps.",
)
parser.add_argument(
    "--ppisp_responsivity",
    type=float,
    default=None,
    help="Override the USD-authored PPISP responsivity. If omitted, the scene-authored value is used.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write comparison images. Defaults to scripts/demos/sensors/output/ppisp_camera.",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Profile simulation, camera updates, GPU transfers, and disk writes.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# enable cameras by default; the QA workflow validates saved camera outputs and does not require a visualizer.
parser.set_defaults(enable_cameras=True)
# parse the arguments
args_cli = parser.parse_args()
if "://" not in args_cli.input_scene:
    args_cli.input_scene = os.path.abspath(os.path.expanduser(args_cli.input_scene))
    if not os.path.exists(args_cli.input_scene):
        parser.error(f"--input_scene does not exist: {args_cli.input_scene}")
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.image_width < 1:
    parser.error("--image_width must be at least 1.")
if args_cli.image_height is not None and args_cli.image_height < 1:
    parser.error("--image_height must be at least 1.")
if args_cli.warmup_steps is None:
    args_cli.warmup_steps = 32 if args_cli.renderer == "isaac_rtx" else 0
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative.")
if args_cli.fps <= 0.0:
    parser.error("--fps must be positive.")
if args_cli.write_fps is None:
    args_cli.write_fps = args_cli.fps
if args_cli.write_fps <= 0.0:
    parser.error("--write_fps must be positive.")
if args_cli.isaacrtx_gaussian_max_intersections is not None and args_cli.isaacrtx_gaussian_max_intersections < -1:
    parser.error("--isaacrtx_gaussian_max_intersections must be -1 or non-negative.")
if (
    args_cli.isaacrtx_gaussian_self_shadow_distance is not None
    and args_cli.isaacrtx_gaussian_self_shadow_distance < 0.0
):
    parser.error("--isaacrtx_gaussian_self_shadow_distance must be non-negative.")
if args_cli.isaacrtx_accumulation_limit is not None and args_cli.isaacrtx_accumulation_limit < 1:
    parser.error("--isaacrtx_accumulation_limit must be positive.")
isaacrtx_settings_requested = any(
    [
        args_cli.isaacrtx_gaussian_max_intersections is not None,
        args_cli.isaacrtx_gaussian_self_shadow_distance is not None,
        args_cli.isaacrtx_enable_accumulation,
        args_cli.isaacrtx_accumulation_limit is not None,
        args_cli.isaacrtx_gaussian_depth_all_hits,
        args_cli.isaacrtx_gaussian_accumulated_albedo,
        args_cli.isaacrtx_gaussian_skip_tonemapping,
        args_cli.isaacrtx_render_mode is not None,
    ]
)
if isaacrtx_settings_requested and args_cli.renderer != "isaac_rtx":
    parser.error("Isaac RTX settings require --renderer isaac_rtx.")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from isaaclab_ppisp._demo_utils import (
    find_ppisp_camera_bindings,
    format_available_ppisp_cameras,
    order_ppisp_bindings_by_camera,
)
from isaaclab_ppisp.cfg import PpispCfg, ppisp_cfg_from_usd_camera

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.configclass import configclass


@configclass
class PpispCameraSceneCfg(InteractiveSceneCfg):
    """Minimal scene cfg that references the input USD under each env."""

    env_spacing: float = 20.0

    input_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )

    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


def make_renderer_cfg() -> Any:
    """Create the selected camera renderer cfg."""
    if args_cli.renderer == "newton_renderer":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg()
    else:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg()


def make_sim_cfg() -> sim_utils.SimulationCfg:
    """Create the simulation cfg matching the selected renderer."""
    physics_cfg = None
    if args_cli.renderer == "newton_renderer":
        from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

        physics_cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)

    return sim_utils.SimulationCfg(
        dt=1.0 / args_cli.fps,
        device=args_cli.device,
        physics=physics_cfg,
        use_fabric=not args_cli.disable_fabric,
    )


def log_current_isaacrtx_settings(prefix: str = "[INFO] Verified settings") -> None:
    """Log the current Isaac RTX settings after USD and renderer initialization."""
    from isaaclab.app.settings_manager import get_settings_manager

    settings = get_settings_manager()
    if args_cli.renderer != "isaac_rtx":
        return
    aa_op = settings.get("/rtx/post/aa/op")
    aa_op_names = {0: "None", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "RTXAA"}
    aa_op_name = aa_op_names.get(aa_op, "unknown")
    dlss_denoiser_enabled = settings.get("/rtx-transient/dldenoiser/enabled")
    registered_invert_color_correction = settings.get("/rtx/post/registeredCompositing/invertColorCorrection")
    print(
        f"{prefix}: "
        f"renderMode={settings.get('/rtx/rendermode')!r}, "
        f"aaOp={aa_op_name}({aa_op!r}), "
        f"dlssDenoiserEnabled={dlss_denoiser_enabled!r}, "
        f"dlssHistoryCandidate={aa_op in (3, 4) and dlss_denoiser_enabled!r}, "
        f"pathTracingDlssEnabled={settings.get('/rtx/pathtracing/dlss/enabled')!r}, "
        f"rt2Enabled={settings.get('/rtx-transient/rt2Enabled')!r}, "
        f"gaussianEnabled={settings.get('/rtx/geometry/gaussian/enabled')!r}, "
        f"maxIntersections={settings.get('/rtx/raytracing/gaussian/maxIntersections')!r}, "
        f"selfShadowDistance={settings.get('/rtx/raytracing/gaussian/selfShadowDistance')!r}, "
        f"accumulationEnabled={settings.get('/rtx/raytracing/enableAccumulation')!r}, "
        f"accumulationLimit={settings.get('/rtx/raytracing/accumulationLimit')!r}, "
        f"gaussianDepthAllHits={settings.get('/rtx/rtpt/gaussian/accumulatedDepth/allHits/enabled')!r}, "
        f"gaussianAccumulatedAlbedo={settings.get('/rtx/rtpt/gaussian/accumulatedAlbedo/enabled')!r}, "
        f"gaussianSkipTonemapping={settings.get('/rtx/rtpt/gaussian/skipTonemapping/enabled')!r}, "
        f"disableNuRecPostProcessings={settings.get('/omni/rtx/nre/compositing/disableNuRecPostProcessings')!r}, "
        f"nurecCompositingLogLevel={settings.get('/omni/rtx/nre/compositing/logLevel')!r}, "
        f"registeredCompositingInvertColorCorrection={registered_invert_color_correction!r}, "
        f"registeredCompositingInvertToneMap={settings.get('/rtx/post/registeredCompositing/invertToneMap')!r}",
        flush=True,
    )


def apply_rtx_settings() -> None:
    """Apply the default and optional Isaac RTX settings requested on the command line."""
    from isaaclab.app.settings_manager import get_settings_manager

    if args_cli.renderer != "isaac_rtx":
        return

    settings = get_settings_manager()
    settings.set_bool("/omni/rtx/nre/compositing/disableNuRecPostProcessings", True)
    settings.set_int("/omni/rtx/nre/compositing/logLevel", 4)
    settings.set_int("/omni/rtx/nre/compositing/rendererHints", 0)
    settings.set_bool("/rtx/post/registeredCompositing/invertColorCorrection", False)
    settings.set_bool("/rtx/post/registeredCompositing/invertToneMap", False)
    applied = [
        "disableNuRecPostProcessings=true",
        "nreCompositingLogLevel=4",
        "nreCompositingRendererHints=0",
        "registeredCompositingInvertColorCorrection=false",
        "registeredCompositingInvertToneMap=false",
    ]

    if args_cli.isaacrtx_gaussian_max_intersections is not None:
        settings.set_int("/rtx/raytracing/gaussian/maxIntersections", args_cli.isaacrtx_gaussian_max_intersections)
        applied.append(f"maxIntersections={args_cli.isaacrtx_gaussian_max_intersections}")
    if args_cli.isaacrtx_gaussian_self_shadow_distance is not None:
        settings.set_float(
            "/rtx/raytracing/gaussian/selfShadowDistance", args_cli.isaacrtx_gaussian_self_shadow_distance
        )
        applied.append(f"selfShadowDistance={args_cli.isaacrtx_gaussian_self_shadow_distance:g}")
    if args_cli.isaacrtx_enable_accumulation:
        settings.set_bool("/rtx/raytracing/enableAccumulation", True)
        applied.append("enableAccumulation=true")
    if args_cli.isaacrtx_accumulation_limit is not None:
        settings.set_int("/rtx/raytracing/accumulationLimit", args_cli.isaacrtx_accumulation_limit)
        applied.append(f"accumulationLimit={args_cli.isaacrtx_accumulation_limit}")
    if args_cli.isaacrtx_gaussian_depth_all_hits:
        settings.set_bool("/rtx/rtpt/gaussian/accumulatedDepth/allHits/enabled", True)
        applied.append("gaussianDepthAllHits=true")
    if args_cli.isaacrtx_gaussian_accumulated_albedo:
        settings.set_bool("/rtx/rtpt/gaussian/accumulatedAlbedo/enabled", True)
        applied.append("gaussianAccumulatedAlbedo=true")
    if args_cli.isaacrtx_gaussian_skip_tonemapping:
        settings.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", True)
        applied.append("gaussianSkipTonemapping=true")
    if args_cli.isaacrtx_render_mode is not None:
        settings.set_string("/rtx/rendermode", args_cli.isaacrtx_render_mode)
        applied.append(f"renderMode={args_cli.isaacrtx_render_mode}")

    if applied:
        print("[INFO] Applied RTX/Gaussian settings: " + ", ".join(applied), flush=True)


def resolve_source_camera_binding(source_stage: Usd.Stage) -> tuple[str, Usd.Prim | None, Usd.Prim]:
    """Resolve the source camera and PPISP camera binding from CLI or source stage metadata."""
    ppisp_bindings = order_ppisp_bindings_by_camera(source_stage, find_ppisp_camera_bindings(source_stage))
    if not ppisp_bindings:
        raise RuntimeError("No cameras with PPISP camera attributes found in input scene.")

    if args_cli.camera_prim_path is not None:
        camera_prim_path = args_cli.camera_prim_path
        if not camera_prim_path.startswith("/"):
            camera_prim_path = f"/{camera_prim_path}"
    else:
        camera_prim_path = ppisp_bindings[0][0]
        print(f"[INFO] Auto-selected camera prim: {camera_prim_path}", flush=True)

    camera_prim = source_stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        available = format_available_ppisp_cameras(ppisp_bindings)
        raise RuntimeError(
            f"Camera prim not found: {camera_prim_path}\n"
            "Omit --camera_prim_path to auto-select a camera with PPISP attributes, or use one of:\n"
            f"  {available}"
        )
    if camera_prim.GetTypeName() != "Camera":
        available = format_available_ppisp_cameras(ppisp_bindings)
        raise RuntimeError(
            f"Prim is not a Camera: {camera_prim_path} ({camera_prim.GetTypeName()})\n"
            "Omit --camera_prim_path to auto-select a camera with PPISP attributes, or use one of:\n"
            f"  {available}"
        )

    for binding in ppisp_bindings:
        if binding[0] == camera_prim_path:
            return binding

    available = format_available_ppisp_cameras(ppisp_bindings)
    raise RuntimeError(
        f"Selected camera has no PPISP camera attributes: {camera_prim_path}\n"
        "Omit --camera_prim_path to auto-select a camera with PPISP attributes, or use one of:\n"
        f"  {available}"
    )


def source_camera_path_to_default_rel_path(source_stage: Usd.Stage, source_camera_prim_path: str) -> str:
    """Return the source camera path relative to the source defaultPrim."""
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Input scene must have a defaultPrim so it can be referenced under each env.")

    default_prim_path = default_prim.GetPath().pathString
    default_prefix = f"{default_prim_path}/"
    if not source_camera_prim_path.startswith(default_prefix):
        raise RuntimeError(
            f"Camera path {source_camera_prim_path} is not under source defaultPrim {default_prim_path}."
        )
    return source_camera_prim_path[len(default_prefix) :]


def source_camera_path_to_env_regex(source_stage: Usd.Stage, source_camera_prim_path: str) -> str:
    """Map a source camera path to the duplicated-env camera regex."""
    camera_rel_path = source_camera_path_to_default_rel_path(source_stage, source_camera_prim_path)
    return f"/World/envs/env_.*/Scene/{camera_rel_path}"


def get_trajectory_time_samples(source_stage: Usd.Stage, source_camera_prim_path: str) -> list[float]:
    """Return uniformly spaced USD times spanning the camera and parent-rig xform samples."""
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Input scene must have a defaultPrim so it can be referenced under each env.")

    prim = source_stage.GetPrimAtPath(source_camera_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {source_camera_prim_path}")

    time_samples = set()
    while prim and prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        for xform_op in xformable.GetOrderedXformOps():
            time_samples.update(float(value) for value in xform_op.GetAttr().GetTimeSamples())
        if prim == default_prim:
            break
        prim = prim.GetParent()

    authored_times = sorted(time_samples)
    if not authored_times:
        return []
    start_time = authored_times[0]
    end_time = authored_times[-1]
    time_codes_per_second = source_stage.GetTimeCodesPerSecond()
    if time_codes_per_second <= 0.0:
        time_codes_per_second = 24.0
    time_step = time_codes_per_second / args_cli.fps
    sample_count = int(np.floor((end_time - start_time) / time_step)) + 1
    trajectory_times = [start_time + index * time_step for index in range(sample_count)]
    if trajectory_times[-1] < end_time:
        trajectory_times.append(end_time)
    return trajectory_times


def bake_source_camera_pose_to_envs(
    source_stage: Usd.Stage, source_camera_prim_path: str, time_code: float | None = None, *, log: bool = True
) -> None:
    """Bake a USD camera pose at ``time_code`` into duplicated env camera prims."""
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Input scene must have a defaultPrim so it can be referenced under each env.")

    source_camera_prim = source_stage.GetPrimAtPath(source_camera_prim_path)
    if not source_camera_prim or not source_camera_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {source_camera_prim_path}")

    if time_code is None:
        time_code = args_cli.camera_time_code
    source_cache = UsdGeom.XformCache(Usd.TimeCode(time_code))
    source_default_world = source_cache.GetLocalToWorldTransform(default_prim)
    source_camera_world = source_cache.GetLocalToWorldTransform(source_camera_prim)
    source_camera_in_default = source_camera_world * source_default_world.GetInverse()

    stage = sim_utils.get_current_stage()
    target_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    camera_rel_path = source_camera_path_to_default_rel_path(source_stage, source_camera_prim_path)
    authored_count = 0
    for env_id in range(args_cli.num_envs):
        scene_path = f"/World/envs/env_{env_id}/Scene"
        target_camera_path = f"{scene_path}/{camera_rel_path}"
        scene_prim = stage.GetPrimAtPath(scene_path)
        target_camera_prim = stage.GetPrimAtPath(target_camera_path)
        if not scene_prim or not scene_prim.IsValid():
            raise RuntimeError(f"Duplicated scene prim not found: {scene_path}")
        if not target_camera_prim or not target_camera_prim.IsValid():
            raise RuntimeError(f"Duplicated camera prim not found: {target_camera_path}")

        target_scene_world = target_cache.GetLocalToWorldTransform(scene_prim)
        target_parent_world = target_cache.GetLocalToWorldTransform(target_camera_prim.GetParent())
        target_camera_world = source_camera_in_default * target_scene_world
        target_camera_local = target_camera_world * target_parent_world.GetInverse()
        target_camera_local.Orthonormalize()

        xformable = UsdGeom.Xformable(target_camera_prim)
        xformable.ClearXformOpOrder()
        xform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ppispCameraPose")
        xform_op.Set(target_camera_local, Usd.TimeCode.Default())
        xformable.SetXformOpOrder([xform_op])
        authored_count += 1

    if log:
        print(
            f"[INFO] Baked camera pose at USD time {time_code:g} into {authored_count} env camera(s).",
            flush=True,
        )


def get_render_product_resolution(render_product_prim: Usd.Prim | None) -> tuple[int, int] | None:
    """Return ``(width, height)`` from a RenderProduct ``resolution`` attribute."""
    if render_product_prim is None:
        return None
    resolution_attr = render_product_prim.GetAttribute("resolution")
    if not resolution_attr:
        return None
    resolution = resolution_attr.Get()
    if resolution is None or len(resolution) != 2:
        return None
    return int(resolution[0]), int(resolution[1])


def resolve_image_shape(render_product_prim: Usd.Prim | None) -> tuple[int, int]:
    """Resolve demo output ``(width, height)`` preserving source aspect when height is omitted."""
    width = args_cli.image_width
    height = args_cli.image_height
    if height is not None:
        return width, height

    source_resolution = get_render_product_resolution(render_product_prim)
    if source_resolution is None:
        return width, width

    source_width, source_height = source_resolution
    height = max(1, round(width * source_height / source_width))
    return width, height


def make_ppisp_cfg(camera_prim: Usd.Prim, num_ppisp_bindings: int) -> PpispCfg:
    """Parse the selected source PPISP camera into an explicit cfg for duplicated envs."""
    ppisp_cfg = ppisp_cfg_from_usd_camera(camera_prim)
    # The duplicated stage can remap source camera paths; keep the parsed inputs
    # as explicit values instead of resolving the original camera path later.
    ppisp_cfg.camera_prim_path = None
    if args_cli.ppisp_responsivity is None:
        print(f"[INFO] Using USD-authored PPISP values from {num_ppisp_bindings} PPISP camera(s).", flush=True)
    else:
        ppisp_cfg.inputs["responsivity"] = float(args_cli.ppisp_responsivity)
        print(
            f"[INFO] Applied PPISP responsivity={args_cli.ppisp_responsivity:g} to duplicated env PPISP cfg.",
            flush=True,
        )
    return ppisp_cfg


def create_duplicated_env_scene() -> InteractiveScene:
    """Create a production-style duplicated-env scene for tiled camera rendering."""
    scene_cfg = PpispCameraSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene_cfg.input_scene.spawn = sim_utils.UsdFileCfg(usd_path=args_cli.input_scene)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] Referenced input scene into {args_cli.num_envs} env(s).", flush=True)
    return scene


def make_camera(camera_prim_path: str, *, ppisp_cfg: PpispCfg | None, width: int, height: int) -> Camera:
    """Create a baseline or PPISP camera sensor for the duplicated-env camera batch."""
    return Camera(
        CameraCfg(
            prim_path=camera_prim_path,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=None,
            isp_cfg=ppisp_cfg,
            renderer_cfg=make_renderer_cfg(),
        )
    )


@wp.kernel
def compute_rgb_diff(
    baseline: wp.array(dtype=wp.uint8, ndim=4),
    ppisp: wp.array(dtype=wp.uint8, ndim=4),
    diff: wp.array(dtype=wp.float32, ndim=4),
) -> None:
    """Compute normalized RGB absolute difference on the camera array's device."""
    env_id, row, col, channel = wp.tid()
    diff[env_id, row, col, channel] = (
        wp.abs(wp.float32(ppisp[env_id, row, col, channel]) - wp.float32(baseline[env_id, row, col, channel])) / 255.0
    )


def save_images_grid(
    images: list[np.ndarray],
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
) -> None:
    """Save images in a grid with optional subtitles and title."""
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    for idx, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    plt.close()


def make_tiled_image(images: np.ndarray) -> np.ndarray:
    """Stack a camera batch vertically into one image."""
    return np.concatenate([image for image in images], axis=0)


def save_array_image(image: np.ndarray, filename: str) -> None:
    """Save an array image in [0, 1] without axes, titles, or layout scaling."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.imsave(filename, np.clip(image, 0.0, 1.0))


def corner_center_ratio(rgb: np.ndarray) -> float:
    """Return the average corner brightness divided by center brightness for one RGB image."""
    h, w = rgb.shape[:2]
    patch = max(4, min(h, w) // 8)
    cy, cx = h // 2 - patch // 2, w // 2 - patch // 2
    center = rgb[cy : cy + patch, cx : cx + patch, :3].astype(np.float32).mean()
    corners = np.stack(
        [
            rgb[:patch, :patch, :3].astype(np.float32).mean(),
            rgb[:patch, -patch:, :3].astype(np.float32).mean(),
            rgb[-patch:, :patch, :3].astype(np.float32).mean(),
            rgb[-patch:, -patch:, :3].astype(np.float32).mean(),
        ]
    ).mean()
    return float(corners / max(center, 1.0))


def profile_sync(device: str | None) -> None:
    """Synchronize a device only when profiling GPU work."""
    if args_cli.profile and device is not None and device.startswith("cuda"):
        wp.synchronize_device(device)


def profile_record(profile: dict[str, dict[str, float]], name: str, elapsed: float) -> None:
    """Accumulate profiling measurements in seconds."""
    if not args_cli.profile:
        return
    entry = profile.setdefault(name, {"count": 0.0, "total_seconds": 0.0, "max_seconds": 0.0})
    entry["count"] += 1.0
    entry["total_seconds"] += elapsed
    entry["max_seconds"] = max(entry["max_seconds"], elapsed)


def report_profile(profile: dict[str, dict[str, float]], output_dir: str, num_steps: int) -> None:
    """Print and save profiling results when profiling is enabled."""
    if not args_cli.profile:
        return
    results = {
        "steps": num_steps,
        "sections": {
            name: {
                **values,
                "average_seconds": values["total_seconds"] / values["count"] if values["count"] else 0.0,
            }
            for name, values in profile.items()
        },
    }
    print("[PROFILE] section                         total(s)   avg(ms)   max(ms)", flush=True)
    for name, values in results["sections"].items():
        print(
            f"[PROFILE] {name:<32} {values['total_seconds']:>8.3f} "
            f"{values['average_seconds'] * 1000.0:>9.3f} {values['max_seconds'] * 1000.0:>9.3f}",
            flush=True,
        )
    profile_path = os.path.join(output_dir, "profile.json")
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as profile_file:
        json.dump(results, profile_file, indent=2)
    print(f"[PROFILE] Saved profile to {profile_path}", flush=True)


def run_simulator(
    sim: sim_utils.SimulationContext,
    baseline_camera: Camera | None,
    ppisp_camera: Camera,
    source_stage: Usd.Stage,
    source_camera_prim_path: str,
    trajectory_times: list[float],
) -> None:
    """Run the simulator through the trajectory and periodically save rendered images."""
    sim_dt = sim.get_physics_dt()
    write_interval = max(1, int(round(1.0 / (args_cli.write_fps * sim_dt))))
    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ppisp_camera")
    os.makedirs(output_dir, exist_ok=True)
    comparison_dir = os.path.join(output_dir, "comparison")
    baseline_dir = os.path.join(output_dir, "baseline")
    ppisp_dir = os.path.join(output_dir, "ppisp")
    diff_dir = os.path.join(output_dir, "diff")

    if args_cli.warmup_steps > 0:
        print(f"[INFO] Running {args_cli.warmup_steps} warmup step(s) before saving images.", flush=True)
    bake_source_camera_pose_to_envs(source_stage, source_camera_prim_path, trajectory_times[0], log=False)
    for _ in range(args_cli.warmup_steps):
        sim.step()
        if baseline_camera is not None:
            baseline_camera.update(sim_dt, force_recompute=True)
        ppisp_camera.update(sim_dt, force_recompute=True)

    count = 0
    reported_shape = False
    profile: dict[str, dict[str, float]] = {}
    profile_device = str(ppisp_camera.data.output["rgb"].warp.device)
    reported_post_load_settings = False
    target_steps = max(len(trajectory_times), write_interval)
    next_write_step = write_interval
    print(
        f"[INFO] Simulation dt={sim_dt:g}s ({1.0 / sim_dt:g} FPS), "
        f"writing at {args_cli.write_fps:g} FPS (every {write_interval} render step(s)).",
        flush=True,
    )
    while simulation_app.is_running():
        step_start = time.perf_counter() if args_cli.profile else 0.0
        section_start = time.perf_counter() if args_cli.profile else 0.0
        trajectory_index = min(count, len(trajectory_times) - 1)
        bake_source_camera_pose_to_envs(
            source_stage, source_camera_prim_path, trajectory_times[trajectory_index], log=False
        )
        profile_sync(profile_device)
        profile_record(profile, "trajectory_bake", time.perf_counter() - section_start)

        profile_sync(profile_device)
        section_start = time.perf_counter() if args_cli.profile else 0.0
        sim.step()
        profile_sync(profile_device)
        profile_record(profile, "simulation_step", time.perf_counter() - section_start)

        if baseline_camera is not None:
            section_start = time.perf_counter() if args_cli.profile else 0.0
            baseline_camera.update(sim_dt, force_recompute=True)
            profile_sync(profile_device)
            profile_record(profile, "baseline_camera_update", time.perf_counter() - section_start)

        section_start = time.perf_counter() if args_cli.profile else 0.0
        ppisp_camera.update(sim_dt, force_recompute=True)
        profile_sync(profile_device)
        profile_record(profile, "ppisp_camera_update", time.perf_counter() - section_start)
        if not reported_post_load_settings:
            log_current_isaacrtx_settings("[INFO] Post-load verified settings")
            reported_post_load_settings = True
        count += 1

        is_final_step = count >= target_steps
        if count == next_write_step or is_final_step:
            ppisp_wp = ppisp_camera.data.output["rgb"].warp
            if args_cli.render_only:
                section_start = time.perf_counter() if args_cli.profile else 0.0
                ppisp = ppisp_wp.numpy()
                profile_record(profile, "gpu_to_cpu_transfer", time.perf_counter() - section_start)
                if not reported_shape:
                    print(f"[INFO] camera batch rgb shape={tuple(ppisp.shape)}", flush=True)
                    reported_shape = True
                per_env_ppisp_mean = ppisp.astype(np.float32).mean(axis=(1, 2, 3))
                print(
                    f"[INFO] step={count} mean_ppisp="
                    + ", ".join(f"{value:.2f}" for value in per_env_ppisp_mean.tolist()),
                    flush=True,
                )
                section_start = time.perf_counter() if args_cli.profile else 0.0
                save_array_image(
                    make_tiled_image(ppisp / 255.0),
                    os.path.join(ppisp_dir, f"ppisp_camera_{count:06d}.png"),
                )
                profile_record(profile, "disk_writes", time.perf_counter() - section_start)
                profile_record(profile, "total_step", time.perf_counter() - step_start)
                if count >= target_steps:
                    break
                next_write_step += write_interval
                continue

            baseline_wp = baseline_camera.data.output["rgb"].warp
            section_start = time.perf_counter() if args_cli.profile else 0.0
            diff_wp = wp.empty(ppisp_wp.shape, dtype=wp.float32, device=ppisp_wp.device)
            wp.launch(
                compute_rgb_diff,
                dim=ppisp_wp.shape,
                inputs=[baseline_wp, ppisp_wp, diff_wp],
                device=ppisp_wp.device,
            )
            profile_sync(profile_device)
            profile_record(profile, "warp_difference", time.perf_counter() - section_start)

            # Transfer image buffers to NumPy only at the disk-writing boundary.
            section_start = time.perf_counter() if args_cli.profile else 0.0
            baseline = baseline_wp.numpy()
            ppisp = ppisp_wp.numpy()
            diff = diff_wp.numpy()
            profile_record(profile, "gpu_to_cpu_transfer", time.perf_counter() - section_start)
            if not reported_shape:
                print(f"[INFO] camera batch rgb shape={tuple(ppisp.shape)}", flush=True)
                reported_shape = True
            mean_abs_delta = float(diff.mean()) * 255.0
            ratios = [corner_center_ratio(ppisp[i]) for i in range(ppisp.shape[0])]
            ratio = sum(ratios) / len(ratios)
            per_env_delta = diff.mean(axis=(1, 2, 3)) * 255.0
            per_env_ppisp_mean = ppisp.astype(np.float32).mean(axis=(1, 2, 3))
            print(
                f"[INFO] step={count} mean_abs_delta={mean_abs_delta:.2f} mean_ppisp_corner_center_ratio={ratio:.3f}",
                flush=True,
            )
            print(
                "[INFO] per-env mean_abs_delta=" + ", ".join(f"{value:.2f}" for value in per_env_delta.tolist()),
                flush=True,
            )
            print(
                "[INFO] per-env ppisp_mean=" + ", ".join(f"{value:.2f}" for value in per_env_ppisp_mean.tolist()),
                flush=True,
            )
            images = []
            subtitles = []
            for env_id in range(ppisp.shape[0]):
                images.extend(
                    [
                        baseline[env_id] / 255.0,
                        ppisp[env_id] / 255.0,
                        diff[env_id],
                    ]
                )
                subtitles.extend([f"env {env_id} baseline", f"env {env_id} PPISP", f"env {env_id} diff"])
            section_start = time.perf_counter() if args_cli.profile else 0.0
            save_images_grid(
                images,
                nrow=ppisp.shape[0],
                subtitles=subtitles,
                title="USD-authored PPISP on duplicated Gaussian scene envs",
                filename=os.path.join(comparison_dir, f"ppisp_camera_{count:06d}.png"),
            )
            save_array_image(
                make_tiled_image(baseline / 255.0),
                os.path.join(baseline_dir, f"ppisp_camera_{count:06d}.png"),
            )
            save_array_image(
                make_tiled_image(ppisp / 255.0),
                os.path.join(ppisp_dir, f"ppisp_camera_{count:06d}.png"),
            )
            save_array_image(
                make_tiled_image(diff),
                os.path.join(diff_dir, f"ppisp_camera_{count:06d}.png"),
            )
            profile_record(profile, "disk_writes", time.perf_counter() - section_start)

            next_write_step += write_interval

        profile_record(profile, "total_step", time.perf_counter() - step_start)

        if count >= target_steps:
            break
    report_profile(profile, output_dir, count)


def main() -> None:
    """Main function."""
    source_stage = Usd.Stage.Open(args_cli.input_scene)
    if source_stage is None:
        raise RuntimeError(f"Failed to open input scene: {args_cli.input_scene}")
    source_camera_prim_path, render_product_prim, ppisp_camera_prim = resolve_source_camera_binding(source_stage)
    ppisp_cfg = make_ppisp_cfg(ppisp_camera_prim, len(find_ppisp_camera_bindings(source_stage)))
    camera_prim_path = source_camera_path_to_env_regex(source_stage, source_camera_prim_path)
    width, height = resolve_image_shape(render_product_prim)

    sim_utils.create_new_stage()
    sim_cfg = make_sim_cfg()
    apply_rtx_settings()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    scene = create_duplicated_env_scene()
    trajectory_times = get_trajectory_time_samples(source_stage, source_camera_prim_path)
    if not trajectory_times:
        trajectory_times = [args_cli.camera_time_code]
        print(
            f"[INFO] No USD xform time samples found; rendering the static camera for "
            f"at least one image at {args_cli.write_fps:g} write FPS.",
            flush=True,
        )
    else:
        print(
            f"[INFO] Found {len(trajectory_times)} camera trajectory time sample(s): "
            f"{trajectory_times[0]:g}..{trajectory_times[-1]:g}.",
            flush=True,
        )
    baseline_camera = None
    if not args_cli.render_only:
        baseline_camera = make_camera(camera_prim_path, ppisp_cfg=None, width=width, height=height)
    ppisp_camera = make_camera(camera_prim_path, ppisp_cfg=ppisp_cfg, width=width, height=height)
    # Apply after RTX/Replicator camera construction, but before reset, warmup, and the first render.
    apply_rtx_settings()
    print(f"[INFO] Duplicated-env camera regex: {camera_prim_path}", flush=True)
    print(f"[INFO] Rendering {width}x{height} from source camera {source_camera_prim_path}.", flush=True)

    sim.reset()
    print("[INFO]: Setup complete. Saving comparison images during simulation.", flush=True)
    run_simulator(
        sim,
        baseline_camera,
        ppisp_camera,
        source_stage,
        source_camera_prim_path,
        trajectory_times,
    )
    del scene


if __name__ == "__main__":
    main()
    simulation_app.close()
