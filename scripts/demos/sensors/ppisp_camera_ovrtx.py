# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates USD-authored PPISP on a Gaussian scene through the kit-less OVRTX renderer.

It mirrors ``ppisp_camera.py``: the same CLI arguments, trajectory playback, and output layout, minus the
Kit-only options. OVRTX settings are configured through ``OVRTX_*`` environment variables instead.

.. code-block:: bash

    # Run a finite smoke and save comparison images.
    OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled=0 \
    uv run python scripts/demos/sensors/ppisp_camera_ovrtx.py \
        --input_scene /path/to/scene.usdz --num_frames 3 --num_envs 8

    # Follow xform time samples on the selected camera or its parent rig, rendering at 30 FPS and
    # writing every third frame.
    uv run python scripts/demos/sensors/ppisp_camera_ovrtx.py \
        --input_scene /path/to/scene.usdz --fps 30 --write_fps 10

OVRTX must run kit-less: launch this script with ``uv run python``.
"""

import argparse
import contextlib
import json
import os
import time
from collections.abc import Iterator
from typing import Any

import gaussian_animation as gaussian_anim
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_ppisp._demo_utils import (
    find_ppisp_camera_bindings,
    format_available_ppisp_cameras,
    order_ppisp_bindings_by_camera,
)
from isaaclab_ppisp.cfg import PpispCfg, ppisp_cfg_from_usd_camera

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

DEFAULT_INPUT_SCENE = f"{ISAAC_NUCLEUS_DIR}/Samples/Scene_ParticleField/valiant_auto.usdz"

STAGE_TIME_CODE = 0.0
"""USD time code at which the simulation stage is evaluated.

The referenced scene -- including any animated ancestors of the camera -- is composed at this time code
for the whole demo.
"""

parser = argparse.ArgumentParser(description="Kit-less OVRTX demo for USD-authored PPISP camera output.")
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
    help="Rendered-frame rate in FPS, used to resample the USD camera trajectory. Does not change the physics rate.",
)
parser.add_argument(
    "--physics_dt",
    type=float,
    default=0.005,
    help="Physics step size in seconds. Physics steps are batched per rendered frame via render_interval.",
)
parser.add_argument(
    "--num_frames",
    type=int,
    default=None,
    help="Number of frames to render. Defaults to the resampled USD camera trajectory length, or 1 if static.",
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
parser.add_argument("--device", type=str, default="cuda:0", help="Torch/Warp device to render into.")
parser.add_argument(
    "--viz",
    type=str,
    choices=["none"],
    default="none",
    help="Accepted for CLI parity with the Kit demo. OVRTX runs kit-less, so only 'none' is supported.",
)
parser.add_argument(
    "--render_only",
    action="store_true",
    help="Render and save only the PPISP camera output; skip baseline, comparison, and diff outputs.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=32,
    help="Simulation/render steps to run before saving images.",
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
    help="Directory to write comparison images. Defaults to scripts/demos/sensors/output/ppisp_camera_ovrtx.",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Profile simulation, camera updates, GPU transfers, and disk writes.",
)
parser.add_argument("--ovrtx_log_level", type=str, default="verbose", help="OVRTX carb log level.")
parser.add_argument("--ovrtx_log_file", type=str, default="/tmp/ovrtx_renderer.log", help="OVRTX log file path.")

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
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative.")
if args_cli.fps <= 0.0:
    parser.error("--fps must be positive.")
if args_cli.write_fps is None:
    args_cli.write_fps = args_cli.fps
if args_cli.write_fps <= 0.0:
    parser.error("--write_fps must be positive.")
if args_cli.physics_dt <= 0.0:
    parser.error("--physics_dt must be positive.")
if args_cli.num_frames is not None and args_cli.num_frames < 1:
    parser.error("--num_frames must be at least 1.")


@configclass
class PpispCameraOvrtxSceneCfg(InteractiveSceneCfg):
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
    """Create the OVRTX camera renderer cfg."""
    try:
        from isaaclab_ov.renderers import OVRTXRendererCfg
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ppisp_camera_ovrtx.py requires the optional OVRTX renderer stack. "
            "Run it from an environment with isaaclab_ov and ovrtx installed."
        ) from exc

    return OVRTXRendererCfg(log_level=args_cli.ovrtx_log_level, log_file_path=args_cli.ovrtx_log_file)


def make_sim_cfg() -> sim_utils.SimulationCfg:
    """Create the kit-less Newton simulation cfg required by OVRTX."""
    render_interval = max(1, int(round((1.0 / args_cli.fps) / args_cli.physics_dt)))
    return sim_utils.SimulationCfg(
        dt=args_cli.physics_dt,
        device=args_cli.device,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1),
        use_fabric=not args_cli.disable_fabric,
        render_interval=render_interval,
    )


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
    # Keep the resampled cadence strictly uniform so the written sequence plays back at a constant
    # rate; this drops at most one frame of the authored tail.
    time_step = time_codes_per_second / args_cli.fps
    sample_count = int(np.floor((end_time - start_time) / time_step)) + 1
    return [start_time + index * time_step for index in range(sample_count)]


def get_source_camera_in_default(
    source_stage: Usd.Stage, source_camera_prim_path: str, time_code: float
) -> Gf.Matrix4d:
    """Return the source camera transform relative to the source defaultPrim at ``time_code``."""
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Input scene must have a defaultPrim so it can be referenced under each env.")

    source_camera_prim = source_stage.GetPrimAtPath(source_camera_prim_path)
    if not source_camera_prim or not source_camera_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {source_camera_prim_path}")

    source_cache = UsdGeom.XformCache(Usd.TimeCode(time_code))
    source_default_world = source_cache.GetLocalToWorldTransform(default_prim)
    source_camera_world = source_cache.GetLocalToWorldTransform(source_camera_prim)
    return source_camera_world * source_default_world.GetInverse()


def get_env_scene_world_transforms() -> list[Gf.Matrix4d]:
    """Return the world transform of each duplicated env scene prim, as the renderer evaluates it."""
    stage = sim_utils.get_current_stage()
    target_cache = UsdGeom.XformCache(Usd.TimeCode(STAGE_TIME_CODE))
    transforms = []
    for env_id in range(args_cli.num_envs):
        scene_path = f"/World/envs/env_{env_id}/Scene"
        scene_prim = stage.GetPrimAtPath(scene_path)
        if not scene_prim or not scene_prim.IsValid():
            raise RuntimeError(f"Duplicated scene prim not found: {scene_path}")
        transforms.append(target_cache.GetLocalToWorldTransform(scene_prim))
    return transforms


def freeze_env_camera_ancestor_xforms(source_stage: Usd.Stage, source_camera_prim_path: str) -> None:
    """Collapse the xform chain above each duplicated env camera to a static pose at :data:`STAGE_TIME_CODE`.

    Capture scenes animate the camera rig rather than the camera prim itself. Re-evaluating those animated
    xforms per frame would overwrite the camera poses written by :func:`set_env_camera_world_poses`, so each
    ancestor is rewritten with the static transform already evaluated at :data:`STAGE_TIME_CODE`. This makes
    the runtime pose writes authoritative without changing the pose that is rendered.
    """
    stage = sim_utils.get_current_stage()
    time_code = Usd.TimeCode(STAGE_TIME_CODE)
    camera_rel_path = source_camera_path_to_default_rel_path(source_stage, source_camera_prim_path)
    ancestor_rel_parts = camera_rel_path.split("/")[:-1]
    frozen_count = 0
    for env_id in range(args_cli.num_envs):
        for depth in range(1, len(ancestor_rel_parts) + 1):
            ancestor_path = "/".join([f"/World/envs/env_{env_id}/Scene", *ancestor_rel_parts[:depth]])
            ancestor_prim = stage.GetPrimAtPath(ancestor_path)
            if not ancestor_prim or not ancestor_prim.IsValid():
                raise RuntimeError(f"Duplicated camera ancestor prim not found: {ancestor_path}")
            # Scopes and other non-Xformable groupings carry no transform to freeze.
            if not ancestor_prim.IsA(UsdGeom.Xformable):
                continue
            local_transform = Gf.Transform(UsdGeom.Xformable(ancestor_prim).GetLocalTransformation(time_code))
            rotation = local_transform.GetRotation().GetQuat()
            imaginary = rotation.GetImaginary()
            sim_utils.standardize_xform_ops(
                ancestor_prim,
                translation=tuple(local_transform.GetTranslation()),
                orientation=(imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()),
                scale=tuple(local_transform.GetScale()),
            )
            frozen_count += 1
    print(
        f"[INFO] Froze {frozen_count} camera ancestor xform(s) at USD time {STAGE_TIME_CODE:g}.",
        flush=True,
    )


def resolve_animated_gaussian_tracks(source_stage: Usd.Stage) -> list[gaussian_anim.AnimatedGaussianTrack]:
    """Discover the animated Gaussian tracks to play back on OVRTX."""
    tracks = gaussian_anim.find_animated_gaussian_tracks(source_stage)
    if tracks:
        print(f"[INFO] Animated Gaussian track(s): {gaussian_anim.format_tracks(tracks)}", flush=True)
    return tracks


def get_ovrtx_renderer(sim: sim_utils.SimulationContext) -> Any:
    """Return the simulation's shared OVRTX renderer after a camera initialized it.

    The cfg must match the one the camera sensors were created with.
    """
    try:
        from isaaclab_ov.renderers import OVRTXRenderer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Animated Gaussian playback requires the optional OVRTX renderer stack.") from exc

    renderer = sim.render_context.get_renderer(make_renderer_cfg())
    if not isinstance(renderer, OVRTXRenderer):
        raise RuntimeError(f"Expected the OVRTX renderer, received {type(renderer).__name__}.")
    return renderer


def play_animated_gaussian_tracks(
    renderer: Any,
    source_stage: Usd.Stage,
    tracks: list[gaussian_anim.AnimatedGaussianTrack],
    time_code: float,
) -> None:
    """Advance the Gaussian tracks to ``time_code`` through OVRTX's public update hooks.

    Rigid tracks are played by writing the animated ancestor's local transform; deformable tracks by
    writing the sampled per-particle arrays, which every duplicated env shares because the samples
    are in the Gaussian prim's local frame.
    """
    if not tracks:
        return
    default_prim = gaussian_anim.require_default_prim(source_stage)
    default_prefix = f"{default_prim.GetPath().pathString}/"
    for track in tracks:
        for xform_rel_path in track.animated_xform_rel_paths:
            source_prim = source_stage.GetPrimAtPath(f"{default_prefix}{xform_rel_path}")
            local_transform = UsdGeom.Xformable(source_prim).GetLocalTransformation(Usd.TimeCode(time_code))
            paths = [gaussian_anim.env_prim_path(env_id, xform_rel_path) for env_id in range(args_cli.num_envs)]
            transforms = np.broadcast_to(np.asarray(local_transform, dtype=np.float64), (len(paths), 4, 4)).copy()
            renderer.update_gaussian_splat_transforms(paths, transforms)

        if not track.is_deformable:
            continue
        positions, orientations = gaussian_anim.sample_track_particles(source_stage, track, time_code)
        paths = [gaussian_anim.env_prim_path(env_id, track.gaussian_rel_path) for env_id in range(args_cli.num_envs)]
        renderer.update_gaussian_splat_particles(
            paths,
            positions=None if positions is None else [positions] * len(paths),
            orientations=None if orientations is None else [orientations] * len(paths),
        )


def bake_source_camera_pose_to_envs(source_stage: Usd.Stage, source_camera_prim_path: str, time_code: float) -> None:
    """Bake a USD camera pose at ``time_code`` into duplicated env camera prims.

    This seeds the initial pose on the stage and must run before the camera sensors are created and before
    :meth:`SimulationContext.reset`. The Newton backend samples the camera prim's USD transform once while
    building its model, and the Fabric path is populated from USD at reset, so later USD edits do not reach
    the renderer. Runtime trajectory playback goes through :func:`set_env_camera_world_poses` instead.

    The pose is written through :func:`~isaaclab.sim.utils.standardize_xform_ops` so the prims keep the
    canonical ``[translate, orient, scale]`` op order that the sensor frame views require; authoring a
    single ``xformOp:transform`` here would silently discard every later view-side pose write.
    """
    source_camera_in_default = get_source_camera_in_default(source_stage, source_camera_prim_path, time_code)
    stage = sim_utils.get_current_stage()
    # The renderer evaluates the referenced scene at the stage's current timeline time, so the ancestor
    # chain must be sampled there too. Sampling it at the default time code instead resolves animated
    # ancestor xforms as unauthored, which double-counts their contribution in the composed camera pose.
    target_cache = UsdGeom.XformCache(Usd.TimeCode(STAGE_TIME_CODE))
    camera_rel_path = source_camera_path_to_default_rel_path(source_stage, source_camera_prim_path)
    scene_world_transforms = get_env_scene_world_transforms()
    for env_id, target_scene_world in enumerate(scene_world_transforms):
        target_camera_path = f"/World/envs/env_{env_id}/Scene/{camera_rel_path}"
        target_camera_prim = stage.GetPrimAtPath(target_camera_path)
        if not target_camera_prim or not target_camera_prim.IsValid():
            raise RuntimeError(f"Duplicated camera prim not found: {target_camera_path}")

        target_parent_world = target_cache.GetLocalToWorldTransform(target_camera_prim.GetParent())
        target_camera_world = source_camera_in_default * target_scene_world
        target_camera_local = target_camera_world * target_parent_world.GetInverse()
        target_camera_local.Orthonormalize()
        translation = target_camera_local.ExtractTranslation()
        rotation = target_camera_local.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        sim_utils.standardize_xform_ops(
            target_camera_prim,
            translation=(translation[0], translation[1], translation[2]),
            orientation=(imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()),
            scale=(1.0, 1.0, 1.0),
        )

    print(
        f"[INFO] Baked camera pose at USD time {time_code:g} into {len(scene_world_transforms)} env camera(s).",
        flush=True,
    )


def compute_env_camera_world_poses(
    source_stage: Usd.Stage,
    source_camera_prim_path: str,
    time_code: float,
    scene_world_transforms: list[Gf.Matrix4d],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-env camera world poses at ``time_code``.

    Returns:
        A tuple of camera positions [m] with shape (num_envs, 3) and OpenGL-convention quaternion
        orientations in (x, y, z, w) order with shape (num_envs, 4).
    """
    source_camera_in_default = get_source_camera_in_default(source_stage, source_camera_prim_path, time_code)
    positions = np.empty((len(scene_world_transforms), 3), dtype=np.float32)
    orientations = np.empty((len(scene_world_transforms), 4), dtype=np.float32)
    for env_id, target_scene_world in enumerate(scene_world_transforms):
        target_camera_world = source_camera_in_default * target_scene_world
        target_camera_world.Orthonormalize()
        translation = target_camera_world.ExtractTranslation()
        rotation = target_camera_world.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        positions[env_id] = (translation[0], translation[1], translation[2])
        orientations[env_id] = (imaginary[0], imaginary[1], imaginary[2], rotation.GetReal())
    return positions, orientations


def set_env_camera_world_poses(cameras: list[Camera], positions: np.ndarray, orientations: np.ndarray) -> None:
    """Write camera world poses through the sensor view so the render backend observes them.

    A USD camera transform is expressed in the OpenGL convention (forward -Z, up +Y). Every camera
    sensor keeps its own view state, so a shared prim batch has to be updated once per sensor.
    """
    for camera in cameras:
        camera.set_world_poses(positions, orientations, convention="opengl")


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
    scene_cfg = PpispCameraOvrtxSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene_cfg.input_scene.spawn = sim_utils.UsdFileCfg(usd_path=args_cli.input_scene)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] Referenced input scene into {args_cli.num_envs} env(s).", flush=True)
    return scene


def make_matched_camera_prims_visible(stage: Usd.Stage, camera_prim_path: str) -> None:
    """Make duplicated camera prims visible for OVRTX render product discovery."""
    for prim in sim_utils.find_matching_prims(camera_prim_path, stage):
        UsdGeom.Imageable(prim).MakeVisible()


def make_camera(
    camera_prim_path: str,
    *,
    ppisp_cfg: PpispCfg | None,
    width: int,
    height: int,
) -> Camera:
    """Create a baseline or PPISP camera sensor for the duplicated-env camera batch."""
    return Camera(
        CameraCfg(
            prim_path=camera_prim_path,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=None,
            # Required for trajectory playback: without it, the per-frame pose written by
            # set_world_poses() stays in the sensor view and never reaches the renderer.
            update_latest_camera_pose=True,
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


@contextlib.contextmanager
def profile_section(profile: dict[str, dict[str, float]], name: str, device: str) -> Iterator[None]:
    """Time a section of the render loop, bracketing it with device synchronization when profiling.

    Synchronizing on both ends keeps asynchronous GPU work attributed to the section that queued it,
    so section timings sum to the measured frame time.
    """
    if not args_cli.profile:
        yield
        return
    is_cuda = device.startswith("cuda")
    if is_cuda:
        wp.synchronize_device(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if is_cuda:
            wp.synchronize_device(device)
        entry = profile.setdefault(name, {"count": 0.0, "total_seconds": 0.0, "max_seconds": 0.0})
        elapsed = time.perf_counter() - start
        entry["count"] += 1.0
        entry["total_seconds"] += elapsed
        entry["max_seconds"] = max(entry["max_seconds"], elapsed)


def report_profile(profile: dict[str, dict[str, float]], output_dir: str, num_frames: int) -> None:
    """Print and save profiling results when profiling is enabled."""
    if not args_cli.profile:
        return
    results = {
        "frames": num_frames,
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


def resolve_output_dir() -> str:
    """Resolve the demo output directory, creating it if needed."""
    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ppisp_camera_ovrtx")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_simulator(
    sim: sim_utils.SimulationContext,
    baseline_camera: Camera | None,
    ppisp_camera: Camera,
    ovrtx_renderer: Any,
    source_stage: Usd.Stage,
    source_camera_prim_path: str,
    frame_time_codes: list[float],
    gaussian_tracks: list[gaussian_anim.AnimatedGaussianTrack],
) -> None:
    """Play the camera trajectory and periodically save rendered images."""
    render_dt = 1.0 / args_cli.fps
    physics_steps_per_frame = sim.cfg.render_interval
    write_interval = max(1, int(round(args_cli.fps / args_cli.write_fps)))
    # The baseline and PPISP sensors share the same camera prims but keep independent view state.
    cameras = [ppisp_camera] if baseline_camera is None else [baseline_camera, ppisp_camera]
    scene_world_transforms = get_env_scene_world_transforms()
    output_dir = resolve_output_dir()
    comparison_dir = os.path.join(output_dir, "comparison")
    baseline_dir = os.path.join(output_dir, "baseline")
    ppisp_dir = os.path.join(output_dir, "ppisp")
    diff_dir = os.path.join(output_dir, "diff")
    profile: dict[str, dict[str, float]] = {}
    profile_device = str(sim.device)
    diff_wp: wp.array | None = None
    reported_shape = False
    frame_index = 0

    print(
        f"[INFO] Rendering {len(frame_time_codes)} frame(s) at {args_cli.fps:g} FPS with "
        f"{physics_steps_per_frame} physics step(s) of {args_cli.physics_dt:g}s per frame; "
        f"writing every {write_interval} frame(s) at {args_cli.write_fps:g} FPS.",
        flush=True,
    )
    if args_cli.warmup_steps > 0:
        print(f"[INFO] Running {args_cli.warmup_steps} warmup step(s) before saving images.", flush=True)
    # Seed the first pose before warmup, so a nonzero first USD animation sample is represented in
    # all renderer-startup frames too.
    play_animated_gaussian_tracks(ovrtx_renderer, source_stage, gaussian_tracks, frame_time_codes[0])
    for _ in range(args_cli.warmup_steps):
        sim.step()
        for camera in cameras:
            camera.update(render_dt, force_recompute=True)

    for frame_index, time_code in enumerate(frame_time_codes):
        with profile_section(profile, "total_frame", profile_device):
            with profile_section(profile, "camera_pose_update", profile_device):
                positions, orientations = compute_env_camera_world_poses(
                    source_stage, source_camera_prim_path, time_code, scene_world_transforms
                )
                set_env_camera_world_poses(cameras, positions, orientations)

            with profile_section(profile, "gaussian_animation_update", profile_device):
                play_animated_gaussian_tracks(ovrtx_renderer, source_stage, gaussian_tracks, time_code)

            with profile_section(profile, "simulation_step", profile_device):
                for physics_step in range(physics_steps_per_frame):
                    sim.step(render=physics_step == physics_steps_per_frame - 1)

            if baseline_camera is not None:
                with profile_section(profile, "baseline_camera_update", profile_device):
                    baseline_camera.update(render_dt, force_recompute=True)
            with profile_section(profile, "ppisp_camera_update", profile_device):
                ppisp_camera.update(render_dt, force_recompute=True)

            if frame_index % write_interval != 0:
                continue

            ppisp_wp = ppisp_camera.data.output["rgb"].warp
            baseline_wp = None if baseline_camera is None else baseline_camera.data.output["rgb"].warp
            if baseline_wp is not None:
                with profile_section(profile, "warp_difference", profile_device):
                    if diff_wp is None:
                        diff_wp = wp.empty(ppisp_wp.shape, dtype=wp.float32, device=ppisp_wp.device)
                    wp.launch(
                        compute_rgb_diff,
                        dim=ppisp_wp.shape,
                        inputs=[baseline_wp, ppisp_wp, diff_wp],
                        device=ppisp_wp.device,
                    )

            # Transfer image buffers to NumPy only at the disk-writing boundary.
            with profile_section(profile, "gpu_to_cpu_transfer", profile_device):
                ppisp = ppisp_wp.numpy()
                baseline = None if baseline_wp is None else baseline_wp.numpy()
                diff = None if diff_wp is None else diff_wp.numpy()
            if not reported_shape:
                print(f"[INFO] camera batch rgb shape={tuple(ppisp.shape)}", flush=True)
                reported_shape = True

            ratios = [corner_center_ratio(ppisp[env_id]) for env_id in range(ppisp.shape[0])]
            per_env_ppisp_mean = ppisp.astype(np.float32).mean(axis=(1, 2, 3))
            summary = (
                f"[INFO] frame={frame_index} usd_time={time_code:g} "
                f"mean_ppisp_corner_center_ratio={sum(ratios) / len(ratios):.3f}"
            )
            if diff is not None:
                summary += f" mean_abs_delta={float(diff.mean()) * 255.0:.2f}"
            print(summary, flush=True)
            print(
                "[INFO] per-env ppisp_mean=" + ", ".join(f"{value:.2f}" for value in per_env_ppisp_mean.tolist()),
                flush=True,
            )
            if diff is not None:
                per_env_delta = diff.mean(axis=(1, 2, 3)) * 255.0
                print(
                    "[INFO] per-env mean_abs_delta=" + ", ".join(f"{value:.2f}" for value in per_env_delta.tolist()),
                    flush=True,
                )

            with profile_section(profile, "disk_writes", profile_device):
                save_array_image(
                    make_tiled_image(ppisp / 255.0),
                    os.path.join(ppisp_dir, f"ppisp_camera_ovrtx_{frame_index:06d}.png"),
                )
                if baseline is not None and diff is not None:
                    save_array_image(
                        make_tiled_image(baseline / 255.0),
                        os.path.join(baseline_dir, f"ppisp_camera_ovrtx_{frame_index:06d}.png"),
                    )
                    save_array_image(
                        make_tiled_image(diff), os.path.join(diff_dir, f"ppisp_camera_ovrtx_{frame_index:06d}.png")
                    )
                    images = []
                    subtitles = []
                    for env_id in range(ppisp.shape[0]):
                        images.extend([baseline[env_id] / 255.0, ppisp[env_id] / 255.0, diff[env_id]])
                        subtitles.extend([f"env {env_id} baseline", f"env {env_id} PPISP", f"env {env_id} diff"])
                    save_images_grid(
                        images,
                        nrow=ppisp.shape[0],
                        subtitles=subtitles,
                        title="USD-authored PPISP on duplicated Gaussian scene envs through OVRTX",
                        filename=os.path.join(comparison_dir, f"ppisp_camera_ovrtx_{frame_index:06d}.png"),
                    )

    report_profile(profile, output_dir, frame_index + 1)


def main() -> None:
    """Main function."""
    args_cli.input_scene = retrieve_file_path(args_cli.input_scene)
    source_stage = Usd.Stage.Open(args_cli.input_scene)
    if source_stage is None:
        raise RuntimeError(f"Failed to open input scene: {args_cli.input_scene}")
    source_camera_prim_path, render_product_prim, ppisp_camera_prim = resolve_source_camera_binding(source_stage)
    ppisp_cfg = make_ppisp_cfg(ppisp_camera_prim, len(find_ppisp_camera_bindings(source_stage)))
    camera_prim_path = source_camera_path_to_env_regex(source_stage, source_camera_prim_path)
    width, height = resolve_image_shape(render_product_prim)

    sim_utils.create_new_stage()
    sim_cfg = make_sim_cfg()
    sim = sim_utils.SimulationContext(sim_cfg)

    scene = create_duplicated_env_scene()
    gaussian_tracks = resolve_animated_gaussian_tracks(source_stage)
    trajectory_times = get_trajectory_time_samples(source_stage, source_camera_prim_path)
    if trajectory_times:
        print(
            f"[INFO] Resampled {len(trajectory_times)} camera trajectory time sample(s) at {args_cli.fps:g} FPS: "
            f"{trajectory_times[0]:g}..{trajectory_times[-1]:g}.",
            flush=True,
        )
    else:
        # A scene may animate only its Gaussians, in which case their time samples drive the frames.
        trajectory_times = gaussian_anim.collect_authored_times(source_stage, gaussian_tracks)
        if trajectory_times:
            print(
                f"[INFO] Static camera; playing {len(trajectory_times)} Gaussian animation time sample(s): "
                f"{trajectory_times[0]:g}..{trajectory_times[-1]:g}.",
                flush=True,
            )
    if not trajectory_times:
        trajectory_times = [args_cli.camera_time_code]
        print(
            f"[INFO] No USD xform time samples found; playing the static camera at USD time "
            f"{args_cli.camera_time_code:g}.",
            flush=True,
        )
    frame_time_codes = trajectory_times
    if args_cli.num_frames is not None:
        # Hold the last trajectory pose when more frames are requested than the trajectory provides.
        last_index = len(trajectory_times) - 1
        frame_time_codes = [trajectory_times[min(index, last_index)] for index in range(args_cli.num_frames)]

    # Prepare the camera prims before the sensors and their frame views are built: freeze the animated
    # rig above them so runtime pose writes stick, then seed the first trajectory pose.
    freeze_env_camera_ancestor_xforms(source_stage, source_camera_prim_path)
    bake_source_camera_pose_to_envs(source_stage, source_camera_prim_path, frame_time_codes[0])
    make_matched_camera_prims_visible(sim_utils.get_current_stage(), camera_prim_path)

    # Create the PPISP sensor before the baseline sensor: when two OVRTX camera sensors share the same
    # camera prims, only the first one created gets a working HDR source, and a PPISP sensor created
    # second renders black.
    camera_kwargs = {"width": width, "height": height}
    ppisp_camera = make_camera(camera_prim_path, ppisp_cfg=ppisp_cfg, **camera_kwargs)
    baseline_camera = None
    if not args_cli.render_only:
        baseline_camera = make_camera(camera_prim_path, ppisp_cfg=None, **camera_kwargs)
    print(f"[INFO] Duplicated-env camera regex: {camera_prim_path}", flush=True)
    print(f"[INFO] Rendering {width}x{height} from source camera {source_camera_prim_path}.", flush=True)

    try:
        sim.reset()
        print("[INFO]: Setup complete. Saving comparison images during simulation.", flush=True)
        ovrtx_renderer = get_ovrtx_renderer(sim)
        run_simulator(
            sim,
            baseline_camera,
            ppisp_camera,
            ovrtx_renderer,
            source_stage,
            source_camera_prim_path,
            frame_time_codes,
            gaussian_tracks,
        )
    finally:
        del ppisp_camera
        del baseline_camera
        del scene
        sim.stop()
        sim.clear_instance()


if __name__ == "__main__":
    main()
