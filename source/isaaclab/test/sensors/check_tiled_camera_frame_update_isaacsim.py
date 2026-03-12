# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone repro script that mimics the Isaac-Cartpole-RGB-Camera-Direct-v0 environment
and saves out frames of images after applying forces to the cartpoles.

Uses only Isaac Sim and Replicator APIs — no IsaacLab dependencies.

Key differences from check_tiled_camera_frame_update.py:
  - Uses a single ``rep.create.render_product_tiled`` covering all cameras at once
    (exactly as IsaacLab's TiledCamera does internally).
  - One annotator attached to that single tiled render product path.
  - The tiled image is de-tiled with the warp kernel from isaacsim.sensors.camera.
  - Uses ``omni.physics.tensors`` directly instead of ``isaacsim.core.prims.Articulation``.
  - Adds an explicit ``UsdPhysics.Scene`` prim so the physics engine always has a
    valid scene regardless of whether any articulation references are resolved yet.
  - Physics stepping uses ``omni.physx.get_physx_simulation_interface()`` directly
    (``simulate`` + ``fetch_results``), mirroring ``PhysxManager.step()``.
  - Rendering uses ``omni.kit.app.get_app().update()`` with
    ``/app/player/playSimulations`` toggled, mirroring ``RenderingManager.render()``.

Fabric-mode rendering (matches IsaacLab's PhysxManager._load_fabric / use_fabric=True):
  - Enables the ``omni.physx.fabric`` extension and acquires ``physxfabric`` interface.
  - Disables the USD sync path (``/physics/updateToUsd = False``), so physics
    transforms only reach the RTX renderer via an explicit ``physxfabric.update()``
    call — exactly what ``PhysxManager.forward()`` does in IsaacLab.
  - Calls ``_update_fabric(0, 0)`` before each render to ensure RTX sees current
    physics transforms (the fix applied in SimulationContext.render()).

Usage:

.. code-block:: bash

    python source/isaaclab/test/sensors/check_tiled_camera_frame_update_isaacsim.py --headless

    python source/isaaclab/test/sensors/check_tiled_camera_frame_update_isaacsim.py

    python source/isaaclab/test/sensors/check_tiled_camera_frame_update_isaacsim.py --headless --num_envs 4

Output images are saved to ``outputs/camera_frame_repro/``.
"""

import argparse
import math
import os

# ---------------------------------------------------------------------------
# CLI (must come before SimulationApp)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Repro: tiled camera frames not updating after physics steps (Isaac Sim only)."
)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4, help="Number of cartpole environments.")
parser.add_argument("--num_frames", type=int, default=5, help="Frames to capture and compare.")
parser.add_argument(
    "--physics_steps_between_frames",
    type=int,
    default=10,
    help="Physics-only steps (no render) between each captured frame.",
)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
import omni.physics.tensors
import omni.physx
import omni.timeline
import omni.usd
import torch
import warp as wp
import omni.replicator.core as rep
from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage_id
from isaacsim.sensors.camera.camera_view import reshape_tiled_image  # warp de-tiling kernel
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

# ---------------------------------------------------------------------------
# Carb settings — mirror what AppLauncher sets for --enable_cameras
# ---------------------------------------------------------------------------
_carb = carb.settings.get_settings()
# Allow tiled camera usage (guard check in isaaclab camera classes)
_carb.set_bool("/isaaclab/cameras_enabled", True)
# Match AppLauncher's fixed-time-stepping preference for standalone scripts
_carb.set_bool("/app/player/useFixedTimeStepping", False)
# NOTE: We intentionally do NOT set /physics/fabricUpdateTransformations here.
# That global shortcut masks the bug; instead we replicate IsaacLab's explicit
# physxfabric.update() path to demonstrate and fix the stale-frame issue.

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
ENV_SPACING = 20.0
IMG_WIDTH = 100
IMG_HEIGHT = 100
CAMERA_LOCAL_POS = (-5.0, 0.0, 2.0)  # relative to env origin
CARTPOLE_Z = 2.0
PHYSICS_DT = 1.0 / 120.0


# ---------------------------------------------------------------------------
# Direct omni API handles (populated in main after physics initialises)
# ---------------------------------------------------------------------------
# These mirror the handles used in PhysxManager:
#   _physx_sim  ↔  cls._physx_sim  (IPhysxSimulation)
#   _app        ↔  omni.kit.app.get_app()
_physx_sim = None  # IPhysxSimulation — set in main()
_app = None  # IApp — set in main()


def step_physics(steps: int = 1) -> None:
    """Step physics ``steps`` times via the raw IPhysxSimulation interface.

    Mirrors ``PhysxManager.step()`` / ``SimulationManager.step(steps=N)``:
        physx_sim.simulate(dt, 0.0)
        physx_sim.fetch_results()
    """
    for _ in range(steps):
        _physx_sim.simulate(PHYSICS_DT, 0.0)
        _physx_sim.fetch_results()


def render_no_physics() -> None:
    """Pump the Kit app loop without stepping physics.

    Mirrors ``RenderingManager.render()`` / the inner block of
    ``SimulationContext.render()``:
        set /app/player/playSimulations False
        app.update()
        restore /app/player/playSimulations
    """
    play = _carb.get_as_bool("/app/player/playSimulations")
    if play:
        _carb.set_bool("/app/player/playSimulations", False)
    _app.update()
    if play:
        _carb.set_bool("/app/player/playSimulations", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def grid_origin(env_idx: int, num_envs: int) -> tuple[float, float, float]:
    cols = max(1, math.ceil(math.sqrt(num_envs)))
    row, col = divmod(env_idx, cols)
    return (col * ENV_SPACING, row * ENV_SPACING, 0.0)


def tiling_grid_shape(num_cameras: int) -> tuple[int, int]:
    """Return (num_cols, num_rows) for the tiled render product grid.

    Mirrors TiledCamera._tiling_grid_shape().
    """
    cols = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / cols)
    return cols, rows


def build_scene(stage, num_envs: int, cartpole_usd: str) -> list[str]:
    """Populate the stage and return sorted camera prim paths.

    Creates an explicit ``UsdPhysics.Scene`` with the correct physics dt and
    applies ``PhysxSceneAPI`` so ``force_load_physics_from_usd()`` always
    finds a valid physics scene regardless of whether USD references have
    resolved yet.
    """
    # Physics scene with explicit dt — mirrors PhysxManager._configure_physics().
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    scene_prim.AddAppliedSchema("PhysxSceneAPI")
    scene_prim.CreateAttribute(
        "physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Float
    ).Set(float(round(1.0 / PHYSICS_DT)))

    dome = UsdLux.DomeLight.Define(stage, "/World/Light")
    dome.CreateIntensityAttr(2000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

    cam_paths = []
    for idx in range(num_envs):
        ox, oy, _ = grid_origin(idx, num_envs)
        env_path = f"/World/envs/env_{idx}"

        env_xf = UsdGeom.Xform.Define(stage, env_path)
        UsdGeom.XformCommonAPI(env_xf).SetTranslate(Gf.Vec3d(ox, oy, 0.0))

        # Cartpole (USD reference, offset 2 m above ground)
        robot_path = f"{env_path}/Robot"
        add_reference_to_stage(usd_path=cartpole_usd, prim_path=robot_path)
        UsdGeom.XformCommonAPI(UsdGeom.Xformable(stage.GetPrimAtPath(robot_path))).SetTranslate(
            Gf.Vec3d(0.0, 0.0, CARTPOLE_Z)
        )

        # Camera — child of env xform so it inherits the grid offset.
        # USD cameras look along local -Z; rotating -90° around Y maps -Z → +X,
        # so the camera faces the cartpole (which is at local origin + height).
        cam_path = f"{env_path}/Camera"
        cam_obj = UsdGeom.Camera.Define(stage, cam_path)
        cam_obj.GetFocalLengthAttr().Set(24.0)
        cam_obj.GetHorizontalApertureAttr().Set(20.955)
        cam_obj.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20.0))
        cam_obj.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_LOCAL_POS))
        cam_obj.AddRotateYOp().Set(-90.0)

        cam_paths.append(cam_path)

    return cam_paths


def detile_rgba(
    tiled_data: wp.array,
    num_cameras: int,
    img_height: int,
    img_width: int,
) -> torch.Tensor:
    """Convert the flat tiled RGBA buffer to a [N, H, W, 4] uint8 torch tensor.

    Uses the same warp kernel as isaacsim.sensors.camera.CameraView.get_data().
    """
    cols, _ = tiling_grid_shape(num_cameras)
    out = wp.zeros((num_cameras, img_height, img_width, 4), dtype=wp.uint8, device=tiled_data.device)
    wp.launch(
        kernel=reshape_tiled_image,
        dim=(num_cameras, img_height, img_width),
        inputs=[
            tiled_data.flatten(),
            out,
            img_height,   # image_height
            img_width,    # image_width
            4,            # num_channels  (rgba annotator → 4 channels)
            4,            # num_output_channels
            cols,         # num_tiles_x
            0,            # offset
        ],
        device=tiled_data.device,
    )
    return wp.to_torch(out)  # [N, H, W, 4]


def read_rgb(annotator, num_envs: int) -> torch.Tensor | None:
    """Read the tiled annotator and return [N, H, W, 3] uint8 on CUDA, or None."""
    raw = annotator.get_data()
    if isinstance(raw, dict):
        tiled_wp = raw.get("data")
    else:
        tiled_wp = raw
    if tiled_wp is None or tiled_wp.size == 0:
        return None
    rgba = detile_rgba(tiled_wp, num_envs, IMG_HEIGHT, IMG_WIDTH)
    return rgba[:, :, :, :3].contiguous()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _physx_sim, _app

    num_envs = args_cli.num_envs
    num_frames = args_cli.num_frames
    physics_steps_between = args_cli.physics_steps_between_frames

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "camera_frame_repro")
    os.makedirs(output_dir, exist_ok=True)

    # Acquire direct omni API handles.
    # IPhysxSimulation: low-level simulate/fetch_results — same object as
    # PhysxManager._physx_sim / SimulationManager._physics_sim_interface.
    _physx_sim = omni.physx.get_physx_simulation_interface()
    _app = omni.kit.app.get_app()

    # Read the asset root directly from carb settings (mirrors isaaclab.utils.assets).
    # Using the "cloud" key avoids the live connectivity check done by get_assets_root_path(),
    # which fails when the configured default points to an inaccessible Nucleus server.
    _FALLBACK = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    assets_root = (
        _carb.get("/persistent/isaac/asset_root/cloud")
        or _carb.get("/persistent/isaac/asset_root/default")
        or _FALLBACK
    )
    cartpole_usd = f"{assets_root}/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd"

    # ---- Stage --------------------------------------------------------------
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    cam_paths = build_scene(stage, num_envs, cartpole_usd)

    # Wait for the first cartpole USD reference to fully load.
    # timeline.play() fires _on_play() → force_load_physics_from_usd()
    # synchronously; if USD content is not yet resolved (async streaming from
    # Nucleus/S3) the physics engine will see no articulations.
    # Pumping updates until the reference prim has children guarantees the USD
    # content is present before we start physics.
    _first_robot = "/World/envs/env_0/Robot"
    for _ in range(200):
        _app.update()
        _prim = stage.GetPrimAtPath(_first_robot)
        if _prim.IsValid() and list(_prim.GetChildren()):
            break
    else:
        raise RuntimeError(
            f"Timed out waiting for '{_first_robot}' USD to load. "
            "Check your Isaac Sim asset root configuration."
        )

    # Resolve the exact ArticulationRootAPI prim path for the first robot.
    # The cartpole USD may nest the ArticulationRootAPI at a child prim
    # (e.g. /World/envs/env_0/Robot/CartRail), so we must pass the resolved
    # path to create_articulation_view() rather than the reference root.
    _resolved = get_articulation_root_api_prim_path(_first_robot)
    if _resolved != _first_robot:
        _suffix = _resolved[len(_first_robot):]   # e.g. "/CartRail"
        _artic_pattern = f"/World/envs/env_*/Robot{_suffix}"
    else:
        _artic_pattern = "/World/envs/env_*/Robot"
    print(f"[INFO] Articulation pattern: {_artic_pattern}")

    # ---- Single tiled render product (mirrors TiledCamera._initialize_impl) -
    tiled_rp = rep.create.render_product_tiled(
        cameras=cam_paths,
        tile_resolution=(IMG_WIDTH, IMG_HEIGHT),
    )
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda", do_array_copy=False)
    rgb_annotator.attach([tiled_rp.path])

    # ---- Enable Fabric before physics starts (mirrors PhysxManager._load_fabric) ---
    # Must happen before timeline.play() so physxfabric registers its scene handles
    # when force_load_physics_from_usd() fires.  Enabling it after play() means the
    # extension has no valid articulation handles and _update_fabric() is a no-op.
    ext_mgr = _app.get_extension_manager()
    if not ext_mgr.is_extension_enabled("omni.physx.fabric"):
        ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", True)
    from omni.physxfabric import get_physx_fabric_interface

    _fabric = get_physx_fabric_interface()
    _update_fabric = getattr(_fabric, "force_update", _fabric.update)

    for _key in ("updateToUsd", "updateParticlesToUsd", "updateVelocitiesToUsd",
                 "updateForceSensorsToUsd", "updateResidualsToUsd"):
        _carb.set_bool(f"/physics/{_key}", False)
    print("[INFO] Fabric enabled, USD sync disabled — IsaacLab use_fabric=True mode active.")

    # ---- Start physics ------------------------------------------------------
    # timeline.play() fires the Kit extension callbacks (e.g. SimulationManager's
    # _on_play) that call force_load_physics_from_usd() + start_simulation().
    # After this, _physx_sim is ready for direct simulate/fetch_results calls.
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    _app.update()

    # ---- Create omni.physics.tensors views ----------------------------------
    # One physics step to ensure PhysX has fully settled.
    step_physics()

    stage_id = get_current_stage_id()
    sim_view = omni.physics.tensors.create_simulation_view("torch", stage_id=stage_id)
    sim_view.set_subspace_roots("/")

    artic_view = sim_view.create_articulation_view(_artic_pattern)
    if artic_view is None or artic_view._backend is None or artic_view.count == 0:
        raise RuntimeError(
            f"create_articulation_view found no articulations matching '{_artic_pattern}'. "
            "Check that cartpole.usd loaded and that the ArticulationRootAPI prim is correct."
        )

    meta = artic_view.shared_metatype
    dof_names = list(meta.dof_names)
    print(f"[INFO] {num_envs} environments ready. DOF names: {dof_names}")
    cart_idx = dof_names.index("slider_to_cart")
    pole_idx = dof_names.index("cart_to_pole")
    num_dofs = len(dof_names)

    _idx = torch.arange(num_envs, dtype=torch.int32)
    _dev = artic_view.get_dof_positions().device

    # ---- Warm-up: textures stream, renderer settles -------------------------
    for _ in range(5):
        step_physics()
        _update_fabric(0.0, 0.0)
        render_no_physics()

    # ---- Initial pole kick --------------------------------------------------
    joint_pos = artic_view.get_dof_positions().clone()
    joint_pos[:, pole_idx] = torch.linspace(-math.pi / 4, math.pi / 4, num_envs, device=_dev)
    artic_view.set_dof_positions(joint_pos, _idx)
    artic_view.set_dof_velocities(torch.zeros(num_envs, num_dofs, device=_dev), _idx)

    for _ in range(5):
        step_physics()
        _update_fabric(0.0, 0.0)
        render_no_physics()

    # ---- Capture loop -------------------------------------------------------
    print(f"\n{'='*70}")
    print("Capturing frames — applying random cart forces between each frame")
    print(f"{'='*70}\n")

    prev_images: torch.Tensor | None = None

    for frame_idx in range(num_frames):
        efforts = torch.zeros(num_envs, num_dofs, device=_dev)
        efforts[:, cart_idx] = 200.0 * (2.0 * torch.rand(num_envs, device=_dev) - 1.0)
        artic_view.set_dof_actuation_forces(efforts, _idx)

        step_physics(physics_steps_between)
        step_physics()

        joint_pos = artic_view.get_dof_positions()
        cart_pos = joint_pos[:, cart_idx].cpu().numpy()
        pole_ang = joint_pos[:, pole_idx].cpu().numpy()
        print(f"Frame {frame_idx}: cart_pos={cart_pos}, pole_angle={pole_ang}")

        # Sync Fabric → RTX before rendering (mirrors PhysxManager.forward()).
        _update_fabric(0.0, 0.0)
        render_no_physics()
        images = read_rgb(rgb_annotator, num_envs)

        if images is None:
            print("  annotator returned no data — skipping.\n")
            continue

        print(f"  Mean pixel: {images.float().mean(dim=(1, 2, 3)).cpu().numpy()}")

        if prev_images is not None:
            diff = (images.float() - prev_images.float()).abs().mean(dim=(1, 2, 3))
            print(f"  Abs diff vs prev frame: {diff.cpu().numpy()}")
            if diff.max().item() < 1.0:
                print("  *** WARNING: render appears stale — physics may not be moving. ***")
        else:
            print("  (first frame — no diff)")

        prev_images = images.clone()

        try:
            from torchvision.utils import make_grid, save_image

            nrow = max(1, math.ceil(math.sqrt(num_envs)))
            imgs_f = images.float() / 255.0
            imgs_nchw = imgs_f.permute(0, 3, 1, 2).cpu()
            for env_idx in range(num_envs):
                save_image(
                    imgs_nchw[env_idx],
                    os.path.join(output_dir, f"frame_{frame_idx:03d}_env_{env_idx:03d}.png"),
                )
            grid_path = os.path.join(output_dir, f"frame_{frame_idx:03d}_grid.png")
            save_image(make_grid(imgs_nchw, nrow=nrow), grid_path)
            print(f"  Saved grid → {grid_path}")
        except ImportError:
            torch.save(images.cpu(), os.path.join(output_dir, f"frame_{frame_idx:03d}.pt"))
            print(f"  torchvision not available; saved raw tensor → {output_dir}/")

        print()

    # ---- Summary ------------------------------------------------------------
    print(f"{'='*70}")
    print(f"Done. {num_frames} frames saved to: {output_dir}")
    print(f"{'='*70}")

    rgb_annotator.detach([tiled_rp.path])
    tiled_rp.destroy()
    timeline.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()
