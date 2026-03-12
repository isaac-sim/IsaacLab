# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone repro script that mimics the Isaac-Cartpole-RGB-Camera-Direct-v0 environment
and saves out frames of images after applying forces to the cartpoles.

Uses only Isaac Sim and Replicator APIs — no IsaacLab dependencies.

This demonstrates the issue where rendered camera frames do NOT update after physics
has stepped — the cartpole moves but the camera image remains stale.

Usage:

.. code-block:: bash

    # Run headless (GPU rendering still happens for cameras)
    python source/isaaclab/test/sensors/check_tiled_camera_frame_update.py --headless

    # Run with viewer
    python source/isaaclab/test/sensors/check_tiled_camera_frame_update.py

    # Adjust number of environments
    python source/isaaclab/test/sensors/check_tiled_camera_frame_update.py --headless --num_envs 4

Output images are saved to ``outputs/camera_frame_repro/``.
"""

import argparse
import math
import os

# ---------------------------------------------------------------------------
# CLI  (must be parsed before SimulationApp is created)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Repro: camera frames not updating after physics steps (Isaac Sim only)."
)
parser.add_argument("--headless", action="store_true", default=False, help="Run without a viewer.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of cartpole environments.")
parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to capture.")
parser.add_argument(
    "--physics_steps_between_frames",
    type=int,
    default=10,
    help="Physics-only steps (no render) between each captured frame.",
)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Launch Isaac Sim (must happen before any omni.* / isaacsim.* imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import torch
from isaacsim.core.prims import Articulation
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom, UsdLux

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_SPACING = 20.0  # metres between environment grid origins
IMG_WIDTH = 100
IMG_HEIGHT = 100
# Camera sits 5 m behind and 2 m above the cartpole base, looking along +X
CAMERA_LOCAL_POS = (-5.0, 0.0, 2.0)
CARTPOLE_Z = 2.0  # spawn height above ground


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def grid_origin(env_idx: int, num_envs: int) -> tuple[float, float, float]:
    """Return the world XYZ origin for environment *env_idx* in a square grid."""
    cols = max(1, math.ceil(math.sqrt(num_envs)))
    row, col = divmod(env_idx, cols)
    return (col * ENV_SPACING, row * ENV_SPACING, 0.0)


def build_scene(stage, num_envs: int, cartpole_usd: str) -> None:
    """Populate the USD stage with cartpoles, cameras, and a dome light."""
    # Dome light
    dome = UsdLux.DomeLight.Define(stage, "/World/Light")
    dome.CreateIntensityAttr(2000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

    for idx in range(num_envs):
        ox, oy, _ = grid_origin(idx, num_envs)
        env_path = f"/World/envs/env_{idx}"

        # Environment root xform — all children inherit the grid offset
        env_xf = UsdGeom.Xform.Define(stage, env_path)
        UsdGeom.XformCommonAPI(env_xf).SetTranslate(Gf.Vec3d(ox, oy, 0.0))

        # Cartpole articulation (USD reference)
        robot_path = f"{env_path}/Robot"
        add_reference_to_stage(usd_path=cartpole_usd, prim_path=robot_path)
        robot_prim = stage.GetPrimAtPath(robot_path)
        UsdGeom.XformCommonAPI(UsdGeom.Xformable(robot_prim)).SetTranslate(Gf.Vec3d(0.0, 0.0, CARTPOLE_Z))

        # Camera (child of env xform so it inherits the grid offset)
        cam_path = f"{env_path}/Camera"
        cam_obj = UsdGeom.Camera.Define(stage, cam_path)
        cam_obj.GetFocalLengthAttr().Set(24.0)
        cam_obj.GetHorizontalApertureAttr().Set(20.955)
        cam_obj.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20.0))

        # Position the camera and orient it to look along +X toward the cartpole.
        # USD cameras look along their local -Z axis.  Rotating -90° around Y maps
        # local -Z → world +X, so the camera faces the cartpole.
        cam_obj.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_LOCAL_POS))
        cam_obj.AddRotateYOp().Set(-90.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    num_envs = args_cli.num_envs
    num_frames = args_cli.num_frames
    physics_steps_between = args_cli.physics_steps_between_frames

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "camera_frame_repro")
    os.makedirs(output_dir, exist_ok=True)

    # Read the asset root directly from carb settings (avoids the live Nucleus
    # connectivity check that get_assets_root_path() performs).
    _FALLBACK = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    _carb_settings = carb.settings.get_settings()
    assets_root = (
        _carb_settings.get("/persistent/isaac/asset_root/cloud")
        or _carb_settings.get("/persistent/isaac/asset_root/default")
        or _FALLBACK
    )
    cartpole_usd = f"{assets_root}/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd"

    # ---- Stage setup --------------------------------------------------------
    import omni.usd

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    build_scene(stage, num_envs, cartpole_usd)

    # Allow USD references to fully resolve before physics initialization.
    simulation_app.update()

    # ---- Replicator: render products and annotators -------------------------
    # One per-environment camera → one render product → one RGB annotator
    render_products = []
    annotators = []
    for idx in range(num_envs):
        cam_path = f"/World/envs/env_{idx}/Camera"
        rp = rep.create.render_product(cam_path, resolution=(IMG_WIDTH, IMG_HEIGHT))
        render_products.append(rp)
        ann = rep.annotators.get("rgb")
        ann.attach(rp)
        annotators.append(ann)

    # ---- Articulation view --------------------------------------------------
    cartpole_view = Articulation("/World/envs/env_*/Robot")

    # ---- Configure physics --------------------------------------------------
    SimulationManager.set_physics_dt(1.0 / 120.0)

    # ---- Start timeline and warm up ----------------------------------------
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()  # first app frame — lets physics initialise

    cartpole_view.initialize()
    simulation_app.update()

    dof_names = list(cartpole_view.dof_names)
    print(f"[INFO] Scene created with {num_envs} environments.")
    print(f"[INFO] Cartpole DOF names: {dof_names}")
    cart_idx = dof_names.index("slider_to_cart")
    pole_idx = dof_names.index("cart_to_pole")
    num_dofs = len(dof_names)

    # Warm-up: let textures load and renderer settle
    for _ in range(5):
        SimulationManager.step()
        RenderingManager.render()

    # ---- Initial pole kick --------------------------------------------------
    joint_pos = cartpole_view.get_joint_positions()  # [N, num_dofs] numpy array
    joint_pos[:, pole_idx] = torch.linspace(-math.pi / 4, math.pi / 4, num_envs).numpy()
    cartpole_view.set_joint_positions(joint_pos)
    cartpole_view.set_joint_velocities(torch.zeros(num_envs, num_dofs).numpy())

    for _ in range(5):
        SimulationManager.step()
        RenderingManager.render()

    # ---- Capture loop -------------------------------------------------------
    print(f"\n{'='*70}")
    print("Capturing frames — applying random cart forces between each frame")
    print(f"{'='*70}\n")

    prev_images: torch.Tensor | None = None

    for frame_idx in range(num_frames):
        # Apply random effort to the cart joint only
        efforts = torch.zeros(num_envs, num_dofs)
        efforts[:, cart_idx] = 200.0 * (2.0 * torch.rand(num_envs) - 1.0)
        cartpole_view.set_joint_efforts(efforts)

        # Physics-only steps (no render)
        SimulationManager.step(steps=physics_steps_between)

        # One more physics step + render; RenderingManager.render() calls
        # app.update() with physics paused so replicator annotators update.
        SimulationManager.step()
        RenderingManager.render()

        # Collect RGB from each annotator: numpy array [H, W, 4] (RGBA uint8)
        rgb_list = []
        for ann in annotators:
            data = ann.get_data()
            if data is not None and "data" in data and data["data"] is not None:
                arr = data["data"].reshape(IMG_HEIGHT, IMG_WIDTH, 4)
                rgb_list.append(torch.from_numpy(arr[:, :, :3].copy()))  # drop alpha → [H,W,3]
            else:
                rgb_list.append(torch.zeros(IMG_HEIGHT, IMG_WIDTH, 3, dtype=torch.uint8))

        rgb_images = torch.stack(rgb_list, dim=0)  # [N, H, W, 3]

        # Read joint state (get_joint_positions returns numpy)
        joint_pos = cartpole_view.get_joint_positions()
        cart_pos = joint_pos[:, cart_idx]
        pole_ang = joint_pos[:, pole_idx]

        print(f"Frame {frame_idx}: cart_pos={cart_pos}, pole_angle={pole_ang}")

        mean_px = rgb_images.float().mean(dim=(1, 2, 3))
        print(f"  Mean pixel per env: {mean_px.numpy()}")

        if prev_images is not None:
            diff = (rgb_images.float() - prev_images.float()).abs().mean(dim=(1, 2, 3))
            print(f"  Abs diff from prev frame (per env): {diff.numpy()}")
            if diff.max().item() < 1.0:
                print("  *** WARNING: Images appear IDENTICAL to previous frame despite physics update! ***")
        else:
            print("  (first frame — no diff)")

        prev_images = rgb_images.clone()

        # Save images
        try:
            from torchvision.utils import make_grid, save_image

            imgs_f = rgb_images.float() / 255.0  # [N, H, W, 3]
            imgs_nchw = imgs_f.permute(0, 3, 1, 2)  # [N, 3, H, W]
            for env_idx in range(num_envs):
                save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}_env_{env_idx:03d}.png")
                save_image(imgs_nchw[env_idx], save_path)
            nrow = max(1, int(math.ceil(math.sqrt(num_envs))))
            grid = make_grid(imgs_nchw, nrow=nrow)
            grid_path = os.path.join(output_dir, f"frame_{frame_idx:03d}_grid.png")
            save_image(grid, grid_path)
            print(f"  Saved grid → {grid_path}")
        except ImportError:
            save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}_raw.pt")
            torch.save(rgb_images, save_path)
            print(f"  torchvision not available; saved raw tensor → {save_path}")

        print()

    # ---- Summary ------------------------------------------------------------
    print(f"{'='*70}")
    print(f"Done. {num_frames} frames saved to: {output_dir}")
    print(f"{'='*70}")
    print("\nIf 'Abs diff from prev frame' is near-zero despite joint positions changing,")
    print("this confirms the camera frame update bug — renderer returning stale images.")

    timeline.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()
