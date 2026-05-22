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
        --input_scene /path/to/scene.usd --renderer newton --visualizer none --max_steps 60

    # Run the same saved-image workflow with Isaac RTX.
    ./isaaclab.sh -p scripts/demos/sensors/ppisp_camera.py \
        --input_scene /path/to/scene.usd --renderer isaac_rtx --visualizer none --max_steps 60

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
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
    help="Optional camera prim path override. Omit to auto-select the first PPISP-bound camera.",
)
parser.add_argument(
    "--camera_time_code",
    type=float,
    default=0.0,
    help="USD time code used to bake the selected camera pose into duplicated env cameras for Newton.",
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
    choices=["newton", "isaac_rtx"],
    default="newton",
    help="Camera renderer backend to use. Newton Warp is the default for this PPISP smoke.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=None,
    help="Simulation/render steps to run before saving images. Defaults to 32 for Isaac RTX and 0 for Newton.",
)
parser.add_argument("--max_steps", type=int, default=120, help="Maximum simulation steps before exiting.")
parser.add_argument("--save_interval", type=int, default=20, help="Interval, in steps, for saving comparison images.")
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
if args_cli.max_steps < 1:
    parser.error("--max_steps must be at least 1.")
if args_cli.save_interval < 1:
    parser.error("--save_interval must be at least 1.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab_ppisp.cfg import PpispCfg, ppisp_cfg_from_usd_shader

from pxr import Usd, UsdGeom, UsdShade

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
    if args_cli.renderer == "newton":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg()
    else:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg()


def make_sim_cfg() -> sim_utils.SimulationCfg:
    """Create the simulation cfg matching the selected renderer."""
    physics_cfg = None
    if args_cli.renderer == "newton":
        from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

        physics_cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)

    return sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physics=physics_cfg,
        use_fabric=not args_cli.disable_fabric,
    )


def find_ppisp_camera_bindings(stage: Usd.Stage) -> list[tuple[str, Usd.Prim, Usd.Prim]]:
    """Return camera, RenderProduct, and PPISP shader bindings from a source stage."""
    bindings = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "RenderProduct":
            continue
        shader_prim = stage.GetPrimAtPath(prim.GetPath().AppendChild("PPISP"))
        if not shader_prim or not shader_prim.IsValid():
            continue
        camera_rel = prim.GetRelationship("camera")
        if not camera_rel:
            continue
        for target in camera_rel.GetTargets():
            camera_prim = stage.GetPrimAtPath(target)
            if camera_prim and camera_prim.IsValid():
                bindings.append((str(target), prim, shader_prim))
    return bindings


def order_ppisp_bindings_by_camera(
    stage: Usd.Stage, ppisp_bindings: list[tuple[str, Usd.Prim, Usd.Prim]]
) -> list[tuple[str, Usd.Prim, Usd.Prim]]:
    """Return PPISP bindings ordered by source Camera prim traversal."""
    binding_by_camera_path = {}
    for binding in ppisp_bindings:
        binding_by_camera_path.setdefault(binding[0], binding)

    ordered_bindings = []
    seen_paths = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Camera":
            continue
        camera_path = str(prim.GetPath())
        binding = binding_by_camera_path.get(camera_path)
        if binding is not None:
            ordered_bindings.append(binding)
            seen_paths.add(camera_path)

    for binding in ppisp_bindings:
        if binding[0] not in seen_paths:
            ordered_bindings.append(binding)
            seen_paths.add(binding[0])
    return ordered_bindings


def format_available_ppisp_cameras(ppisp_bindings: list[tuple[str, Usd.Prim, Usd.Prim]]) -> str:
    """Format PPISP-bound camera paths for CLI error messages."""
    return "\n  ".join(dict.fromkeys(binding[0] for binding in ppisp_bindings))


def resolve_source_camera_binding(source_stage: Usd.Stage) -> tuple[str, Usd.Prim, Usd.Prim]:
    """Resolve the source camera and PPISP shader binding from CLI or source stage metadata."""
    ppisp_bindings = order_ppisp_bindings_by_camera(source_stage, find_ppisp_camera_bindings(source_stage))
    if not ppisp_bindings:
        raise RuntimeError("No PPISP RenderProduct bindings found in input scene.")

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
            "Omit --camera_prim_path to auto-select a PPISP-bound camera, or use one of:\n"
            f"  {available}"
        )
    if camera_prim.GetTypeName() != "Camera":
        available = format_available_ppisp_cameras(ppisp_bindings)
        raise RuntimeError(
            f"Prim is not a Camera: {camera_prim_path} ({camera_prim.GetTypeName()})\n"
            "Omit --camera_prim_path to auto-select a PPISP-bound camera, or use one of:\n"
            f"  {available}"
        )

    for binding in ppisp_bindings:
        if binding[0] == camera_prim_path:
            return binding

    available = format_available_ppisp_cameras(ppisp_bindings)
    raise RuntimeError(
        f"Selected camera has no PPISP RenderProduct binding: {camera_prim_path}\n"
        "Omit --camera_prim_path to auto-select a PPISP-bound camera, or use one of:\n"
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


def bake_source_camera_pose_to_envs(source_stage: Usd.Stage, source_camera_prim_path: str) -> None:
    """Bake the selected USD camera pose at ``camera_time_code`` into duplicated env camera prims."""
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Input scene must have a defaultPrim so it can be referenced under each env.")

    source_camera_prim = source_stage.GetPrimAtPath(source_camera_prim_path)
    if not source_camera_prim or not source_camera_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {source_camera_prim_path}")

    time_code = Usd.TimeCode(args_cli.camera_time_code)
    source_cache = UsdGeom.XformCache(time_code)
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

    print(
        f"[INFO] Baked camera pose at USD time {args_cli.camera_time_code:g} into {authored_count} env camera(s).",
        flush=True,
    )


def get_render_product_resolution(render_product_prim: Usd.Prim) -> tuple[int, int] | None:
    """Return ``(width, height)`` from a RenderProduct ``resolution`` attribute."""
    resolution_attr = render_product_prim.GetAttribute("resolution")
    if not resolution_attr:
        return None
    resolution = resolution_attr.Get()
    if resolution is None or len(resolution) != 2:
        return None
    return int(resolution[0]), int(resolution[1])


def resolve_image_shape(render_product_prim: Usd.Prim) -> tuple[int, int]:
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


def make_ppisp_cfg(shader_prim: Usd.Prim, num_ppisp_bindings: int) -> PpispCfg:
    """Parse the selected source PPISP shader into an explicit cfg for duplicated envs."""
    ppisp_cfg = ppisp_cfg_from_usd_shader(UsdShade.Shader(shader_prim))
    # The duplicated stage does not carry the source /Render scope; keep the
    # parsed inputs as explicit values instead of resolving this source path later.
    ppisp_cfg.shader_prim_path = None
    if args_cli.ppisp_responsivity is None:
        print(f"[INFO] Using USD-authored PPISP values from {num_ppisp_bindings} PPISP shader(s).", flush=True)
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


def save_images_grid(
    images: list[torch.Tensor],
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
        ax.imshow(img.detach().cpu().clamp(0.0, 1.0).numpy())
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


def make_tiled_image(images: torch.Tensor) -> torch.Tensor:
    """Stack a camera batch vertically into one image."""
    return torch.cat([image for image in images], dim=0)


def save_tensor_image(image: torch.Tensor, filename: str) -> None:
    """Save a tensor image in [0, 1] without axes, titles, or layout scaling."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.imsave(filename, image.detach().cpu().clamp(0.0, 1.0).numpy())


def corner_center_ratio(rgb: torch.Tensor) -> float:
    """Return the average corner brightness divided by center brightness for one RGB image."""
    h, w = rgb.shape[:2]
    patch = max(4, min(h, w) // 8)
    cy, cx = h // 2 - patch // 2, w // 2 - patch // 2
    center = rgb[cy : cy + patch, cx : cx + patch, :3].float().mean()
    corners = torch.stack(
        [
            rgb[:patch, :patch, :3].float().mean(),
            rgb[:patch, -patch:, :3].float().mean(),
            rgb[-patch:, :patch, :3].float().mean(),
            rgb[-patch:, -patch:, :3].float().mean(),
        ]
    ).mean()
    return (corners / center.clamp_min(1.0)).item()


def run_simulator(sim: sim_utils.SimulationContext, baseline_camera: Camera, ppisp_camera: Camera) -> None:
    """Run the simulator and periodically save baseline-vs-PPISP images."""
    sim_dt = sim.get_physics_dt()
    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ppisp_camera")
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.warmup_steps > 0:
        print(f"[INFO] Running {args_cli.warmup_steps} warmup step(s) before saving images.", flush=True)
    for _ in range(args_cli.warmup_steps):
        sim.step()
        baseline_camera.update(sim_dt)
        ppisp_camera.update(sim_dt)

    count = 0
    reported_shape = False
    while simulation_app.is_running():
        sim.step()
        baseline_camera.update(sim_dt)
        ppisp_camera.update(sim_dt)
        count += 1

        if count % args_cli.save_interval == 0:
            baseline = baseline_camera.data.output["rgb"][..., :3]
            ppisp = ppisp_camera.data.output["rgb"][..., :3]
            diff = (ppisp.float() - baseline.float()).abs() / 255.0
            if not reported_shape:
                print(f"[INFO] camera batch rgb shape={tuple(ppisp.shape)}", flush=True)
                reported_shape = True
            mean_abs_delta = diff.mean().item() * 255.0
            ratios = [corner_center_ratio(ppisp[i]) for i in range(ppisp.shape[0])]
            ratio = sum(ratios) / len(ratios)
            per_env_delta = diff.mean(dim=(1, 2, 3)) * 255.0
            per_env_ppisp_mean = ppisp.float().mean(dim=(1, 2, 3))
            print(
                f"[INFO] step={count} mean_abs_delta={mean_abs_delta:.2f} mean_ppisp_corner_center_ratio={ratio:.3f}",
                flush=True,
            )
            print(
                "[INFO] per-env mean_abs_delta="
                + ", ".join(f"{value:.2f}" for value in per_env_delta.detach().cpu().tolist()),
                flush=True,
            )
            print(
                "[INFO] per-env ppisp_mean="
                + ", ".join(f"{value:.2f}" for value in per_env_ppisp_mean.detach().cpu().tolist()),
                flush=True,
            )
            images = []
            subtitles = []
            for env_id in range(ppisp.shape[0]):
                images.extend(
                    [
                        baseline[env_id].float() / 255.0,
                        ppisp[env_id].float() / 255.0,
                        diff[env_id],
                    ]
                )
                subtitles.extend([f"env {env_id} baseline", f"env {env_id} PPISP", f"env {env_id} diff"])
            save_images_grid(
                images,
                nrow=ppisp.shape[0],
                subtitles=subtitles,
                title="USD-authored PPISP on duplicated Gaussian scene envs",
                filename=os.path.join(output_dir, f"ppisp_camera_{count:04d}.png"),
            )
            save_tensor_image(
                make_tiled_image(baseline.float() / 255.0),
                os.path.join(output_dir, f"ppisp_camera_{count:04d}_baseline_tiled.png"),
            )
            save_tensor_image(
                make_tiled_image(ppisp.float() / 255.0),
                os.path.join(output_dir, f"ppisp_camera_{count:04d}_ppisp_tiled.png"),
            )
            save_tensor_image(
                make_tiled_image(diff),
                os.path.join(output_dir, f"ppisp_camera_{count:04d}_diff_tiled.png"),
            )

        if args_cli.max_steps is not None and count >= args_cli.max_steps:
            break


def main() -> None:
    """Main function."""
    source_stage = Usd.Stage.Open(args_cli.input_scene)
    if source_stage is None:
        raise RuntimeError(f"Failed to open input scene: {args_cli.input_scene}")
    source_camera_prim_path, render_product_prim, shader_prim = resolve_source_camera_binding(source_stage)
    ppisp_cfg = make_ppisp_cfg(shader_prim, len(find_ppisp_camera_bindings(source_stage)))
    camera_prim_path = source_camera_path_to_env_regex(source_stage, source_camera_prim_path)
    width, height = resolve_image_shape(render_product_prim)

    sim_utils.create_new_stage()
    sim_cfg = make_sim_cfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    scene = create_duplicated_env_scene()
    if args_cli.renderer == "newton":
        bake_source_camera_pose_to_envs(source_stage, source_camera_prim_path)
    baseline_camera = make_camera(camera_prim_path, ppisp_cfg=None, width=width, height=height)
    ppisp_camera = make_camera(camera_prim_path, ppisp_cfg=ppisp_cfg, width=width, height=height)
    print(f"[INFO] Duplicated-env camera regex: {camera_prim_path}", flush=True)
    print(f"[INFO] Rendering {width}x{height} from source camera {source_camera_prim_path}.", flush=True)

    sim.reset()
    print("[INFO]: Setup complete. Saving comparison images during simulation.", flush=True)
    run_simulator(sim, baseline_camera, ppisp_camera)
    del scene


if __name__ == "__main__":
    main()
    simulation_app.close()
