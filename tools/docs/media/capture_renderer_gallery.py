# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the renderer documentation gallery from an editable USD scene."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GalleryMode:
    """One visually distinct camera output included in the gallery."""

    output_name: str
    label: str
    animated: bool = False


_COMMON_MODES = (
    GalleryMode("rgb", "RGB", animated=True),
    GalleryMode("albedo", "Albedo"),
    GalleryMode("depth", "Depth"),
    GalleryMode("normals", "Normals"),
    GalleryMode("semantic_segmentation", "Semantic segmentation"),
    GalleryMode("instance_segmentation", "Instance segmentation"),
)
_RTX_ONLY_MODES = (
    GalleryMode("motion_vectors", "Motion vectors"),
    GalleryMode("simple_shading_constant_diffuse", "Constant diffuse"),
    GalleryMode("simple_shading_diffuse_mdl", "Diffuse MDL"),
    GalleryMode("simple_shading_full_mdl", "Full MDL"),
)
_RENDERER_SLUGS = {"newton": "newton", "ovrtx": "ovrtx", "isaac_rtx": "isaac-rtx"}
_SIMPLE_SHADING_MODES = tuple(mode.output_name for mode in _RTX_ONLY_MODES if mode.output_name.startswith("simple_"))


def gallery_modes(renderer: str) -> tuple[GalleryMode, ...]:
    """Return the visually distinct outputs documented for a renderer backend."""
    if renderer == "newton":
        return _COMMON_MODES
    if renderer in {"ovrtx", "isaac_rtx"}:
        return (*_COMMON_MODES, *_RTX_ONLY_MODES)
    raise ValueError(f"Unknown renderer: {renderer}")


def capture_data_types(renderer: str, capture_group: str) -> tuple[str, ...]:
    """Return the camera outputs captured together for one renderer process."""
    modes = gallery_modes(renderer)
    if capture_group == "standard":
        return tuple(mode.output_name for mode in modes if not mode.output_name.startswith("simple_shading_"))
    if any(mode.output_name == capture_group for mode in modes) and capture_group.startswith("simple_shading_"):
        return (capture_group,)
    raise ValueError(f"Capture group {capture_group!r} is not available for renderer {renderer!r}.")


def gallery_asset_name(renderer: str, output_name: str) -> str:
    """Return the published documentation asset name for one renderer output."""
    try:
        renderer_slug = _RENDERER_SLUGS[renderer]
    except KeyError as exc:
        raise ValueError(f"Unknown renderer: {renderer}") from exc
    suffix = "webp" if output_name == "rgb" else "png"
    output_suffix = "" if output_name == "rgb" else f"-{output_name.replace('_', '-')}"
    return f"camera-renderer-{renderer_slug}{output_suffix}.{suffix}"


def snapshot_camera_tensor(data: Any) -> Any:
    """Copy a renderer-owned tensor to stable CPU storage."""
    return data.detach().to(device="cpu", copy=True)


def poster_frame_index(frame_count: int) -> int:
    """Select the still-image frame after two thirds of the fall."""
    return min(frame_count - 1, frame_count * 2 // 3)


def renderer_requires_kit(renderer: str) -> bool:
    """Return whether a renderer must run inside Isaac Sim Kit."""
    return renderer == "isaac_rtx"


def gallery_stage_paths() -> tuple[str, str]:
    """Return the single-environment scene path and matching camera expression."""
    return "/World/envs/env_0/Scene", "/World/envs/env_.*/Scene/Camera"


def add_gallery_arguments(parser: argparse.ArgumentParser) -> None:
    """Add renderer-gallery arguments without colliding with AppLauncher options."""
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--renderer-backend", choices=tuple(_RENDERER_SLUGS), required=True)
    parser.add_argument("--capture-group", choices=("standard", *_SIMPLE_SHADING_MODES), default="standard")
    parser.add_argument("--scene", type=Path, default=script_dir / "renderer_gallery_scene.usda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parents[2] / "docs" / "source" / "_static" / "overview" / "sensors",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=37)
    parser.add_argument("--physics-steps-per-frame", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=24)


def _parse_args() -> argparse.Namespace:
    """Parse capture and Isaac Lab launcher arguments."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    add_gallery_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)
    args = parser.parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.scene.is_file():
        parser.error(f"Scene does not exist: {args.scene}")
    if args.width < 1 or args.height < 1:
        parser.error("Image width and height must be positive.")
    if args.frames < 2:
        parser.error("At least two animation frames are required.")
    if args.physics_steps_per_frame < 1:
        parser.error("Physics steps per frame must be positive.")
    if args.warmup_steps < 0:
        parser.error("Warmup steps must be non-negative.")
    try:
        capture_data_types(args.renderer_backend, args.capture_group)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _make_renderer_cfg(renderer: str) -> Any:
    """Create the selected renderer configuration."""
    if renderer == "newton":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg()
    if renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if renderer == "isaac_rtx":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg()
    raise ValueError(f"Unknown renderer: {renderer}")


def _capture(args: argparse.Namespace) -> None:
    """Load the USD scene, render the selected outputs, and write documentation assets."""
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab import cloner
    from isaaclab.envs.utils.camera_colorizer import CameraFrameColorizer
    from isaaclab.sensors import Camera, CameraCfg

    if renderer_requires_kit(args.renderer_backend):
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, device=args.device, use_fabric=False)
    else:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        sim_utils.create_new_stage()
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            device=args.device,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1),
        )

    data_types = capture_data_types(args.renderer_backend, args.capture_group)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_path, camera_path = gallery_stage_paths()
    stage = sim_utils.get_current_stage()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    scene_cfg = sim_utils.UsdFileCfg(usd_path=str(args.scene))
    scene_cfg.func(scene_path, scene_cfg)
    env_positions = torch.zeros((1, 3), device=args.device)
    clone_plan = cloner.clone_plan_from_env_0("/World/envs/env_0", "/World/envs/env_{}", 1, args.device, env_positions)
    cloner.replicate(clone_plan, stage=stage)
    camera = Camera(
        CameraCfg(
            prim_path=camera_path,
            update_period=0.0,
            width=args.width,
            height=args.height,
            data_types=list(data_types),
            spawn=None,
            renderer_cfg=_make_renderer_cfg(args.renderer_backend),
        )
    )

    sim.reset()

    def tensor_to_image(data: torch.Tensor, output_name: str) -> Image.Image:
        data = snapshot_camera_tensor(data)
        if output_name == "depth":
            array = CameraFrameColorizer.colorize(data, "depth", depth_min=2.0, depth_max=13.0)
        elif output_name == "normals":
            array = CameraFrameColorizer.colorize(data, "normals")
        elif output_name in {"semantic_segmentation", "instance_segmentation"}:
            if data.ndim == 3 and data.shape[-1] >= 3 and data.dtype == torch.uint8:
                array = data[..., :3].numpy()
            else:
                array = CameraFrameColorizer.colorize(data, "segmentation")
        elif output_name == "motion_vectors":
            uv = data[..., :2].float()
            max_magnitude = max(float(uv.abs().quantile(0.99)), 1.0e-6)
            uv = (uv / max_magnitude).clamp(-1.0, 1.0)
            magnitude = torch.linalg.vector_norm(uv, dim=-1, keepdim=True).clamp(0.0, 1.0)
            array = torch.cat(((uv + 1.0) * 0.5, magnitude), dim=-1).mul(255).to(torch.uint8).numpy()
        else:
            array = data[..., :3].numpy()
            if array.dtype != np.uint8:
                array = np.clip(array, 0.0, 1.0)
                array = (array * 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(array).copy(), mode="RGB")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(args.warmup_steps):
            camera.update(dt=0.0, force_recompute=True)

        poster_index = poster_frame_index(args.frames)
        if args.capture_group == "standard":
            rgb_frames: list[Image.Image] = []
            poster_frames: dict[str, Image.Image] = {}
            for frame_index in range(args.frames):
                if frame_index > 0:
                    for _ in range(args.physics_steps_per_frame):
                        sim.step(render=False)
                camera.update(
                    dt=sim.get_physics_dt() * args.physics_steps_per_frame,
                    force_recompute=True,
                )
                if "rgb" in data_types:
                    rgb_frames.append(tensor_to_image(camera.data.output["rgb"].torch[0], "rgb"))
                if frame_index == poster_index:
                    for output_name in data_types:
                        if output_name != "rgb":
                            poster_frames[output_name] = tensor_to_image(
                                camera.data.output[output_name].torch[0], output_name
                            )

            rgb_path = args.output_dir / gallery_asset_name(args.renderer_backend, "rgb")
            rgb_frames[0].save(
                rgb_path,
                save_all=True,
                append_images=rgb_frames[1:],
                duration=round(1000 / 12),
                loop=0,
                quality=86,
                method=6,
            )
            print(f"[INFO] Wrote {rgb_path}", flush=True)
            for output_name, image in poster_frames.items():
                output_path = args.output_dir / gallery_asset_name(args.renderer_backend, output_name)
                image.save(output_path, optimize=True)
                print(f"[INFO] Wrote {output_path}", flush=True)
        else:
            for _ in range(poster_index * args.physics_steps_per_frame):
                sim.step(render=False)
            camera.update(
                dt=sim.get_physics_dt() * args.physics_steps_per_frame,
                force_recompute=True,
            )
            output_name = data_types[0]
            image = tensor_to_image(camera.data.output[output_name].torch[0], output_name)
            output_path = args.output_dir / gallery_asset_name(args.renderer_backend, output_name)
            image.save(output_path, optimize=True)
            print(f"[INFO] Wrote {output_path}", flush=True)
    finally:
        sim.stop()
        sim.clear_instance()


def main() -> None:
    """Launch Isaac Sim and capture the selected renderer gallery group."""
    from isaaclab.app import AppLauncher

    args = _parse_args()
    if not renderer_requires_kit(args.renderer_backend):
        _capture(args)
    else:
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        try:
            _capture(args)
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
