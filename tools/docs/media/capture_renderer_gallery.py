# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the renderer documentation gallery from an editable USD scene."""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable, Iterator
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
_OVRTX_AMBIENT_LIGHT_SETTING = "float omni:rtx:rt:ambientLight:intensity = 1.0"


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


def depth_display_bounds(data: Any) -> tuple[float, float]:
    """Return the minimum and maximum finite depths in a frame."""
    import torch

    finite_depth = data[torch.isfinite(data)]
    if finite_depth.numel() == 0:
        raise ValueError("Depth frame does not contain finite samples.")
    return float(finite_depth.min()), float(finite_depth.max())


def motion_vectors_to_image(data: Any) -> Any:
    """Colorize motion vectors and overlay sparse image-space direction arrows."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    data = snapshot_camera_tensor(data)
    uv = data[..., :2].float()
    raw_magnitude = torch.linalg.vector_norm(uv, dim=-1)
    max_magnitude = max(float(raw_magnitude.quantile(0.99)), 1.0e-6)
    normalized_uv = (uv / max_magnitude).clamp(-1.0, 1.0)
    normalized_magnitude = torch.linalg.vector_norm(normalized_uv, dim=-1).clamp(0.0, 1.0)
    array = (
        torch.cat(((normalized_uv + 1.0) * 0.5, normalized_magnitude.unsqueeze(-1)), dim=-1)
        .mul(255)
        .to(torch.uint8)
        .numpy()
    )
    image = Image.fromarray(np.ascontiguousarray(array).copy(), mode="RGB")
    draw = ImageDraw.Draw(image)

    uv_array = normalized_uv.numpy()
    magnitude_array = normalized_magnitude.numpy()
    height, width = magnitude_array.shape
    grid_spacing = max(min(height, width) // 10, 16)
    arrow_length = grid_spacing * 0.45
    head_length = max(grid_spacing * 0.18, 4.0)

    for cell_y in range(0, height, grid_spacing):
        for cell_x in range(0, width, grid_spacing):
            cell = magnitude_array[
                cell_y : min(cell_y + grid_spacing, height),
                cell_x : min(cell_x + grid_spacing, width),
            ]
            if cell.size == 0 or float(cell.max()) < 0.08:
                continue
            local_y, local_x = np.unravel_index(int(cell.argmax()), cell.shape)
            y = cell_y + int(local_y)
            x = cell_x + int(local_x)
            direction = uv_array[y, x].copy()
            # Image-space v points up, while raster screen y points down.
            direction[1] *= -1.0
            direction_length = float(np.linalg.norm(direction))
            if direction_length < 1.0e-6:
                continue

            unit = direction / direction_length
            end = np.array((x, y), dtype=np.float32) + unit * arrow_length * float(magnitude_array[y, x])
            start_xy = (float(x), float(y))
            end_xy = (float(end[0]), float(end[1]))
            draw.line((start_xy, end_xy), fill=(0, 0, 0), width=4)
            draw.line((start_xy, end_xy), fill=(255, 255, 255), width=2)

            perpendicular = np.array((-unit[1], unit[0]), dtype=np.float32)
            head_base = end - unit * head_length
            head_half_width = head_length * 0.55
            draw.polygon(
                (
                    end_xy,
                    tuple(head_base + perpendicular * head_half_width),
                    tuple(head_base - perpendicular * head_half_width),
                ),
                fill=(255, 255, 255),
            )

    return image


def thumbnail_frame_index(frame_count: int) -> int:
    """Select the sixth captured frame so temporal outputs have useful motion history."""
    if frame_count < 6:
        raise ValueError("At least six animation frames are required to capture thumbnails.")
    return 5


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
    parser.add_argument("--newton-shadows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene", type=Path, required=True)
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
    if args.frames < 6:
        parser.error("At least six animation frames are required.")
    if args.physics_steps_per_frame < 1:
        parser.error("Physics steps per frame must be positive.")
    if args.warmup_steps < 0:
        parser.error("Warmup steps must be non-negative.")
    try:
        capture_data_types(args.renderer_backend, args.capture_group)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _make_renderer_cfg(renderer: str, *, enable_shadows: bool = True) -> Any:
    """Create the selected renderer configuration."""
    if renderer == "newton":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg(enable_shadows=enable_shadows, enable_ambient_lighting=False)
    if renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if renderer == "isaac_rtx":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

        return IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(ambient_light_intensity=0.0))
    raise ValueError(f"Unknown renderer: {renderer}")


def override_ovrtx_ambient_light(render_product_usd: str) -> str:
    """Disable renderer-authored ambient fill while preserving the scene authored lights."""
    if _OVRTX_AMBIENT_LIGHT_SETTING not in render_product_usd:
        raise RuntimeError("Expected the OVRTX render product to author its default ambient-light intensity.")
    return render_product_usd.replace(
        _OVRTX_AMBIENT_LIGHT_SETTING,
        "float omni:rtx:rt:ambientLight:intensity = 0.0",
        1,
    )


@contextlib.contextmanager
def gallery_lighting_override(renderer: str) -> Iterator[None]:
    """Apply capture-only renderer lighting overrides for the duration of camera creation."""
    if renderer != "ovrtx":
        yield
        return

    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer

    original_builder = ovrtx_renderer.build_render_product_as_string

    def build_render_product_without_ambient_light(*args: Any, **kwargs: Any) -> tuple[str, str]:
        render_product_usd, render_product_path = original_builder(*args, **kwargs)
        return override_ovrtx_ambient_light(render_product_usd), render_product_path

    ovrtx_renderer.build_render_product_as_string = build_render_product_without_ambient_light
    try:
        yield
    finally:
        ovrtx_renderer.build_render_product_as_string = original_builder


def _create_camera_and_reset(renderer: str, create_camera: Callable[[], Any], reset_sim: Callable[[], None]) -> Any:
    """Create the camera and initialize its renderer under capture lighting overrides."""
    with gallery_lighting_override(renderer):
        camera = create_camera()
        reset_sim()
    return camera


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
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, device=args.device, use_fabric=True)
    else:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        sim_utils.create_new_stage()
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            device=args.device,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"), num_substeps=8),
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
    camera = _create_camera_and_reset(
        args.renderer_backend,
        lambda: Camera(
            CameraCfg(
                prim_path=camera_path,
                update_period=0.0,
                width=args.width,
                height=args.height,
                data_types=list(data_types),
                spawn=None,
                renderer_cfg=_make_renderer_cfg(args.renderer_backend, enable_shadows=args.newton_shadows),
            )
        ),
        sim.reset,
    )

    def tensor_to_image(data: torch.Tensor, output_name: str) -> Image.Image:
        data = snapshot_camera_tensor(data)
        if output_name == "depth":
            depth_min, depth_max = depth_display_bounds(data)
            array = CameraFrameColorizer.colorize(data, "depth", depth_min=depth_min, depth_max=depth_max)
        elif output_name == "normals":
            array = CameraFrameColorizer.colorize(data, "normals")
        elif output_name in {"semantic_segmentation", "instance_segmentation"}:
            if data.ndim == 3 and data.shape[-1] >= 3 and data.dtype == torch.uint8:
                array = data[..., :3].numpy()
            else:
                array = CameraFrameColorizer.colorize(data, "segmentation")
        elif output_name == "motion_vectors":
            return motion_vectors_to_image(data)
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

        thumbnail_index = thumbnail_frame_index(args.frames)
        if args.capture_group == "standard":
            rgb_frames: list[Image.Image] = []
            thumbnail_frames: dict[str, Image.Image] = {}
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
                if frame_index == thumbnail_index:
                    for output_name in data_types:
                        if output_name != "rgb":
                            thumbnail_frames[output_name] = tensor_to_image(
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
            for output_name, image in thumbnail_frames.items():
                output_path = args.output_dir / gallery_asset_name(args.renderer_backend, output_name)
                image.save(output_path, optimize=True)
                print(f"[INFO] Wrote {output_path}", flush=True)
        else:
            for _ in range(thumbnail_index * args.physics_steps_per_frame):
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
