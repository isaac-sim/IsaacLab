# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded GPU check of calibrated dense/Isaac Lab Fovi rendering against analytic geometry.

Run separately for each --backend {rtx,newton} and --sampling {dense,foveated}
with --visualizer none --device cuda:0. Both sampling modes are provided by Isaac Lab Fovi.
The reference uses OpenCV and analytic ray/box intersections, not renderer rays.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from fovi.sensing.projection import CameraCalibration
    from isaaclab.sensors import Camera

HEIGHT, WIDTH = 120, 192
FAR = 500.0
NEAR = 0.01
BOXES = (
    ((0.0, 0.0, -4.05), (200.0, 200.0, 0.1)),
    ((0.8, 0.3, -2.15), (0.6, 0.8, 0.3)),
)


def audit_projection(cameras: dict[str, Camera], output: Path) -> None:
    """Record and verify the camera properties actually seen by the renderer."""
    import isaaclab.sim as sim_utils
    import omni.kit.app
    from pxr import Usd

    records = {}
    for name, camera in cameras.items():
        prim = sim_utils.get_current_stage().GetPrimAtPath(camera.cfg.prim_path)
        schemas = list(prim.GetMetadata("apiSchemas").GetAppliedItems())
        properties = {
            attr.GetName(): str(attr.Get())
            for attr in prim.GetAuthoredAttributes()
            if "Projection" in attr.GetName() or attr.GetName().startswith("omni:lensdistortion:")
        }
        assert "OmniLensDistortionLutAPI" in schemas, (name, schemas)
        assert properties["omni:lensdistortion:model"] == "lut", (name, properties)
        assert properties.get("cameraProjectionType") != "generalizedProjection", (name, properties)
        assert not any("generalizedProjection" in key for key in properties), (name, properties)
        for texture in ("rayEnterDirectionTexture", "rayExitPositionTexture"):
            assert properties[f"omni:lensdistortion:lut:{texture}"], (name, properties)
        records[name] = {"schemas": schemas, "properties": properties}
    runtime = {
        "schema_registered": bool(Usd.SchemaRegistry.GetTypeFromSchemaTypeName("OmniLensDistortionLutAPI")),
        "extension_enabled": omni.kit.app.get_app()
        .get_extension_manager()
        .is_extension_enabled("omni.usd.schema.omni_lens_distortion"),
    }
    output.write_text(json.dumps({"runtime": runtime, "cameras": records}, indent=2) + "\n")


def video_maps(cameras: dict[str, Camera], size: int) -> dict[str, np.ndarray]:
    """Nearest-sample visual-field reconstruction; the periphery stays coarse."""
    from scipy.spatial import cKDTree

    y, x = np.mgrid[:size, :size]
    points = np.stack((2 * (x + 0.5) / size - 1, 1 - 2 * (y + 0.5) / size), axis=-1)
    maps = {}
    for name, camera in cameras.items():
        coords = camera.sampling_coords.cartesian.cpu().numpy()
        indices = cKDTree(coords).query(points.reshape(-1, 2))[1].reshape(size, size)
        indices[np.linalg.norm(points, axis=-1) > 1] = -1
        maps[name] = indices
    return maps


def video_frame(
    frames: dict[str, dict[str, np.ndarray]],
    maps: dict[str, np.ndarray],
    phase: int,
    mode: str,
    fixation: tuple[float, float],
) -> np.ndarray:
    """Pair measured RGB with analytic foreground/mask reference for every lens."""
    import cv2

    size = next(iter(maps.values())).shape[0]
    tile_width, tile_height = size * 2 + 16, size + 32
    canvas = np.full((tile_height * 3 + 64, tile_width * 2, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, f"Foveated | {mode} | frame {phase - 4}", (12, 24), 0, 0.6, (240,) * 3, 1)
    cv2.putText(
        canvas,
        f"Fixation y,x: {fixation[0]:.3f}, {fixation[1]:.3f} | RGB / analytic reference",
        (12, 48),
        0,
        0.5,
        (240,) * 3,
        1,
    )
    for index, (name, lookup) in enumerate(maps.items()):
        frame = frames[f"{name}_{phase}"]
        actual = frame["rgb"].reshape(-1, 3).astype(np.uint8)
        reference = np.zeros_like(actual)
        reference[:, 2] = 200
        reference[frame["expected"] < 3] = (200, 0, 0)
        reference[~frame["valid"]] = 0
        top, left = 64 + (index // 2) * tile_height, (index % 2) * tile_width
        cv2.putText(canvas, name, (left + 8, top + 22), 0, 0.5, (240,) * 3, 1)
        for column, samples in enumerate((actual, reference)):
            image = samples[lookup].copy()
            image[lookup < 0] = 0
            x = left + 4 + column * (size + 8)
            canvas[top + 32 : top + 32 + size, x : x + size] = image[:, :, ::-1]
    return canvas


def unproject(pixels: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """Independent OpenCV reference, returning unit OpenCV-frame directions."""
    import cv2

    fx, fy, cx, cy = calibration["intrinsics"]
    matrix = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
    coefficients = np.array(calibration["distortion"], dtype=np.float64)
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if calibration["model"] == "fisheye":
        xy = cv2.fisheye.undistortPoints(points, matrix, coefficients)[:, 0]
    else:
        xy = cv2.undistortPointsIter(
            points,
            matrix,
            coefficients,
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12),
        )[:, 0]
    rays = np.column_stack((xy, np.ones(len(xy))))
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def ray_depth(rays: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """First intersection distance [m] with the two axis-aligned scene boxes."""
    result = np.full(len(rays), FAR)
    for center, size in BOXES:
        lo = np.array(center) - np.array(size) / 2
        hi = np.array(center) + np.array(size) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - origin) / rays, (hi - origin) / rays
        near = np.minimum(a, b).max(axis=-1)
        far = np.maximum(a, b).min(axis=-1)
        hit = (far >= near) & (near > 0)
        result = np.minimum(result, np.where(hit, near, FAR))
    return result


def reference_rays(
    camera: Camera, calibration: CameraCalibration, sampling: str, fixation: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical GL-frame rays and lens validity for the public sample grid."""
    import cv2

    y, x = np.mgrid[:HEIGHT, :WIDTH]
    pixels = np.stack((x, y), -1).reshape(-1, 2)
    if sampling == "dense":
        rays = unproject(pixels, calibration)
        forward_cosine = rays[:, 2].copy()
    else:
        coords = camera.sampling_coords.cartesian.cpu().numpy().astype(np.float64)
        # The short-side retinal diameter is the angle between these image rays.
        edges = np.array((((WIDTH - 1) / 2, -0.5), ((WIDTH - 1) / 2, HEIGHT - 0.5)))
        edge_rays = unproject(edges, calibration)
        half_angle = np.arccos(np.clip(edge_rays[0] @ edge_rays[1], -1, 1)) / 2
        xy = coords * (1, -1)
        radius = np.linalg.norm(xy, axis=-1)
        scale = half_angle * np.sinc(radius * half_angle / np.pi)
        rays = np.column_stack((xy * scale[:, None], np.cos(radius * half_angle)))
        forward_cosine = rays[:, 2].copy()
        target = unproject(np.array(((fixation[1] * WIDTH - 0.5, fixation[0] * HEIGHT - 0.5),)), calibration)[0]
        pan = np.arctan2(target[0], target[2])
        tilt = np.arctan2(-target[1], np.hypot(target[0], target[2]))
        rx, _ = cv2.Rodrigues(np.array((tilt, 0, 0)))
        ry, _ = cv2.Rodrigues(np.array((0, pan, 0)))
        rays = rays @ (ry @ rx).T
        fx, fy, cx, cy = calibration["intrinsics"]
        matrix = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)))
        project = cv2.fisheye.projectPoints if calibration["model"] == "fisheye" else cv2.projectPoints
        pixels = project(rays[:, None], np.zeros(3), np.zeros(3), matrix, np.array(calibration["distortion"]))[0][:, 0]
    valid = (
        (pixels[:, 0] >= -0.5) & (pixels[:, 0] < WIDTH - 0.5) & (pixels[:, 1] >= -0.5) & (pixels[:, 1] < HEIGHT - 0.5)
    )
    if "image_circle" in calibration:
        cx, cy, radius = calibration["image_circle"]
        valid &= np.linalg.norm(pixels - (cx, cy), axis=-1) <= radius
        if sampling == "foveated":
            target_pixel = np.array((fixation[1] * WIDTH - 0.5, fixation[0] * HEIGHT - 0.5))
            # An invalid fixation disables the whole calibrated view, including
            # rays that would otherwise land inside the lens's usable circle.
            valid &= np.linalg.norm(target_pixel - (cx, cy)) <= radius
    return rays * (1, -1, -1), valid, forward_cosine


def main(args: argparse.Namespace) -> None:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.utils.warp import ProxyArray

    from isaaclab_fovi.sensors.calibrated_camera import CalibratedTiledCamera, CalibratedTiledCameraCfg

    if args.backend in ("newton", "ovrtx"):
        from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)
        if args.backend == "newton":
            renderer = NewtonWarpRendererCfg(render_order="pixel_priority")
        else:
            from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg

            renderer = OVRTXRendererCfg(log_level="warn")
    else:
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import (
            IsaacRtxRendererGlobalSettingsCfg,
        )

        physics, renderer = (
            PhysxCfg(),
            IsaacRtxRendererCfg(
                global_settings=IsaacRtxRendererGlobalSettingsCfg(
                    antialiasing_mode="Off",
                    enable_dl_denoiser=False,
                    enable_global_illumination=False,
                    enable_reflections=False,
                    enable_ambient_occlusion=False,
                    enable_shadows=False,
                )
            ),
        )
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device=args.device, physics=physics, gravity=(0.0, 0.0, 0.0))
    )
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    for i, (center, size) in enumerate(BOXES):
        cfg = sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.0, 0.0) if i else (0.0, 0.0, 0.8),
                emissive_color=(0.8, 0.0, 0.0) if i else (0.0, 0.0, 0.8),
            ),
        )
        cfg.func(f"/World/envs/env_0/Box{i}", cfg, translation=center)
    light = sim_utils.DistantLightCfg(intensity=2000)
    light.func("/World/Light", light)
    base = dict(
        model="fisheye",
        image_size=(HEIGHT, WIDTH),
        intrinsics=(80.0, 85.0, 88.0, 63.0),
        distortion=(0.0,) * 4,
    )
    cases = {
        "equidistant": base,
        "intrinsics": dict(base, intrinsics=(105.0, 95.0, 102.0, 53.0)),
        "distortion": dict(base, distortion=(0.14, -0.025, 0.004, -0.0002)),
        "mask": dict(base, image_circle=(88.0, 63.0, 68.0)),
        "pinhole": dict(base, model="pinhole"),
        "brown": dict(base, model="pinhole", distortion=(0.12, -0.01, 0.005, -0.003)),
    }
    cameras = {}
    for name, calibration in cases.items():
        common = dict(
            prim_path=f"/World/envs/env_0/{name}",
            data_types=["rgb", "distance_to_camera", "depth"],
            camera_calibration=calibration,
            renderer_cfg=renderer,
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(NEAR, FAR), lock_camera=False),
            offset=CalibratedTiledCameraCfg.OffsetCfg(rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
            update_latest_camera_pose=True,
        )
        if args.sampling == "dense":
            camera = CalibratedTiledCamera(CalibratedTiledCameraCfg(height=HEIGHT, width=WIDTH, **common))
        else:
            from isaaclab_fovi.sensors import FoveatedTiledCamera, FoveatedTiledCameraCfg

            camera = FoveatedTiledCamera(
                FoveatedTiledCameraCfg(
                    field_geometry="spherical",
                    foveated_res=24,
                    cmf_a=0.1,
                    fixed_software_fov=True,
                    software_fov_image_height=HEIGHT,
                    software_fov_image_width=WIDTH,
                    **common,
                )
            )
        cameras[name] = camera
    if args.backend == "ovrtx":
        from isaaclab.cloner import ClonePlan

        sim.set_clone_plan(
            ClonePlan(
                sources=("/World/envs/env_0",),
                destinations=("/World/envs/env_{}",),
                clone_mask=np.ones((1, 1), dtype=bool),
                env_ids=np.array([0]),
                positions=np.zeros((1, 3)),
            )
        )
    sim.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "rtx" and args.sampling == "foveated":
        audit_projection(cameras, args.output.with_suffix(".projection.json"))
    records, frames = [], {}
    fixations = [(0.5, 0.5), (0.4, 0.65), (0.4, 0.65), (0.5, 0.5)]
    maps = {}
    if args.video_frames:
        maps = video_maps(cameras, 224)
        for angle in np.linspace(0, 2 * np.pi, args.video_frames):
            fixations.append((0.5 + 0.2 * np.sin(angle), 0.5 + 0.3 * np.sin(2 * angle)))
    with ExitStack() as resources:
        video = None
        if args.video_frames:
            import imageio.v2 as imageio

            video = resources.enter_context(
                imageio.get_writer(
                    args.output.with_suffix(".mp4"),
                    fps=24,
                    codec="libx264",
                    macro_block_size=1,
                    ffmpeg_params=["-crf", "18", "-preset", "medium"],
                )
            )
        # Repeated fixation and return exercise pose updates without accumulated rotation.
        for phase, fixation in enumerate(fixations):
            origin = np.array((0.15, -0.1, 0.0)) if phase in (1, 2) else np.zeros(3)
            for camera in cameras.values():
                camera.set_world_poses(positions=torch.tensor(origin[None], device=sim.device, dtype=torch.float32))
                if args.sampling == "foveated":
                    camera.set_calibrated_fixation(torch.tensor((fixation,), device=sim.device, dtype=torch.float32))
            for _ in range(8):
                sim.step()
                if args.backend == "ovrtx":
                    from isaaclab.sensors import SensorBase

                    SensorBase.update_batch(list(cameras.values()), sim.get_physics_dt())
                else:
                    for camera in cameras.values():
                        camera.update(sim.get_physics_dt(), force_recompute=True)
            for name, camera in cameras.items():
                rays, valid, forward_cosine = reference_rays(camera, cases[name], args.sampling, fixation)
                expected = ray_depth(rays, origin)
                output = camera.data.output["distance_to_camera"]
                measured = (output.torch if isinstance(output, ProxyArray) else output).cpu().numpy().reshape(-1)
                depth = camera.data.output["depth"]
                depth = (depth.torch if isinstance(depth, ProxyArray) else depth).cpu().numpy().reshape(-1)
                # Native RTX annotators measure ray range after advancing by near clip.
                # Forward-axis depth still measures from the optical center.
                expected_range = expected - NEAR if args.backend == "rtx" else expected
                error = np.abs(measured[valid] - expected_range[valid])
                depth_error = np.abs(depth[valid] - expected[valid] * forward_cosine[valid])
                record = dict(
                    case=name,
                    phase=phase,
                    median_error_m=float(np.median(error)) if valid.any() else None,
                    p95_error_m=float(np.quantile(error, 0.95)) if valid.any() else None,
                    p99_error_m=float(np.quantile(error, 0.99)) if valid.any() else None,
                    max_error_m=float(error.max()) if valid.any() else None,
                    depth_p99_error_m=float(np.quantile(depth_error, 0.99)) if valid.any() else None,
                    valid_samples=int(valid.sum()),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
                rgb = camera.data.output["rgb"]
                rgb = (rgb.torch if isinstance(rgb, ProxyArray) else rgb).cpu().numpy()
                rgb_flat = rgb.reshape(-1, 3)
                foreground = rgb_flat[:, 0] > rgb_flat[:, 2]
                record["rgb_foreground_agreement"] = (
                    float(np.mean(foreground[valid] == (expected[valid] < 3))) if valid.any() else None
                )
                frames[f"{name}_{phase}"] = dict(distance=measured.copy(), expected=expected, valid=valid, rgb=rgb)
                if not valid.all():
                    assert np.all(rgb.reshape(-1, 3)[~valid] == 0), (name, phase, "color mask")
                    masked_depth = FAR if args.sampling == "dense" else 0
                    assert np.all(measured[~valid] == masked_depth), (name, phase, "distance mask")
                    assert np.all(depth[~valid] == masked_depth), (name, phase, "depth mask")
            if video is not None and phase >= 4:
                frame = video_frame(frames, maps, phase, args.backend, fixation)
                video.append_data(frame[:, :, ::-1])
    if args.video_frames:
        import imageio_ffmpeg

        video_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(args.output.with_suffix(".mp4")))
        if video_frames != args.video_frames:
            raise AssertionError(f"Expected {args.video_frames} video frames, decoded {video_frames}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **{f"{key}_{channel}": value for key, frame in frames.items() for channel, value in frame.items()},
    )
    sim.stop()
    for record in records:
        if record["valid_samples"] == 0:
            # RGB, range, and depth were required to be entirely zero above.
            continue
        assert record["p95_error_m"] < 0.01, record
        assert record["p99_error_m"] < 0.03, record
        assert record["depth_p99_error_m"] < 0.03, record
        assert record["rgb_foreground_agreement"] > 0.99, record
    for name in cases:
        np.testing.assert_allclose(frames[f"{name}_0"]["distance"], frames[f"{name}_3"]["distance"], atol=0.001)
        np.testing.assert_allclose(frames[f"{name}_1"]["distance"], frames[f"{name}_2"]["distance"], atol=0.001)
    for name in ("intrinsics", "distortion", "pinhole", "brown"):
        frame, baseline = frames[f"{name}_0"], frames["equidistant_0"]
        assert abs(frame["distance"][frame["valid"]].mean() - baseline["distance"][baseline["valid"]].mean()) > 0.01, (
            name
        )
    print("CALIBRATED RENDER CHECK PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rtx", "newton", "ovrtx"), required=True)
    parser.add_argument("--sampling", choices=("dense", "foveated"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video_frames", type=int, default=0)
    parser.add_argument("--near", type=float, default=NEAR)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    NEAR = args.near
    if args.video_frames < 0 or (args.video_frames and args.sampling != "foveated"):
        parser.error("--video_frames must be nonnegative and requires --sampling foveated")
    args.enable_cameras = True
    faulthandler.dump_traceback_later(60, repeat=True)
    launcher = None if args.backend == "ovrtx" else AppLauncher(args)
    exit_code = 0
    try:
        main(args)
    except Exception:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=exit_code)
