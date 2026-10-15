# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate small environment JPEG previews from deterministic reset snapshots.

Each task runs in a separate process so Kit and physics resources are released between
captures. Manifest entries contain ``path`` and ``task``; optional ``eye``, ``lookat``,
``focal_length``, ``physics``, ``overrides``, and ``num_envs`` control the scene and framing.
The output retains its existing aspect ratio and is limited to 960 pixels on its longest
edge. Rendering warms up without advancing physics or applying actions.

Examples::

    uv run --extra isaacsim python tools/docs/media/capture_environment_previews.py \
        --task Isaac-Velocity-Flat-AnymalD --output /tmp/anymal.jpg \
        --eye 2.4 -2.4 1.6 --lookat 0 0 0.4 --direct-s3-origin

    uv run --extra isaacsim python tools/docs/media/capture_environment_previews.py \
        --manifest /tmp/preview-manifest.json --shard 0/2 --direct-s3-origin
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="Registered environment ID for a single capture.")
    source.add_argument(
        "--manifest", type=Path, help="JSON list of capture entries, or an object with an entries list."
    )
    parser.add_argument("--output", type=Path, help="JPEG destination for a single capture.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--log-dir", type=Path, default=Path("logs/environment-preview-refresh"))
    parser.add_argument("--shard", default="0/1", help="Zero-based partition index/count, for example 0/2.")
    parser.add_argument("--only", action="append", default=[], help="Capture only these task IDs or output basenames.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries with a successful capture receipt.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Maximum seconds per isolated capture.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-size", type=int, default=960)
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--eye", type=float, nargs=3)
    parser.add_argument("--lookat", type=float, nargs=3)
    parser.add_argument("--focal-length", type=float)
    parser.add_argument("--camera-origin", choices=("world", "env", "asset"), help="Camera coordinate frame.")
    parser.add_argument("--track-asset", help="Scene asset name or asset/body path when using the asset camera frame.")
    parser.add_argument("--physics", help="Optional task physics preset; otherwise retain the task default.")
    parser.add_argument("--override", action="append", default=[], help="Hydra override, repeatable.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--direct-s3-origin", action="store_true", help="Bypass a stale asset CDN for this process only."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _entry_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _image_size(path: Path, maximum: int) -> tuple[int, int]:
    from PIL import Image

    size = (960, 540)
    if path.is_file():
        with Image.open(path) as original:
            size = original.size
    scale = min(1.0, maximum / max(size))
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def _configure_asset_origin() -> dict[str, str]:
    """Use the public origin without changing the user's persisted SDK configuration."""
    import omni.client

    origin = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    if not omni.client.initialize():
        raise RuntimeError("Failed to initialize the asset client.")
    result = omni.client.set_s3_configuration(
        origin,
        bucket="omniverse-content-production",
        region="us-west-2",
        cloudfrontUrl=origin + "/",
        writeConfig=False,
    )
    if result != omni.client.Result.OK:
        raise RuntimeError(f"Failed to configure public asset origin: {result}")

    from isaaclab.sim import GroundPlaneCfg
    from isaaclab.utils.assets import retrieve_file_path

    # Refresh these two small files even when the batch reuses other downloaded assets.
    base = GroundPlaneCfg().usd_path.rsplit("/", 1)[0] + "/Materials/Textures/"
    hashes = {}
    for filename in ("default_ground_plane_albedo.png", "default_ground_plane_roughness.png"):
        downloaded = Path(retrieve_file_path(base + filename, force_download=True))
        hashes[filename] = hashlib.sha256(downloaded.read_bytes()).hexdigest()
    return hashes


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg

    # Some task configurations import Newton runtime modules while they resolve.
    # Start Kit first so its USD libraries take precedence over standalone USD.
    # The task's launch below reuses this runtime and selects its actual physics.
    launcher_args = {
        "physics": "isaacsim_physx",
        "visualizer": ["kit"],
        "headless": True,
        "enable_cameras": True,
        "device": args.device,
    }
    print("Starting Kit before resolving the task configuration", flush=True)
    with launch_simulation(PhysicsCfg(), launcher_args):
        return _capture_task(args)


def _capture_task(args: argparse.Namespace) -> dict[str, Any]:
    import warp as wp

    wp.config.enable_backward = False

    import gymnasium as gym
    import numpy as np
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from PIL import Image

    from isaaclab.app import launch_simulation
    from isaaclab.utils import validate
    from isaaclab.visualizers import WindowCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    overrides = list(args.override)
    if args.physics:
        overrides.insert(0, f"physics={args.physics}")
    print(f"Resolving {args.task} with {overrides}", flush=True)
    cfg, _ = resolve_task_config(args.task, "", play_mode=True, overrides=overrides)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.video_recorders = []

    width, height = _image_size(args.output, args.max_size)
    base = cfg.sim.default_visualizer_cfg
    if base is None:
        base = next(iter(cfg.sim.visualizer_cfgs or []), None)
    viewer = getattr(cfg, "viewer", None)
    eye = args.eye or getattr(base, "eye", None) or getattr(viewer, "eye", (3.0, -3.0, 2.0))
    lookat = args.lookat or getattr(base, "lookat", None) or getattr(viewer, "lookat", (0.0, 0.0, 0.5))
    focal_length = args.focal_length if args.focal_length is not None else 18.14756202697754
    visualizer_cfg = KitVisualizerCfg(
        eye=tuple(eye),
        lookat=tuple(lookat),
        focal_length=focal_length,
        window=WindowCfg(size=(width, height)),
        headless=True,
        origin_type=args.camera_origin or getattr(base, "origin_type", "world"),
        origin_track_path=args.track_asset or getattr(base, "origin_track_path", None) or "robot",
    )
    # Explicitly configure the selected visualizer before launch; this also supports
    # tasks whose deprecated ViewerCfg would otherwise replace the capture dimensions.
    cfg.sim.default_visualizer_cfg = visualizer_cfg
    cfg.sim.visualizer_cfgs = [visualizer_cfg]
    validate(cfg)
    launcher_args = {"visualizer": ["kit"], "headless": True, "enable_cameras": True, "device": args.device}
    print("Launching headless Kit", flush=True)
    with launch_simulation(cfg, launcher_args):
        # Kit must load its own omni.client before this import. Importing the standalone
        # SDK earlier would shadow APIs expected by Kit extensions.
        ground_hashes = _configure_asset_origin() if args.direct_s3_origin else {}
        print(f"Runtime ready; ground texture hashes: {ground_hashes}", flush=True)
        env = gym.make(args.task, cfg=cfg)
        try:
            env.reset(seed=args.seed)
            print("Environment reset; warming up rendering without advancing physics", flush=True)
            sim = env.unwrapped.sim
            visualizer = next(viz for viz in sim.visualizers if viz.cfg.visualizer_type == "kit")
            from pxr import Gf, UsdGeom

            origin = [0.0, 0.0, 0.0]
            if visualizer_cfg.origin_type == "env":
                origin = env.unwrapped.scene.env_origins[0].tolist()
            elif visualizer_cfg.origin_type == "asset":
                asset_name, _, body_name = visualizer_cfg.origin_track_path.partition("/")
                asset = env.unwrapped.scene[asset_name]
                if body_name:
                    body_index = asset.find_bodies(body_name)[0][0]
                    origin = asset.data.body_pos_w.torch[0, body_index].tolist()
                else:
                    origin = asset.data.root_pos_w.torch[0].tolist()
            world_eye = tuple(float(value + offset) for value, offset in zip(eye, origin))
            world_lookat = tuple(float(value + offset) for value, offset in zip(lookat, origin))
            camera_path = "/World/DocumentationCaptureCamera"
            camera = UsdGeom.Camera.Define(sim.stage, camera_path)
            camera.GetFocalLengthAttr().Set(focal_length)
            camera.GetHorizontalApertureAttr().Set(20.955)
            camera.GetVerticalApertureAttr().Set(20.955 * height / width)
            camera.GetClippingRangeAttr().Set((0.01, 1000000.0))
            camera_transform = (
                Gf.Matrix4d()
                .SetLookAt(Gf.Vec3d(*world_eye), Gf.Vec3d(*world_lookat), Gf.Vec3d(0.0, 0.0, 1.0))
                .GetInverse()
            )
            UsdGeom.Xformable(camera).AddTransformOp().Set(camera_transform)
            # A dedicated camera avoids Kit's stronger session-layer opinions on its
            # interactive perspective camera, including delayed viewport pose changes.
            visualizer._controlled_camera_path = camera_path
            frame = None
            for _ in range(args.warmup_frames):
                sim.render()
                frame = visualizer.render_rgb_array()
            if frame is None or frame.shape != (height, width, 3) or np.std(frame) < 2.0:
                raise RuntimeError("The capture did not produce a nonblank RGB frame.")
            if abs(float(camera.GetFocalLengthAttr().Get()) - focal_length) > 1.0e-4:
                raise RuntimeError("The capture camera focal length changed during rendering warmup.")
            actual_transform = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
            if not Gf.IsClose(actual_transform, camera_transform, 1.0e-5):
                raise RuntimeError("The capture camera transform changed during rendering warmup.")
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = args.output.with_name(args.output.name + ".tmp")
            image.save(temporary_output, format="JPEG", quality=args.quality, optimize=True, progressive=True)
            temporary_output.replace(args.output)
            print(f"Saved {args.output}", flush=True)
            result = {
                "status": "success",
                "task": args.task,
                "path": str(args.output),
                "size": [width, height],
                "bytes": args.output.stat().st_size,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "eye": list(eye),
                "lookat": list(lookat),
                "focal_length": focal_length,
                "focal_length_verified": True,
                "camera_prim_path": camera_path,
                "camera_transform_verified": True,
                "camera_horizontal_aperture": float(camera.GetHorizontalApertureAttr().Get()),
                "camera_vertical_aperture": float(camera.GetVerticalApertureAttr().Get()),
                "camera_clipping_range": list(camera.GetClippingRangeAttr().Get()),
                "world_eye": list(world_eye),
                "world_lookat": list(world_lookat),
                "camera_origin": visualizer_cfg.origin_type,
                "camera_track_asset": visualizer_cfg.origin_track_path,
                "seed": args.seed,
                "device": args.device,
                "num_envs": args.num_envs,
                "physics_steps_after_reset": 0,
                "warmup_frames": args.warmup_frames,
                "ground_texture_sha256": ground_hashes,
            }
            # Kit can terminate the Python process during shutdown. Persist the completed
            # capture before releasing the environment and runtime.
            _write_json(args.log_dir / "result.json", result)
        finally:
            env.close()
    return result


def _worker_command(args: argparse.Namespace, entry: dict[str, Any], receipt: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--task",
        entry["task"],
        "--output",
        str(_entry_path(args.root, entry["path"])),
        "--root",
        str(args.root),
        "--log-dir",
        str(receipt.parent),
        "--device",
        entry.get("device", args.device),
        "--seed",
        str(entry.get("seed", args.seed)),
        "--max-size",
        str(args.max_size),
        "--warmup-frames",
        str(args.warmup_frames),
        "--quality",
        str(args.quality),
        "--num-envs",
        str(entry.get("num_envs", args.num_envs)),
    ]
    for field in ("eye", "lookat"):
        value = entry.get(field) or getattr(args, field)
        if value:
            command.extend([f"--{field}", *(str(coordinate) for coordinate in value)])
    for field in ("physics", "focal_length", "camera_origin", "track_asset"):
        value = entry.get(field) or getattr(args, field)
        if value:
            command.extend([f"--{field.replace('_', '-')}", str(value)])
    for override in [*args.override, *entry.get("overrides", [])]:
        command.append(f"--override={override}")
    if args.direct_s3_origin:
        command.append("--direct-s3-origin")
    return command


def _run_batch(args: argparse.Namespace) -> int:
    if args.manifest:
        loaded = json.loads(args.manifest.read_text())
        entries = loaded["entries"] if isinstance(loaded, dict) else loaded
    else:
        entries = [{"task": args.task, "path": str(args.output)}]
    shard, count = (int(value) for value in args.shard.split("/"))
    if count < 1 or not 0 <= shard < count:
        raise ValueError("--shard must have 0 <= index < count.")
    selected = [entry for index, entry in enumerate(entries) if index % count == shard]
    if args.only:
        selected = [entry for entry in selected if entry["task"] in args.only or Path(entry["path"]).name in args.only]
    results = []
    basename_counts = Counter(Path(entry["path"]).stem for entry in entries)
    cache_dir = args.log_dir / "asset-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(selected):
        name = Path(entry["path"]).stem
        if basename_counts[name] > 1:
            name += "-" + hashlib.sha256(entry["path"].encode()).hexdigest()[:8]
        job_dir = args.log_dir / name
        receipt = job_dir / "result.json"
        command = _worker_command(args, entry, receipt)
        capture_options = command[3:]
        for option in ("--root", "--log-dir"):
            option_index = capture_options.index(option)
            del capture_options[option_index : option_index + 2]
        if args.skip_existing and receipt.is_file():
            previous = json.loads(receipt.read_text())
            output = _entry_path(args.root, entry["path"])
            if (
                previous.get("status") == "success"
                and previous.get("entry") == entry
                and previous.get("capture_options") == capture_options
                and output.is_file()
                and hashlib.sha256(output.read_bytes()).hexdigest() == previous.get("sha256")
            ):
                results.append(previous)
                continue
        job_dir.mkdir(parents=True, exist_ok=True)
        receipt.unlink(missing_ok=True)
        print(f"[{index + 1}/{len(selected)}] {entry['task']} -> {entry['path']}", flush=True)
        environment = {**os.environ, "TMPDIR": str(cache_dir), "OMNI_KIT_ACCEPT_EULA": "YES"}
        started = time.monotonic()
        with (job_dir / "capture.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=args.root, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                returncode = process.wait(timeout=args.timeout)
                failure = f"Capture exited with code {returncode}." if returncode else "Worker did not write a receipt."
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                failure = f"Capture exceeded {args.timeout:g} seconds."
        result = (
            json.loads(receipt.read_text()) if receipt.is_file() else {**entry, "status": "failed", "error": failure}
        )
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        result["log"] = str(job_dir / "capture.log")
        result["entry"] = entry
        result["capture_options"] = capture_options
        result["process_returncode"] = process.returncode
        _write_json(receipt, result)
        results.append(result)
        _write_json(args.log_dir / f"summary-{shard}-of-{count}.json", results)
        print(f"  {result['status']} ({result['elapsed_seconds']} s)", flush=True)
    return int(any(result["status"] != "success" for result in results))


def main() -> int:
    """Capture one task or a partition of a preview manifest."""
    parser = _parser()
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.log_dir = _entry_path(args.root, args.log_dir).resolve()
    if args.output:
        args.output = _entry_path(args.root, args.output).resolve()
    if args.task and args.output is None:
        parser.error("--task requires --output.")
    if args.max_size < 1 or args.warmup_frames < 1 or not 1 <= args.quality <= 100:
        parser.error("Image dimensions and warmup frames must be positive; JPEG quality must be in 1..100.")
    if not args.worker:
        return _run_batch(args)
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    try:
        result = {"status": "success", **_capture(args)}
    except Exception:
        result = {"status": "failed", "task": args.task, "path": str(args.output), "error": traceback.format_exc()}
        traceback.print_exc()
    _write_json(args.log_dir / "result.json", result)
    return int(result["status"] != "success")


if __name__ == "__main__":
    raise SystemExit(main())
