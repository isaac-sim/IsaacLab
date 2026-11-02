# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run SONIC over gravel and sand using Newton MPM or PhysX PBD.

Usage:
    uv sync --extra isaacsim --extra video
    uv pip install "huggingface-hub>=0.34" "onnxruntime-gpu>=1.22"
    uv run --extra isaacsim --extra video isaaclab demo sonic
    uv run isaaclab demo sonic --viz newton_gl
    uv run --extra video isaaclab demo sonic --viz newton_gl --video

The released planner and tracker are downloaded from nvidia/GEAR-SONIC on Hugging
Face under their upstream model license. The robot USD is downloaded from the
public Isaac asset server. Newton runs offscreen and records with Kit/RTX by default,
saving logs/sonic/sonic_0000.mp4. The full MPM scene can run much slower than real time.
Pass ``--viz`` to select a visualizer; recording then requires ``--video``. PhysX opens
Kit by default and also requires ``--video`` to record.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg

wp.config.enable_backward = False

from isaaclab_visualizers.kit import KitBackendCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_tasks.contrib.g1_terrain.agents.sonic_policy import SonicPolicy
from isaaclab_tasks.utils.hydra import resolve_task_config, setup_preset_cli


def command_speed(step: int) -> float:
    """Scripted forward velocity, updated every 100 ms."""
    tick = step // 5
    if tick < 30:
        return 0.0
    if tick < 35:
        return 0.1 * (tick - 29)
    if tick < 126:
        return 0.6
    return round(max(0.0, 0.592 - (tick - 126) * (0.6 / 45.0)), 3)


def main() -> None:
    """Download SONIC models and run a scripted traversal with optional video recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physics",
        dest="physics_preset",
        choices=("newton_mjwarp", "isaacsim_physx"),
        help="Physics backend (default: newton_mjwarp)",
    )
    parser.add_argument(
        "--video", action="store_true", help="Record an MP4 clip (automatic for Newton when --viz is omitted)"
    )
    parser.add_argument("--output", type=Path, default=Path("logs/sonic"))
    parser.add_argument("--seconds", type=float, help="Simulation duration in seconds, independent of wall time")
    parser.add_argument("--max_steps", type=int, help="Stop after at most this many control steps")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=None)
    args, overrides = setup_preset_cli(parser)
    use_default_visualizer = not any(
        arg.partition("=")[0] in ("visualizer", "env.sim.visualizer_cfgs", "env.sim.visualizer_cfgs.backend")
        for arg in overrides
    )
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        parser.error("Capture dimensions must be positive even integers")
    if args.physics_preset is not None:
        overrides.insert(0, f"physics={args.physics_preset}")
    task = "IsaacContrib-G1-Terrain"
    overrides.insert(0, "presets=granular")
    cfg, _ = resolve_task_config(task, None, overrides=overrides)
    is_physx = isinstance(cfg.sim.physics, PhysxCfg)
    if not isinstance(cfg.sim.visualizer_cfgs, list):
        cfg.sim.visualizer_cfgs = [cfg.sim.visualizer_cfgs]
    offline_default = (
        use_default_visualizer and not is_physx and not any(viewer.backend for viewer in cfg.sim.visualizer_cfgs)
    )
    args.video = args.video or offline_default
    duration = args.seconds or cfg.duration
    step_dt = cfg.sim.dt * cfg.decimation
    num_steps = math.ceil(duration / step_dt)
    if args.max_steps is not None:
        num_steps = min(num_steps, args.max_steps)
    if args.video:
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="viz:kit" if is_physx or offline_default else "viz",
                output_dir=str(args.output),
                output_filename_prefix="sonic",
                video_length=num_steps,
                fps=round(1 / step_dt),
            )
        ]
    if offline_default:
        print(
            f"Running SONIC offscreen with Kit/RTX; recording to {args.output.resolve() / 'sonic_0000.mp4'}.\n"
            "No simulation window will open. MPM can run much slower than real time; watch the terminal for progress.",
            flush=True,
        )

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    model_dir = snapshot_download(
        repo_id=cfg.policy_repo,
        revision=cfg.policy_revision,
        allow_patterns=[
            "model_encoder.onnx",
            "model_decoder.onnx",
            "planner_sonic.onnx",
            "observation_config.yaml",
            "LICENSE",
        ],
    )
    aperture = 20.955 * args.height / args.width
    for viewer in cfg.sim.visualizer_cfgs:
        viewer.window.size = (args.width, args.height)
        viewer.eye, viewer.lookat = cfg.camera_eye, cfg.camera_target
        viewer.focal_length = aperture / (2 * math.tan(math.radians(cfg.camera_vfov) / 2))
        viewer.enable_markers = viewer.enable_live_plots = viewer.streaming_view = False
        if use_default_visualizer and is_physx and not viewer.backend:
            viewer.backend = KitBackendCfg()
    torch.set_num_threads(1)
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            kit_visualizer = next(
                (visualizer for visualizer in env.sim.visualizers if visualizer.cfg.backend.visualizer_type == "kit"),
                None,
            )
            env.prepare_view(kit_visualizer)
            policy_ids = [robot.joint_names.index(name) for name in SonicPolicy.joint_names]
            policy = SonicPolicy(model_dir, robot.data.joint_pos.torch.cpu().numpy()[0, policy_ids])
            action = robot.data.default_joint_pos.torch.clone()
            print(f"Running {duration:.2f} simulation seconds with scripted commands; no keyboard control.", flush=True)
            started = time.perf_counter()
            last_log = started
            for step in range(num_steps):
                if env.sim.is_stopped():
                    break
                t = step * env.step_dt
                quat = robot.data.root_quat_w.torch.cpu().numpy()[0]
                joint_pos = robot.data.joint_pos.torch.cpu().numpy()[0, policy_ids]
                joint_vel = robot.data.joint_vel.torch.cpu().numpy()[0, policy_ids]
                gyro = robot.data.root_ang_vel_b.torch.cpu().numpy()[0]
                speed = command_speed(step)
                target = policy.step(joint_pos, joint_vel, quat, gyro, speed)
                if not np.isfinite(target).all():
                    raise RuntimeError(f"Non-finite policy targets at t={t:.2f}s")
                action[0, policy_ids] = torch.as_tensor(target, device=env.device)
                now = time.perf_counter()
                if step % 100 == 0 or now - last_log >= 10.0 or step == num_steps - 1:
                    position = robot.data.root_pos_w.torch.cpu().numpy()[0].tolist()
                    print(
                        f"sim={t:.2f}/{duration:.2f}s wall={now - started:.1f}s"
                        f" vx_cmd={speed:.2f}m/s position={position}",
                        flush=True,
                    )
                    last_log = now
                env.step(action)
        finally:
            env.close()
        if args.video:
            print(f"Video saved under {args.output}", flush=True)


if __name__ == "__main__":
    main()
