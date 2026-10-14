# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run SONIC over gravel and sand using Newton MPM or PhysX PBD.

Usage:
    uv sync --extra isaacsim --extra video
    uv pip install "huggingface-hub>=0.34" "onnxruntime-gpu>=1.22"
    uv run --extra isaacsim --extra video isaaclab demo sonic

The released planner and tracker are downloaded from nvidia/GEAR-SONIC on Hugging
Face under their upstream model license. The robot USD is downloaded from the
public Isaac asset server. Recording advances in simulation time; Newton records
offscreen, while PhysX opens a window to synchronize robot and particle visuals.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg

wp.config.enable_backward = False

from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.contrib.g1_terrain.agents.sonic_policy import SonicPolicy
from isaaclab_tasks.utils.hydra import resolve_task_config


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
    """Download SONIC models and record a scripted traversal with the selected backend."""
    parser = argparse.ArgumentParser(description="Run SONIC over gravel and sand and save an RTX video.")
    parser.add_argument(
        "--physics", choices=("newton_mjwarp", "isaacsim_physx"), help="Physics backend (default: newton_mjwarp)"
    )
    parser.add_argument("--output", type=Path, default=Path("logs/sonic"))
    parser.add_argument("--seconds", type=float, help="Simulation duration in seconds, independent of wall time")
    parser.add_argument("--max_steps", type=int, help="Stop after at most this many control steps")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--info", action="store_true", help="Show simulator startup messages")
    args, overrides = parser.parse_known_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        parser.error("Capture dimensions must be positive even integers")
    if args.physics is not None:
        overrides.insert(0, f"physics={args.physics}")
    task = "IsaacContrib-G1-Terrain"
    overrides.insert(0, "presets=granular")
    cfg, _ = resolve_task_config(task, None, overrides=overrides)
    headless = not isinstance(cfg.sim.physics, PhysxCfg)

    import imageio.v2 as imageio  # noqa: PLC0415
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
    cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            headless=headless,
            window=WindowCfg(size=(args.width, args.height)),
            eye=cfg.camera_eye,
            lookat=cfg.camera_target,
            streaming_view=False,
            enable_markers=False,
            enable_live_plots=False,
        )
    ]
    torch.set_num_threads(1)
    if headless:
        os.environ["HEADLESS"] = "1"
    with launch_simulation(cfg, {"visualizer": ["kit"], "enable_cameras": True}):
        env = gym.make(task, cfg=cfg).unwrapped
        writer = None
        try:
            env.reset()
            robot = env.scene["robot"]
            visualizer = env.sim.visualizers[0]
            env.prepare_view(visualizer)
            policy_ids = [robot.joint_names.index(name) for name in SonicPolicy.joint_names]
            policy = SonicPolicy(model_dir, robot.data.joint_pos.torch.cpu().numpy()[0, policy_ids])
            action = robot.data.default_joint_pos.torch.clone()
            args.output.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(args.output / "video.mp4"),
                fps=round(1 / env.step_dt),
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
            duration = args.seconds or cfg.duration
            print(f"Running {duration:.2f} simulation seconds with scripted commands; no keyboard control.", flush=True)
            started = time.perf_counter()
            last_log = started
            num_steps = math.ceil(duration / env.step_dt)
            if args.max_steps is not None:
                num_steps = min(num_steps, args.max_steps)
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
                writer.append_data(visualizer.render_rgb_array()[..., :3])
                env.step(action)
        finally:
            if writer:
                writer.close()
            env.close()
        print(f"Video saved to {args.output / 'video.mp4'}", flush=True)


if __name__ == "__main__":
    main()
