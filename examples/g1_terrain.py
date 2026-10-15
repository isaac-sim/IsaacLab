# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the pretrained G1 depth policy on platforms or ramps."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.utils.hydra import resolve_task_config


def main() -> None:
    """Run the pretrained policy in simulation time and optionally record RTX frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("platforms", "ramps"), required=True)
    parser.add_argument("--checkpoint", help="Override the pretrained TorchScript checkpoint path or URL")
    parser.add_argument("--video", action="store_true", help="Record video.mp4")
    parser.add_argument("--output", type=Path, default=Path("logs/g1_terrain"))
    parser.add_argument("--seconds", type=float, help="Simulation duration in seconds, independent of wall time")
    parser.add_argument("--max_steps", type=int, help="Stop after at most this many control steps")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args, overrides = parser.parse_known_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        parser.error("Capture dimensions must be positive even integers")
    task = "IsaacContrib-G1-Terrain"
    overrides.insert(0, f"presets={args.task}")
    cfg, _ = resolve_task_config(task, None, overrides=overrides)
    if args.checkpoint is not None:
        cfg.policy.checkpoint = args.checkpoint
    cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            headless=False,
            window=WindowCfg(size=(args.width, args.height)),
            eye=cfg.camera_eye,
            lookat=cfg.camera_target,
            streaming_view=False,
            enable_markers=False,
            enable_live_plots=False,
        )
    ]
    torch.set_num_threads(1)
    with launch_simulation(cfg, {"visualizer": ["kit"], "enable_cameras": True}):
        env = gym.make(task, cfg=cfg).unwrapped
        writer = None
        try:
            env.reset()
            robot = env.scene["robot"]
            visualizer = env.sim.visualizers[0]
            env.prepare_view(visualizer)
            from isaaclab_tasks.contrib.g1_terrain.agents.depth_policy import DepthStudentPolicy  # noqa: PLC0415

            policy = DepthStudentPolicy(cfg.policy)
            policy_ids = [robot.joint_names.index(name) for name in policy.joint_names]
            action = robot.data.default_joint_pos.torch.clone()
            if args.video:
                import imageio.v2 as imageio  # noqa: PLC0415

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
                x, y, z, w = quat
                yaw = math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
                yaw_error = (cfg.heading - yaw + math.pi) % (2 * math.pi) - math.pi
                speed = 0.6 * min(t / 0.3, 1.0)
                command = np.array([speed, 0.0, np.clip(0.5 * yaw_error, -1.0, 1.0)], np.float32)
                depth = env.scene["depth_camera"].data.output["distance_to_image_plane"].torch.cpu().numpy()[0, ..., 0]
                target = policy.step(joint_pos, joint_vel, quat, gyro, command, depth[::-1, ::-1])
                action[0, policy_ids] = torch.as_tensor(target, device=env.device)
                if not np.isfinite(target).all():
                    raise RuntimeError(f"Non-finite policy targets at t={t:.2f}s")
                now = time.perf_counter()
                if step % 100 == 0 or now - last_log >= 10.0 or step == num_steps - 1:
                    position = robot.data.root_pos_w.torch.cpu().numpy()[0].tolist()
                    print(
                        f"sim={t:.2f}/{duration:.2f}s wall={now - started:.1f}s"
                        f" vx_cmd={speed:.2f}m/s position={position}",
                        flush=True,
                    )
                    last_log = now
                if writer:
                    writer.append_data(visualizer.render_rgb_array()[..., :3])
                env.step(action)
        finally:
            if writer:
                writer.close()
            env.close()


if __name__ == "__main__":
    main()
