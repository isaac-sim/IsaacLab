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
from isaaclab_physx.physics import PhysxCfg

wp.config.enable_backward = False

from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.physics import PhysxAutoCfg

from isaaclab_tasks.utils.hydra import resolve_task_config, setup_preset_cli


def main() -> None:
    """Run the pretrained policy in simulation time and optionally record visualizer frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("platforms", "ramps"), required=True)
    parser.add_argument("--checkpoint", help="Override the pretrained TorchScript checkpoint path or URL")
    parser.add_argument("--video", action="store_true", help="Record the run to an MP4 clip")
    parser.add_argument("--output", type=Path, default=Path("logs/g1_terrain"))
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
    task = "IsaacContrib-G1-Terrain"
    overrides.insert(0, f"presets={args.task}")
    cfg, _ = resolve_task_config(task, None, overrides=overrides)
    if args.checkpoint is not None:
        cfg.policy.checkpoint = args.checkpoint
    aperture = 20.955 * args.height / args.width
    if not isinstance(cfg.sim.visualizer_cfgs, list):
        cfg.sim.visualizer_cfgs = [cfg.sim.visualizer_cfgs]
    for viewer in cfg.sim.visualizer_cfgs:
        viewer.window.size = (args.width, args.height)
        viewer.eye, viewer.lookat = cfg.camera_eye, cfg.camera_target
        viewer.focal_length = aperture / (2 * math.tan(math.radians(cfg.camera_vfov) / 2))
        viewer.enable_markers = viewer.enable_live_plots = False
        if use_default_visualizer and not viewer.backend:
            viewer.backend = (
                KitBackendCfg() if isinstance(cfg.sim.physics, (PhysxCfg, PhysxAutoCfg)) else NewtonGLBackendCfg()
            )
    duration = args.seconds or cfg.duration
    step_dt = cfg.sim.dt * cfg.decimation
    num_steps = math.ceil(duration / step_dt)
    if args.max_steps is not None:
        num_steps = min(num_steps, args.max_steps)
    if args.video:
        cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=str(args.output),
                output_filename_prefix="g1_terrain",
                video_length=num_steps,
                fps=round(1 / step_dt),
            )
        ]
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
            from isaaclab_tasks.contrib.g1_terrain.agents.depth_policy import DepthStudentPolicy  # noqa: PLC0415

            policy = DepthStudentPolicy(cfg.policy)
            policy_ids = [robot.joint_names.index(name) for name in policy.joint_names]
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
                env.step(action)
        finally:
            env.close()


if __name__ == "__main__":
    main()
