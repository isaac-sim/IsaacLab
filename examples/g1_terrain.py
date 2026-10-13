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
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.utils.hydra import resolve_task_config

if TYPE_CHECKING:
    from isaaclab_visualizers.kit import KitVisualizer

    from isaaclab_tasks.contrib.g1_terrain.terrain_env import G1TerrainEnv


def _prepare_view(env: G1TerrainEnv, visualizer: KitVisualizer) -> None:
    """Configure the spectator camera after initializing physics and depth geometry."""
    import omni.kit.app  # noqa: PLC0415
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    from isaaclab.sim import get_current_stage  # noqa: PLC0415

    from isaaclab_tasks.contrib.g1_terrain.terrain import separate_ramp_base  # noqa: PLC0415

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep  # noqa: PLC0415

    rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    stage = get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetName().startswith("imu_in_"):
            UsdGeom.Imageable(prim).MakeInvisible()
    # Kit's viewport camera opinions in the session layer override the scene layer.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera_path = "/OmniverseKit_Persp"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        transform = UsdGeom.Xformable(camera.GetPrim())
        transform.ClearXformOpOrder()
        transform.SetResetXformStack(True)
        view = Gf.Matrix4d(1.0).SetLookAt(
            Gf.Vec3d(*env.cfg.camera_eye), Gf.Vec3d(*env.cfg.camera_target), Gf.Vec3d(0.0, 0.0, 1.0)
        )
        transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(view.GetInverse())
        aperture = 20.955 * visualizer.cfg.window.size[1] / visualizer.cfg.window.size[0]
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(aperture)
        camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(env.cfg.camera_vfov) / 2)))
    for _ in range(60):
        env.sim.render()
        visualizer.render_rgb_array()
    env.sim.forward()
    env.scene.update(0.0)
    if env.cfg.scene.depth_camera is not None:
        env.scene["depth_camera"].update(0.0, force_recompute=True)
        separate_ramp_base(stage)
    for _ in range(30):
        env.sim.render()
        visualizer.render_rgb_array()


def main() -> None:
    """Run the pretrained policy in simulation time and optionally record RTX frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("platforms", "ramps"), required=True)
    parser.add_argument("--checkpoint", help="Override the pretrained TorchScript checkpoint path or URL")
    parser.add_argument("--video", action="store_true", help="Record video.mp4")
    parser.add_argument("--output", type=Path, default=Path("logs/g1_terrain"))
    parser.add_argument("--seconds", type=float, help="Simulation duration in seconds, independent of wall time")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args, overrides = parser.parse_known_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        parser.error("Capture dimensions must be positive even integers")
    task = f"IsaacContrib-G1-Terrain-{args.task.title()}"
    cfg, _ = resolve_task_config(task, None, overrides=overrides)
    if args.checkpoint is not None:
        cfg.policy_path = args.checkpoint
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
            _prepare_view(env, visualizer)
            from isaaclab_tasks.contrib.g1_terrain.depth_policy import DepthStudentPolicy  # noqa: PLC0415

            policy = DepthStudentPolicy(cfg.policy_path)
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
