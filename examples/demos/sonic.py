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
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from scipy.spatial.transform import Rotation

wp.config.enable_backward = False

from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.contrib.g1_terrain.sonic_parameters import ACTION_SCALE, DEFAULT_POS, JOINT_NAMES, POLICY_ORDER
from isaaclab_tasks.utils.hydra import resolve_task_config

if TYPE_CHECKING:
    from isaaclab_visualizers.kit import KitVisualizer

    from isaaclab_tasks.contrib.g1_terrain.terrain_env import G1TerrainEnv


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


def _slerp(a, b, weight):
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sine = np.maximum(np.sin(angle), 1e-12)
    result = np.where(
        angle < 1e-6,
        (1 - weight) * a + weight * b,
        np.sin((1 - weight) * angle) / sine * a + np.sin(weight * angle) / sine * b,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _sample_motion(motion, frames):
    frames = np.clip(frames, 0, len(motion) - 1)
    lower = frames.astype(int)
    upper = np.minimum(lower + 1, len(motion) - 1)
    weight = (frames - lower)[:, None]
    result = (1 - weight) * motion[lower] + weight * motion[upper]
    result[:, 3:7] = _slerp(motion[lower, 3:7], motion[upper, 3:7], weight)
    return result


class SonicPolicy:
    """Released g1-mode encoder/decoder with a 30 Hz planner resampled to 50 Hz."""

    joint_names = JOINT_NAMES

    def __init__(self, model_dir: str, joint_pos: np.ndarray):
        import onnxruntime as ort  # noqa: PLC0415

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model_dir = Path(model_dir)
        providers = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(model_dir / "model_encoder.onnx"), options, providers=providers)
        self.decoder = ort.InferenceSession(str(model_dir / "model_decoder.onnx"), options, providers=providers)
        self.planner = ort.InferenceSession(str(model_dir / "planner_sonic.onnx"), options, providers=providers)
        self.planner_inputs = {
            "context_mujoco_qpos": np.zeros((1, 4, 36), np.float32),
            "target_vel": np.zeros(1, np.float32),
            "mode": np.zeros(1, np.int64),
            "movement_direction": np.zeros((1, 3), np.float32),
            "facing_direction": np.array([[1, 0, 0]], np.float32),
            "random_seed": np.array([1234], np.int64),
            "height": np.array([-1], np.float32),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
        }
        context = self.planner_inputs["context_mujoco_qpos"]
        context[..., 2] = 0.788740
        context[..., 3] = 1.0
        context[..., 7:] = joint_pos
        self.motion, self.velocity = self._plan(0.0)
        self.frame = 1
        self.step_count = 0
        self.speed = 0.0
        self.history = np.zeros((10, 93), np.float32)
        self.action = np.zeros(29, np.float32)
        self.encoder_obs = np.zeros((1, 1762), np.float32)
        self.inverse_order = np.argsort(POLICY_ORDER)
        self.heading_offset = None
        self.initial_heading = None
        self.idle_target = None
        self.idle_state = "idle"

    def _plan(self, speed):
        self.planner_inputs["target_vel"][0] = speed
        self.planner_inputs["mode"][0] = int(speed > 0.0)
        self.planner_inputs["movement_direction"][0, 0] = float(speed > 0.0)
        poses, count = self.planner.run(["mujoco_qpos", "num_pred_frames"], self.planner_inputs)
        count = int(count.item())
        poses = poses[0, :count]
        motion = _sample_motion(poses, np.arange(int(count * 50 / 30)) * (30 / 50))
        velocity = np.diff(motion[:, 7:], axis=0) * 50
        return motion, np.concatenate((velocity, velocity[-1:]))

    def _replan(self, speed):
        context = _sample_motion(self.motion, self.frame + 2 + np.arange(4) * (50 / 30))
        self.planner_inputs["context_mujoco_qpos"][0] = context
        motion, velocity = self._plan(speed)
        frames = np.arange(len(motion) + 2)
        old = np.minimum(self.frame + frames, len(self.motion) - 1)
        new = np.clip(frames - 2, 0, len(motion) - 1)
        weight = np.clip((frames - 2) / 8, 0.0, 1.0)[:, None]
        blended = (1 - weight) * self.motion[old] + weight * motion[new]
        blended[:, 3:7] = _slerp(self.motion[old, 3:7], motion[new, 3:7], weight)
        blended_velocity = (1 - weight) * self.velocity[old] + weight * velocity[new]
        self.speed = speed
        return blended, blended_velocity

    def _adapt_idle(self, joint_pos):
        if self.idle_target is None:
            self.idle_target = self.motion[-1, 7:19].copy()
            self.idle_state = "idle"
        target = self.motion[-1, 7:19]
        error = np.mean(np.abs(target - joint_pos[:12]))
        if error > 0.10:
            self.idle_state = "adapting"
        elif self.idle_state == "adapting" and error < 0.05:
            self.idle_state = "idle"
        elif self.idle_state == "idle" and error < 0.045:
            self.idle_state = "recovering"
        if self.idle_state == "adapting":
            target[:] = 0.98 * target + 0.02 * joint_pos[:12]
        elif self.idle_state == "recovering":
            target[:] = 0.98 * target + 0.02 * self.idle_target

    def step(self, joint_pos, joint_vel, quat_xyzw, gyro, speed):
        """Run one 20 ms closed-loop control step and return 29 joint targets."""
        pending = None
        if self.step_count % 5 == 0 and (speed != self.speed or (speed > 0 and self.step_count % 50 == 45)):
            pending = self._replan(speed)
        orientation = Rotation.from_quat(quat_xyzw)
        if self.initial_heading is None:
            forward = orientation.apply([1, 0, 0])
            self.initial_heading = np.arctan2(forward[1], forward[0])
        if self.heading_offset is None:
            forward = Rotation.from_quat(self.motion[self.frame, [4, 5, 6, 3]]).apply([1, 0, 0])
            self.heading_offset = Rotation.from_rotvec(
                [0, 0, self.initial_heading - np.arctan2(forward[1], forward[0])]
            )
        gravity = orientation.inv().apply([0, 0, -1])
        self.history[:-1] = self.history[1:]
        self.history[-1] = np.concatenate(
            (gyro, (joint_pos - DEFAULT_POS)[POLICY_ORDER], joint_vel[POLICY_ORDER], self.action, gravity)
        )
        future = np.minimum(self.frame + np.arange(10) * 5, len(self.motion) - 1)
        self.encoder_obs[0, 4:294] = self.motion[future, 7:][:, POLICY_ORDER].ravel()
        self.encoder_obs[0, 294:584] = self.velocity[future][:, POLICY_ORDER].ravel()
        reference_quat = self.motion[future, 3:7][:, [1, 2, 3, 0]]
        relative = orientation.inv() * self.heading_offset * Rotation.from_quat(reference_quat)
        self.encoder_obs[0, 601:661] = relative.as_matrix()[..., :2].ravel()
        tokens = self.encoder.run(["encoded_tokens"], {"obs_dict": self.encoder_obs})[0]
        terms = np.split(self.history, [3, 32, 61, 90], axis=1)
        obs = np.concatenate((tokens.ravel(), *(term.ravel() for term in terms))).astype(np.float32)[None]
        self.action = self.decoder.run(["action"], {"obs_dict": obs})[0][0]
        if pending is not None:
            self.motion, self.velocity = pending
            self.frame = 0
            self.idle_target = None
        self.frame += 1
        if self.frame >= len(self.motion):
            self.frame = len(self.motion) - 1
            if speed == 0.0:
                self._adapt_idle(joint_pos)
        self.step_count += 1
        return DEFAULT_POS + ACTION_SCALE * self.action[self.inverse_order]


def _prepare_view(env: G1TerrainEnv, visualizer: KitVisualizer) -> None:
    """Set the RTX camera and the unbranded, matte robot appearance."""
    import omni.kit.app  # noqa: PLC0415
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    from isaaclab.sim import bind_visual_material, get_current_stage  # noqa: PLC0415

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep  # noqa: PLC0415

    rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    stage = get_current_stage()
    for name, material_cfg in env.cfg.robot_visual_materials.items():
        material_cfg.func(f"/World/SonicLooks/{name}", material_cfg)
    dark_parts = {
        "head_link",
        "pelvis",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    }
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetName().startswith("imu_in_"):
            UsdGeom.Imageable(prim).MakeInvisible()
        if prim.GetName() == "visuals":
            prim.SetInstanceable(False)
            for part in prim.GetChildren():
                if part.GetName() == "logo_link":
                    UsdGeom.Imageable(part).MakeInvisible()
                else:
                    material = "/World/SonicLooks/" + ("dark" if part.GetName() in dark_parts else "light")
                    bind_visual_material(part.GetPath(), material)
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
    for _ in range(30):
        env.sim.render()
        visualizer.render_rgb_array()


def main() -> None:
    """Download SONIC models and record a scripted traversal with the selected backend."""
    parser = argparse.ArgumentParser(description="Run SONIC over gravel and sand and save an RTX video.")
    parser.add_argument(
        "--physics", choices=("newton_mjwarp", "isaacsim_physx"), help="Physics backend (default: newton_mjwarp)"
    )
    parser.add_argument("--output", type=Path, default=Path("logs/sonic"))
    parser.add_argument("--seconds", type=float, help="Simulation duration in seconds, independent of wall time")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--info", action="store_true", help="Show simulator startup messages")
    args, overrides = parser.parse_known_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
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
            _prepare_view(env, visualizer)
            policy_ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
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
