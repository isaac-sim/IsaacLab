# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner and tracking-policy inference for the released SONIC G1 models."""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..sonic_parameters import ACTION_SCALE, DEFAULT_POS, JOINT_NAMES, POLICY_ORDER


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
        providers = [
            ("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(), "use_tf32": 0}),
            "CPUExecutionProvider",
        ]
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
