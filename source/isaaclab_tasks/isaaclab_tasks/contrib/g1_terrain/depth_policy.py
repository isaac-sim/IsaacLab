# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment observations for the G1 depth-based locomotion policy."""

import re

import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path

from .depth_parameters import ACTION_SCALE, DEFAULT_JOINT_POS, DEPTH_MAX_RANGE, DEPTH_SHAPE, HISTORY_LENGTH, JOINT_NAMES


class DepthStudentPolicy:
    """Run the exported student with term-major history and three depth frames."""

    def __init__(self, checkpoint: str):
        self.model = torch.jit.load(retrieve_file_path(checkpoint), map_location="cpu").eval()
        self.joint_names = JOINT_NAMES
        self.default = np.array(
            [next((v for p, v in DEFAULT_JOINT_POS.items() if re.fullmatch(p, name)), 0.0) for name in JOINT_NAMES],
            dtype=np.float32,
        )
        self.body = np.array(["_hand_" not in name for name in self.joint_names])
        self.scale = np.array(
            [
                next(value for pattern, value in ACTION_SCALE.items() if re.fullmatch(pattern, name))
                for name in self.joint_names
            ],
            dtype=np.float32,
        )
        self.history_length = HISTORY_LENGTH
        self.depth_range = DEPTH_MAX_RANGE
        self.reset()

    def reset(self):
        """Clear recurrent observations at the start of a traversal."""
        self.history = None
        self.depth_history = None
        self.action = np.zeros(len(self.joint_names), dtype=np.float32)

    @torch.inference_mode()
    def step(self, joint_pos, joint_vel, quat_xyzw, gyro, command, depth):
        """Return targets in policy joint order; absent finger feedback stays zero."""
        q_rel = np.zeros_like(self.default)
        dq = np.zeros_like(self.default)
        q_rel[self.body] = joint_pos[self.body] - self.default[self.body]
        dq[self.body] = joint_vel[self.body]
        xyz, w = np.asarray(quat_xyzw[:3], dtype=np.float32), float(quat_xyzw[3])
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        cross = 2.0 * np.cross(xyz, gravity)
        gravity = gravity - w * cross + np.cross(xyz, cross)
        frame = np.concatenate((gyro, gravity, command, q_rel, dq, self.action)).astype(np.float32)
        if self.history is None:
            self.history = np.repeat(frame[None], self.history_length, axis=0)
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame
        terms = np.split(self.history, [3, 6, 9, 52, 95], axis=1)
        obs = np.concatenate([term.ravel() for term in terms])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, self.depth_range)
        depth = np.clip(depth, 0.0, self.depth_range).astype(np.float32) / self.depth_range
        if self.depth_history is None:
            self.depth_history = np.repeat(depth[None], DEPTH_SHAPE[0], axis=0)
        else:
            self.depth_history[:-1] = self.depth_history[1:]
            self.depth_history[-1] = depth
        self.action = self.model(torch.from_numpy(obs[None]), [torch.from_numpy(self.depth_history[None])])[0].numpy()
        target = self.default.copy()
        target[self.body] += self.scale[self.body] * self.action[self.body]
        return target
