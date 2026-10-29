# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load named reference arrays once, in the runtime robot's joint and body order."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


class MotionLoader:
    """Reference trajectory sampled at the policy rate.

    NPZ files contain ``fps``, ``joint_names``, ``body_names``, ``joint_pos`` [rad],
    ``joint_vel`` [rad/s], ``body_pos_w`` [m], ``body_quat_w`` (WXYZ),
    ``body_lin_vel_w`` [m/s] and ``body_ang_vel_w`` [rad/s]. Joint arrays have shape
    (frames, joints); body arrays have shape (frames, bodies, 3 or 4). Positions
    use a local world origin; the command adds each environment's origin.

    Legacy files may provide names in a same-stem JSON file with a matching
    ``sha256``. Loaded quaternions use Isaac Lab's XYZW convention.
    """

    def __init__(self, motion_file: str, body_names: Sequence[str], joint_names: Sequence[str], device: str):
        path = Path(motion_file)
        with np.load(path, allow_pickle=False) as data:
            if "body_names" in data and "joint_names" in data:
                names = {key: data[key].tolist() for key in ("body_names", "joint_names")}
            else:
                names = json.loads(path.with_suffix(".json").read_text())
                if names["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"Motion metadata does not match {path}.")
            for key in ("body_names", "joint_names"):
                if len(names[key]) != len(set(names[key])):
                    raise ValueError(f"Duplicate {key} in {path}.")
            body_indices = [names["body_names"].index(name) for name in body_names]
            joint_indices = [names["joint_names"].index(name) for name in joint_names]
            self.fps = float(data["fps"].item())
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError(f"Invalid motion FPS: {self.fps}.")
            self.time_step_total = len(data["joint_pos"])
            if self.time_step_total < 2:
                raise ValueError("A motion must contain at least two frames.")
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                values = data[key]
                is_joint = key.startswith("joint_")
                count = len(names["joint_names" if is_joint else "body_names"])
                shape = (self.time_step_total, count)
                if not is_joint:
                    shape += (4 if key == "body_quat_w" else 3,)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {key} in {path}: expected finite values with shape {shape}.")
                values = values[:, joint_indices if is_joint else body_indices]
                if key == "body_quat_w":
                    if not np.allclose(np.linalg.norm(values, axis=-1), 1.0, atol=1e-3):
                        raise ValueError(f"Non-unit body quaternions in {path}.")
                    values = values[..., [1, 2, 3, 0]]
                setattr(self, key, torch.tensor(values, dtype=torch.float32, device=device))
