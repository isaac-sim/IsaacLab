# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Reference tensors and differentiable kinematics in the algorithm's WXYZ convention."""

from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.datasets.motion_loader import CsvMotionCfg, load_csv_motion, resample_motion
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from isaaclab_rl.model_based.kinematics import ForwardKinematics, from_native, so3_log

CONTROL_DT = 0.02


def _load_pose_motion(path: Path, cfg) -> dict[str, torch.Tensor]:
    """Adapt a read-only pose clip to the task's 50 Hz WXYZ state layout."""
    if path.suffix == ".csv":
        motion = load_csv_motion(path, cfg.joint_names, getattr(cfg, "csv", CsvMotionCfg()))
    else:
        raw = torch.tensor(np.load(path, allow_pickle=False), dtype=torch.float32)
        width = 7 + len(cfg.joint_names)
        if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != width:
            raise ValueError(f"Motion {path} must contain at least two {width}-channel frames.")
        if not torch.isfinite(raw).all() or (raw[:, 3:7].norm(dim=-1) < 1e-6).any():
            raise ValueError(f"Motion {path} contains nonfinite values or invalid quaternions.")
        motion = {"root_pos_w": raw[:, :3], "root_quat_w": raw[:, 3:7], "joint_pos": raw[:, 7:]}
    motion = resample_motion(motion, cfg.source_fps, 1.0 / CONTROL_DT)
    for velocity, position in (("root_lin_vel_w", "root_pos_w"), ("joint_vel", "joint_pos")):
        if velocity not in motion:
            motion[velocity] = torch.gradient(motion[position], spacing=CONTROL_DT, dim=0)[0]
    R = matrix_from_quat(motion["root_quat_w"])
    if "root_ang_vel_w" not in motion:
        w = torch.zeros_like(motion["root_pos_w"])
        w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
        w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
        w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
        motion["root_ang_vel_w"] = w
    motion["root_quat_w"] = from_native(quat_from_matrix(R))
    return motion


class MotionReference:
    """Concatenated clips with exclusive ends, retaining supplied reference velocities.

    NPZ clips contain ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``
    (WXYZ), ``root_lin_vel_w``, and ``root_ang_vel_w`` at 50 Hz. CSV inputs use
    the configured read-only adapter; NPY inputs use metric XYZW poses. Missing
    velocities are estimated within each clip, never across clip boundaries.
    """

    def __init__(self, cfg, device: str):
        names = cfg.joint_names
        bodies = cfg.tracked_bodies
        self.fk = ForwardKinematics(cfg.kinematics_path, names, bodies, device)
        self.device = device
        files = sorted(Path(cfg.motion_dir).rglob("*.npz"))
        if not files:
            files = sorted((*Path(cfg.motion_dir).rglob("*.csv"), *Path(cfg.motion_dir).rglob("*.npy")))
        if not files:
            raise ValueError(f"No motion NPZ/CSV/NPY files found in {cfg.motion_dir}.")
        clip_names = [str(path.relative_to(cfg.motion_dir).with_suffix("")) for path in files]
        if len(set(clip_names)) != len(clip_names):
            raise ValueError("Motion names must be unique across files.")
        shard_count = getattr(cfg, "motion_shard_count", 1)
        shard_index = getattr(cfg, "motion_shard_index", 0)
        if not 0 <= shard_index < shard_count:
            raise ValueError("Motion shard index must be in [0, motion_shard_count).")
        if shard_count > 1:
            files = files[shard_index::shard_count]
            if not files:
                raise ValueError("Sharded references require at least one clip per rank.")
        fields = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
        clips = []
        for path in files:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    clip = {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in fields}
                    if "fps" in data and float(data["fps"]) != 50.0:
                        raise ValueError(f"Reference clip {path} must be sampled at 50 Hz.")
            else:
                clip = {key: value.to(device) for key, value in _load_pose_motion(path, cfg).items()}
            length = len(clip["joint_pos"])
            widths = (len(names), len(names), 3, 4, 3, 3)
            if length < 2 or any(clip[key].shape != (length, width) for key, width in zip(fields, widths)):
                raise ValueError(f"Invalid reference layout in {path}.")
            if any(not torch.isfinite(value).all() for value in clip.values()):
                raise ValueError(f"Nonfinite reference in {path}.")
            clips.append(clip)
        lengths = [len(clip["joint_pos"]) for clip in clips]
        self.names = [str(path.relative_to(cfg.motion_dir).with_suffix("")) for path in files]
        for key in fields:
            setattr(self, key, torch.cat([clip[key] for clip in clips]))
        self.motion_ends = torch.tensor(lengths, device=device).cumsum(0)
        self.motion_starts = self.motion_ends - torch.tensor(lengths, device=device)
        self.time_step_total = sum(lengths)

    def body_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute tracked body poses and velocities without retaining an all-body dataset."""
        return self.fk(
            self.root_pos_w[frames],
            self.root_quat_w[frames],
            self.joint_pos[frames],
            self.root_lin_vel_w[frames],
            self.root_ang_vel_w[frames],
            self.joint_vel[frames],
        )
