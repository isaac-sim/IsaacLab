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
import torch.nn.functional as F

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from isaaclab_rl.model_based.kinematics import ArticulationKinematics, ForwardKinematics, so3_log
from isaaclab_rl.model_based.math import from_native

CONTROL_DT = 0.02


def resample_motion(raw: torch.Tensor, src_fps: float):
    """Resample metric/xyzw/radian G1 poses to the control rate using shortest-arc SLERP."""
    p, quat, j = raw[:, :3], raw[:, 3:7], raw[:, 7:]
    # resample to 50 Hz
    T_src = raw.shape[0]
    dur = (T_src - 1) / src_fps
    n = int(np.floor(dur / CONTROL_DT)) + 1
    if n < 2:
        raise ValueError("Motion is shorter than one control transition.")
    t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
    i0 = torch.clamp(t.floor().long(), max=T_src - 2)
    a = (t - i0.float()).clamp(0, 1)
    p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
    j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
    # Batched shortest-arc SLERP; the native helper accepts a single interpolation pair.
    q0, q1 = quat[i0], quat[i0 + 1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    theta = torch.acos(dot.abs().clamp(max=1.0))
    sine, weight = torch.sin(theta), a[:, None]
    w0 = torch.where(sine < 1e-6, 1 - weight, torch.sin((1 - weight) * theta) / (sine + 1e-12))
    w1 = torch.where(sine < 1e-6, weight, torch.sin(weight * theta) / (sine + 1e-12))
    quat = F.normalize(w0 * q0 + w1 * q1, dim=-1)
    return p, quat, j


class _PoseMotion:
    @torch.no_grad()
    def __init__(self, path: str, kin: ArticulationKinematics, device: str, src_fps: float = 30.0):
        if not np.isfinite(src_fps) or src_fps <= 0:
            raise ValueError("Source frame rate must be finite and positive.")
        files = sorted((*Path(path).rglob("*.csv"), *Path(path).rglob("*.npy")))
        if not files:
            raise ValueError(f"No motion CSV/NumPy files found in {path}.")
        self.device = device
        self.names = [f.relative_to(path).with_suffix("").as_posix() for f in files]
        if len(set(self.names)) != len(self.names):
            raise ValueError("Motion names must be unique across CSV and NumPy files.")
        ps, qs, js = [], [], []
        lens = []
        for f in files:
            raw = torch.tensor(
                np.load(f, allow_pickle=False) if f.suffix == ".npy" else np.loadtxt(f, delimiter=","),
                dtype=torch.float32,
            )
            width = 7 + len(kin.q_lo)
            if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != width:
                raise ValueError(f"Motion {f} must contain at least two {width}-channel frames.")
            if not torch.isfinite(raw).all() or (raw[:, 3:7].norm(dim=-1) < 1e-6).any():
                raise ValueError(f"Motion {f} contains nonfinite values or invalid quaternions.")
            if src_fps == 1.0 / CONTROL_DT:
                p, quat, j = raw[:, :3], F.normalize(raw[:, 3:7], dim=-1), raw[:, 7:]
            else:
                p, quat, j = resample_motion(raw, src_fps)
            n = len(p)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = matrix_from_quat(quat)
            v = torch.gradient(p, spacing=CONTROL_DT, dim=0)[0]
            qd = torch.gradient(j, spacing=CONTROL_DT, dim=0)[0]
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)


class MotionReference:
    """Concatenated clips with exclusive ends, retaining supplied reference velocities.

    NPZ clips contain ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``
    (WXYZ), ``root_lin_vel_w``, and ``root_ang_vel_w`` at 50 Hz. CSV/NPY inputs use
    the existing XYZW pose preprocessing; their inferred velocities are not a
    substitute for the reference project's processed motion dataset.
    """

    def __init__(self, cfg, device: str):
        names = cfg.joint_names
        bodies = cfg.tracked_bodies
        self.fk = ForwardKinematics(cfg.kinematics_path, names, bodies, device)
        self.device = device
        files = sorted(Path(cfg.motion_dir).rglob("*.npz"))
        shard_count = getattr(cfg, "motion_shard_count", 1)
        shard_index = getattr(cfg, "motion_shard_index", 0)
        if not 0 <= shard_index < shard_count:
            raise ValueError("Motion shard index must be in [0, motion_shard_count).")
        if shard_count > 1:
            files = files[shard_index::shard_count]
            if not files:
                raise ValueError("Sharded references require at least one NPZ clip per rank.")
        fields = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
        if files:
            clips = []
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    clip = {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in fields}
                    if "fps" in data and float(data["fps"]) != 50.0:
                        raise ValueError(f"Reference clip {path} must be sampled at 50 Hz.")
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
        else:
            source = _PoseMotion(cfg.motion_dir, self.fk.kinematics, device, cfg.source_fps)
            self.joint_pos, self.joint_vel = source.q, source.qd
            self.root_pos_w, self.root_quat_w = source.root_p, from_native(quat_from_matrix(source.root_R))
            self.root_lin_vel_w, self.root_ang_vel_w = source.root_v, source.root_w
            lengths, self.names = source.clip_len.tolist(), source.names
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
