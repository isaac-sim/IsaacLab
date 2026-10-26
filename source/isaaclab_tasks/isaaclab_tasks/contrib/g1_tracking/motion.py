# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""G1 reference motions resampled to 50 Hz, with bounded-memory FK construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.utils.math import matrix_from_quat

from .kinematics import G1Kinematics, mat_to_6d, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


def resample_motion(raw: torch.Tensor, src_fps: float):
    """Resample metric/xyzw/radian G1 poses to the control rate using shortest-arc SLERP."""
    p, quat, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
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


class MotionLib:
    def __init__(self, path: str, kin: G1Kinematics, device: str, src_fps: float = 30.0):
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
            if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 36:
                raise ValueError(f"Motion {f} must contain at least two 36-channel frames.")
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
        self.clip_start = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0
        ).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

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
        with torch.no_grad():
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(
                    self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e]
                )
                if s == 0:
                    self.body_p, self.body_R, self.body_v, self.body_w = (
                        value.new_empty((self.num_frames, *value.shape[1:])) for value in out
                    )
                for destination, value in zip((self.body_p, self.body_R, self.body_v, self.body_w), out, strict=True):
                    destination[s:e].copy_(value)
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start, frame_bins = [], [], []
        frame_offset = 0
        for c in range(len(lens)):
            nb = (lens[c] - 2) // BIN_FRAMES + 1
            frame_bins.append((torch.arange(lens[c]) // BIN_FRAMES).clamp_max(nb - 1) + len(bin_start))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(frame_offset + b * BIN_FRAMES)
            frame_offset += lens[c]
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.cat(frame_bins).to(device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        # Beyond the recorded clip, the padded pose is held still rather than continuing at its endpoint velocity.
        recorded = (fut == f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)).unsqueeze(-1)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (
                dp,
                rot,
                self.root_v_local[fut] * recorded,
                self.root_w_local[fut] * recorded,
                self.q[fut],
                self.qd[fut] * (0.1 * recorded),
            ),
            -1,
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]
