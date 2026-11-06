# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: pelvis position, rotation matrix, world linear/angular
velocity of the pelvis link frame, joint positions and velocities.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

from .kinematics import TRACKED_BODIES, G1Kinematics, mat_to_6d, rot_angle_sq, yaw_mat
from .motion import MotionLib

STATE_DIM = 71
OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6
XI_DIM = 33
NUM_J = 29

MAX_REWARD_RATE = 5.0
MAX_VELOCITY_REWARD_RATE = 3.0


@configclass
class VelocityCfg:
    """Optional velocity interface, using the command distribution and objective in Appendix G."""

    fraction: float = 0.0
    """Fraction of physical worlds and imagined starts assigned to velocity control."""
    resampling_steps: int = 250
    reference_probability: float = 0.5
    standing_probability: float = 0.1
    ranges: tuple[tuple[float, float], ...] = ((-1.5, 2.0), (-1.32, 1.32), (-3.35, 3.35))
    fall_height: float = 0.35
    fall_tilt: float = 0.7


@configclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], matrix_from_quat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


class TrackingObjective:
    def __init__(
        self,
        kin: G1Kinematics,
        motion: MotionLib,
        default_q: torch.Tensor,
        cfg: RewardCfg | None = None,
        velocity_cfg: VelocityCfg | None = None,
    ):
        self.kin, self.motion, self.cfg = kin, motion, cfg if cfg is not None else RewardCfg()
        self.velocity_cfg = velocity_cfg if velocity_cfg is not None else VelocityCfg()
        cfg = self.cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f, velocity: bool = False) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1:30]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = torch.zeros_like(R[..., :2, :]).flatten(-2) if velocity else mat_to_6d(Rt @ self.motion.root_R[f])
        return torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Reward rate comparing S with reference frame f; the caller applies its step duration."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (
            (dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        )
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim
        if terms:
            return r, {
                "track": r_track,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (
            1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt)
        )
        return 1 - alive

    def sample_velocity_command(self, f, command, reference, resample):
        """Advance reference commands; replace held commands and their source only at resampling boundaries."""
        cfg = self.velocity_cfg
        reference = torch.where(
            resample, torch.rand_like(f, dtype=torch.float32) < cfg.reference_probability, reference
        )
        bounds = command.new_tensor(cfg.ranges)
        sampled = bounds[:, 0] + torch.rand_like(command) * (bounds[:, 1] - bounds[:, 0])
        sampled = torch.where(
            (torch.rand_like(f, dtype=torch.float32) < cfg.standing_probability).unsqueeze(-1), 0.0, sampled
        )
        command = torch.where(resample.unsqueeze(-1), sampled, command)
        return torch.where(reference.unsqueeze(-1), self.motion.velocity_command[f], command), reference

    def velocity_error(self, S, command) -> torch.Tensor:
        """Signed planar velocity errors in the root heading frame and world yaw-rate error."""
        _, R, v, w, _, _ = S
        v_heading = (yaw_mat(R).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        return torch.cat((v_heading[..., :2], w[..., 2:3]), -1) - command

    def velocity_reward(self, S, command, a, a_prev) -> torch.Tensor:
        """Velocity reward rate from Table A.6, including torso and joint regularization."""
        error = self.velocity_error(S, command)
        _, body_R, _, body_w = self.kin.forward(*S)
        torso = TRACKED_BODIES.index("torso_link")
        R = body_R[..., torso, :, :]
        tilt = yaw_mat(R).transpose(-1, -2) @ R
        q = S[4]
        return (
            2 * torch.exp(-error[..., :2].square().sum(-1) / 0.5**2)
            + torch.exp(-error[..., 2].square() / 0.5)
            - 0.3 * (a - a_prev).square().sum(-1)
            - 2 * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
            - 4 * tilt[..., 1, 2].square()
            - 0.05 * body_w[..., torso, :2].square().sum(-1)
        )

    def velocity_termination(self, S, soft: bool = False) -> torch.Tensor:
        """Fall-only termination, with a differentiable counterpart for imagination."""
        height_error = self.velocity_cfg.fall_height - S[0][..., 2]
        tilt_error = 1 - S[1][..., 2, 2] - self.velocity_cfg.fall_tilt
        if soft:
            return 1 - (1 - torch.sigmoid(height_error / 0.02)) * (1 - torch.sigmoid(tilt_error / 0.04))
        return (height_error > 0) | (tilt_error > 0)
