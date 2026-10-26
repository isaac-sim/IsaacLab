# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from physicsnemo_robotics.dynamics.normalization import RunningNormalizer

from isaaclab_tasks.contrib.g1_tracking.kinematics import mat_to_6d, so3_exp, so3_log, yaw_mat
from isaaclab_tasks.contrib.g1_tracking.tracking import NUM_J, XI_DIM

INV_DIM = 1 + 6 + 3 + 3 + 29 + 29
CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05), -1)


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


WM_IN = INV_DIM + NUM_J + XI_DIM
WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J


def run_network(net: nn.Module, x: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    if dtype is None:
        return net(x)
    with torch.autocast("cuda", dtype=dtype):
        return net(x).float()


def state_difference(S, S1):
    """Frame-local increments from S to S1 (the regression targets)."""
    p, R, v, w, q, qd = S
    p1, R1, v1, w1, q1, qd1 = S1
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
            so3_log(Rt @ R1),
            (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
            (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
            q1 - q,
            qd1 - qd,
        ),
        -1,
    )


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(self, hidden=(2048, 2048, 2048), amp_dtype=None, network: nn.Module | None = None):
        super().__init__()
        self.amp_dtype = amp_dtype
        self.net = network if network is not None else mlp(WM_IN, list(hidden), WM_OUT, act=nn.SiLU, layer_norm=True)
        if network is None:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNormalizer(WM_IN)
        self.out_norm = RunningNormalizer(WM_OUT, eps=1e-4)

    def inputs(self, S, a, xi_n):
        return torch.cat((invariant_features(S), a, xi_n), -1)

    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        tokens = self.in_norm(self.inputs(S, a, xi_n)).unsqueeze(-2)
        y = run_network(self.net, tokens, self.amp_dtype).squeeze(-2)
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32, amp_dtype=None):
        super().__init__()
        self.hist = hist
        self.amp_dtype = amp_dtype
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNormalizer(obs_dim)
        self.goal_norm = RunningNormalizer(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_network(self.enc, self.goal_norm(goal), self.amp_dtype))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_network(self.dec, torch.cat((z, o), -1), self.amp_dtype)


class Critic(nn.Module):
    def __init__(self, in_dim: int, amp_dtype=None):
        super().__init__()
        self.amp_dtype = amp_dtype
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNormalizer(in_dim)

    def forward(self, x):
        return run_network(self.net, self.norm(x), self.amp_dtype).squeeze(-1)


def critic_features(mdp, S, f, xi) -> torch.Tensor:
    """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
    m = mdp.motion
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Ht = H.transpose(-1, -2)
    rp, rR = m.root_p[f], m.root_R[f]
    bp = mdp.kin.forward(p, R, v, w, q, qd)[0]
    dR = yaw_mat(R @ rR.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
    ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
    e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
    e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
    rel = mat_to_6d(R.transpose(-1, -2) @ rR)
    return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)
