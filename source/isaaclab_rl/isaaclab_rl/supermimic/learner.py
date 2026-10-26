# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Replay ownership, supervised dynamics fitting and differentiable actor/critic updates."""

from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab_tasks.contrib.g1_tracking.tracking import MAX_REWARD_RATE, OBS_DIM, unflatten_state

from .networks import CRITIC_EXTRA, INV_DIM, Critic, Policy, WorldModel, critic_features, normalize_xi, state_difference
from .supermimic_cfg import SuperMimicCfg


class ReplayBuffer:
    """Learner-owned time/world ring; sampled model windows cannot cross physical discontinuities."""

    def __init__(self, capacity: int, transition: dict[str, torch.Tensor]):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity, self.size, self.pointer = (capacity, 0, 0)
        self.num_envs, self.device = (transition["s"].shape[0], transition["s"].device)
        fields = ("s", "s1", "a", "a_prev", "xi", "f", "ep_step", "done")
        self.data = {key: transition[key].new_empty((capacity, *transition[key].shape)) for key in fields}

    @torch.no_grad()
    def add(self, transition):
        for key, data in self.data.items():
            data[self.pointer].copy_(transition[key])
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _row(self, age):
        return (self.pointer - self.size + age) % self.capacity

    def sample_windows(self, batch_size, length):
        if self.size < length:
            raise ValueError("Replay does not yet contain a full model window.")
        start = torch.randint(self.size - length + 1, (batch_size,), device=self.device)
        env = torch.randint(self.num_envs, (batch_size, 1), device=self.device)
        rows = self._row(start[:, None] + torch.arange(length, device=self.device))
        out = {key: self.data[key][rows, env] for key in ("s", "s1", "a", "xi", "done")}
        steps, frames = (self.data["ep_step"][rows, env], self.data["f"][rows, env])
        connected = (
            ~out["done"][:, :-1]
            & (steps[:, 1:] == steps[:, :-1] + 1)
            & (frames[:, 1:] == frames[:, :-1] + 1)
            & (out["s"][:, 1:] == out["s1"][:, :-1]).all(-1)
        )
        out["mask"] = torch.cat((torch.ones(batch_size, 1, device=self.device), connected.float()), -1).cumprod(-1)
        return out

    def sample_starts(self, batch_size, history_length):
        if not self.size:
            raise ValueError("Cannot sample an empty replay buffer.")
        age = torch.randint(self.size, (batch_size,), device=self.device)
        env = torch.randint(self.num_envs, (batch_size,), device=self.device)
        rows = self._row(age)
        steps = self.data["ep_step"][rows, env].long()
        back = torch.arange(history_length - 1, -1, -1, device=self.device)
        back = torch.minimum(torch.minimum(back[None], steps[:, None]), age[:, None])
        history_rows = self._row(age[:, None] - back)
        fields = ("s", "a_prev", "xi", "f")
        start = {key: self.data[key][rows, env] for key in fields}
        history = {key: self.data[key][history_rows, env[:, None]] for key in fields}
        return (start, history)


class Learner:
    """Own models, normalization, replay and optimization, independent of simulator stepping."""

    def __init__(
        self,
        cfg: SuperMimicCfg,
        mdp,
        transition,
        noise_terms,
        reward_scale,
        world_model: WorldModel,
        *,
        goal_frames: int,
        history: int,
        gamma: float,
    ):
        self.cfg, self.mdp, self.motion = (cfg, mdp, mdp.motion)
        self.gamma = gamma
        self.device = transition["s"].device
        self.noise_terms, self.reward_scale = (noise_terms, reward_scale)
        if not 0 <= gamma < 1:
            raise ValueError("Discount must lie in [0, 1).")
        self.goal_frames, self.history, self.horizon = (goal_frames, history, cfg.horizon)
        dtype = torch.bfloat16 if cfg.amp == "bf16" else None
        self.wm = world_model
        self.pi = Policy(OBS_DIM, 73 * goal_frames, hist=history, amp_dtype=dtype).to(self.device)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * goal_frames, amp_dtype=dtype).to(self.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.learning_rate)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.buffer_steps, transition)
        self.error_floor = torch.cat(
            (
                torch.full((3,), 0.003),
                torch.full((3,), 0.005),
                torch.full((3,), 0.02),
                torch.full((3,), 0.05),
                torch.full((29,), 0.003),
                torch.full((29,), 0.05),
            )
        ).to(self.device)
        self.rollout = torch.compile(self._wm_rollout, disable=not cfg.compile)
        self.error = torch.compile(self._wm_err, disable=not cfg.compile)
        self.imagine = torch.compile(self._imag_step, disable=not cfg.compile)
        self.reward_termination = torch.compile(self._reward_term, disable=not cfg.compile)
        self.critic_input_compiled = torch.compile(self._critic_input, disable=not cfg.compile)

    def add_noise(self, observation):
        return torch.cat(
            [
                observation[..., region] if fn is None else fn(observation[..., region], cfg)
                for region, fn, cfg in self.noise_terms
            ],
            -1,
        )

    @torch.no_grad()
    def act(self, observations):
        self.pi.obs_norm.update(observations["policy"][:, -1])
        self.pi.goal_norm.update(observations["goal"])
        action = self.pi(observations["policy"], observations["goal"])
        return (action + self.cfg.action_noise * torch.randn_like(action)).clamp(-20, 20)

    @torch.no_grad()
    def observe(self, transition):
        self.buffer.add(transition)
        state, successor = (unflatten_state(transition["s"]), unflatten_state(transition["s1"]))
        self.wm.in_norm.update(self.wm.inputs(state, transition["a"], normalize_xi(transition["xi"])))
        self.wm.out_norm.update(state_difference(state, successor))

    def _wm_rollout(self, s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.cfg.world_model_window):
            S = self.wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))

    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.error_floor)
        e = (state_difference(unflatten_state(s1), S_hat) / std) ** 2
        e = e.mean(-1)
        return ((e * mask).sum() / s1.shape[0], e[:, -1])

    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g):
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = self.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return (S1, a, o_hist, f1, self.motion.goal(f1, self.goal_frames))

    def _reward_term(self, S, f, a, a_prev):
        return (self.reward_scale * self.mdp.reward(S, f, a, a_prev), self.mdp.soft_termination(S, f))

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((critic_features(self.mdp, S, f, xi_n), g), -1)

    def wm_update(self):
        w = self.buffer.sample_windows(self.cfg.world_model_batch_size, self.cfg.world_model_window)
        S_hat = self.rollout(w["s"], w["a"], w["xi"])
        loss, e_last = self.error(S_hat, w["s1"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0, error_if_nonfinite=True)
        self.opt_wm.step()
        m = w["mask"][:, -1]
        return (loss.detach() / self.cfg.world_model_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1))

    def sample_valid_starts(self, B: int):
        starts, hists, n = ([], [], 0)
        attempts = 0
        while n < B:
            attempts += 1
            if attempts > 32:
                raise RuntimeError("Replay has too few valid imagination starts for the configured horizon.")
            s, h = self.buffer.sample_starts(2 * B, self.history)
            f = s["f"]
            ok = f + self.horizon <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()})
            hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        return tuple(
            {key: torch.cat([sample[key] for sample in samples]) for key in samples[0]} for samples in (starts, hists)
        )

    def policy_update(self):
        s, h = self.sample_valid_starts(self.cfg.policy_batch_size)
        S = unflatten_state(s["s"])
        xi, f, a_prev = (s["xi"], s["f"], s["a_prev"])
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = self.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = self.add_noise(o_hist)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        B = self.cfg.policy_batch_size
        g0 = self.motion.goal(f, self.goal_frames)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = ([], [], [], [], [])
        for _ in range(self.horizon):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = self.imagine(S, o_hist, a_prev, f, xi, xi_n, g)
            Ss.append(S)
            As.append(a_prev)
            Fs.append(f)
            Gs.append(g)

        def flat(xs):
            return torch.stack(xs).flatten(0, 1)

        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, d = self.reward_termination(S_all, f_all, flat(As), flat(Aps))
        r, d = (r.view(self.horizon, B), d.view(self.horizon, B))
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=self.device), alive[:-1]), 0)
        disc = self.gamma ** torch.arange(self.horizon, device=self.device, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = (r.detach(), d.detach())
        terminal_value = self.critic(self._critic_input(S, f, xi_n, g))
        J = J + self.gamma**self.horizon * c * terminal_value
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        with torch.no_grad():
            S_c = tuple(
                torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1)
                for i in range(6)
            )
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
            X = self.critic_input_compiled(S_c, f_c, xi_c, g_c).view(self.horizon + 1, B, -1)
        self.critic.norm.update(X)
        with torch.no_grad():
            V = self.critic(X)
            R = V[-1]
            targets = []
            for t in range(self.horizon - 1, -1, -1):
                R = rews[t] + self.gamma * (1 - dones[t]) * (
                    (1 - self.cfg.gae_lambda) * V[t + 1] + self.cfg.gae_lambda * R
                )
                targets.append(R)
            targets = torch.stack(targets[::-1])
        Xf, Yf = (X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1))
        n = Xf.shape[0]
        mb = n // self.cfg.critic_minibatches
        for _ in range(self.cfg.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                prediction = self.critic(Xf[idx])
                v_loss = ((prediction - Yf[idx]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=True)
                self.opt_v.step()
        return (
            loss.detach(),
            v_loss.detach(),
            (r_sum / self.horizon / (MAX_REWARD_RATE * self.reward_scale)).mean(),
            gn.detach(),
            c.mean().detach(),
        )
