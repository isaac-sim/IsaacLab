# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

"""Native PPO with the merged source post-PPO two-teacher BC pass.

The BC pass is adapted from Pollen Robotics microduck_rl at bae39f5900b318f7e87fc485c30a31213cfa14eb,
Apache-2.0 (LICENSE-microduck_rl.md). The runner, PPO update and checkpoint schemas are native RSL-RL.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .artifacts import ARTIFACTS, REVISION, resolve_artifact
from .distill import (
    _with_flat,
    body_frame_masks,
    expert_input,
    fallen_mask_from_obs,
    remap_anchor_yaw,
    turn_in_place_mask,
)

CONTRACT = "velstand-upstream-bae39f5-actor61-critic76-v1"


class MicroDuckVelStandPPO(PPO):
    """Single-GPU source recipe; frozen teachers use their own checkpoint normalizers."""

    def __init__(self, *args, bc_cfg=None, checkpoint_mode="auto", artifact_cache_dir=None, **kwargs):
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("VelStand teacher BC currently requires a single GPU")
        if checkpoint_mode not in ("auto", "warm_start", "resume"):
            raise ValueError(f"Unsupported VelStand checkpoint_mode: {checkpoint_mode}")
        super().__init__(*args, **kwargs)
        self.checkpoint_mode = checkpoint_mode
        self.env = None
        self.bc_cfg = bc_cfg
        self.expert = self.anchor = None
        self.artifact_cache_dir = artifact_cache_dir
        self.artifact_provenance = {"revision": REVISION, "sha256": {k: v[1] for k, v in ARTIFACTS.items()}}
        if bc_cfg:
            self.expert = self._load_teacher(
                resolve_artifact("stand", artifact_cache_dir, bc_cfg.get("checkpoint_path"))
            )
            self.anchor = self._load_teacher(
                resolve_artifact("walk", artifact_cache_dir, bc_cfg.get("anchor_checkpoint_path"))
            )
            self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=bc_cfg["learning_rate"])

    def _load_teacher(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        teacher = copy.deepcopy(self._raw_actor)
        # Actor-only import: the stand expert's privileged critic has 74, not 76 inputs.
        teacher.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        return teacher.to(self.device).eval().requires_grad_(False)

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        algorithm.env = env.unwrapped
        if algorithm.bc_cfg:
            # WARM_START=True in the source factory. Plain training must not silently start
            # from random weights; a subsequent native runner.load still supports resume.
            path = resolve_artifact(
                "walk", algorithm.artifact_cache_dir, algorithm.bc_cfg.get("anchor_checkpoint_path")
            )
            walk = torch.load(path, map_location=device, weights_only=True)
            PPO.load(algorithm, walk, None, True)
            algorithm.env.common_step_counter = 0
        return algorithm

    def process_env_step(self, obs, rewards, dones, extras):
        self._raw_actor.update_rollout_normalization(obs)
        self._raw_critic.update_rollout_normalization(obs)
        super().process_env_step(obs, rewards, dones, extras)

    def _environment_contract(self):
        if self.env is None or not hasattr(self.env, "cfg"):
            return None
        return {
            "terrain": self.env.cfg.scene.terrain.terrain_type,
            "selected_presets": list(self.env.cfg.selected_presets),
        }

    def save(self):
        saved = super().save()
        saved["microduck_velstand_state"] = {
            "contract": CONTRACT,
            "common_step_counter": int(self.env.common_step_counter) if self.env is not None else 0,
            "teachers": self.artifact_provenance,
            "environment": self._environment_contract(),
        }
        # Source BC Adam is intentionally fresh on resume; upstream never checkpoints it.
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        state = loaded_dict.get("microduck_velstand_state")
        resume = self.checkpoint_mode == "resume" or (self.checkpoint_mode == "auto" and state is not None)
        if resume:
            if state is None or state.get("contract") != CONTRACT:
                raise ValueError("Resume requires a matching VelStand observation/learning contract")
            if state.get("environment") != self._environment_contract():
                raise ValueError("VelStand resume terrain/backlash domain differs from the checkpoint")
            if state.get("teachers") != self.artifact_provenance:
                raise ValueError("VelStand resume teacher provenance differs from the checkpoint")
        else:
            if state is not None or loaded_dict.get("iter") != 3750:
                raise ValueError("Warm start requires the pinned walking checkpoint at iteration 3750")
            if self.anchor is not None:
                expected = self.anchor.state_dict()
                actual = loaded_dict["actor_state_dict"]
                if expected.keys() != actual.keys() or any(
                    not torch.equal(value, actual[key].to(value.device)) for key, value in expected.items()
                ):
                    raise ValueError("Warm-start actor is not the pinned walking teacher")
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if self.env is not None:
            self.env.common_step_counter = int(state["common_step_counter"]) if resume else 0
            # Native training enters learn() directly after load(). Recompute the
            # restored curricula/reset mixture and refresh cached observations now,
            # before the first resumed action can use the constructor's stage zero.
            self.env.reset()
        return load_iteration if resume else False

    def _bc_update(self) -> dict[str, float]:
        cfg = self.bc_cfg
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)  # (T*N, ...)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        gsl = tuple(cfg["gravity_slice"])
        fallen = fallen_mask_from_obs(flat, gsl, cfg["gate_tilt_deg"])
        upright = (
            ~fallen_mask_from_obs(flat, gsl, cfg["anchor_tilt_deg"])
            if self.anchor is not None
            else torch.zeros_like(fallen)
        )
        body_teach = torch.zeros_like(fallen)
        if cfg.get("body_slice") is not None:
            body_frames, body_teach = body_frame_masks(
                flat,
                tuple(cfg["twist_slice"]),
                tuple(cfg["body_slice"]),
                tuple(cfg["body_active_axes"]),
                tuple(cfg["body_no_teacher_axes"]),
            )
            upright = upright & ~body_frames
            body_teach = body_teach & ~fallen
        turn_free = torch.zeros_like(fallen)
        if cfg.get("unanchor_turn_in_place"):
            turn_free = turn_in_place_mask(flat, tuple(cfg["twist_slice"]))
            upright = upright & ~turn_free
        stats = {
            "expert_bc_fallen_frac": fallen.float().mean().item(),
            "expert_bc_anchor_frac": upright.float().mean().item(),
            "expert_bc_body_teach_frac": body_teach.float().mean().item(),
            "expert_bc_turn_free_frac": turn_free.float().mean().item(),
        }
        tilted = (
            fallen_mask_from_obs(flat, gsl, cfg.get("expert_mask_tilt_deg", 75.0)) & fallen
        )  # expert_mask_cols rows
        fallen = fallen | body_teach
        if fallen.sum().item() < cfg["min_samples"]:
            fallen = torch.zeros_like(fallen)  # too few fallen frames: anchor-only pass (or nothing)
        use = fallen | upright
        idx = use.nonzero().flatten()
        if idx.numel() == 0:
            stats["expert_bc"] = 0.0
            return stats
        obs_td = obs_td[idx]
        fallen = fallen[idx]
        tilted = tilted[idx]
        weight = torch.where(
            fallen,
            torch.full_like(fallen, cfg["coef"], dtype=torch.float),
            torch.full_like(fallen, cfg.get("anchor_coef", 0.0), dtype=torch.float),
        )
        with torch.no_grad():
            target = torch.zeros(idx.numel(), self.storage.actions.shape[-1], device=self.device)
            if fallen.any():
                exp_td = obs_td[fallen].clone()
                exp_flat = expert_input(
                    torch.cat([exp_td[g] for g in groups], dim=-1),
                    tuple(cfg["twist_slice"]),
                    tuple(cfg.get("expert_mask_cols", ())),
                    tilted[fallen],
                )
                target[fallen] = self.expert(_with_flat(exp_td, groups, exp_flat))
            if (~fallen).any():
                anc_td = obs_td[~fallen]
                if cfg.get("anchor_yaw_remap"):
                    anc_td = anc_td.clone()
                    anc_flat = remap_anchor_yaw(
                        torch.cat([anc_td[g] for g in groups], dim=-1),
                        tuple(cfg["twist_slice"]),
                        cfg["anchor_yaw_remap"],
                    )
                    anc_td = _with_flat(anc_td, groups, anc_flat)
                target[~fallen] = self.anchor(anc_td)
        n = idx.numel()
        mb = max(cfg.get("min_mini_batch", 1), n // cfg["mini_batches"])
        total = 0.0
        steps = 0
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s : s + mb]
                pred = self.actor(obs_td[sel])
                loss = (weight[sel] * (pred - target[sel]).pow(2).mean(dim=1)).mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += loss.item()
                steps += 1
        stats["expert_bc"] = total / max(steps, 1)
        return stats

    def update(self):
        # Native storage.clear only resets its cursor; raw observations remain valid for BC.
        loss_dict = super().update()
        if self.expert is not None:
            loss_dict.update(self._bc_update())
        return loss_dict
