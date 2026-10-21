# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict
from torch import nn

from isaaclab_tasks.core.velocity.mdp.symmetry.g1 import _action_joint_map, _mirror_group


class Mirror:
    def __init__(self, env):
        self.env = env.unwrapped
        self.perm, self.sign = _action_joint_map(self.env)

    @torch.no_grad()
    def observations(self, obs):
        assert set(obs.keys()) == {"policy", "teacher", "depth", "depth_mirror"}
        return TensorDict(
            {
                "policy": _mirror_group(self.env, "policy", obs["policy"]),
                "teacher": _mirror_group(self.env, "teacher", obs["teacher"]),
                "depth": obs["depth_mirror"].flip(-1),
                "depth_mirror": obs["depth"].flip(-1),
            },
            batch_size=obs.batch_size,
        )

    @torch.no_grad()
    def actions(self, actions):
        return actions[..., self.perm] * self.sign


class AugmentedStorage:
    def __init__(self, storage, mirror):
        self.storage = storage
        self.mirror = mirror

    def __getattr__(self, name):
        return getattr(self.storage, name)

    def generator(self):
        for batch in self.storage.generator():
            out = copy(batch)
            out.observations = torch.cat([batch.observations, self.mirror.observations(batch.observations)], dim=0)
            out.privileged_actions = torch.cat(
                [batch.privileged_actions, self.mirror.actions(batch.privileged_actions)]
            )
            out.dones = batch.dones.repeat(2, 1)
            yield out


class MirrorDistillation(Distillation):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = Distillation.construct_algorithm(obs, env, cfg, device)
        assert not alg.student.is_recurrent and not alg.teacher.is_recurrent
        alg.mirror = Mirror(env)
        alg.storage = AugmentedStorage(alg.storage, alg.mirror)
        return alg

    def process_env_step(self, obs, rewards, dones, extras):
        mirrored = TensorDict(
            {"policy": _mirror_group(self.mirror.env, "policy", obs["policy"])}, batch_size=obs.batch_size
        )
        self.student.update_normalization(mirrored)
        self.student.update_normalization(obs)
        super().process_env_step(obs, rewards, dones, extras)

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if "actor_state_dict" in loaded_dict:
            load_cfg = {"teacher": True, "student": False, "optimizer": False, "iteration": False}
        return super().load(loaded_dict, load_cfg, strict)

    def update(self):
        """Update the student with accumulated gradients over rollout steps."""
        self.num_updates += 1
        mean_loss = 0.0
        loss = 0.0
        count = 0
        for _ in range(self.num_learning_epochs):
            for batch in self.storage.generator():
                behavior_loss = self.loss_fn(self.student(batch.observations), batch.privileged_actions)
                loss = loss + behavior_loss
                mean_loss += behavior_loss.item()
                count += 1
                if count % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    loss = 0.0
        self.storage.clear()
        return {"behavior": mean_loss / count}
