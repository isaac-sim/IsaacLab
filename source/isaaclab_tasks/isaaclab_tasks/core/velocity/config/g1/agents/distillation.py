# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feed-forward G1 distillation with paired original and mirrored rollout samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict
from torch import nn

from isaaclab_tasks.core.velocity.mdp.symmetry.g1 import _action_joint_map, _mirror_group

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


class Mirror:
    """Reflect G1 proprioception, paired depth cameras, and joint-position actions.

    Args:
        env: Wrapped G1 environment supplying joint and observation layouts.
    """

    def __init__(self, env: VecEnv) -> None:
        self.env = env.unwrapped
        self.perm, self.sign = _action_joint_map(self.env)

    @torch.no_grad()
    def observations(self, obs: TensorDict) -> TensorDict:
        """Return reflected observations without modifying the input.

        Args:
            obs: Batched proprioception, height scan, history, and paired depth images.

        Returns:
            The same observation groups and batch shape. Depth images swap camera sides
            and flip horizontally; joint and vector terms reflect across the sagittal plane.
        """
        assert set(obs.keys()) == {"proprio", "height_scan", "proprio_history", "depth", "depth_mirror"}
        return TensorDict(
            {
                "proprio": _mirror_group(self.env, "proprio", obs["proprio"]),
                "height_scan": _mirror_group(self.env, "height_scan", obs["height_scan"]),
                "proprio_history": _mirror_group(self.env, "proprio_history", obs["proprio_history"]),
                "depth": obs["depth_mirror"].flip(-1),
                "depth_mirror": obs["depth"].flip(-1),
            },
            batch_size=obs.batch_size,
        )

    @torch.no_grad()
    def actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Reflect joint-position actions in the environment's action order.

        Args:
            actions: Actions with the joint dimension last.

        Returns:
            Reflected actions with the same shape, dtype, and device.
        """
        return actions[..., self.perm] * self.sign


class MirrorDistillation(Distillation):
    """Train a feed-forward student on original and sagittally reflected teacher targets.

    Each rollout batch is doubled during the update; stored transitions remain unchanged.
    Gradients accumulate over ``gradient_length`` batches before each optimizer step.
    """

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MirrorDistillation:
        """Construct the RSL-RL algorithm and its environment-specific reflection maps.

        Args:
            obs: Initial observations used to determine model input shapes.
            env: Wrapped G1 environment.
            cfg: RSL-RL runner configuration selecting this algorithm class.
            device: Device for models and rollout storage.

        Returns:
            Configured distillation algorithm. Both models must be feed-forward.
        """
        alg = Distillation.construct_algorithm(obs, env, cfg, device)
        assert not alg.student.is_recurrent and not alg.teacher.is_recurrent
        alg.mirror = Mirror(env)
        return alg

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record the transition and update normalization with both observation orientations.

        Args:
            obs: Observations after the environment step.
            rewards: Per-environment rewards.
            dones: Per-environment termination flags.
            extras: Additional environment information passed to RSL-RL.
        """
        mirrored = TensorDict(
            {"proprio_history": _mirror_group(self.mirror.env, "proprio_history", obs["proprio_history"])},
            batch_size=obs.batch_size,
        )
        self.student.update_normalization(mirrored)
        self.student.update_normalization(obs)
        super().process_env_step(obs, rewards, dones, extras)

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load a distillation checkpoint, or only the teacher from a PPO checkpoint.

        Args:
            loaded_dict: Serialized model, optimizer, and iteration state.
            load_cfg: RSL-RL component selection, overridden for PPO actor checkpoints.
            strict: Whether model state keys must match exactly.

        Returns:
            Whether the runner should restore the checkpoint's iteration counter.
        """
        if "actor_state_dict" in loaded_dict:
            load_cfg = {"teacher": True, "student": False, "optimizer": False, "iteration": False}
        return super().load(loaded_dict, load_cfg, strict)

    def update(self) -> dict[str, float]:
        """Optimize paired rollout samples, then clear storage.

        Returns:
            Mean behavior-cloning loss across rollout steps and learning epochs.
        """
        self.num_updates += 1
        mean_loss = 0.0
        loss = 0.0
        count = 0
        for _ in range(self.num_learning_epochs):
            for batch in self.storage.generator():
                observations = torch.cat([batch.observations, self.mirror.observations(batch.observations)], dim=0)
                targets = torch.cat([batch.privileged_actions, self.mirror.actions(batch.privileged_actions)])
                behavior_loss = self.loss_fn(self.student(observations), targets)
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
