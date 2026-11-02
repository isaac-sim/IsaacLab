# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed configuration for model-based SuperMimic learning."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg, RslRlMLPModelCfg


@configclass
class RslRlSuperMimicTaskCfg:
    """Task-owned differentiable state, reward and deployment features."""

    class_name: str = MISSING
    """The task class resolved by RSL-RL."""


@configclass
class RslRlSuperMimicActorCfg:
    """History-conditioned actor with a goal encoder."""

    class_name: str = "SuperMimicActor"
    """The actor class resolved by RSL-RL."""

    history_length: int = 5
    """Number of actor frames in the deployment history."""

    latent_dim: int = 32
    """Width of the encoded goal."""

    encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    """Hidden widths of the goal encoder."""

    decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    """Hidden widths of the history-conditioned action decoder."""


@configclass
class RslRlWorldModelCfg:
    """Learned state increments and optional event-label predictions."""

    class_name: str = "WorldModel"
    """The world-model class resolved by RSL-RL."""

    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    """Hidden widths of the dynamics network."""

    label_hidden_dims: tuple[int, ...] = (512, 512)
    """Hidden widths of the optional event-label head."""


@configclass
class RslRlSuperMimicAlgorithmCfg:
    """Arguments forwarded to SuperMimic, including task and replay construction."""

    class_name: str = "SuperMimic"
    """The algorithm class resolved by RSL-RL."""

    task_cfg: RslRlSuperMimicTaskCfg = MISSING
    """Differentiable task configured by the environment owner."""

    replay_buffer_size: int = 1_000_000
    """Replay capacity in transitions across all environments."""

    max_replay_memory_fraction: float | None = None
    """Optional fraction of free CUDA memory available to replay at first insertion."""

    updates_per_step: float = 0.4
    """Learner iterations accrued per collected environment step."""

    num_world_model_updates: int = 10
    """Dynamics optimizer steps per learner iteration."""

    num_policy_updates: int = 1
    """Actor updates and associated critic fits per learner iteration."""

    world_model_window: int = 8
    """Maximum autoregressive dynamics fitting horizon."""

    world_model_batch_size: int = 2048
    """Replay anchors per dynamics optimizer step."""

    horizon: int = 32
    """Number of imagined policy steps."""

    policy_batch_size: int = 4096
    """Starts per actor update."""

    learning_rate: float = 2e-4
    """Initial learning rate of each optimizer."""

    lr_decay_iterations: int = 250_000
    """Learner iterations in the shared cosine learning-rate schedule."""

    lr_final_ratio: float = 0.03
    """Final learning rate relative to its initial value."""

    gamma: float = 0.99
    """Reward discount."""

    gae_lambda: float = 0.95
    """TD(lambda) mixing coefficient for value targets."""

    critic_epochs: int = 5
    """Epochs per critic fit."""

    critic_minibatches: int = 4
    """Minibatches per critic epoch."""

    warmup_steps: int = 64
    """Stored steps per environment required before learning."""

    world_model_warmup_iterations: int = 500
    """Learner iterations fitting only dynamics."""

    action_noise: float = 0.1
    """Standard deviation of additive collection noise."""

    max_step_reward: float | None = None
    """Optional per-step reward bound used to cap values at the maximum discounted return."""

    use_mixed_precision: bool = False
    """Whether to use BF16 matrix operations during learning."""

    max_grad_norm: float = 1.0
    """Global gradient-norm limit for each model."""


@configclass
class RslRlSuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    """Standard replay runner with physical successors on every step."""

    class_name: str = "OffPolicyRunner"
    """The runner class resolved by RSL-RL."""

    final_obs_every_step: bool = True
    """Capture physical observations before interval events on continuing steps."""

    actor: RslRlSuperMimicActorCfg = MISSING
    """History-conditioned deployment actor."""

    world_model: RslRlWorldModelCfg = MISSING
    """Learned physical dynamics and event predictions."""

    critic: RslRlMLPModelCfg = MISSING
    """Scalar value model consuming task-owned privileged features."""

    algorithm: RslRlSuperMimicAlgorithmCfg = MISSING
    """SuperMimic learning, task and replay configuration."""
