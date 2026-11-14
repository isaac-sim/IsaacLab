# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning and checkpoint settings for the native SuperMimic workflow."""

from isaaclab.utils import configclass


@configclass
class SuperMimicCfg:
    """SuperMimic settings; task timing, observations and reference goals belong to the environment."""

    seed: int = 0
    experiment_name: str = "g1_tracking"
    max_iterations: int = 14_000
    buffer_steps: int = 1000
    """Replay capacity in control steps per world."""
    update_to_data_ratio: float = 0.4
    """Learning iterations per physical control step."""
    horizon: int = 32
    """Differentiable imagined rollout length."""
    world_model_window: int = 8
    world_model_batches: int = 10
    world_model_batch_size: int = 2048
    policy_batch_size: int = 4096
    learning_rate: float = 2e-4
    gae_lambda: float = 0.95
    action_noise: float = 0.1
    velocity_alignment_weight: float = 0.1
    """Velocity-to-motion latent alignment on reference-command rollouts only."""
    critic_epochs: int = 5
    critic_minibatches: int = 4
    warmup_steps: int = 64
    world_model_only_iterations: int = 500
    learning_rate_decay_iterations: int = 250_000
    world_model: str = "mlp"
    """Dynamics architecture: ``mlp`` or ``nerd-one-frame`` (optional PhysicsNeMo Robotics)."""
    amp: str = "bf16"
    """Neural projection precision: ``bf16`` or ``none``. Geometry always uses float32."""
    compile: bool = True
    """Compile the tensor rollout, objective and regression functions with Torch."""
    save_interval: int = 2000
    save_interval_seconds: float = 120.0
    """Maximum time between checkpoint requests, checked at iteration boundaries."""
    log_interval: int = 50
    max_hours: float | None = None
    """Cumulative training-hour budget, including prior progress when resuming."""
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_name: str | None = None
