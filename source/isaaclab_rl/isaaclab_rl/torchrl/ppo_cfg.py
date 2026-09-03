# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class TorchRlPpoCfg:
    """Configuration of :func:`~isaaclab_rl.torchrl.train_ppo`."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    """Environment steps collected from every environment per iteration."""
    max_iterations: int = MISSING
    save_interval: int = MISSING
    """Iterations between checkpoints."""
    experiment_name: str = MISSING
    """Name of the experiment folder under ``logs/torchrl``."""
    run_name: str = ""
    """Optional suffix of the timestamped run folder."""
    clip_actions: float | None = None
    """Clipping range applied to actions before they reach the environment; ``None`` disables it."""

    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    """The critic reads the ``"critic"`` observation group when the task defines one, else ``"policy"``."""
    activation: str = "ELU"
    """Name of a :mod:`torch.nn` activation class."""
    init_noise_std: float = 1.0
    """Initial standard deviation of the Gaussian policy."""

    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
