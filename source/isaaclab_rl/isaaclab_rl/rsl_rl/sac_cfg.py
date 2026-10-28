# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed configuration for native RSL-RL SAC and its opt-in FlashSAC components."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg, RslRlMLPModelCfg, RslRlQMLPModelCfg
from .symmetry_cfg import RslRlSymmetryCfg


@configclass
class RslRlSacAlgorithmCfg:
    """Arguments forwarded to SAC, including replay and critic construction."""

    class_name: str = "SAC"
    """The algorithm class resolved by RSL-RL."""

    num_critics: int = 2
    """Number of independently initialized state-action critics."""

    replay_buffer_size: int = 100_000
    """Replay capacity in transitions across all environments."""

    max_replay_memory_fraction: float | None = None
    """Optional fraction of free GPU memory used to cap replay on its first insertion."""

    num_updates: int = MISSING
    """Number of critic updates per collection iteration."""

    batch_size: int = MISSING
    """Number of replay anchors sampled per update."""

    n_steps: int = 1
    """Maximum number of transitions in each bootstrapped return."""

    gamma: float = MISSING
    """Discount factor."""

    tau: float = MISSING
    """Interpolation weight of online parameters in each target update."""

    actor_update_interval: int = 1
    """Number of critic updates between actor and temperature updates."""

    learning_starts: int = 1000
    """Collected transition count before updates; also sets uniform-action warm-up for a new actor."""

    actor_learning_rate: float = MISSING
    """Learning rate for the actor optimizer."""

    critic_learning_rate: float = MISSING
    """Learning rate for the critic optimizer."""

    alpha_learning_rate: float = MISSING
    """Learning rate for the entropy-temperature optimizer."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """Optimizer used by the actor and critics; temperature uses Adam."""

    weight_decay: float = 0.0
    """Weight decay for the actor and critic optimizers."""

    max_grad_norm: float | None = None
    """Maximum actor and critic gradient norm; None disables clipping."""

    init_alpha: float = 0.2
    """Initial entropy temperature."""

    target_entropy: float | None = None
    """Target entropy; None uses minus the action dimension unless target_entropy_std is set."""

    target_entropy_std: float | None = None
    """Gaussian standard deviation used to derive target entropy when target_entropy is None."""

    critic_loss: Literal["mse", "categorical"] = "mse"
    """Scalar mean-squared error or a categorical return distribution."""

    num_bins: int = 101
    """Number of support points for categorical critics."""

    value_range: tuple[float, float] = (-5.0, 5.0)
    """Lower and upper support bounds for categorical returns."""

    reward_normalization: bool = False
    """Whether to scale replay rewards using discounted-return statistics."""

    reward_normalization_max_return: float = 5.0
    """Sets the scale floor to the largest observed absolute return divided by this value."""

    exploration: Literal["gaussian", "held"] = "gaussian"
    """Independent Gaussian noise or noise held for a sampled number of environment steps."""

    noise_repeat_exponent: float = 2.0
    """Power-law exponent of the held-noise duration distribution."""

    noise_repeat_max: int = 16
    """Maximum number of environment steps sharing one held-noise sample."""

    learning_rate_schedule: Literal["constant", "cosine"] = "constant"
    """Constant learning rates or cosine annealing for all three optimizers."""

    final_learning_rate_ratio: float = 0.5
    """Learning-rate ratio reached at the end of cosine annealing."""

    schedule_updates: int | None = None
    """Number of critic updates in the cosine schedule; required for cosine annealing."""

    use_mixed_precision: bool = False
    """Whether to use BF16 autocast during updates."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Optional replay augmentation and actor mirror-loss configuration."""

    use_cuda_graph: bool = False
    """Whether to capture update steps with CUDA graphs; independent of torch_compile_mode."""


@configclass
class RslRlSacRunnerCfg(RslRlBaseRunnerCfg):
    """Off-policy runner with typed models and terminal observations enabled."""

    class_name: str = "OffPolicyRunner"
    """The runner class resolved by RSL-RL."""

    compute_final_obs: bool = True
    """Capture pre-reset observations for replay successors."""

    actor: RslRlMLPModelCfg = MISSING
    """Actor with a squashed Gaussian action distribution."""

    critic: RslRlQMLPModelCfg = MISSING
    """Configuration used to construct each state-action critic."""

    algorithm: RslRlSacAlgorithmCfg = MISSING
    """SAC learning and replay configuration."""
