# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-based training settings matching the pinned SuperMimic reference."""

from isaaclab.utils import configclass

from .bptt import ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg
from .world_model_learner import WorldModelLearnerCfg


@configclass
class ModelBasedRunnerCfg:
    """Learning iterations count vectorized physical environment steps."""

    seed: int = 42
    max_iterations: int = 1000000
    buffer_size: int = 1000
    replay_ratio: int = 16
    dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    latent_dim: int = 32
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    latent_bc_weight: float = 0.1
    history_length: int = 1
    allow_tf32: bool = False
    lr_decay_iterations: int | None = None
    lr_final_factor: float = 0.03
    termination_softness: float = 0.05
    wm: WorldModelLearnerCfg = WorldModelLearnerCfg()
    policy: BpttPolicyLearnerCfg | ActorCriticPolicyLearnerCfg = ActorCriticPolicyLearnerCfg()
    log_interval: int = 10
    save_interval: int = 1000
    wandb_project: str | None = None
    wandb_name: str | None = None


REFERENCE_REVISION = "5efbf114e11f725c31d83e217be987c13c0e4ce2"
