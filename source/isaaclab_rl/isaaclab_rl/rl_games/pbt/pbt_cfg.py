# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class PbtCfg:
    """
    Population-Based Training (PBT) configuration.

    leaders are policies with score > max(mean + threshold_std*std, mean + threshold_abs).
    underperformers are policies with score < min(mean - threshold_std*std, mean - threshold_abs).
    On replacement, selected hyperparameters are mutated multiplicatively in [change_min, change_max].
    """

    enabled: bool = False
    """Enable/disable PBT logic."""

    policy_idx: int = 0
    """Index of this learner in the population (unique in [0, num_policies-1])."""

    num_policies: int = 8
    """Total number of learners participating in PBT."""

    directory: str = ""
    """Root directory for PBT artifacts (checkpoints, metadata)."""

    workspace: str = "pbt_workspace"
    """Subfolder under the training dir to isolate this PBT run."""

    objective: str = "Episode_Reward/success"
    """The key in info returned by env.step that pbt measures to determine leaders and underperformers,
    If reward is stationary, using the term that corresponds to task success is usually enough, when reward
    are non-stationary, consider uses better objectives.
    """

    interval_steps: int = 100_000
    """Environment steps between PBT iterations (save, compare, replace/mutate)."""

    threshold_std: float = 0.10
    """Std-based margin k in max(mean ± k·std, mean ± threshold_abs) for leader/underperformer cuts."""

    threshold_abs: float = 0.05
    """Absolute margin A in max(mean ± threshold_std·std, mean ± A) for leader/underperformer cuts."""

    mutation_rate: float = 0.25
    """Per-parameter probability of mutation when a policy is replaced."""

    change_range: tuple[float, float] = (1.1, 2.0)
    """Lower and upper bound of multiplicative change factor (sampled in [change_min, change_max])."""

    mutation: dict[str, str] = {}
    """Mutation strings indicating which parameter will be mutated when pbt restart
    example:
        {
            "agent.params.config.learning_rate": "mutate_float"
            "agent.params.config.grad_norm": "mutate_float"
            "agent.params.config.entropy_coef": "mutate_float"
        }
    """
