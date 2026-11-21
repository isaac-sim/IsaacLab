# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-tracking settings matching SuperMimic revision 5efbf114."""

from isaaclab.utils import configclass

from isaaclab_rl.model_based.learners import ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg, WorldModelLearnerCfg
from isaaclab_rl.model_based.runner import ModelBasedRunnerCfg

from ....mdp.model_based.commands_cfg import ReturnSamplerCfg, TasksCfg


@configclass
class TrackingModelBasedCfg(ModelBasedRunnerCfg):
    representation: str = "gencoord"
    actor_obs: str = "rep_state"
    tasks: TasksCfg = TasksCfg()
    slot_mix: dict[str, float] = {"motion": 1.0, "velocity": 0.0}
    sampling_mode: str = "return"
    return_sampler: ReturnSamplerCfg = ReturnSamplerCfg()


@configclass
class MaxCoordModelBasedCfg(TrackingModelBasedCfg):
    representation: str = "maxcoord"


@configclass
class PaperTrackingModelBasedCfg(TrackingModelBasedCfg):
    """Published Appendix C/D settings, distinct from the pinned source defaults."""

    actor_obs: str = "proprio"
    history_length: int = 10
    max_iterations: int = 320000
    save_interval: int = 4000
    log_interval: int = 100
    allow_tf32: bool = True
    lr_decay_iterations: int = 250000
    # Eight predicted transitions require the initial snapshot plus eight targets.
    wm: WorldModelLearnerCfg = WorldModelLearnerCfg(optimizer="adam", horizon=9)
    replay_ratio: int = 18
    policy: ActorCriticPolicyLearnerCfg = ActorCriticPolicyLearnerCfg(optimizer="adam")


@configclass
class PaperNoCriticModelBasedCfg(PaperTrackingModelBasedCfg):
    """The published horizon ablation with the terminal value removed."""

    policy: BpttPolicyLearnerCfg = BpttPolicyLearnerCfg(optimizer="adam", gamma=0.99)
