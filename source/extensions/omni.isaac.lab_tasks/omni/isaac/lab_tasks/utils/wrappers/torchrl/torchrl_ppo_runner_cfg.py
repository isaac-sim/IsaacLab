# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass


@configclass
class DistributionCfg:

    distribution_class: torch.distributions.Distribution = MISSING
    """A torch.distributions.Distribution class to be used for sampling. """

    distribution_kwargs: dict | None = None
    """Keyword arguments to be passed to the distribution. """

    return_log_prob: bool = True
    """If ``True``, the log-probability of the distribution sample will be written in the tensordict with the key
    `'sample_log_prob'`. Default is ``True``."""


@configclass
class ProbabilisticActorCfg:
    """Configuration for the Actor network."""

    class_name: str = "ProbabilisticActor"
    """The actor class name. Default is ProbabilisticActor."""

    actor_network: object = MISSING
    """Actor network to use for value estimation"""

    in_keys: list[str] = ["policy"]
    """Key(s) that will be read from the input TensorDict and used to build the distribution. Importantly, if it's an
    iterable of string or a string, those keys must match the keywords used by the distribution class of interest,
    e.g. :obj:`"loc"` and :obj:`"scale"` for the Normal distribution and similar. If in_keys is a dictionary,
    the keys are the keys of the distribution and the values are the keys in the tensordict that will get match to the
    corresponding distribution keys.
    """

    out_keys: list[str] = ["loc", "scale"]
    """Keys where the sampled values will be written.
    Importantly, if these keys are found in the input TensorDict, the sampling step will be skipped.
    """

    distribution: DistributionCfg = MISSING
    """Distribution module Cfg used for sampling policy actions. """

    init_noise_std: float = 1.0
    """The standard deviation of the Gaussian noise added to the policy actions during exploration. """


@configclass
class ValueOperatorCfg:
    """Configuration for the Critic network."""

    critic_network: object = MISSING
    """Critic network to use for value estimation"""

    in_keys: list[str] | None = ["policy"]
    """Keys to be read from input tensordict and passed to the module.
    If it contains more than one element, the values will be passed in the order given by the in_keys iterable.
    Defaults to ``["policy"]``.
    """

    out_keys: list[str] | None = None

    """Keys to be written to the input tensordict.
    The length of out_keys must match the
    number of tensors returned by the embedded module. Using "_" as a
    key avoid writing tensor to output.
    Defaults to ``["state_value"]`` or  ``["state_action_value"]`` if ``"action"`` is part of the ``in_keys``.
    """


@configclass
class ClipPPOLossCfg:
    """Configuration for the TorchRL loss module. Defines policy model architecture and sets PPO parameters."""

    class_name: str = "CLipPPOLoss"
    """The loss module class name. Default is ClipPPOLoss."""

    actor_network: ProbabilisticActorCfg = MISSING
    """The model architecture configuration for the actor network."""

    value_network: ValueOperatorCfg = MISSING
    """The model architecture configuration for the critic network."""

    value_key: str = "state_value"
    """The input tensordict key where the state value is expected to be written. Defaults to ``"state_value"``."""

    desired_kl: float = MISSING
    """The target KL divergence."""

    value_loss_coef: float = MISSING
    """Critic loss multiplier when computing the total loss."""

    clip_param: float = MISSING
    """The PPO epsilon clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    entropy_bonus: bool = False
    """If ``True``, an entropy bonus will be added to the loss to favour exploratory policies.."""

    loss_critic_type: Literal["l1", "l2", "smooth_l1"] = "l2"
    """loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1"."""

    normalize_advantage: bool = False
    """Normalize advantages by subtracting the mean and dividing by its std before computing loss. Defaults to False."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    max_grad_norm: float = MISSING
    """value to be used for clipping gradients. ."""


@configclass
class CollectorCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "SyncDataCollector"
    """The collector class name. Default is SyncDataCollector."""

    actor_network: ProbabilisticActorCfg = MISSING
    """The model architecture configuration for the actor network."""

    split_trajs: bool = False


@configclass
class OnPolicyPPORunnerCfg:
    """Configuration of the PPO torchRL runner."""

    loss_module: ClipPPOLossCfg = MISSING
    """The loss module configuration."""

    collector_module: CollectorCfg = MISSING
    """The collector module configuration."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    num_epochs: int = MISSING
    """The number of model optimizations to do per batch of experiences."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    lr_schedule: Literal["fixed", "adaptive"] = "adaptive"
    """The learning rate schedule. "fixed" for no learning rate annealing, "adaptive" to use a kl-based scheduler."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    min_sub_traj_len: int = -1
    """Minimum value of :obj:`sub_traj_len`, in case some elements of the batch contain few steps. Default is -1 (i.e. no minimum value)"""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    save_trainer_interval: int = 100
    """"How often to save the current policy to disk, in number of optimization steps"""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
