# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from omni.isaac.lab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

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

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

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

    params: DictConfig = OmegaConf.create()
    """Additional agent parameters."""
    
    def __post_init__(self):
        # update the configuration with the additional parameters
        self.num_steps_per_env = self.params.get("agent.num_steps_per_env", self.num_steps_per_env)
        self.max_iterations = self.params.get("agent.max_iterations", self.max_iterations)
        self.save_interval = self.params.get("agent.save_interval", self.save_interval)
        self.experiment_name = self.params.get("agent.experiment_name", self.experiment_name)
        self.run_name = self.params.get("agent.run_name", self.run_name)
        self.logger = self.params.get("agent.logger", self.logger)
        self.wandb_project = self.params.get("agent.wandb_project", self.wandb_project)
        self.resume = self.params.get("agent.resume", self.resume)
        self.load_run = self.params.get("agent.load_run", self.load_run)
        self.load_checkpoint = self.params.get("agent.load_checkpoint", self.load_checkpoint)
        
        self.empirical_normalization = self.params.get("agent.empirical_normalization", self.empirical_normalization)
        self.algorithm.entropy_coef = self.params.get("agent.algorithm.entropy_coef", self.algorithm.entropy_coef)
        self.algorithm.learning_rate = self.params.get("agent.algorithm.learning_rate", self.algorithm.learning_rate)
        self.algorithm.gamma = self.params.get("agent.algorithm.gamma", self.algorithm.gamma)
        self.algorithm.lam = self.params.get("agent.algorithm.lam", self.algorithm.lam)
        self.algorithm.desired_kl = self.params.get("agent.algorithm.desired_kl", self.algorithm.desired_kl)
        self.policy.init_noise_std = self.params.get("agent.policy.init_noise_std", self.policy.init_noise_std)
        self.policy.actor_hidden_dims = self.params.get("agent.policy.actor_hidden_dims", self.policy.actor_hidden_dims)
        self.policy.critic_hidden_dims = self.params.get("agent.policy.critic_hidden_dims", self.policy.critic_hidden_dims)
        self.policy.activation = self.params.get("agent.policy.activation", self.policy.activation)
