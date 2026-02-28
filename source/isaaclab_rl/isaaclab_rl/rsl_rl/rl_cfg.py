# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Model configurations #
#########################


@configclass
class RslRlMLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = "MLPModel"
    """The model class name. Default is MLPModel."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP network."""

    activation: str = MISSING
    """The activation function for the MLP network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Default is False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution. Default is None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""

    stochastic: bool = MISSING
    """Whether the model output is stochastic.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and set it to None
    for deterministic output or to a valid configuration class, e.g., `GaussianDistributionCfg` for stochastic output.
    """

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the model.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `init_std` field of the distribution configuration to specify the initial noise standard deviation.
    """

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the model. Default is scalar.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `std_type` field of the distribution configuration to specify the type of noise standard deviation.
    """

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use
    the `HeteroscedasticGaussianDistributionCfg` if state-dependent standard deviation is desired.
    """


@configclass
class RslRlRNNModelCfg(RslRlMLPModelCfg):
    """Configuration for RNN model."""

    class_name: str = "RNNModel"
    """The model class name. Default is RNNModel."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlCNNModelCfg(RslRlMLPModelCfg):
    """Configuration for CNN model."""

    class_name: str = "CNNModel"
    """The model class name. Default is CNNModel."""

    @configclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = MISSING
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = 1
        """The stride for the CNN."""

        dilation: int | tuple[int] | list[int] = 1
        """The dilation for the CNN."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
        """The padding for the CNN."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN."""

        activation: str = MISSING
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = False
        """Whether to use max pooling for the CNN."""

        global_pool: Literal["none", "max", "avg"] = "none"
        """The global pooling for the CNN."""

        flatten: bool = True
        """Whether to flatten the output of the CNN."""

    cnn_cfg: CNNCfg = MISSING
    """The configuration for the CNN(s)."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

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

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False."""

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """This parameter is deprecated and will be removed in the future.

    For rsl-rl < 4.0.0, use `actor_obs_normalization` and `critic_obs_normalization` of the policy instead.
    For rsl-rl >= 4.0.0, use `obs_normalization` of the model instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "actor": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the actor will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    check_for_nan: bool = True
    """Whether to check for NaN values coming from the environment."""

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

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""

    actor: RslRlMLPModelCfg = MISSING
    """The actor configuration."""

    critic: RslRlMLPModelCfg = MISSING
    """The critic configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is is deprecated. Please use `actor` and `critic` model configurations
    instead.
    """


#############################
# Deprecated configurations #
#############################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead.
    """

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlRNNModelCfg` instead.
    """

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""
