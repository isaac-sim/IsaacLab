# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Literal

from isaaclab.utils import config_field

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Model configurations #
#########################


@dataclass
class RslRlMLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = config_field("MLPModel")
    """The model class name. Defaults to MLPModel."""

    hidden_dims: list[int] = config_field(MISSING)
    """The hidden dimensions of the MLP network."""

    activation: str = config_field(MISSING)
    """The activation function for the MLP network."""

    obs_normalization: bool = config_field(False)
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = config_field(None)
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

    @dataclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = config_field(MISSING)
        """The distribution class name."""

    @dataclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = config_field("GaussianDistribution")
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = config_field(MISSING)
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = config_field("scalar")
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @dataclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = config_field("HeteroscedasticGaussianDistribution")
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""

    stochastic: bool = config_field(MISSING)
    """Whether the model output is stochastic.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and set it to None
    for deterministic output or to a valid configuration class, e.g., `GaussianDistributionCfg` for stochastic output.
    """

    init_noise_std: float = config_field(MISSING)
    """The initial noise standard deviation for the model.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `init_std` field of the distribution configuration to specify the initial noise standard deviation.
    """

    noise_std_type: Literal["scalar", "log"] = config_field("scalar")
    """The type of noise standard deviation for the model. Defaults to scalar.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `std_type` field of the distribution configuration to specify the type of noise standard deviation.
    """

    state_dependent_std: bool = config_field(False)
    """Whether to use state-dependent standard deviation for the policy. Defaults to False.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use
    the `HeteroscedasticGaussianDistributionCfg` if state-dependent standard deviation is desired.
    """


@dataclass
class RslRlRNNModelCfg(RslRlMLPModelCfg):
    """Configuration for RNN model."""

    class_name: str = config_field("RNNModel")
    """The model class name. Defaults to RNNModel."""

    rnn_type: str = config_field(MISSING)
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = config_field(MISSING)
    """The dimension of the RNN layers."""

    rnn_num_layers: int = config_field(MISSING)
    """The number of RNN layers."""


@dataclass
class RslRlCNNModelCfg(RslRlMLPModelCfg):
    """Configuration for CNN model."""

    class_name: str = config_field("CNNModel")
    """The model class name. Defaults to CNNModel."""

    @dataclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = config_field(MISSING)
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = config_field(MISSING)
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = config_field(1)
        """The stride for the CNN. Defaults to 1."""

        dilation: int | tuple[int] | list[int] = config_field(1)
        """The dilation for the CNN. Defaults to 1."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = config_field("none")
        """The padding for the CNN. Defaults to none."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = config_field("none")
        """The normalization for the CNN. Defaults to none."""

        activation: str = config_field(MISSING)
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = config_field(False)
        """Whether to use max pooling for the CNN. Defaults to False."""

        global_pool: Literal["none", "max", "avg"] = config_field("none")
        """The global pooling for the CNN. Defaults to none."""

        flatten: bool = config_field(True)
        """Whether to flatten the output of the CNN. Defaults to True."""

    cnn_cfg: CNNCfg = config_field(MISSING)
    """The configuration for the CNN(s)."""


############################
# Algorithm configurations #
############################


@dataclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = config_field("PPO")
    """The algorithm class name. Defaults to PPO."""

    num_learning_epochs: int = config_field(MISSING)
    """The number of learning epochs per update."""

    num_mini_batches: int = config_field(MISSING)
    """The number of mini-batches per update."""

    learning_rate: float = config_field(MISSING)
    """The learning rate for the policy."""

    schedule: str = config_field(MISSING)
    """The learning rate schedule."""

    gamma: float = config_field(MISSING)
    """The discount factor."""

    lam: float = config_field(MISSING)
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = config_field(MISSING)
    """The coefficient for the entropy loss."""

    desired_kl: float = config_field(MISSING)
    """The desired KL divergence."""

    max_grad_norm: float = config_field(MISSING)
    """The maximum gradient norm."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = config_field("adam")
    """The optimizer to use. Defaults to adam."""

    value_loss_coef: float = config_field(MISSING)
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = config_field(MISSING)
    """Whether to use clipped value loss."""

    clip_param: float = config_field(MISSING)
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = config_field(False)
    """Whether to normalize the advantage per mini-batch. Defaults to False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    share_cnn_encoders: bool = config_field(False)
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False."""

    rnd_cfg: RslRlRndCfg | None = config_field(None)
    """The RND configuration. Defaults to None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = config_field(None)
    """The symmetry configuration. Defaults to None, in which case symmetry is not used."""


#########################
# Runner configurations #
#########################


@dataclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = config_field(42)
    """The seed for the experiment. Defaults to 42."""

    device: str = config_field("cuda:0")
    """The device for the rl-agent. Defaults to cuda:0."""

    num_steps_per_env: int = config_field(MISSING)
    """The number of steps per environment per update."""

    init_at_random_ep_len: bool = config_field(True)
    """Whether to randomize each environment's episode length before learning.

    Defaults to True. Disable this for curricula whose first recorded outcomes must come from
    complete episodes.
    """

    max_iterations: int = config_field(MISSING)
    """The maximum number of iterations."""

    empirical_normalization: bool = config_field(MISSING)
    """This parameter is deprecated and will be removed in the future.

    For rsl-rl < 4.0.0, use `actor_obs_normalization` and `critic_obs_normalization` of the policy instead.
    For rsl-rl >= 4.0.0, use `obs_normalization` of the model instead.
    """

    obs_groups: dict[str, list[str]] = config_field(MISSING)
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

    clip_actions: float | None = config_field(None)
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    check_for_nan: bool = config_field(True)
    """Whether to check for NaN values coming from the environment."""

    save_interval: int = config_field(MISSING)
    """The number of iterations between saves."""

    experiment_name: str = config_field(MISSING)
    """The experiment name."""

    run_name: str = config_field("")
    """The run name. Defaults to empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = config_field("tensorboard")
    """The logger to use. Defaults to tensorboard."""

    neptune_project: str = config_field("isaaclab")
    """The neptune project name. Defaults to "isaaclab"."""

    wandb_project: str = config_field("isaaclab")
    """The wandb project name. Defaults to "isaaclab"."""

    resume: bool = config_field(False)
    """Whether to resume a previous training. Defaults to False.

    This flag will be ignored for distillation.
    """

    load_run: str = config_field(".*")
    """The run directory to load. Defaults to ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = config_field("model_.*.pt")
    """The checkpoint file to load. Defaults to ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = config_field("OnPolicyRunner")
    """The runner class name. Defaults to OnPolicyRunner."""

    actor: RslRlMLPModelCfg = config_field(MISSING)
    """The actor configuration."""

    critic: RslRlMLPModelCfg = config_field(MISSING)
    """The critic configuration."""

    algorithm: RslRlPpoAlgorithmCfg = config_field(MISSING)
    """The algorithm configuration."""

    policy: RslRlPpoActorCriticCfg = config_field(MISSING)
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is is deprecated. Please use `actor` and `critic` model configurations
    instead.
    """


#############################
# Deprecated configurations #
#############################


@dataclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead.
    """

    class_name: str = config_field("ActorCritic")
    """The policy class name. Defaults to ActorCritic."""

    init_noise_std: float = config_field(MISSING)
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = config_field("scalar")
    """The type of noise standard deviation for the policy. Defaults to scalar."""

    state_dependent_std: bool = config_field(False)
    """Whether to use state-dependent standard deviation for the policy. Defaults to False."""

    actor_obs_normalization: bool = config_field(MISSING)
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = config_field(MISSING)
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = config_field(MISSING)
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = config_field(MISSING)
    """The hidden dimensions of the critic network."""

    activation: str = config_field(MISSING)
    """The activation function for the actor and critic networks."""


@dataclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlRNNModelCfg` instead.
    """

    class_name: str = config_field("ActorCriticRecurrent")
    """The policy class name. Defaults to ActorCriticRecurrent."""

    rnn_type: str = config_field(MISSING)
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = config_field(MISSING)
    """The dimension of the RNN layers."""

    rnn_num_layers: int = config_field(MISSING)
    """The number of RNN layers."""
