# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from dataclasses import MISSING
from typing import TYPE_CHECKING

from packaging import version

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


_V4_0_0 = version.parse("4.0.0")
_V5_0_0 = version.parse("5.0.0")
_MODEL_CFG_NAMES = ("actor", "critic", "student", "teacher")


def handle_deprecated_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, installed_version) -> RslRlBaseRunnerCfg:
    """Handle deprecated RSL-RL configurations across version boundaries.

    This function mutates ``agent_cfg`` to keep configurations compatible with the installed ``rsl-rl`` version:

    - For ``rsl-rl < 4.0.0``, ``policy`` is required; new model configs (``actor``, ``critic``, ``student``,
        ``teacher``) are ignored and cleared.
    - For ``rsl-rl >= 4.0.0``, deprecated ``policy`` can be used to infer missing model configs, then ``policy`` is
        cleared.
    - For ``rsl-rl >= 5.0.0``, legacy stochastic parameters are migrated to ``distribution_cfg`` when needed; for
        ``4.0.0 <= rsl-rl < 5.0.0``, those legacy parameters are validated instead.

    Raises:
        ValueError: If required legacy parameters are missing for the selected ``rsl-rl`` version.
    """
    installed_version = version.parse(installed_version)

    # Handle configurations for rsl-rl < 4.0.0
    if installed_version < _V4_0_0:
        # exit if no policy configuration is present
        if not hasattr(agent_cfg, "policy") or _is_missing(agent_cfg.policy):
            raise ValueError(
                "The `policy` configuration is required for rsl-rl < 4.0.0. Please specify the `policy` configuration"
                " or update rsl-rl."
            )

        # handle deprecated obs_normalization argument
        if _has_non_missing_attr(agent_cfg, "empirical_normalization"):
            _handle_empirical_normalization(agent_cfg.policy, agent_cfg)

        # remove optimizer argument for PPO only available in rsl-rl >= 4.0.0
        from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

        if hasattr(agent_cfg.algorithm, "optimizer") and isinstance(agent_cfg.algorithm, RslRlPpoAlgorithmCfg):
            if agent_cfg.algorithm.optimizer != "adam":
                print(
                    "[WARNING]: The `optimizer` parameter for PPO is only available for rsl-rl >= 4.0.0. Consider"
                    " updating rsl-rl to use this feature. Defaulting to `adam` optimizer."
                )
            del agent_cfg.algorithm.optimizer

        # warn about model configurations only used in rsl-rl >= 4.0.0
        for model_name in _MODEL_CFG_NAMES:
            if _has_non_missing_attr(agent_cfg, model_name):
                _clear_new_model_cfg(agent_cfg, model_name)

    # Handle configurations for rsl-rl >= 4.0.0
    else:
        # Handle deprecated policy configuration
        if _has_non_missing_attr(agent_cfg, "policy"):
            print(
                "[WARNING]: The `policy` configuration is deprecated for rsl-rl >= 4.0.0. Please use, e.g., `actor` and"
                " `critic` model configurations instead."
            )

            # handle deprecated obs_normalization argument
            if _has_non_missing_attr(agent_cfg, "empirical_normalization"):
                _handle_empirical_normalization(agent_cfg.policy, agent_cfg)

            # import old and new config classes
            from isaaclab_rl.rsl_rl import (
                RslRlDistillationStudentTeacherCfg,
                RslRlDistillationStudentTeacherRecurrentCfg,
                RslRlMLPModelCfg,
                RslRlPpoActorCriticCfg,
                RslRlPpoActorCriticRecurrentCfg,
                RslRlRNNModelCfg,
            )

            # set actor model configuration if missing
            if hasattr(agent_cfg, "actor") and _is_missing(agent_cfg.actor):
                print("[WARNING]: The `policy` configuration is used to infer the `actor` model configuration.")
                if type(agent_cfg.policy) is RslRlPpoActorCriticCfg:
                    agent_cfg.actor = RslRlMLPModelCfg(
                        hidden_dims=agent_cfg.policy.actor_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.actor_obs_normalization,
                        stochastic=True,
                        init_noise_std=agent_cfg.policy.init_noise_std,
                        noise_std_type=agent_cfg.policy.noise_std_type,
                        state_dependent_std=agent_cfg.policy.state_dependent_std,
                    )
                elif type(agent_cfg.policy) is RslRlPpoActorCriticRecurrentCfg:
                    agent_cfg.actor = RslRlRNNModelCfg(
                        hidden_dims=agent_cfg.policy.actor_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.actor_obs_normalization,
                        stochastic=True,
                        init_noise_std=agent_cfg.policy.init_noise_std,
                        noise_std_type=agent_cfg.policy.noise_std_type,
                        state_dependent_std=agent_cfg.policy.state_dependent_std,
                        rnn_type=agent_cfg.policy.rnn_type,
                        rnn_hidden_dim=agent_cfg.policy.rnn_hidden_dim,
                        rnn_num_layers=agent_cfg.policy.rnn_num_layers,
                    )
            # set critic model configuration if missing
            if hasattr(agent_cfg, "critic") and _is_missing(agent_cfg.critic):
                print("[WARNING]: The `policy` configuration is used to infer the `critic` model configuration.")
                if type(agent_cfg.policy) is RslRlPpoActorCriticCfg:
                    agent_cfg.critic = RslRlMLPModelCfg(
                        hidden_dims=agent_cfg.policy.critic_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.critic_obs_normalization,
                        stochastic=False,
                    )
                elif type(agent_cfg.policy) is RslRlPpoActorCriticRecurrentCfg:
                    agent_cfg.critic = RslRlRNNModelCfg(
                        hidden_dims=agent_cfg.policy.critic_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.critic_obs_normalization,
                        stochastic=False,
                        rnn_type=agent_cfg.policy.rnn_type,
                        rnn_hidden_dim=agent_cfg.policy.rnn_hidden_dim,
                        rnn_num_layers=agent_cfg.policy.rnn_num_layers,
                    )
            # set student model configuration if missing
            if hasattr(agent_cfg, "student") and _is_missing(agent_cfg.student):
                print("[WARNING]: The `policy` configuration is used to infer the `student` model configuration.")
                if type(agent_cfg.policy) is RslRlDistillationStudentTeacherCfg:
                    agent_cfg.student = RslRlMLPModelCfg(
                        hidden_dims=agent_cfg.policy.student_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.student_obs_normalization,
                        stochastic=True,
                        init_noise_std=agent_cfg.policy.init_noise_std,
                        noise_std_type=agent_cfg.policy.noise_std_type,
                    )
                elif type(agent_cfg.policy) is RslRlDistillationStudentTeacherRecurrentCfg:
                    agent_cfg.student = RslRlRNNModelCfg(
                        hidden_dims=agent_cfg.policy.student_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.student_obs_normalization,
                        stochastic=True,
                        init_noise_std=agent_cfg.policy.init_noise_std,
                        noise_std_type=agent_cfg.policy.noise_std_type,
                        rnn_type=agent_cfg.policy.rnn_type,
                        rnn_hidden_dim=agent_cfg.policy.rnn_hidden_dim,
                        rnn_num_layers=agent_cfg.policy.rnn_num_layers,
                    )
            # set teacher model configuration if missing
            if hasattr(agent_cfg, "teacher") and _is_missing(agent_cfg.teacher):
                print("[WARNING]: The `policy` configuration is used to infer the `teacher` model configuration.")
                if type(agent_cfg.policy) is RslRlDistillationStudentTeacherCfg:
                    agent_cfg.teacher = RslRlMLPModelCfg(
                        hidden_dims=agent_cfg.policy.teacher_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.teacher_obs_normalization,
                        stochastic=True,
                        init_noise_std=0.0,
                    )
                elif type(agent_cfg.policy) is RslRlDistillationStudentTeacherRecurrentCfg:
                    agent_cfg.teacher = RslRlRNNModelCfg(
                        hidden_dims=agent_cfg.policy.teacher_hidden_dims,
                        activation=agent_cfg.policy.activation,
                        obs_normalization=agent_cfg.policy.teacher_obs_normalization,
                        stochastic=True,
                        init_noise_std=0.0,
                        rnn_type=agent_cfg.policy.rnn_type,
                        rnn_hidden_dim=agent_cfg.policy.rnn_hidden_dim,
                        rnn_num_layers=agent_cfg.policy.rnn_num_layers,
                    )

            # remove deprecated policy configuration
            agent_cfg.policy = MISSING

        # Handle new distribution configuration
        if installed_version < _V5_0_0:
            for model_name in _MODEL_CFG_NAMES:
                if _has_non_missing_attr(agent_cfg, model_name):
                    _validate_old_stochastic_cfg(getattr(agent_cfg, model_name))
        else:  # rsl-rl >= 5.0.0
            # import new distribution config classes
            from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

            for model_name in _MODEL_CFG_NAMES:
                if _has_non_missing_attr(agent_cfg, model_name):
                    _update_distribution_cfg(getattr(agent_cfg, model_name), RslRlMLPModelCfg)

    return agent_cfg


def _is_missing(value) -> bool:
    return isinstance(value, type(MISSING))


def _has_non_missing_attr(obj, attr_name: str) -> bool:
    return hasattr(obj, attr_name) and not _is_missing(getattr(obj, attr_name))


def _handle_empirical_normalization(policy_cfg, agent_cfg):
    print(
        "[WARNING]: The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and"
        " `critic_obs_normalization` as part of the `policy` configuration instead."
    )
    if _is_missing(policy_cfg.actor_obs_normalization):
        policy_cfg.actor_obs_normalization = agent_cfg.empirical_normalization
    if _is_missing(policy_cfg.critic_obs_normalization):
        policy_cfg.critic_obs_normalization = agent_cfg.empirical_normalization
    agent_cfg.empirical_normalization = MISSING


def _clear_new_model_cfg(agent_cfg, model_name: str):
    print(
        f"[WARNING]: The `{model_name}` model configuration is only used for rsl-rl >= 4.0.0. Consider updating rsl-rl"
        " or use the `policy` configuration for rsl-rl < 4.0.0."
    )
    setattr(agent_cfg, model_name, MISSING)


def _validate_old_stochastic_cfg(model_cfg):
    if not hasattr(model_cfg, "stochastic") or _is_missing(model_cfg.stochastic):
        raise ValueError(
            "Please parameterize the output distribution using the old parameters `stochastic`, `init_noise_std`,"
            " `noise_std_type`, and `state_dependent_std` or update rsl-rl."
        )
    # remove new distribution configuration
    if hasattr(model_cfg, "distribution_cfg"):
        del model_cfg.distribution_cfg


def _update_distribution_cfg(model_cfg, rsl_rl_mlp_model_cfg_cls):
    if model_cfg.distribution_cfg is not None:
        pass  # new distribution configuration is used, no need to handle deprecated configurations
    elif model_cfg.stochastic is True:  # distribution config is None but stochastic output is requested
        print(
            "[WARNING]: The `distribution_cfg` configuration is now used to specify the output distribution for"
            " stochastic policies. Consider updating the configuration to use `distribution_cfg` instead of"
            " `stochastic`, `init_noise_std`, `noise_std_type`, and `state_dependent_std` parameters."
        )
        if model_cfg.state_dependent_std is False:  # gaussian distribution
            model_cfg.distribution_cfg = rsl_rl_mlp_model_cfg_cls.GaussianDistributionCfg(
                init_std=model_cfg.init_noise_std, std_type=model_cfg.noise_std_type
            )
        elif model_cfg.state_dependent_std is True:  # heteroscedastic gaussian distribution
            model_cfg.distribution_cfg = rsl_rl_mlp_model_cfg_cls.HeteroscedasticGaussianDistributionCfg(
                init_std=model_cfg.init_noise_std, std_type=model_cfg.noise_std_type
            )
    # remove deprecated stochastic parameters
    if hasattr(model_cfg, "stochastic"):
        del model_cfg.stochastic
    if hasattr(model_cfg, "init_noise_std"):
        del model_cfg.init_noise_std
    if hasattr(model_cfg, "noise_std_type"):
        del model_cfg.noise_std_type
    if hasattr(model_cfg, "state_dependent_std"):
        del model_cfg.state_dependent_std
