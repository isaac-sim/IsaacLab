# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from dataclasses import MISSING
from packaging import version
from typing import TYPE_CHECKING

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


def handle_deprecated_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, installed_version) -> RslRlBaseRunnerCfg:
    """Handle deprecated configurations for RSL-RL."""

    # Handle configurations for rsl-rl < 4.0.0
    if version.parse(installed_version) < version.parse("4.0.0"):
        # exit if no policy configuration is present
        if not hasattr(agent_cfg, "policy") or isinstance(agent_cfg.policy, type(MISSING)):
            raise ValueError(
                "The `policy` configuration is required for rsl-rl < 4.0.0. Please specify the `policy` configuration"
                " or update rsl-rl."
            )

        # handle deprecated obs_normalization argument
        if hasattr(agent_cfg, "empirical_normalization") and not isinstance(
            agent_cfg.empirical_normalization, type(MISSING)
        ):
            print(
                "[WARNING]: The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization`"
                " and `critic_obs_normalization` as part of the `policy` configuration instead."
            )
            if isinstance(agent_cfg.policy.actor_obs_normalization, type(MISSING)):
                agent_cfg.policy.actor_obs_normalization = agent_cfg.empirical_normalization
            if isinstance(agent_cfg.policy.critic_obs_normalization, type(MISSING)):
                agent_cfg.policy.critic_obs_normalization = agent_cfg.empirical_normalization
            agent_cfg.empirical_normalization = MISSING

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
        if hasattr(agent_cfg, "actor") and not isinstance(agent_cfg.actor, type(MISSING)):
            print(
                "[WARNING]: The `actor` model configuration is only used for rsl-rl >= 4.0.0. Consider updating rsl-rl"
                " or use the `policy` configuration for rsl-rl < 4.0.0."
            )
            agent_cfg.actor = MISSING
        if hasattr(agent_cfg, "critic") and not isinstance(agent_cfg.critic, type(MISSING)):
            print(
                "[WARNING]: The `critic` model configuration is only used for rsl-rl >= 4.0.0. Consider updating rsl-rl"
                " or use the `policy` configuration for rsl-rl < 4.0.0."
            )
            agent_cfg.critic = MISSING
        if hasattr(agent_cfg, "student") and not isinstance(agent_cfg.student, type(MISSING)):
            print(
                "[WARNING]: The `student` model configuration is only used for rsl-rl >= 4.0.0. Consider updating"
                " rsl-rl or use the `policy` configuration for rsl-rl < 4.0.0."
            )
            agent_cfg.student = MISSING
        if hasattr(agent_cfg, "teacher") and not isinstance(agent_cfg.teacher, type(MISSING)):
            print(
                "[WARNING]: The `teacher` model configuration is only used for rsl-rl >= 4.0.0. Consider updating"
                " rsl-rl or use the `policy` configuration for rsl-rl < 4.0.0."
            )
            agent_cfg.teacher = MISSING

    # Handle deprecated configurations for rsl-rl >= 4.0.0
    else:
        if hasattr(agent_cfg, "policy") and not isinstance(agent_cfg.policy, type(MISSING)):
            print(
                "[WARNING]: The `policy` configuration is deprecated for rsl-rl >= 4.0.0. Please use, e.g., `actor` and"
                " `critic` model configurations instead."
            )

            # handle deprecated obs_normalization argument
            if hasattr(agent_cfg, "empirical_normalization") and not isinstance(
                agent_cfg.empirical_normalization, type(MISSING)
            ):
                print(
                    "[WARNING]: The `empirical_normalization` parameter is deprecated. Please set"
                    " `actor_obs_normalization` and `critic_obs_normalization` as part of the `policy` configuration"
                    " instead."
                )
                if isinstance(agent_cfg.policy.actor_obs_normalization, type(MISSING)):
                    agent_cfg.policy.actor_obs_normalization = agent_cfg.empirical_normalization
                if isinstance(agent_cfg.policy.critic_obs_normalization, type(MISSING)):
                    agent_cfg.policy.critic_obs_normalization = agent_cfg.empirical_normalization
                agent_cfg.empirical_normalization = MISSING

            # import relevant config classes
            from isaaclab_rl.rsl_rl import (
                RslRlDistillationStudentTeacherCfg,
                RslRlDistillationStudentTeacherRecurrentCfg,
                RslRlMLPModelCfg,
                RslRlPpoActorCriticCfg,
                RslRlPpoActorCriticRecurrentCfg,
                RslRlRNNModelCfg,
            )

            # set actor model configuration if missing
            if hasattr(agent_cfg, "actor") and isinstance(agent_cfg.actor, type(MISSING)):
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
            if hasattr(agent_cfg, "critic") and isinstance(agent_cfg.critic, type(MISSING)):
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
            if hasattr(agent_cfg, "student") and isinstance(agent_cfg.student, type(MISSING)):
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
            if hasattr(agent_cfg, "teacher") and isinstance(agent_cfg.teacher, type(MISSING)):
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

    return agent_cfg
