# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool

from common import (
    add_common_train_args,
    add_isaaclab_launcher_args,
    apply_env_overrides,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    set_hydra_args,
    validate_distributed_device,
    wrap_record_video,
)

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RL-Games training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
    add_common_train_args(
        parser,
        agent_default="rl_games_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
    parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    from isaaclab_tasks.utils import fold_preset_tokens, setup_preset_cli

    add_isaaclab_launcher_args(parser)
    # setup_preset_cli registers preset-selection help text + runs parse_known_args;
    # fold_preset_tokens rewrites typed selectors (physics=, renderer=, presets=) post-argparse.
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(fold_preset_tokens(hydra_args))
    return args_cli


def run(argv: list[str]) -> None:
    """Train an RL-Games agent."""
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab.app import launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

    from isaaclab_tasks.utils import resolve_task_config

    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        apply_env_overrides(args_cli, env_cfg)
        validate_distributed_device(args_cli)

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
        agent_cfg["params"]["config"]["max_epochs"] = (
            args_cli.max_iterations
            if args_cli.max_iterations is not None
            else agent_cfg["params"]["config"]["max_epochs"]
        )
        if args_cli.checkpoint is not None:
            resume_path = retrieve_file_path(args_cli.checkpoint)
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = resume_path
            print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
        train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

        if args_cli.distributed:
            agent_cfg["params"]["seed"] += int(os.getenv("RANK", "0"))
            agent_cfg["params"]["config"]["device"] = env_cfg.sim.device
            agent_cfg["params"]["config"]["device_name"] = env_cfg.sim.device
            agent_cfg["params"]["config"]["multi_gpu"] = True

        env_cfg.seed = agent_cfg["params"]["seed"]

        config_name = agent_cfg["params"]["config"]["name"]
        log_root_path = os.path.join("logs", "rl_games", config_name)
        if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
            log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
        else:
            log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = agent_cfg["params"]["config"].get(
            "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        agent_cfg["params"]["config"]["train_dir"] = log_root_path
        agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
        wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
        experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

        run_log_dir = os.path.join(log_root_path, log_dir)
        dump_train_configs(run_log_dir, env_cfg, agent_cfg)
        print(f"Exact experiment name requested from command line: {run_log_dir}")

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        configure_io_descriptors(env_cfg, args_cli, logger)
        env_cfg.log_dir = run_log_dir

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )
        env = wrap_record_video(env, run_log_dir, args_cli)

        start_time = time.time()
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
            observers = MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
            runner = Runner(observers)
        else:
            runner = Runner(IsaacAlgoObserver())

        runner.load(agent_cfg)
        runner.reset()

        global_rank = int(os.getenv("RANK", "0"))
        if args_cli.track and global_rank == 0:
            if args_cli.wandb_entity is None:
                raise ValueError("Weights and Biases entity must be specified for tracking.")
            import wandb

            wandb.init(
                project=wandb_project,
                entity=args_cli.wandb_entity,
                name=experiment_name,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
            if not wandb.run.resumed:
                wandb.config.update({"env_cfg": env_cfg.to_dict()})
                wandb.config.update({"agent_cfg": agent_cfg})

        try:
            if args_cli.checkpoint is not None:
                runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
            else:
                runner.run({"train": True, "play": False, "sigma": train_sigma})
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            env.close()
        except KeyboardInterrupt:
            pass
