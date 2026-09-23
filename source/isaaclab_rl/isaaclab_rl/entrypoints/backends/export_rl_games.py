# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games LEAPP export backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

import gymnasium as gym
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks.registry  # noqa: F401

from ...rl_games import RlGamesVecEnvWrapper, register_rl_games_env
from ..common import resolve_published_checkpoint
from .export_common import (
    add_common_export_args,
    finalize_export_args,
    get_checkpoint_path,
    is_two_tensor_lstm_state,
    leapp_capture,
    prepare_export_env,
    resolve_export_save_path,
    run_export,
    state_dict_from_sequence,
    state_sequence_from_registered,
)


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return the remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Export an RL agent with RL-Games.")
    add_common_export_args(parser, agent_default="rl_games_cfg_entry_point")
    parser.add_argument(
        "--use_last_checkpoint",
        action="store_true",
        help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
    )
    return finalize_export_args(parser, argv)


def is_rl_games_lstm_policy(agent: BasePlayer) -> bool:
    """Return whether the RL-Games player exposes supported actor-side LSTM feedback state."""
    return bool(getattr(agent, "is_rnn", False) and is_two_tensor_lstm_state(getattr(agent, "states", None)))


def get_rl_games_policy_states(agent: BasePlayer) -> Any:
    """Return RL-Games actor-side recurrent state."""
    return getattr(agent, "states", None)


def set_rl_games_policy_states(agent: BasePlayer, states: Any) -> None:
    """Assign RL-Games actor-side recurrent state."""
    agent.states = list(states)


def _validate_rl_games_recurrent_support(agent: BasePlayer) -> None:
    """Raise when the RL-Games recurrent state is present but is not supported."""
    if getattr(agent, "is_rnn", False) and not is_rl_games_lstm_policy(agent):
        raise NotImplementedError("Only RL-Games LSTM recurrent policies are supported for LEAPP export.")


def _required_obs_groups(agent_cfg: dict) -> set[str]:
    """Return Isaac Lab observation groups consumed by the RL-Games actor."""
    obs_groups = agent_cfg["params"].get("env", {}).get("obs_groups")
    if obs_groups is None:
        return {"policy"}
    return set(obs_groups.get("obs", ["policy"]))


def _resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg: dict, env_cfg: Any, log_root_path: str) -> str | None:
    """Resolve the checkpoint to export, or None when no published checkpoint exists."""
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rl_games", args_cli.task, env_cfg)
    if args_cli.checkpoint is None:
        config = agent_cfg["params"]["config"]
        run_dir = config.get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{config['name']}.pth"
        return get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    return retrieve_file_path(args_cli.checkpoint)


def export_rl_games_agent(args_cli: argparse.Namespace, env_cfg: Any, agent_cfg: dict) -> bool:
    """Export an RL-Games agent; returns whether a graph was written."""
    # concrete environment classes and the LEAPP runtime load simulation modules, so import them
    # only after launch_simulation has initialized the selected backend
    from leapp import annotate

    from isaaclab.envs import multi_agent_to_single_agent

    params = agent_cfg["params"]
    env_cfg.scene.num_envs = 1
    env_cfg.seed = params["seed"]
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", params["config"]["name"]))
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {args_cli.task} in directory: {log_root_path}")
        return False
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        policy_node_name = prepare_export_env(env, args_cli, required_obs_groups=_required_obs_groups(agent_cfg))
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            env = multi_agent_to_single_agent(env)
        env = RlGamesVecEnvWrapper.from_agent_cfg(env, agent_cfg)
        register_rl_games_env(env)

        params["load_checkpoint"] = True
        params["load_path"] = resume_path
        params["config"]["num_actors"] = env.unwrapped.num_envs
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = Runner()
        # configure_seed must run after Runner() so torch determinism does not disturb its initialization
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        save_path = resolve_export_save_path(args_cli, "rl_games", log_dir)
        with leapp_capture(args_cli, save_path=save_path, env_cfg=env_cfg) as num_steps:
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
            _validate_rl_games_recurrent_support(agent)
            recurrent = is_rl_games_lstm_policy(agent)

            for _ in range(num_steps):
                with torch.inference_mode():
                    if recurrent:
                        actor_states = get_rl_games_policy_states(agent)
                        state_dict = state_dict_from_sequence(actor_states)
                        registered_state = annotate.state_tensors(policy_node_name, state_dict)
                        set_rl_games_policy_states(
                            agent, state_sequence_from_registered(registered_state, list(state_dict), actor_states)
                        )
                    obs = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                    if recurrent:
                        annotate.update_state(
                            policy_node_name, state_dict_from_sequence(get_rl_games_policy_states(agent))
                        )
                    obs, _, _, _ = env.step(actions)
    finally:
        env.close()
    return True


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export(args_cli, hydra_args, export_rl_games_agent)


if __name__ == "__main__":
    raise SystemExit(run())
