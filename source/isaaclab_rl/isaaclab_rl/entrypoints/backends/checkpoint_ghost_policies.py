# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint Ghosts playback for each training library.

Each ``run_<library>`` builds the library's policy factory, which loads a checkpoint into its own policy so that every
slot keeps its own, and hands it to :func:`isaaclab.visualizers.checkpoint_ghost_play.run`. A play script calls it in
place of its normal playback loop when ``--checkpoint_ghosts`` is set. The library imports are local so that a
library only has to be installed to play with it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from isaaclab.visualizers.checkpoint_ghost_play import PolicyAdapter, run


def run_rsl_rl(
    env: Any,
    args_cli: argparse.Namespace,
    agent_cfg: Any,
    reference_path: str,
    log_root_path: str,
    run_playback: Callable[..., None],
) -> None:
    """Play RSL-RL checkpoints, each loaded into its own runner."""
    from tensordict import TensorDict
    from torch.utils._pytree import tree_leaves

    from .runner_rsl_rl import create_rsl_rl_runner

    def policy_factory(path: str) -> PolicyAdapter:
        runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
        runner.load(path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        return PolicyAdapter(
            lambda obs: policy(TensorDict(obs, batch_size=[tree_leaves(obs)[0].shape[0]])),
            policy.reset,
            agent_cfg.clip_actions,
        )

    run(
        env,
        args_cli,
        policy_factory=policy_factory,
        checkpoint_pattern=r"model_.*\.pt",
        reference_path=reference_path,
        log_root_path=log_root_path,
        run_playback=run_playback,
    )


def run_rl_games(
    env: Any,
    args_cli: argparse.Namespace,
    runner: Any,
    params: dict,
    reference_path: str,
    log_root_path: str,
    run_playback: Callable[..., None],
) -> None:
    """Play rl_games checkpoints, each loaded into its own player with its own recurrent state."""
    from isaaclab.utils import index_fill_

    def policy_factory(path: str) -> PolicyAdapter:
        player = runner.create_player()
        player.restore(path)
        player.reset()

        def act(obs):
            obs = player.obs_to_torch(env._process_obs(obs)["obs"])
            player.get_batch_size(obs, 1)
            if player.is_rnn and player.states is None:
                player.init_rnn()
            return player.get_action(obs, is_deterministic=player.is_deterministic)

        def reset(dones):
            if dones is None:  # the slot was resized: its recurrent states are rebuilt on the next act
                player.states = None
            elif player.states is not None:
                for state in player.states:
                    index_fill_(state, dones, 0.0, dim=1)

        return PolicyAdapter(act, reset, params.get("env", {}).get("clip_actions"))

    run(
        env,
        args_cli,
        policy_factory=policy_factory,
        checkpoint_pattern=r".*\.pth",
        reference_path=reference_path,
        log_root_path=log_root_path,
        run_playback=run_playback,
    )


def run_skrl(
    env: Any,
    args_cli: argparse.Namespace,
    runner_cls: type,
    agent_cfg: dict,
    reference_path: str,
    log_root_path: str,
    run_playback: Callable[..., None],
) -> None:
    """Play skrl checkpoints, each loaded into its own agent."""

    def policy_factory(path: str) -> PolicyAdapter:
        agent = runner_cls(env, agent_cfg).agent  # one agent per checkpoint, since they run side by side
        agent.load(path)
        agent.enable_training_mode(False, apply_to_models=True)

        def act(obs):
            outputs = agent.act(obs["policy"], obs.get("critic"), timestep=0, timesteps=0)
            return outputs[-1].get("mean_actions", outputs[0])

        return PolicyAdapter(act)

    run(
        env,
        args_cli,
        policy_factory=policy_factory,
        checkpoint_pattern=r".*\.pt",
        reference_path=reference_path,
        log_root_path=log_root_path,
        run_playback=run_playback,
    )


def run_sb3(
    env: Any,
    args_cli: argparse.Namespace,
    agent_cfg: dict,
    reference_path: str,
    log_root_path: str,
    run_playback: Callable[..., None],
) -> None:
    """Play SB3 checkpoints, each with the normalization statistics saved beside it."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    def policy_factory(path: str) -> PolicyAdapter:
        model = PPO.load(path, device=agent_cfg["device"])
        name = Path(path).name.replace("model", "model_vecnormalize", 1).replace(".zip", ".pkl")
        stats = Path(path).with_name(name)
        norm = VecNormalize.load(stats, env) if stats.exists() else None
        if norm is None and agent_cfg.get("normalize_input"):
            # plain playback would normalize with live statistics instead, which no checkpoint was trained with
            raise FileNotFoundError(f"{stats.name} is missing, and this task normalizes observations")

        def act(obs):
            obs = env._process_obs(obs)
            actions, _ = model.predict(norm.normalize_obs(obs) if norm else obs, deterministic=True)
            return torch.as_tensor(actions, device=env.unwrapped.device)

        return PolicyAdapter(act)

    run(
        env,
        args_cli,
        policy_factory=policy_factory,
        checkpoint_pattern=r"model(?:_.*)?\.zip",
        reference_path=reference_path,
        log_root_path=log_root_path,
        run_playback=run_playback,
    )
