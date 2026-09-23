# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl LEAPP export backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks.registry  # noqa: F401

from ...skrl import SkrlVecEnvWrapper, check_skrl_version, import_skrl_runner, resolve_skrl_algorithm
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
    parser = argparse.ArgumentParser(description="Export an RL agent with skrl.")
    add_common_export_args(parser, agent_default="skrl_cfg_entry_point")
    return finalize_export_args(parser, argv)


def is_skrl_lstm_policy(agent: Any) -> bool:
    """Return whether the skrl agent exposes supported actor-side LSTM feedback state."""
    states = get_skrl_policy_states(agent)
    spec_sizes = agent.policy.get_specification().get("rnn", {}).get("sizes", [])
    return bool(getattr(agent, "_rnn", False) and is_two_tensor_lstm_state(states) and len(spec_sizes) == 2)


def get_skrl_policy_states(agent: Any) -> Any:
    """Return skrl actor-side recurrent state."""
    return getattr(agent, "_rnn_initial_states", {}).get("policy", [])


def set_skrl_policy_states(agent: Any, states: Any) -> None:
    """Assign skrl actor-side recurrent state."""
    agent._rnn_initial_states["policy"] = list(states)


def get_skrl_policy_output_states(agent: Any, outputs_dict: dict) -> Any:
    """Return updated skrl actor-side recurrent state after an action call."""
    output_states = outputs_dict.get("rnn")
    if output_states is not None:
        return output_states
    return getattr(agent, "_rnn_final_states", {}).get("policy", [])


def _validate_skrl_recurrent_support(agent: Any) -> None:
    """Raise when the skrl recurrent state is present but is not supported."""
    if getattr(agent, "_rnn", False) and not is_skrl_lstm_policy(agent):
        raise NotImplementedError("Only skrl LSTM recurrent policies are supported for LEAPP export.")


def _resolve_checkpoint(args_cli: argparse.Namespace, env_cfg: Any, log_root_path: str, algorithm: str) -> str | None:
    """Resolve the checkpoint to export, or None when no published checkpoint exists."""
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("skrl", args_cli.task, env_cfg)
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_torch", other_dirs=["checkpoints"])


def export_skrl_agent(args_cli: argparse.Namespace, env_cfg: Any, agent_cfg: dict) -> bool:
    """Export a skrl agent; returns whether a graph was written."""
    # concrete environment classes and the LEAPP runtime load simulation modules, so import them
    # only after launch_simulation has initialized the selected backend
    from leapp import annotate

    from isaaclab.envs import multi_agent_to_single_agent

    check_skrl_version()
    runner_cls = import_skrl_runner("torch")
    algorithm = resolve_skrl_algorithm(agent_cfg)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    experiment_cfg = agent_cfg["agent"]["experiment"]
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["directory"]))
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    resume_path = _resolve_checkpoint(args_cli, env_cfg, log_root_path, algorithm)
    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {args_cli.task} in directory: {log_root_path}")
        return False
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        policy_node_name = prepare_export_env(env, args_cli, required_obs_groups={"policy"})
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm == "ppo":
            env = multi_agent_to_single_agent(env)
        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        agent_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["write_interval"] = 0
        experiment_cfg["checkpoint_interval"] = 0
        runner = runner_cls(env, agent_cfg)
        # configure_seed must run after Runner() so torch determinism does not disturb its initialization
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent = runner.agent
        agent.load(resume_path)
        agent.enable_training_mode(False, apply_to_models=True)
        _validate_skrl_recurrent_support(agent)
        recurrent = is_skrl_lstm_policy(agent)

        save_path = resolve_export_save_path(args_cli, "skrl", log_dir)
        with leapp_capture(args_cli, save_path=save_path, env_cfg=env_cfg) as num_steps:
            obs, _ = env.reset()
            states = env.state()
            for _ in range(num_steps):
                with torch.inference_mode():
                    if recurrent:
                        actor_states = get_skrl_policy_states(agent)
                        state_dict = state_dict_from_sequence(actor_states)
                        registered_state = annotate.state_tensors(policy_node_name, state_dict)
                        set_skrl_policy_states(
                            agent, state_sequence_from_registered(registered_state, list(state_dict), actor_states)
                        )
                    outputs = agent.act(obs, states, timestep=0, timesteps=0)
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    if recurrent:
                        actor_states_after = get_skrl_policy_output_states(agent, outputs[-1])
                        annotate.update_state(policy_node_name, state_dict_from_sequence(actor_states_after))
                        set_skrl_policy_states(agent, actor_states_after)
                    obs, _, _, _, _ = env.step(actions)
                    states = env.state()
    finally:
        env.close()
    return True


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export(args_cli, hydra_args, export_skrl_agent)


if __name__ == "__main__":
    raise SystemExit(run())
