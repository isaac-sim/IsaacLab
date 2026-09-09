# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a checkpoint if an RL agent from skrl."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so disable TorchScript before
# importing task or environment modules that compile decorated helpers.
torch.jit._state.disable()

import gymnasium as gym
import leapp
from leapp import annotate
from packaging import version
import skrl
from skrl.utils.runner.torch import Runner

from isaaclab.app import launch_simulation
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.leapp import patch_env_for_export
from isaaclab.utils.leapp.utils import ensure_env_spec_id
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.entrypoints.backends.export_common import (
    add_common_export_args,
    create_graph_configs,
    finalize_export_args,
    get_checkpoint_path,
    is_two_tensor_lstm_state,
    state_dict_from_sequence,
    state_sequence_from_registered,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.utils.hydra import hydra_task_config

SKRL_VERSION = "2.1.0"


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Export an RL agent with skrl.")
    add_common_export_args(parser, agent_default="skrl_cfg_entry_point")
    return finalize_export_args(parser, argv)


def _algorithm_from_agent_entry_point(agent_cfg_entry_point: str) -> str:
    """Derive the skrl algorithm tag used in training run directory names.

    Isaac Lab stores PPO under ``skrl_cfg_entry_point`` and other algorithms under
    ``skrl_<algorithm>_cfg_entry_point``. Training run directories are named with the
    algorithm tag (e.g. ``*_ppo_torch``), so export needs that tag when auto-finding
    checkpoints.
    """
    prefix = agent_cfg_entry_point.split("_cfg")[0]
    if prefix == "skrl":
        return "ppo"
    if prefix.startswith("skrl_"):
        return prefix[len("skrl_") :].lower()
    return prefix.lower()


def is_skrl_lstm_policy(agent) -> bool:
    """Return whether the skrl agent exposes supported actor-side LSTM feedback state."""
    states = getattr(agent, "_rnn_initial_states", {}).get("policy", [])
    spec_sizes = agent.policy.get_specification().get("rnn", {}).get("sizes", [])
    return bool(getattr(agent, "_rnn", False) and is_two_tensor_lstm_state(states) and len(spec_sizes) == 2)


def get_skrl_policy_states(agent):
    """Return skrl actor-side recurrent state."""
    return getattr(agent, "_rnn_initial_states", {}).get("policy", [])


def set_skrl_policy_states(agent, states) -> None:
    """Assign skrl actor-side recurrent state."""
    agent._rnn_initial_states["policy"] = list(states)


def get_skrl_policy_output_states(agent, outputs_dict):
    """Return updated skrl actor-side recurrent state after an action call."""
    output_states = outputs_dict.get("rnn", None)
    if output_states is not None:
        return output_states
    return getattr(agent, "_rnn_final_states", {}).get("policy", [])


def _validate_skrl_recurrent_support(agent) -> None:
    """Raise when the skrl recurrent state is present but is not supported."""
    if getattr(agent, "_rnn", False) and not is_skrl_lstm_policy(agent):
        raise NotImplementedError("Only skrl LSTM recurrent policies are supported for LEAPP export.")


def export_skrl_agent(
    args_cli: argparse.Namespace,
    env_cfg,
    experiment_cfg,
    simulation_app=None,
) -> bool:
    """Export a skrl agent."""
    # Concrete environment classes load simulation modules, so import them
    # only after launch_simulation has initialized the selected backend.
    from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnv, multi_agent_to_single_agent

    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        skrl.logger.error(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")
    algorithm = _algorithm_from_agent_entry_point(args_cli.agent)

    env_cfg.scene.num_envs = 1
    cli_device = getattr(args_cli, "device", None)
    env_cfg.sim.device = cli_device if cli_device is not None else env_cfg.sim.device
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.checkpoint == "pretrained":
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        resume_path = get_published_pretrained_checkpoint("skrl", checkpoint_task_name, *backend_names)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return False
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_torch", other_dirs=["checkpoints"])

    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {checkpoint_task_name} in directory: {log_root_path}")
        return False

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False

    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        policy_node_name = ensure_env_spec_id(env)
        graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name

        if isinstance(env.unwrapped, ManagerBasedRLEnv):
            export_method = "onnx-dynamo" if args_cli.export_method is None else args_cli.export_method
            patch_env_for_export(env, export_method=export_method, required_obs_groups={"policy"})
        elif args_cli.export_method is not None:
            raise ValueError(
                "--export_method is only supported for manager-based environments. For direct environments, "
                "set export_with directly in the annotate.output_tensors() call instead."
            )

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        if getattr(args_cli, "deterministic", False):
            configure_seed(env_cfg.seed, True)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)
        _validate_skrl_recurrent_support(runner.agent)

        if args_cli.export_save_path is not None:
            save_path = args_cli.export_save_path
        elif args_cli.checkpoint == "pretrained":
            save_path = os.path.join(".pretrained_checkpoints", "skrl", checkpoint_task_name)
        else:
            save_path = log_dir
        leapp.start(graph_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True

        obs, _ = env.reset()
        states = env.state()
        if simulation_app is not None:
            while not simulation_app.is_running():
                time.sleep(0.5)

        for _ in range(max(args_cli.validation_steps, 2)):
            with torch.inference_mode():
                if is_skrl_lstm_policy(runner.agent):
                    actor_states = get_skrl_policy_states(runner.agent)
                    state_names = list(state_dict_from_sequence(actor_states).keys())
                    registered_state = annotate.state_tensors(policy_node_name, state_dict_from_sequence(actor_states))
                    set_skrl_policy_states(
                        runner.agent,
                        state_sequence_from_registered(registered_state, state_names, actor_states),
                    )

                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                outputs_dict = outputs[-1]
                actions = outputs_dict.get("mean_actions", outputs[0])

                if is_skrl_lstm_policy(runner.agent):
                    actor_states_after = get_skrl_policy_output_states(runner.agent, outputs_dict)
                    annotate.update_state(policy_node_name, state_dict_from_sequence(actor_states_after))
                    set_skrl_policy_states(runner.agent, actor_states_after)

                obs, _, _, _, _ = env.step(actions)
                states = env.state()

        leapp.stop()
        leapp_started = False
        validate = args_cli.validation_steps > 0
        leapp.compile_graph(
            visualize=not args_cli.disable_graph_visualization,
            validate=validate,
            graph_configs=create_graph_configs(env_cfg),
        )
    finally:
        if leapp_started:
            with contextlib.suppress(Exception):
                leapp.stop()
        if env is not None:
            env.close()

    return True


def run_export_with_hydra(args_cli: argparse.Namespace, hydra_args: list[str]) -> bool:
    """Resolve Hydra task configuration and export one skrl policy."""

    agent_cfg_entry_point = args_cli.agent
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_args
    exported = False

    try:

        @hydra_task_config(args_cli.task, agent_cfg_entry_point)
        def _main(env_cfg, experiment_cfg) -> None:
            nonlocal exported
            with launch_simulation(env_cfg, args_cli):
                exported = export_skrl_agent(args_cli, env_cfg, experiment_cfg)

        _main()
    finally:
        sys.argv = original_argv

    return exported


def main_cli(argv: list[str] | None = None) -> bool:
    """Run the command-line export flow."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export_with_hydra(args_cli, hydra_args)


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    return 0 if main_cli(argv) else 1


if __name__ == "__main__":
    raise SystemExit(run())
