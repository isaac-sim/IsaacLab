# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a checkpoint if an RL agent from skrl."""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import setup_preset_cli

SKRL_VERSION = "2.1.0"
_RUNTIME_IMPORTS_LOADED = False

torch = None
leapp = None
annotate = None
gym = None
skrl = None
version = None
Runner = None
DirectMARLEnvCfg = None
ManagerBasedRLEnv = None
SkrlVecEnvWrapper = None
configure_seed = None
multi_agent_to_single_agent = None
retrieve_file_path = None
patch_env_for_export = None
ensure_env_spec_id = None
get_published_pretrained_checkpoint = None
get_checkpoint_path = None
hydra_task_config = None
is_two_tensor_lstm_state = None
state_dict_from_sequence = None
state_sequence_from_registered = None


def create_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for skrl policy export."""
    parser = argparse.ArgumentParser(description="Export an RL agent with skrl.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help=(
            "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
            "--algorithm is used to determine the default agent configuration entry point."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pre-trained checkpoint from Nucleus.",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="The ML framework used for training the skrl agent.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for training the skrl agent.",
    )

    # LEAPP arguments
    parser.add_argument(
        "--export_task_name",
        type=str,
        default=None,
        help="Name of the exported graph. Defaults to the task name.",
    )
    parser.add_argument(
        "--export_method",
        type=str,
        default="onnx-dynamo",
        choices=["onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace"],
        help="Method to export the policy",
    )
    parser.add_argument(
        "--export_save_path",
        type=str,
        default=None,
        help="Path to save the exported model",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5,
        help="Number of steps to validate the exported model",
    )
    parser.add_argument(
        "--disable_graph_visualization",
        action="store_true",
        default=False,
        help="Disable LEAPP graph visualization during compile_graph().",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    parser = create_arg_parser()
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.headless = True
    return args_cli, hydra_args


def _agent_cfg_entry_point(args_cli: argparse.Namespace) -> tuple[str, str]:
    """Return the skrl config entry point and normalized algorithm name."""
    if args_cli.agent is None:
        algorithm = args_cli.algorithm.lower()
        return ("skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point", algorithm)
    return args_cli.agent, args_cli.agent.split("_cfg")[0].split("skrl_")[-1].lower()


def _load_runtime_dependencies() -> None:
    """Import runtime dependencies after Isaac Sim has been launched."""
    global _RUNTIME_IMPORTS_LOADED
    global DirectMARLEnvCfg, ManagerBasedRLEnv, Runner, SkrlVecEnvWrapper, annotate, get_checkpoint_path, gym, leapp
    global ensure_env_spec_id, get_published_pretrained_checkpoint, hydra_task_config, multi_agent_to_single_agent
    global patch_env_for_export, retrieve_file_path, skrl, torch, version
    global configure_seed, is_two_tensor_lstm_state, state_dict_from_sequence, state_sequence_from_registered

    if _RUNTIME_IMPORTS_LOADED:
        return

    try:
        import leapp as leapp_module
    except ImportError as e:
        raise ImportError("LEAPP package is required for policy export. Install with: pip install leapp") from e
    annotate_module = getattr(leapp_module, "annotate")

    import gymnasium as gym_module
    import skrl as skrl_module
    import torch as torch_module
    from packaging import version as version_module
    from skrl.utils.runner.torch import Runner as RunnerCls

    torch_module.jit._state.disable()

    from isaaclab.envs import DirectMARLEnvCfg as DirectMARLEnvCfgCls
    from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRLEnvCls
    from isaaclab.envs import multi_agent_to_single_agent as multi_agent_to_single_agent_fn
    from isaaclab.utils.assets import retrieve_file_path as retrieve_file_path_fn
    from isaaclab.utils.leapp import patch_env_for_export as patch_env_for_export_fn
    from isaaclab.utils.leapp.utils import ensure_env_spec_id as ensure_env_spec_id_fn
    from isaaclab.utils.seed import configure_seed as configure_seed_fn

    from isaaclab_rl.leapp import (
        is_two_tensor_lstm_state as is_two_tensor_lstm_state_fn,
    )
    from isaaclab_rl.leapp import (
        state_dict_from_sequence as state_dict_from_sequence_fn,
    )
    from isaaclab_rl.leapp import (
        state_sequence_from_registered as state_sequence_from_registered_fn,
    )
    from isaaclab_rl.skrl import SkrlVecEnvWrapper as SkrlVecEnvWrapperCls
    from isaaclab_rl.utils.pretrained_checkpoint import (
        get_published_pretrained_checkpoint as get_published_pretrained_checkpoint_fn,
    )

    __import__("isaaclab_tasks")
    from isaaclab_tasks.utils import get_checkpoint_path as get_checkpoint_path_fn
    from isaaclab_tasks.utils.hydra import hydra_task_config as hydra_task_config_fn

    if version_module.parse(skrl_module.__version__) < version_module.parse(SKRL_VERSION):
        skrl_module.logger.error(
            f"Unsupported skrl version: {skrl_module.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        raise RuntimeError(f"Unsupported skrl version: {skrl_module.__version__}")

    torch = torch_module
    leapp = leapp_module
    annotate = annotate_module
    gym = gym_module
    skrl = skrl_module
    version = version_module
    Runner = RunnerCls
    DirectMARLEnvCfg = DirectMARLEnvCfgCls
    ManagerBasedRLEnv = ManagerBasedRLEnvCls
    SkrlVecEnvWrapper = SkrlVecEnvWrapperCls
    configure_seed = configure_seed_fn
    multi_agent_to_single_agent = multi_agent_to_single_agent_fn
    retrieve_file_path = retrieve_file_path_fn
    patch_env_for_export = patch_env_for_export_fn
    ensure_env_spec_id = ensure_env_spec_id_fn
    get_published_pretrained_checkpoint = get_published_pretrained_checkpoint_fn
    get_checkpoint_path = get_checkpoint_path_fn
    hydra_task_config = hydra_task_config_fn
    is_two_tensor_lstm_state = is_two_tensor_lstm_state_fn
    state_dict_from_sequence = state_dict_from_sequence_fn
    state_sequence_from_registered = state_sequence_from_registered_fn
    _RUNTIME_IMPORTS_LOADED = True


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
    _load_runtime_dependencies()

    if args_cli.ml_framework != "torch":
        raise NotImplementedError("LEAPP export only supports torch for skrl policies.")

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")
    _, algorithm = _agent_cfg_entry_point(args_cli)

    env_cfg.scene.num_envs = 1
    cli_device = getattr(args_cli, "device", None)
    env_cfg.sim.device = cli_device if cli_device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", checkpoint_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return False
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

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
            patch_env_for_export(env, export_method=args_cli.export_method, required_obs_groups={"policy"})

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

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
        elif args_cli.use_pretrained_checkpoint:
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
        leapp.compile_graph(visualize=not args_cli.disable_graph_visualization, validate=validate)
    finally:
        if leapp_started:
            with contextlib.suppress(Exception):
                leapp.stop()
        if env is not None:
            env.close()

    return True


def run_export_with_hydra(args_cli: argparse.Namespace, hydra_args: list[str]) -> bool:
    """Resolve Hydra task configuration and export one skrl policy."""
    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils.hydra import hydra_task_config

    agent_cfg_entry_point, _ = _agent_cfg_entry_point(args_cli)
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


if __name__ == "__main__":
    main_cli()
