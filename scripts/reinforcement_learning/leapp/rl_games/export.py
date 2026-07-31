# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a checkpoint if an RL agent from RL-Games."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time

_RUNTIME_IMPORTS_LOADED = False

torch = None
leapp = None
annotate = None
gym = None
env_configurations = None
vecenv = None
Runner = None
BasePlayer = None
DirectMARLEnvCfg = None
ManagerBasedRLEnv = None
RlGamesGpuEnv = None
RlGamesVecEnvWrapper = None
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


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    _leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _leapp_scripts_dir not in sys.path:
        sys.path.insert(0, _leapp_scripts_dir)
    from export_utils import add_common_export_args, finalize_export_args

    parser = argparse.ArgumentParser(description="Export an RL agent with RL-Games.")
    add_common_export_args(parser, agent_default="rl_games_cfg_entry_point")
    parser.add_argument(
        "--use_last_checkpoint",
        action="store_true",
        help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
    )
    return finalize_export_args(parser, argv)


def _load_runtime_dependencies() -> None:
    """Import runtime dependencies after Isaac Sim has been launched."""
    global _RUNTIME_IMPORTS_LOADED
    global BasePlayer, DirectMARLEnvCfg, ManagerBasedRLEnv, RlGamesGpuEnv, RlGamesVecEnvWrapper, Runner
    global annotate, configure_seed, env_configurations, get_checkpoint_path, gym, leapp
    global ensure_env_spec_id, get_published_pretrained_checkpoint, hydra_task_config, multi_agent_to_single_agent
    global patch_env_for_export, retrieve_file_path, torch, vecenv
    global is_two_tensor_lstm_state, state_dict_from_sequence, state_sequence_from_registered

    if _RUNTIME_IMPORTS_LOADED:
        return

    try:
        import leapp as leapp_module
    except ImportError as e:
        raise ImportError("LEAPP package is required for policy export. Install with: pip install leapp") from e
    annotate_module = getattr(leapp_module, "annotate")

    import gymnasium as gym_module
    import torch as torch_module
    from rl_games.common import env_configurations as env_configurations_module
    from rl_games.common import vecenv as vecenv_module
    from rl_games.common.player import BasePlayer as BasePlayerCls
    from rl_games.torch_runner import Runner as RunnerCls

    from isaaclab.envs import DirectMARLEnvCfg as DirectMARLEnvCfgCls
    from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRLEnvCls
    from isaaclab.envs import multi_agent_to_single_agent as multi_agent_to_single_agent_fn
    from isaaclab.utils.assets import retrieve_file_path as retrieve_file_path_fn
    from isaaclab.utils.leapp import patch_env_for_export as patch_env_for_export_fn
    from isaaclab.utils.leapp.utils import ensure_env_spec_id as ensure_env_spec_id_fn
    from isaaclab.utils.seed import configure_seed as configure_seed_fn

    _leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _leapp_scripts_dir not in sys.path:
        sys.path.insert(0, _leapp_scripts_dir)
    from export_utils import (  # isort: skip
        is_two_tensor_lstm_state as is_two_tensor_lstm_state_fn,
        state_dict_from_sequence as state_dict_from_sequence_fn,
        state_sequence_from_registered as state_sequence_from_registered_fn,
    )

    from isaaclab_rl.rl_games import RlGamesGpuEnv as RlGamesGpuEnvCls
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper as RlGamesVecEnvWrapperCls
    from isaaclab_rl.utils.pretrained_checkpoint import (
        get_published_pretrained_checkpoint as get_published_pretrained_checkpoint_fn,
    )

    __import__("isaaclab_tasks")
    from isaaclab_tasks.utils import get_checkpoint_path as get_checkpoint_path_fn
    from isaaclab_tasks.utils.hydra import hydra_task_config as hydra_task_config_fn

    torch = torch_module
    leapp = leapp_module
    annotate = annotate_module
    gym = gym_module
    env_configurations = env_configurations_module
    vecenv = vecenv_module
    Runner = RunnerCls
    BasePlayer = BasePlayerCls
    DirectMARLEnvCfg = DirectMARLEnvCfgCls
    ManagerBasedRLEnv = ManagerBasedRLEnvCls
    RlGamesGpuEnv = RlGamesGpuEnvCls
    RlGamesVecEnvWrapper = RlGamesVecEnvWrapperCls
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


def is_rl_games_lstm_policy(agent) -> bool:
    """Return whether the RL-Games player exposes supported actor-side LSTM feedback state."""
    return bool(getattr(agent, "is_rnn", False) and is_two_tensor_lstm_state(getattr(agent, "states", None)))


def get_rl_games_policy_states(agent):
    """Return RL-Games actor-side recurrent state."""
    return getattr(agent, "states", None)


def set_rl_games_policy_states(agent, states) -> None:
    """Assign RL-Games actor-side recurrent state."""
    agent.states = list(states)


def _validate_rl_games_recurrent_support(agent) -> None:
    """Raise when the RL-Games recurrent state is present but is not supported."""
    if getattr(agent, "is_rnn", False) and not is_rl_games_lstm_policy(agent):
        raise NotImplementedError("Only RL-Games LSTM recurrent policies are supported for LEAPP export.")


def _required_obs_groups(agent_cfg) -> set[str]:
    """Return Isaac Lab observation groups consumed by the RL-Games actor."""
    obs_groups = agent_cfg["params"].get("env", {}).get("obs_groups")
    if obs_groups is None:
        return {"policy"}
    return set(obs_groups.get("obs", ["policy"]))


def export_rl_games_agent(
    args_cli: argparse.Namespace,
    env_cfg,
    agent_cfg,
    simulation_app=None,
) -> bool:
    """Export an RL-Games agent."""
    _load_runtime_dependencies()

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = 1
    cli_device = getattr(args_cli, "device", None)
    env_cfg.sim.device = cli_device if cli_device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", checkpoint_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return False
    elif args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {checkpoint_task_name} in directory: {log_root_path}")
        return False

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False

    try:
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        policy_node_name = ensure_env_spec_id(env)
        graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name

        if isinstance(env.unwrapped, ManagerBasedRLEnv):
            patch_env_for_export(
                env,
                export_method=args_cli.export_method,
                required_obs_groups=_required_obs_groups(agent_cfg),
            )

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            env = multi_agent_to_single_agent(env)

        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
        runner = Runner()
        if getattr(args_cli, "deterministic", False):
            configure_seed(env_cfg.seed, True)
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        if args_cli.export_save_path is not None:
            save_path = args_cli.export_save_path
        elif args_cli.use_pretrained_checkpoint:
            save_path = os.path.join(".pretrained_checkpoints", "rl_games", checkpoint_task_name)
        else:
            save_path = log_dir
        leapp.start(graph_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
        _validate_rl_games_recurrent_support(agent)

        if simulation_app is not None:
            while not simulation_app.is_running():
                time.sleep(0.5)

        for _ in range(max(args_cli.validation_steps, 2)):
            with torch.inference_mode():
                if is_rl_games_lstm_policy(agent):
                    actor_states = get_rl_games_policy_states(agent)
                    state_names = list(state_dict_from_sequence(actor_states).keys())
                    registered_state = annotate.state_tensors(policy_node_name, state_dict_from_sequence(actor_states))
                    set_rl_games_policy_states(
                        agent,
                        state_sequence_from_registered(registered_state, state_names, actor_states),
                    )

                obs = agent.obs_to_torch(obs)
                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)

                if is_rl_games_lstm_policy(agent):
                    actor_states_after = get_rl_games_policy_states(agent)
                    annotate.update_state(policy_node_name, state_dict_from_sequence(actor_states_after))

                obs, _, _, _ = env.step(actions)

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
    """Resolve Hydra task configuration and export one RL-Games policy."""
    _leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _leapp_scripts_dir not in sys.path:
        sys.path.insert(0, _leapp_scripts_dir)
    from export_utils import disable_torchscript_for_export

    # Must run before the imports below pull in the task modules.
    disable_torchscript_for_export()

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils.hydra import hydra_task_config

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_args
    exported = False

    try:

        @hydra_task_config(args_cli.task, args_cli.agent)
        def _main(env_cfg, agent_cfg) -> None:
            nonlocal exported
            with launch_simulation(env_cfg, args_cli):
                exported = export_rl_games_agent(args_cli, env_cfg, agent_cfg)

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
