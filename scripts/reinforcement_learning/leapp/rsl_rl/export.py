# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import random
import sys
import time
from collections.abc import Mapping

RSL_RL_MIN_VERSION = "5.0.1"
_RUNTIME_IMPORTS_LOADED = False

# Keep heavy/runtime-sensitive imports out of module import time. The CLI needs
# to parse launcher arguments and start Isaac Sim/Kit before importing torch,
# LEAPP, RSL-RL, and task modules; importing them earlier has caused launcher
# import-order failures. ``_load_runtime_dependencies()`` populates these
# globals immediately before export execution.
torch = None
leapp = None
annotate = None
gym = None
DistillationRunner = None
OnPolicyRunner = None
ManagerBasedRLEnv = None
RslRlVecEnvWrapper = None
handle_deprecated_rsl_rl_cfg = None
retrieve_file_path = None
patch_env_for_export = None
ensure_env_spec_id = None
get_published_pretrained_checkpoint = None
get_checkpoint_path = None
hydra_task_config = None
installed_version = None


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    _leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _leapp_scripts_dir not in sys.path:
        sys.path.insert(0, _leapp_scripts_dir)
    from export_utils import add_common_export_args, finalize_export_args

    parser = argparse.ArgumentParser(description="Export an RL agent with RSL-RL.")
    add_common_export_args(parser, agent_default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder used to locate checkpoints."
    )
    parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")
    # setup_preset_cli attaches the preset-selection help group then parses;
    # remainder still carries typed selectors (physics=/renderer=/presets=)
    # verbatim for run_export_with_hydra to fold before invoking Hydra.
    return finalize_export_args(parser, argv)


def _load_runtime_dependencies() -> None:
    """Import runtime dependencies after Isaac Sim has been launched."""
    global _RUNTIME_IMPORTS_LOADED
    global annotate, leapp, torch
    global DistillationRunner, ManagerBasedRLEnv, OnPolicyRunner, RslRlVecEnvWrapper, get_checkpoint_path, gym
    global ensure_env_spec_id, get_published_pretrained_checkpoint, handle_deprecated_rsl_rl_cfg, hydra_task_config
    global installed_version
    global patch_env_for_export, retrieve_file_path

    if _RUNTIME_IMPORTS_LOADED:
        return

    try:
        import leapp as leapp_module
    except ImportError as e:
        raise ImportError("LEAPP package is required for policy export. Install with: pip install leapp") from e
    annotate_module = getattr(leapp_module, "annotate")

    import gymnasium as gym_module
    import torch as torch_module
    from packaging import version as packaging_version_module
    from rsl_rl.runners import DistillationRunner as DistillationRunnerCls
    from rsl_rl.runners import OnPolicyRunner as OnPolicyRunnerCls

    from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRLEnvCls
    from isaaclab.utils.assets import retrieve_file_path as retrieve_file_path_fn
    from isaaclab.utils.leapp import patch_env_for_export as patch_env_for_export_fn
    from isaaclab.utils.leapp.utils import ensure_env_spec_id as ensure_env_spec_id_fn

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as RslRlVecEnvWrapperCls
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg as handle_deprecated_rsl_rl_cfg_fn
    from isaaclab_rl.utils.pretrained_checkpoint import (
        get_published_pretrained_checkpoint as get_published_pretrained_checkpoint_fn,
    )

    __import__("isaaclab_tasks")
    from isaaclab_tasks.utils import get_checkpoint_path as get_checkpoint_path_fn
    from isaaclab_tasks.utils.hydra import hydra_task_config as hydra_task_config_fn

    installed_version = metadata.version("rsl-rl-lib")
    if packaging_version_module.parse(installed_version) < packaging_version_module.parse(RSL_RL_MIN_VERSION):
        print(
            f"[WARNING] LEAPP RSL-RL export is validated with rsl-rl-lib {RSL_RL_MIN_VERSION} or newer. "
            f"Installed version is '{installed_version}'."
        )

    torch = torch_module
    leapp = leapp_module
    annotate = annotate_module
    gym = gym_module
    DistillationRunner = DistillationRunnerCls
    OnPolicyRunner = OnPolicyRunnerCls
    ManagerBasedRLEnv = ManagerBasedRLEnvCls
    RslRlVecEnvWrapper = RslRlVecEnvWrapperCls
    handle_deprecated_rsl_rl_cfg = handle_deprecated_rsl_rl_cfg_fn
    retrieve_file_path = retrieve_file_path_fn
    patch_env_for_export = patch_env_for_export_fn
    ensure_env_spec_id = ensure_env_spec_id_fn
    get_published_pretrained_checkpoint = get_published_pretrained_checkpoint_fn
    get_checkpoint_path = get_checkpoint_path_fn
    hydra_task_config = hydra_task_config_fn
    _RUNTIME_IMPORTS_LOADED = True


def get_actor_memory_module(policy):
    """Return the actor-side RNN module for supported RSL-RL recurrent policies."""
    if hasattr(policy, "rnn"):
        return policy.rnn
    return None


def is_actor_recurrent_policy(policy) -> bool:
    """Return whether the actor policy has a supported recurrent state container."""
    return bool(getattr(policy, "is_recurrent", False) and get_actor_memory_module(policy) is not None)


def get_actor_hidden_state(policy):
    """Return the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    if hasattr(policy, "get_hidden_state"):
        return policy.get_hidden_state()
    memory = get_actor_memory_module(policy)
    return None if memory is None else getattr(memory, "hidden_state", None)


def set_actor_hidden_state(policy, actor_hidden) -> None:
    """Assign the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    memory = get_actor_memory_module(policy)
    if memory is not None:
        memory.hidden_state = actor_hidden


def ensure_actor_hidden_state_initialized(policy, batch_size: int, device, dtype):
    """Initialize and return the actor hidden state when a recurrent policy has not created it yet."""
    # ``torch`` is a lazy runtime global populated by ``_load_runtime_dependencies()``
    # after Isaac Sim launches and before export calls this helper.
    assert torch is not None
    actor_state = get_actor_hidden_state(policy)
    if actor_state is not None:
        return actor_state

    memory = get_actor_memory_module(policy)
    if memory is None or not hasattr(memory, "rnn"):
        return None

    num_layers = memory.rnn.num_layers
    hidden_size = memory.rnn.hidden_size
    zeros = torch.zeros(num_layers, batch_size, hidden_size, device=device, dtype=dtype)
    if isinstance(memory.rnn, torch.nn.LSTM):
        actor_state = (zeros.clone(), zeros.clone())
    else:
        actor_state = zeros
    set_actor_hidden_state(policy, actor_state)
    return actor_state


def state_dict_from_actor_hidden(actor_hidden):
    """Convert the actor hidden state into the named tensor mapping expected by LEAPP state APIs."""
    if actor_hidden is None:
        return {}
    if isinstance(actor_hidden, tuple):
        return {f"actor_state_{idx}": tensor for idx, tensor in enumerate(actor_hidden)}
    return {"actor_state": actor_hidden}


def actor_hidden_from_registered(registered_state, original_hidden):
    """Restore the registered LEAPP state to the hidden-state structure expected by the actor memory module."""
    if isinstance(original_hidden, tuple):
        if isinstance(registered_state, tuple):
            return registered_state
        return (registered_state,)
    return registered_state


def _update_agent_cfg_from_export_args(agent_cfg, args_cli: argparse.Namespace):
    """Apply export-relevant CLI overrides to the RSL-RL agent config."""
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    return agent_cfg


def export_rsl_rl_agent(
    args_cli: argparse.Namespace,
    env_cfg,
    agent_cfg,
    simulation_app=None,
) -> bool:
    """Export a RSL-RL agent."""
    _load_runtime_dependencies()

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")

    agent_cfg = _update_agent_cfg_from_export_args(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", checkpoint_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return False
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {checkpoint_task_name} in directory: {log_root_path}")
        return False

    log_dir = os.path.dirname(resume_path)

    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False

    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        policy_node_name = ensure_env_spec_id(env)

        graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name

        if isinstance(env.unwrapped, ManagerBasedRLEnv):
            # Patch only the observation groups consumed by the actor policy.
            # This filters out the critic and teacher observation groups.
            obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
            if isinstance(obs_groups_cfg, Mapping):
                required_obs_groups = set(obs_groups_cfg.get("actor", ["policy"]))
            else:
                required_obs_groups = {"policy"}
            patch_env_for_export(
                env,
                export_method=args_cli.export_method,
                required_obs_groups=required_obs_groups,
            )

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if args_cli.export_save_path is not None:
            save_path = args_cli.export_save_path
        elif args_cli.use_pretrained_checkpoint:
            # Use a predictable path independent of the Nucleus mirror directory structure.
            save_path = os.path.join(".pretrained_checkpoints", "rsl_rl", checkpoint_task_name)
        else:
            save_path = log_dir
        leapp.start(graph_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True
        obs = env.reset()[0]
        if simulation_app is not None:
            while not simulation_app.is_running():
                time.sleep(0.5)

        for _ in range(max(args_cli.validation_steps, 2)):
            with torch.inference_mode():
                if is_actor_recurrent_policy(policy):
                    actor_hidden = ensure_actor_hidden_state_initialized(
                        policy,
                        batch_size=env.num_envs,
                        device=env.unwrapped.device,
                        dtype=next(policy.parameters()).dtype,
                    )
                    registered_state = annotate.state_tensors(
                        policy_node_name,
                        state_dict_from_actor_hidden(actor_hidden),
                    )
                    set_actor_hidden_state(policy, actor_hidden_from_registered(registered_state, actor_hidden))

                actions = policy(obs)

                if is_actor_recurrent_policy(policy):
                    actor_hidden_after = get_actor_hidden_state(policy)
                    annotate.update_state(
                        policy_node_name,
                        state_dict_from_actor_hidden(actor_hidden_after),
                    )

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
    """Resolve Hydra task configuration and export one RSL-RL policy."""
    _leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _leapp_scripts_dir not in sys.path:
        sys.path.insert(0, _leapp_scripts_dir)
    from export_utils import disable_torchscript_for_export

    # Must run before the imports below pull in the task modules.
    disable_torchscript_for_export()

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils.hydra import hydra_task_config

    original_argv = sys.argv
    # Hydra reads the preset tokens (physics=/renderer=/presets=) from sys.argv directly.
    sys.argv = [sys.argv[0]] + hydra_args
    exported = False

    try:

        @hydra_task_config(args_cli.task, args_cli.agent)
        def _main(env_cfg, agent_cfg) -> None:
            nonlocal exported
            with launch_simulation(env_cfg, args_cli):
                exported = export_rsl_rl_agent(args_cli, env_cfg, agent_cfg)

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
