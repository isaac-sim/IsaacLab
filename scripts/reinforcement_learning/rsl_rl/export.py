# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: E402

"""Script to export a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time
import torch
from collections.abc import Mapping

import leapp
from leapp import annotate

# Disable TorchScript before importing task/environment modules so any
# @torch.jit.script helpers resolve to plain Python functions during export.
torch.jit._state.disable()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)

# LEAPP arguments
parser.add_argument(
    "--export_task_name",
    type=str,
    default=None,
    help="Name of the exported task",
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
parser.add_argument(
    "--disable_automatic_module_annotation",
    action="store_true",
    default=False,
    help="Disables automatic detection and annotation of modules that have internal states",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os

_ANNOTATOR_DIR = os.path.join(
    os.path.dirname(__file__),
    "managed_environment_annotator.py",
)
if _ANNOTATOR_DIR not in sys.path:
    sys.path.insert(0, _ANNOTATOR_DIR)

from export_annotator import patch_env_for_export
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _get_actor_memory_module(policy_nn):
    if hasattr(policy_nn, "memory_a"):
        return policy_nn.memory_a
    if hasattr(policy_nn, "memory_s"):
        return policy_nn.memory_s
    return None


def _ensure_actor_hidden_state_initialized(policy_nn, batch_size: int, device: torch.device, dtype: torch.dtype):
    actor_state, _ = policy_nn.get_hidden_states()
    if actor_state is not None:
        return actor_state

    memory = _get_actor_memory_module(policy_nn)
    if memory is None or not hasattr(memory, "rnn"):
        return None

    num_layers = memory.rnn.num_layers
    hidden_size = memory.rnn.hidden_size
    zeros = torch.zeros(num_layers, batch_size, hidden_size, device=device, dtype=dtype)
    if isinstance(memory.rnn, torch.nn.LSTM):
        actor_state = (zeros.clone(), zeros.clone())
    else:
        actor_state = zeros
    memory.hidden_state = actor_state
    return actor_state


def _state_dict_from_actor_hidden(actor_hidden):
    if actor_hidden is None:
        return {}
    if isinstance(actor_hidden, tuple):
        return {f"actor_state_{idx}": tensor for idx, tensor in enumerate(actor_hidden)}
    return {"actor_state": actor_hidden}


def _actor_hidden_from_registered(registered_state, original_hidden):
    if isinstance(original_hidden, tuple):
        if isinstance(registered_state, tuple):
            return registered_state
        return (registered_state,)
    return registered_state


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Export a RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    # Note: observation functions are already patched at module level (before isaaclab_tasks import)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if not isinstance(env.unwrapped, ManagerBasedRLEnv):
        raise NotImplementedError(
            "Export currently supports only manager-based environments. "
            f"Task '{args_cli.task}' created env type '{type(env.unwrapped).__name__}'."
        )
    export_task_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name

    # required
    obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
    if isinstance(obs_groups_cfg, Mapping):
        required_obs_groups = set(obs_groups_cfg.get("policy", ["policy"]))
    else:
        required_obs_groups = {"policy"}
    patch_env_for_export(
        env,
        task_name=export_task_name,
        export_method=args_cli.export_method,
        required_obs_groups=required_obs_groups,
    )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(policy, "__self__", None)

    # start annotation tracing
    # Note: all patching is done at module/class level before isaaclab_tasks import
    save_path = args_cli.export_save_path if args_cli.export_save_path is not None else log_dir
    leapp.start(export_task_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
    # obs = env.get_observations()
    obs = env.reset()[0]
    # simulate environment
    while not simulation_app.is_running():
        time.sleep(0.5)

    for _ in range(max(args_cli.validation_steps, 2)):
        # run everything in inference mode
        with torch.inference_mode():
            if policy_nn is not None and getattr(policy_nn, "is_recurrent", False):
                actor_hidden = _ensure_actor_hidden_state_initialized(
                    policy_nn,
                    batch_size=env.num_envs,
                    device=env.unwrapped.device,
                    dtype=next(policy_nn.parameters()).dtype,
                )
                registered_state = annotate.state_tensors(
                    export_task_name,
                    _state_dict_from_actor_hidden(actor_hidden),
                )
                actor_memory = _get_actor_memory_module(policy_nn)
                if actor_memory is not None:
                    actor_memory.hidden_state = _actor_hidden_from_registered(registered_state, actor_hidden)
            actions = policy(obs)
            if policy_nn is not None and getattr(policy_nn, "is_recurrent", False):
                actor_hidden_after = policy_nn.get_hidden_states()[0]
                annotate.update_state(
                    export_task_name,
                    _state_dict_from_actor_hidden(actor_hidden_after),
                )
            # env stepping
            obs, _, _, _ = env.step(actions)

    leapp.stop()
    vilidate = args_cli.validation_steps > 0
    leapp.compile_graph(visualize=not args_cli.disable_graph_visualization, validate=vilidate)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
