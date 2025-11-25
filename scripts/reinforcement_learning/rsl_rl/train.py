# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def _resolve_class(path: str):
    module, cls = path.split(":")
    mod = __import__(module, fromlist=[cls])
    return getattr(mod, cls)


def _infer_mp_components(mp_id: str) -> tuple[str | None, str | None, str | None]:
    """Infer base_id, mp_type, wrapper for known MP ids (Box Pushing)."""
    if not mp_id or "Box-Pushing" not in mp_id:
        return None, None, None
    tail = mp_id.split("/")[-1]
    parts = tail.split("-")
    if len(parts) < 6:
        return None, None, None
    reward = parts[2]
    mp_type = parts[3]
    arm = parts[4]
    base_id = f"Isaac-Box-Pushing-{reward}-step-{arm}-v0"
    wrapper = "isaaclab_tasks.manager_based.box_pushing.mp_wrapper:BoxPushingMPWrapper"
    return base_id, mp_type, wrapper


def _is_mp_env(env) -> bool:
    """Heuristic to detect MP-wrapped environments."""
    head = env
    while head is not None:
        if hasattr(head, "traj_gen_action_space") or hasattr(head, "context_mask"):
            return True
        spec = getattr(head, "spec", None)
        if spec is not None and getattr(spec, "id", None) is not None and "Isaac_MP" in spec.id:
            return True
        if not hasattr(head, "env"):
            break
        head = head.env
    return False


def _maybe_register_mp_env():
    """Auto-register MP env ids when --task points to an MP variant."""
    task = args_cli.task or ""
    if "Isaac_MP" not in task:
        return

    mp_id = args_cli.mp_id or task
    base_id = args_cli.mp_base_id
    mp_type = args_cli.mp_type
    mp_wrapper = args_cli.mp_wrapper

    if base_id is None or mp_wrapper is None or mp_type is None:
        guess_base, guess_type, guess_wrapper = _infer_mp_components(mp_id)
        base_id = base_id or guess_base
        mp_type = mp_type or guess_type
        mp_wrapper = mp_wrapper or guess_wrapper

    if base_id is None or mp_wrapper is None or mp_type is None:
        print(
            "[WARN] MP task requested but base_id/mp_wrapper/mp_type not provided. "
            "Please pass --mp_base_id/--mp_wrapper/--mp_type to auto-register."
        )
        return

    device = args_cli.mp_device or args_cli.device
    try:
        mp_wrapper_cls = _resolve_class(mp_wrapper)
        from isaaclab_tasks.utils.mp import upgrade

        upgrade(mp_id=mp_id, base_id=base_id, mp_wrapper_cls=mp_wrapper_cls, mp_type=mp_type, device=device)
        # propagate env/agent cfg entry points from base spec if available
        import gymnasium as gym

        base_spec = gym.spec(base_id)
        mp_spec = gym.spec(mp_id)
        env_entry = base_spec.kwargs.get("env_cfg_entry_point")
        if env_entry is not None:
            mp_spec.kwargs["env_cfg_entry_point"] = env_entry
        # propagate rsl-rl agent cfg entry point
        if CUSTOM_AGENT_CFG_PATH is not None:
            mp_spec.kwargs[AGENT_ENTRY_POINT_KEY] = CUSTOM_AGENT_CFG_PATH
        elif AGENT_ENTRY_POINT_KEY in base_spec.kwargs:
            mp_spec.kwargs[AGENT_ENTRY_POINT_KEY] = base_spec.kwargs[AGENT_ENTRY_POINT_KEY]
        print(f"[INFO] Registered MP env id '{mp_id}' (base: {base_id}, type: {mp_type}, device: {device}).")
    except Exception as exc:
        print(f"[WARN] Failed to auto-register MP env '{mp_id}': {exc}")


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# MP auto-registration (optional)
parser.add_argument(
    "--mp_base_id",
    type=str,
    default=None,
    help="Base step env id for MP registration (e.g., Isaac-Box-Pushing-Dense-step-Franka-v0).",
)
parser.add_argument(
    "--mp_wrapper",
    type=str,
    default=None,
    help="MP wrapper class path (e.g., isaaclab_tasks.manager_based.box_pushing.mp_wrapper:BoxPushingMPWrapper).",
)
parser.add_argument("--mp_type", type=str, default=None, help="MP backend type (e.g., ProDMP, ProMP, DMP).")
parser.add_argument(
    "--mp_id",
    type=str,
    default=None,
    help="MP env id to register; defaults to --task when --task contains Isaac_MP.",
)
parser.add_argument(
    "--mp_device",
    type=str,
    default=None,
    help="Device to place MP components on (defaults to --device).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# agent cfg handling: allow direct module:Class paths while keeping hydra key consistent
AGENT_ENTRY_POINT_KEY = "rsl_rl_cfg_entry_point"
CUSTOM_AGENT_CFG_PATH = None
if args_cli.agent and ":" in args_cli.agent:
    CUSTOM_AGENT_CFG_PATH = args_cli.agent
    args_cli.agent = AGENT_ENTRY_POINT_KEY

# target task for Hydra (defaults to provided --task)
HYDRA_TASK_NAME = args_cli.task
if args_cli.task and "Isaac_MP" in args_cli.task:
    mp_id = args_cli.mp_id or args_cli.task
    guess_base, guess_type, guess_wrapper = _infer_mp_components(mp_id)
    if args_cli.mp_base_id is None:
        args_cli.mp_base_id = guess_base
    if args_cli.mp_type is None:
        args_cli.mp_type = guess_type
    if args_cli.mp_wrapper is None:
        args_cli.mp_wrapper = guess_wrapper
    # Hydra needs a known config name; fall back to base env id
    if args_cli.mp_base_id is not None:
        HYDRA_TASK_NAME = args_cli.mp_base_id

# If a custom agent cfg path is provided, patch the gym spec so Hydra picks it up
if CUSTOM_AGENT_CFG_PATH and HYDRA_TASK_NAME is not None:
    try:
        import gymnasium as gym

        base_spec = gym.spec(HYDRA_TASK_NAME)
        base_spec.kwargs[AGENT_ENTRY_POINT_KEY] = CUSTOM_AGENT_CFG_PATH
    except Exception:
        pass

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime
from math import ceil

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl import RslRlMPVecEnvWrapper
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# MP utils (upgrade imports only when needed to avoid overhead)

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(HYDRA_TASK_NAME, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    # For MP tasks, ensure we honor MP rollout semantics and fancy_gym-equivalent budget.
    if args_cli.task and "Isaac_MP" in args_cli.task:
        agent_cfg.num_steps_per_env = 1
        target_steps = 500_000  # match fancy_gym total timesteps
        agent_cfg.max_iterations = ceil(target_steps / (env_cfg.scene.num_envs * agent_cfg.num_steps_per_env))

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if _is_mp_env(env):
        env = RslRlMPVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    _maybe_register_mp_env()
    # run the main function
    main()
    # close sim app
    simulation_app.close()
