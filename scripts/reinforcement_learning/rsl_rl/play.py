# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
from scripts.reinforcement_learning.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")

parser.add_argument(
    "--import_robot_schema_policy",
    action="store_true",
    default=False,
    help="Import policy using USD Isaac Robot Schema joint order to current engine representation.",
)
parser.add_argument(
    "--export_robot_schema_policy",
    action="store_true",
    default=False,
    help="Export additional JIT/ONNX policies using USD Isaac Robot Schema joint order.",
)
parser.add_argument(
    "--robot_schema_file",
    type=str,
    default=None,
    help=(
        "Path to YAML file containing joint order to treat as Robot Schema order for import/export (uses key"
        " 'robot_schema_joint_names' by default)."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from typing import cast

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

from policy_mapping_helpers import export_robot_schema_policy, import_robot_schema_policy

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent.
    You can use this script to export a policy in robot schema joint order, and import a policy from robot schema order to the current engine representation.
    To export a policy in robot schema order, you can use the following command:
    Example:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py\
    --task=Isaac-Velocity-Flat-Anymal-D-v0 \
    --num_envs=32 \
    --export_robot_schema_policy \
    --robot_schema_file ../IsaacLab/scripts/newton_sim2sim/mappings/sim2sim_anymal_d.yaml

    This will save JIT and runner checkpoint in the exported directory. You can use this to import the policy to the physX-based Isaac Lab.
    To import a policy from robot schema order, you can use the following command:
    Example:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py\
    --task=Isaac-Velocity-Flat-Anymal-D-v0 \
    --num_envs=32 \
    --import_robot_schema_policy \
    --robot_schema_file ../IsaacLab/scripts/newton_sim2sim/mappings/sim2sim_anymal_d.yaml \
    --checkpoint /path/to/exported/policy_runner_schema_order.pt

    """
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        newton_visualizer=args_cli.newton_visualizer,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic
    # Move the policy used by the runner to the env device to avoid mixed devices
    try:
        ppo_runner.alg.to(env.unwrapped.device)  # type: ignore[attr-defined]
    except Exception:
        pass
    inference_device = env.unwrapped.device

    # export policy to onnx/jit (sim joint order)
    # Ensure we don't create nested "exported" directories
    checkpoint_dir = os.path.dirname(resume_path)
    if os.path.basename(checkpoint_dir) == "exported":
        # If resume_path is already in an exported directory, go up one level
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    export_model_dir = os.path.join(checkpoint_dir, "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # Optionally export schema-ordered policy variants
    if args_cli.export_robot_schema_policy:
        export_robot_schema_policy(
            base_env=env.unwrapped,
            runner=ppo_runner,
            policy_nn=policy_nn,
            normalizer=ppo_runner.obs_normalizer,
            export_model_dir=export_model_dir,
            robot_schema_file=args_cli.robot_schema_file,
        )

    # Schema import functionality - remap observations and actions for imported policies
    if args_cli.import_robot_schema_policy:
        obs_remap_fn, action_remap_fn = import_robot_schema_policy(
            base_env=env.unwrapped,
            robot_schema_file=args_cli.robot_schema_file,
        )
    else:
        obs_remap_fn, action_remap_fn = None, None

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    # Align runner/policy devices with observation device
    try:
        if isinstance(obs, dict):
            if "policy" in obs and isinstance(obs["policy"], torch.Tensor):
                target_device = obs["policy"].device
            else:
                # pick first tensor value's device if available, else fallback to env device
                tensor_values = [v for v in obs.values() if isinstance(v, torch.Tensor)]
                target_device = tensor_values[0].device if tensor_values else env.unwrapped.device
        elif isinstance(obs, torch.Tensor):
            target_device = obs.device
        else:
            target_device = env.unwrapped.device

        if hasattr(ppo_runner, "alg") and hasattr(ppo_runner.alg, "to"):
            ppo_runner.alg.to(target_device)
        # Move runner's obs normalizer if present
        try:
            obs_norm = getattr(ppo_runner, "obs_normalizer", None)
            if obs_norm is not None and hasattr(obs_norm, "to"):
                obs_norm.to(target_device)
        except Exception:
            pass
        if hasattr(policy_nn, "to"):
            policy_nn.to(target_device)
        if hasattr(policy_nn, "actor") and isinstance(policy_nn.actor, torch.nn.Module):
            policy_nn.actor.to(target_device)
        if hasattr(policy_nn, "student") and isinstance(policy_nn.student, torch.nn.Module):
            policy_nn.student.to(target_device)
        if hasattr(policy_nn, "memory_a") and hasattr(policy_nn.memory_a, "rnn"):
            policy_nn.memory_a.rnn.to(target_device)
        if hasattr(policy_nn, "memory_s") and hasattr(policy_nn.memory_s, "rnn"):
            policy_nn.memory_s.rnn.to(target_device)

        # Also move the policy function itself
        if hasattr(policy, "to"):
            policy.to(target_device)

        # Derive final inference device from policy params to avoid stale state
        try:
            first_param = next(policy_nn.parameters())
            inference_device = first_param.device
        except Exception:
            inference_device = target_device
    except Exception:
        pass
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # move observations to policy device to avoid CPU/GPU mismatch
            if hasattr(obs, "to"):
                obs = cast(torch.Tensor, obs).to(inference_device)

            # Apply observation remapping if schema import is enabled
            policy_input = obs_remap_fn(obs) if obs_remap_fn else obs
            actions = policy(policy_input)
            # Apply action remapping if schema import is enabled
            env_actions = action_remap_fn(actions) if action_remap_fn else actions
            # ensure actions are on environment device for stepping
            env_actions = env_actions.to(env.unwrapped.device)
            # env stepping
            obs, _, _, _ = env.step(env_actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
