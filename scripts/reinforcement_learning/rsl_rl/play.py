# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

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
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Additional exports: USD Isaac Robot Schema joint order support
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
    help="Export additional JIT policies using USD Isaac Robot Schema joint order.",
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
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from policy_mapping_helpers import export_robot_schema_policy, import_robot_schema_policy
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
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
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # Optionally export schema-ordered policy variant (JIT and runner checkpoint)
    if args_cli.export_robot_schema_policy:
        export_robot_schema_policy(
            base_env=env.unwrapped,
            runner=runner,
            policy_nn=policy_nn,
            normalizer=normalizer,
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
    obs = env.get_observations()
    # Align runner/policy devices with observation device
    try:
        if isinstance(obs, dict) or (hasattr(obs, "__getitem__") and "policy" in obs):
            target_device = obs["policy"].device
        else:
            target_device = obs.device
        if hasattr(runner, "alg") and hasattr(runner.alg, "to"):
            runner.alg.to(target_device)
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
    except Exception:
        pass
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # Apply observation remapping if schema import is enabled
            policy_input = obs_remap_fn(obs) if obs_remap_fn else obs
            actions = policy(policy_input)
            # Apply action remapping if schema import is enabled
            env_actions = action_remap_fn(actions) if action_remap_fn else actions
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
