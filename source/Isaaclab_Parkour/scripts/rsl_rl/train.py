# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import os 
import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
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
import os
import torch
from datetime import datetime

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
# import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from parkour_isaaclab.envs import (
ParkourManagerBasedRLEnv
)
# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point") # 
def main(env_cfg: ParkourManagerBasedRLEnv |ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: ParkourRslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # print("agent_cfg:", agent_cfg)
    # print("agent_cfg end ...")

# agent_cfg: UnitreeGo2ParkourTeacherPPORunnerCfg(
    # seed=1, device='cuda:0', num_steps_per_env=24, max_iterations=50000, 
    # empirical_normalization=False, obs_groups=<dataclasses._MISSING_TYPE object at 0x7f51c6976b60>, 
    # clip_actions=None, save_interval=100, experiment_name='unitree_go2_parkour', run_name='', 
    # logger='tensorboard', neptune_project='isaaclab', wandb_project='isaaclab', resume=False, 
    # load_run='.*', load_checkpoint='model_.*.pt', class_name='OnPolicyRunner', 
    # policy=ParkourRslRlPpoActorCriticCfg(class_name='ActorCriticRMA', init_noise_std=1.0, noise_std_type='scalar', actor_obs_normalization=<dataclasses._MISSING_TYPE object at 0x7f51c6976860>, critic_obs_normalization=<dataclasses._MISSING_TYPE object at 0x7f51c6976770>, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation='elu', tanh_encoder_output=False, scan_encoder_dims=[128, 64, 32], priv_encoder_dims=[64, 20], 
    # actor=ParkourRslRlActorCfg(num_priv_explicit=9, num_priv_latent=29, num_prop=53, num_scan=132, num_hist=10, class_name='Actor', state_history_encoder=ParkourRslRlStateHistEncoderCfg(num_priv_explicit=9, num_priv_latent=29, num_prop=53, num_scan=132, num_hist=10, class_name='StateHistoryEncoder', channel_size=10))), 
    # algorithm=ParkourRslRlPpoAlgorithmCfg(class_name='PPOWithExtractor', num_learning_epochs=5, num_mini_batches=4, learning_rate=0.0002, schedule='adaptive', gamma=0.99, lam=0.95, entropy_coef=0.01, desired_kl=0.01, max_grad_norm=1.0, value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, normalize_advantage_per_mini_batch=False, rnd_cfg=None, symmetry_cfg=None, dagger_update_freq=20, priv_reg_coef_schedual=[0.0, 0.1, 2000.0, 3000.0]), 
    # estimator=ParkourRslRlEstimatorCfg(num_priv_explicit=9, num_priv_latent=29, num_prop=53, num_scan=132, num_hist=10, class_name='DefaultEstimator', train_with_estimated_states=True, learning_rate=0.0001, hidden_dims=[128, 64]), depth_encoder=None)


    
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None) # 
    print("args_cli.task: ", args_cli.task)
    # print("env_cfg:", env_cfg)
    # print("env_cfg end ...")

    
    # # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        print(f"[INFO]: Resuming training from previous checkpoint.")
        if not os.path.exists(log_root_path):
            print(f"{log_root_path} does not exist, creating...")
            os.makedirs(log_root_path, exist_ok=True)
        
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # # wrap for video recording
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
    print(f"Environment {args_cli.task} initialized.")
    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # # create runner from rsl-rl
    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # # # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
