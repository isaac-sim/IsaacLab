# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=8192, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import atexit
import wandb
import signal


def cleanup_wandb(signum=None, frame=None):
    """wandb와 환경을 안전하게 종료하는 함수"""
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

def main():
    """Train with RSL-RL agent."""
    try:
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
            args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

        # wandb 초기화
        wandb.init(
            project="berkeley-humanoid",
            name=agent_cfg.experiment_name,
            config={
                "task": args_cli.task,
                "num_envs": args_cli.num_envs,
                "max_iterations": args_cli.max_iterations
            }
        )

        # 종료 핸들러 등록
        atexit.register(cleanup_wandb)
        signal.signal(signal.SIGINT, cleanup_wandb)
        signal.signal(signal.SIGTERM, cleanup_wandb)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # max iterations for training
        if args_cli.max_iterations:
            agent_cfg.max_iterations = args_cli.max_iterations

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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
        env = RslRlVecEnvWrapper(env)
        

        # create runner from rsl-rl
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # save resume path before creating a new log_dir
        if agent_cfg.resume:
            # get path to previous checkpoint
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)

        # set seed of the environment
        env.seed(agent_cfg.seed)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        # wandb logging callback 정의
        def wandb_log_callback(log_data):
            if log_data is None:
                return
            
            try:
                metrics = {}
                
                # 기본 training 메트릭 로깅
                train_metrics = [
                    "Value function loss",
                    "Surrogate loss", 
                    "Mean action noise std",
                    "Mean reward",
                    "Mean episode length"
                ]
                
                for key, value in log_data.items():
                    # train 메트릭 처리
                    if key in train_metrics:
                        metrics[f"train/{key}"] = value
                    # rewards 메트릭 처리
                    elif key.startswith("Episode_Reward/"):
                        metrics[f"rewards/{key}"] = value  
                    elif key.startswith("Curriculum/"):
                        metrics[f"command_velocity/{key}"] = value
                
                # curriculum 메트릭 직접 로깅
                if hasattr(env.unwrapped, 'reward_manager') and hasattr(env.unwrapped, 'max_episode_length'):
                    try:
                        term_cfg = env.unwrapped.reward_manager.get_term_cfg('joint_pos_pitch_exp')  # velocity_env_cfg.py의 RewardsCfg에 정의된 이름
                        rew = env.unwrapped.reward_manager._episode_sums['joint_pos_pitch_exp']
                        joint_pos_pitch_mean_reward = torch.mean(rew)
                        metrics.update({
                            "curriculum/joint_pos_pitch_mean_reward": joint_pos_pitch_mean_reward,
                            "curriculum/max_episode_length": env.unwrapped.max_episode_length,
                            "curriculum/joint_pos_pitch_reward": joint_pos_pitch_mean_reward/ env.unwrapped.max_episode_length,
                            "curriculum/term_weight": term_cfg.weight,
                            "curriculum/step_dt": env.unwrapped.step_dt
                        })

                        # wandb에 로깅
                        if wandb.run is not None and metrics:
                            wandb.log(metrics)
                            
                    except Exception as e:
                        print(f"Warning: Failed to log curriculum metrics: {str(e)}")

            except Exception as e:
                print(f"Warning: Failed to log to wandb: {str(e)}")

        # wandb 콜백을 환경에 연결
        env.unwrapped.log_callback = wandb_log_callback
        runner.log_callback = wandb_log_callback
        
        # 학습 실행
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, 
            init_at_random_ep_len=True
        )

        # close the simulator
        # close wandb
        wandb.finish()
        env.close()
    except Exception as e:
        print(f"Error: An error occurred during training: {str(e)}")


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
