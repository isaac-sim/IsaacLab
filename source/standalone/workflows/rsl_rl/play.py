# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
from force_tool.utils.data_utils import update_config
import numpy as np
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from force_tool.visualization.plot_utils import get_img_from_fig, save_numpy_as_mp4

def main():
    """Play with RSL-RL agent."""
    cfg = OmegaConf.create()
    vv = {
        "scene.screw_type":  "m8_loose", 
        "scene.robot.collision_approximation": "convexHull"
          }
    cfg = update_config(cfg, vv)
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric, params=cfg
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    record_forces = True
    forces, frames = [], []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            if record_forces:
                frame = env.unwrapped.render()
                frames.append(frame)
                contact_sensor = env.unwrapped.scene["contact_sensor"]
                dt = contact_sensor._sim_physics_dt
                friction_data = contact_sensor.contact_physx_view.get_friction_data(dt)
                contact_data = contact_sensor.contact_physx_view.get_contact_data(dt)
                nforce_mag, npoint, nnormal, ndist, ncount, nstarts = contact_data
                tforce, tpoint, tcount, tstarts = friction_data
                nforce = nnormal * nforce_mag
                nforce = torch.sum(nforce, dim=0)
                tforce = torch.sum(tforce, dim=0)
                total_force = torch.tensor([nforce.norm(), tforce.norm(), torch.norm(nforce + tforce)])
                print(nforce, tforce, total_force)
                print("Total force: ", total_force)
                forces.append(total_force.cpu().numpy())
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                counter = 0
                if record_forces:
                    wrench_frames = []
                    plot_target = np.array(forces)
                    labels = ["Normal Force", "Tangential Force", "Total Force"]
                    max_val = np.max(plot_target)
                    min_val = np.min(plot_target)
                    indices = np.arange(len(plot_target)) + 1
                    num_plots = plot_target.shape[-1]
                    plt.plot(indices, plot_target, label=labels)
                    plt.legend()
                    plt.show()
                    plt.close()
                    for t in tqdm.tqdm(indices):
                        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
                        plt.ylim((min_val, max_val))
                        plt.xlim((0, len(plot_target)))
                        plt.plot(indices[:t], plot_target[:t], label=labels)
                        plt.legend()
                        wrench_frame = get_img_from_fig(fig, width=frame.shape[1] // 2, height=frame.shape[0])
                        wrench_frames.append(wrench_frame)
                        plt.close()
                        # combine frames
                    frames = np.array(frames)
                    wrench_frames = np.array(wrench_frames)
                    combined_frames = np.concatenate([frames, wrench_frames], axis=2)
                    save_numpy_as_mp4(np.array(combined_frames), "nut.mp4")
                    frames = []
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
