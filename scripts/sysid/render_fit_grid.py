# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the parallel FR3 envs replaying a chirp on Newton/mjwarp to a video.

Each env is assigned a different candidate {stiffness, damping} — the population
a CMA-ES generation evaluates in parallel — so the grid shows arms tracking the
same command with visibly different fidelity. Headless (newton.viewer.ViewerGL);
frames encoded to mp4 via imageio's bundled ffmpeg.

    ./isaaclab.sh -p scripts/sysid/render_fit_grid.py --data <chirp_data.pt> \
        --num_envs 16 --out logs/sysid/fr3_render/grid.mp4
"""

# flake8: noqa: E402

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Render parallel FR3 sysid envs to video.")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Sysid-Franka-FR3-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--out", type=str, default="logs/sysid/fr3_render/grid.mp4")
parser.add_argument("--every", type=int, default=6, help="capture one frame per N env steps")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--env_spacing", type=float, default=2.5)
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Override robot USD (e.g. the fr3_visual variant — the fitting default fr3_nomesh has no geometry to draw)",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_contract import load_dataset, validate_contract  # noqa: E402


def main() -> None:
    ds = validate_contract(load_dataset(args_cli.data), allow_stale_fraction=0.2)
    des = ds.des_dof_pos
    meas0 = ds.dof_pos[0]
    T = des.shape[0]
    dt = ds.dt

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.usd is not None:
        env_cfg.scene.robot.spawn.usd_path = os.path.abspath(args_cli.usd)
    n = args_cli.num_envs
    env_cfg.scene.num_envs = n
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.sim.dt = 0.001
    env_cfg.decimation = int(round(dt / 0.001))
    env_cfg.sim.render_interval = env_cfg.decimation
    # CUDA-graph capture can bypass the state buffers the headless viewer reads,
    # freezing the rendered frames while physics advances. Disable for rendering.
    _phys = getattr(env_cfg.sim.physics, "newton_mjwarp", env_cfg.sim.physics)
    if hasattr(_phys, "use_cuda_graph"):
        _phys.use_cuda_graph = False
    env_cfg.episode_length_s = (T + 20) * dt
    # Frame the whole grid: pull the camera back with the grid extent.
    extent = args_cli.env_spacing * (int(np.ceil(np.sqrt(n))) + 1)
    eye = (0.45 * extent, 0.45 * extent, 0.3 * extent)
    lookat = (0.0, 0.0, 0.3)
    env_cfg.viewer.eye = eye
    env_cfg.viewer.lookat = lookat
    # Force the standalone GL capture: it logs the Newton state at capture time,
    # i.e. after our eval_fk refresh. The visualizer-reuse path draws inside
    # env.step(), before the refresh, and returns a stale framebuffer.
    env_cfg.video_recorder.backend_source = "renderer"
    env_cfg.video_recorder.window_width = 1280
    env_cfg.video_recorder.window_height = 720

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        env.reset()
        art = env.unwrapped.scene["robot"]
        dev = env.unwrapped.device
        nj = art.num_joints
        ids = torch.arange(nj, device=dev)

        # Per-env candidate gains: a spread over the search space (log-uniform),
        # env 0 pinned near the converged value as a reference.
        rng = np.random.default_rng(0)
        kp = torch.tensor(rng.uniform(np.log(150.0), np.log(3000.0), n), device=dev).exp()
        kd = torch.tensor(rng.uniform(np.log(1.0), np.log(60.0), n), device=dev).exp()
        kp[0], kd[0] = 629.0, 6.0
        stiff = torch.tensor([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], device=dev).unsqueeze(0).repeat(n, 1)
        damp = torch.tensor([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0], device=dev).unsqueeze(0).repeat(n, 1)
        jid = art.joint_names.index("fr3_joint1")
        stiff[:, jid] = kp
        damp[:, jid] = kd
        art.write_joint_stiffness_to_sim_index(stiffness=stiff, joint_ids=ids)
        art.write_joint_damping_to_sim_index(damping=damp, joint_ids=ids)
        q0 = meas0.to(dev).unsqueeze(0).repeat(n, 1)
        art.write_joint_position_to_sim_index(position=q0, joint_ids=ids)
        art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0), joint_ids=ids)

        default = art.data.default_joint_pos.torch
        des_d = des.to(dev)
        actions = torch.zeros(n, nj, device=dev)
        frames = []
        jmin, jmax = 1e9, -1e9
        with torch.inference_mode():
            for k in range(T):
                actions[:] = des_d[k].unsqueeze(0).repeat(n, 1) - default
                env.step(actions)
                q1 = float(art.data.joint_pos.torch[0, jid])
                jmin, jmax = min(jmin, q1), max(jmax, q1)
                if k % args_cli.every == 0:
                    f = env.render()
                    if f is not None:
                        frames.append(np.array(f, copy=True))  # get_frame() reuses one buffer
        print(f"[RENDER] env0 fr3_joint1 sim range: [{jmin:.4f}, {jmax:.4f}] rad (swing {jmax - jmin:.4f})")
        env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    print(f"[RENDER] {len(frames)} frames {frames[0].shape} -> {args_cli.out}")
    try:
        imageio.mimwrite(args_cli.out, frames, fps=args_cli.fps, codec="libx264", quality=7)
    except Exception as e:
        alt = os.path.splitext(args_cli.out)[0] + ".gif"
        imageio.mimwrite(alt, frames, fps=args_cli.fps)
        print(f"[RENDER] mp4 failed ({e}); wrote {alt}")
        return
    print(f"[RENDER] wrote {args_cli.out} ({os.path.getsize(args_cli.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
