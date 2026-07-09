# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a synthetic chirp dataset with KNOWN gains (recovery gate).

Rolls per-joint decorrelated log chirps through the FR3 sysid env with known
{stiffness, damping} and writes a contract-complete ``chirp_data.pt``
(``shaper_type='none'``, state-before-command sampling like the collector).
Fitting this dataset must recover the known gains — the synthetic-recovery
acceptance gate:

    python scripts/sysid/make_synthetic_dataset.py --out /tmp/synth/chirp_data.pt
    python scripts/sysid/fit.py --data /tmp/synth/chirp_data.pt --num_envs 256
"""

# flake8: noqa: E402

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Synthetic FR3 sysid dataset generator.")
parser.add_argument("--task", type=str, default="Isaac-Sysid-Franka-FR3-v0")
parser.add_argument("--out", type=str, required=True, help="Output .pt path.")
parser.add_argument("--duration", type=float, default=20.0, help="Chirp duration (s).")
parser.add_argument("--sample_rate", type=float, default=200.0, help="Command rate (Hz).")
parser.add_argument("--physics_rate", type=float, default=1000.0, help="Sim physics rate (Hz).")
parser.add_argument("--scale", type=float, default=0.1, help="Chirp amplitude (rad).")
parser.add_argument("--f_min", type=float, default=0.3)
parser.add_argument("--f_max", type=float, default=2.0)
parser.add_argument(
    "--from_dataset",
    type=str,
    default=None,
    help=(
        "Replay the EXACT des_dof_pos command matrix (and sample_rate) of an existing dataset "
        "through the known gains instead of generating chirps — the protocol-recovery cell: "
        "identifiability of the final collection excitation is measured, not inferred."
    ),
)
parser.add_argument("--stiffness", type=float, nargs="+", default=[900.0], help="Known gains (1 or 7 values).")
parser.add_argument("--damping", type=float, nargs="+", default=[45.0], help="Known gains (1 or 7 values).")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import math
import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_contract import CANONICAL_JOINT_ORDER, load_dataset, validate_contract  # noqa: E402


def _expand(values: list[float], n: int) -> torch.Tensor:
    if len(values) == 1:
        return torch.full((n,), values[0])
    if len(values) != n:
        raise ValueError(f"expected 1 or {n} gain values, got {len(values)}")
    return torch.tensor(values)


def main() -> None:
    protocol_des = None
    if args_cli.from_dataset:
        src = validate_contract(load_dataset(args_cli.from_dataset), allow_stale_fraction=0.2)
        if src.joint_names != CANONICAL_JOINT_ORDER:
            raise ValueError(f"--from_dataset joint order {src.joint_names} is not canonical")
        protocol_des = src.des_dof_pos.numpy().astype(float)
        args_cli.sample_rate = src.sample_rate
        args_cli.duration = protocol_des.shape[0] / src.sample_rate
        print(
            f"[INFO]: protocol-recovery mode — replaying {protocol_des.shape[0]} commands from {args_cli.from_dataset}"
        )

    time_steps = int(round(args_cli.duration * args_cli.sample_rate))
    dt = 1.0 / args_cli.sample_rate
    substeps = round(dt * args_cli.physics_rate)
    if abs(dt * args_cli.physics_rate - substeps) > 1e-6:
        raise ValueError("physics_rate must be an integer multiple of sample_rate")

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.dt = 1.0 / args_cli.physics_rate
    env_cfg.decimation = substeps  # plain ZOH — shaper_type 'none'
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.episode_length_s = (time_steps + 10) * dt

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        articulation = env.unwrapped.scene["robot"]
        device = env.unwrapped.device
        n = articulation.num_joints

        stiffness = _expand(args_cli.stiffness, n).to(device)
        damping = _expand(args_cli.damping, n).to(device)
        articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness.unsqueeze(0))
        articulation.write_joint_damping_to_sim_index(damping=damping.unsqueeze(0))

        t = np.arange(time_steps) * dt
        if protocol_des is not None:
            des = protocol_des  # exact final-protocol command matrix
        else:
            # Per-joint decorrelated log chirps around the default (ready) pose —
            # identical zero-phase chirps on every joint are rank-1 (unidentifiable).
            home = articulation.data.default_joint_pos.torch[0].detach().cpu()
            des = np.zeros((time_steps, n))
            for j in range(n):
                f_lo = args_cli.f_min * (1.0 + 0.07 * j)
                f_hi = args_cli.f_max * (1.0 + 0.05 * j)
                k = f_hi / f_lo
                phase = (
                    2.0 * math.pi * f_lo * args_cli.duration * (np.power(k, t / args_cli.duration) - 1.0) / math.log(k)
                )
                des[:, j] = home[j].item() + args_cli.scale * np.sin(phase + j * math.pi / n)

        des_t = torch.as_tensor(des, dtype=torch.float32, device=device)
        default = articulation.data.default_joint_pos.torch
        dof_pos = np.zeros((time_steps, n))
        dof_vel = np.zeros((time_steps, n))
        dof_tau = np.zeros((time_steps, n))

        with torch.inference_mode():
            for i in range(time_steps):
                # state-before-command, like the real collector
                dof_pos[i] = articulation.data.joint_pos.torch[0].cpu().numpy()
                dof_vel[i] = articulation.data.joint_vel.torch[0].cpu().numpy()
                applied = getattr(articulation.data, "applied_torque", None)
                if applied is not None:
                    dof_tau[i] = applied.torch[0].cpu().numpy()
                actions = (des_t[i].unsqueeze(0) - default).clone()
                env.step(actions)

        joint_names = list(articulation.joint_names)
        assert joint_names == CANONICAL_JOINT_ORDER, f"unexpected joint order: {joint_names}"
        out = {
            "time": torch.as_tensor(t, dtype=torch.float32),
            "des_dof_pos": torch.as_tensor(des, dtype=torch.float32),
            "dof_pos": torch.as_tensor(dof_pos, dtype=torch.float32),
            "dof_vel": torch.as_tensor(dof_vel, dtype=torch.float32),
            "dof_tau_est": torch.as_tensor(dof_tau, dtype=torch.float32),
            "joint_names": joint_names,
            "active_joint_names": joint_names,
            "sample_rate": float(args_cli.sample_rate),
            "controller_update_rate_hint": float(args_cli.sample_rate),
            "shaper_type": "none",
            "mode": "synthetic",
            # Contract parity with the collector: synthetic states are fresh by
            # construction, stamped on the command clock.
            "state_fresh": torch.ones(time_steps, dtype=torch.uint8),
            "state_stamps": torch.as_tensor(t, dtype=torch.float64),
            "kp_used": stiffness.cpu(),
            "kd_used": damping.cpu(),
            "kd_estimated": False,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        torch.save(out, args_cli.out)
        print(
            f"[INFO]: wrote synthetic dataset ({time_steps} steps, gains kp={stiffness.tolist()}, "
            f"kd={damping.tolist()}) -> {args_cli.out}"
        )
        env.close()


if __name__ == "__main__":
    main()
