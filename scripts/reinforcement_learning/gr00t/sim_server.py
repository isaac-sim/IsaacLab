# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve one PhysX Franka stacking environment to a local GR00T process."""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing.connection import Listener
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from PIL import Image

wp.config.enable_backward = False

from isaaclab.app import launch_simulation
from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


def _reach_red(env) -> torch.Tensor:
    """Provide a geometric reward for approaching the red cube."""
    distance = torch.linalg.vector_norm(
        env.scene["ee_frame"].data.target_pos_w.torch[:, 0] - env.scene["cube_2"].data.root_pos_w.torch,
        dim=-1,
    )
    return 1.0 - torch.tanh(distance / 0.1)


@configclass
class _RewardsCfg:
    reach_red = RewardTermCfg(func=_reach_red, weight=1.0)


@configclass
class _EventsCfg:
    reset_scene = EventTermCfg(
        func="isaaclab.envs.mdp:reset_scene_to_default", mode="reset", params={"reset_joint_targets": True}
    )


def _observation(obs: dict[str, dict[str, torch.Tensor]], reward: float, terminated: bool, truncated: bool) -> dict:
    policy = obs["policy"]
    # The checkpoint retains roll/pitch/yaw key names but stores principal rotation vectors.
    rotation_vector = axis_angle_from_quat(policy["eef_quat"])
    state = torch.cat((policy["eef_pos"], rotation_vector, policy["gripper_pos"]), dim=-1)
    return {
        "state": state.cpu().numpy().astype(np.float32),
        "table_cam": policy["table_cam"].cpu().numpy().astype(np.uint8),
        "wrist_cam": policy["wrist_cam"].cpu().numpy().astype(np.uint8),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    """Reset and step a single environment through an authenticated Unix socket."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    task = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor"
    cfg, _ = resolve_task_config(task, "", overrides=["physics=isaacsim_physx"])
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.events = _EventsCfg()
    cfg.rewards = _RewardsCfg()
    cfg.compute_final_obs = True
    for camera in (cfg.scene.table_cam, cfg.scene.wrist_cam):
        camera.data_types = ["rgb"]
    with Listener(
        args.socket_path, family="AF_UNIX", authkey=bytes.fromhex(os.environ["GR00T_LOCAL_AUTHKEY"])
    ) as server:
        with launch_simulation(cfg, {"visualizer": None, "device": "cuda:0"}):
            with gym.make(task, cfg=cfg) as env:
                obs, _ = env.reset()
                robot = env.unwrapped.scene["robot"]
                joint_reset_error = float((robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs().max())
                if joint_reset_error > 1e-3:
                    raise RuntimeError(f"Initial joint pose differs from the task configuration: {joint_reset_error}")
                frame = _observation(obs, 0.0, False, False)
                for name in ("table_cam", "wrist_cam"):
                    Image.fromarray(frame[name][0]).save(args.output_dir / f"{name}.png")
                initial_state = frame["state"].copy()
                states = [initial_state[0].tolist()]
                print("Simulation ready: Isaac Sim 6.1, PhysX, one environment, two RGB cameras.", flush=True)
                with server.accept() as client:
                    client.send(frame)
                    while True:
                        action = client.recv()
                        if action is None:
                            break
                        action = np.asarray(action, dtype=np.float32)
                        if action.shape != (1, 7) or not np.isfinite(action).all():
                            raise ValueError(f"Expected finite actions of shape (1, 7), got {action}")
                        with torch.inference_mode():
                            obs, reward, terminated, truncated, info = env.step(
                                torch.from_numpy(action).to(env.unwrapped.device)
                            )
                            terminal_obs = info.get("final_obs")
                            done = bool(terminated[0]) or bool(truncated[0])
                            frame = _observation(obs, float(reward[0]), bool(terminated[0]), bool(truncated[0]))
                            if done and terminal_obs is not None:
                                frame["final_state"] = _observation(terminal_obs, 0.0, False, False)["state"]
                        states.append(frame["state"][0].tolist())
                        client.send(frame)
                movement = float(np.max(np.abs(np.asarray(states)[:, :3] - initial_state[:, :3])))
                (args.output_dir / "simulation.json").write_text(
                    json.dumps(
                        {
                            "task": task,
                            "steps": len(states) - 1,
                            "eef_max_displacement_m": movement,
                            "initial_joint_reset_max_error": joint_reset_error,
                            "states": states,
                            "camera_shapes": {n: list(frame[n].shape) for n in ("table_cam", "wrist_cam")},
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
