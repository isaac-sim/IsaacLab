# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab child process: native PhysX task and bounded local simulation RPC."""

from __future__ import annotations

import argparse
import json
import socket
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import launch_simulation
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_unique

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from scripts.reinforcement_learning.gr00t_skrl.protocol import (
    Observation,
    Request,
    RpcConnection,
    RunConfig,
    Transition,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def proximity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward approach to the actual red cube; native manager multiplies by step_dt [s]."""
    position = env.scene["ee_frame"].data.target_pos_w.torch[:, 0]
    target = env.scene["cube_2"].data.root_pos_w.torch
    return 1.0 - torch.tanh(torch.linalg.vector_norm(target - position, dim=-1) / 0.1)


@configclass
class LocalEvents:
    """Deterministic native reset including joint targets; shared task defaults stay intact."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class LocalRewards:
    """The first runner validates approach, rather than claiming a stacking objective."""

    proximity = RewardTermCfg(func=proximity_reward, weight=1.0)


def critic_state(policy_observations: dict[str, torch.Tensor]) -> np.ndarray:
    """Extract XYZ [m], principal rotation vector [rad] and two finger positions [m]."""
    state = torch.cat(
        (
            policy_observations["eef_pos"],
            axis_angle_from_quat(quat_unique(policy_observations["eef_quat"])),
            policy_observations["gripper_pos"],
        ),
        dim=-1,
    )
    return state[0].detach().float().cpu().numpy().copy()


def observation_packet(observations: dict) -> Observation:
    """Copy native observations to CPU so replies never retain live simulator buffers."""
    policy = observations["policy"]
    packet = Observation(
        policy["table_cam"][0, ..., :3].detach().cpu().numpy().astype(np.uint8, copy=True),
        policy["wrist_cam"][0, ..., :3].detach().cpu().numpy().astype(np.uint8, copy=True),
        critic_state(policy),
    )
    packet.validate()
    return packet


def serve(cfg: RunConfig) -> None:
    """Launch the native task, serve reset/step commands and close on any peer failure."""
    env_cfg = parse_env_cfg(cfg.task, device="cuda:0", num_envs=1, overrides=["physics=isaacsim_physx"])
    env_cfg.seed = cfg.seed
    env_cfg.events = LocalEvents()
    env_cfg.rewards = LocalRewards()
    env_cfg.compute_final_obs = True
    if cfg.episode_steps is not None:
        env_cfg.episode_length_s = cfg.episode_steps * env_cfg.decimation * env_cfg.sim.dt
    for camera in (env_cfg.scene.table_cam, env_cfg.scene.wrist_cam):
        camera.data_types = ["rgb"]
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(cfg.rpc_timeout)
    rpc = None
    try:
        with launch_simulation(
            env_cfg, {"headless": True, "visualizer": None, "visualizer_explicit": True, "enable_cameras": True}
        ):
            env = gym.make(cfg.task, cfg=env_cfg).unwrapped
            try:
                server.bind(cfg.socket_path)
                server.listen(1)
                print("Simulation ready", flush=True)
                connection, _ = server.accept()
                rpc = RpcConnection(connection, cfg.rpc_timeout)
                previous_state = None
                while True:
                    request = rpc.receive()
                    if not isinstance(request, Request):
                        raise ValueError("Invalid RPC request")
                    request.validate()
                    if request.command == "close":
                        break
                    if request.command == "reset":
                        observations, _ = env.reset(seed=cfg.seed)
                        packet = observation_packet(observations)
                        previous_state = packet.state.copy()
                        rpc.send(Transition(packet))
                        continue
                    controls = torch.as_tensor(request.action, device=env.device).reshape(1, 7)
                    observations, reward, terminated, truncated, infos = env.step(controls)
                    packet = observation_packet(observations)
                    done = bool(terminated[0]) or bool(truncated[0])
                    final = critic_state(infos["final_obs"]["policy"]) if done else None
                    physical_state = final if done else packet.state
                    motion = float(np.linalg.norm(physical_state[:3] - previous_state[:3]))
                    previous_state = packet.state.copy()
                    rpc.send(
                        Transition(
                            packet,
                            float(reward[0]),
                            bool(terminated[0]),
                            bool(truncated[0]),
                            final,
                            request.action.copy(),
                            motion,
                        )
                    )
            except BaseException as error:
                if rpc is not None:
                    with suppress(OSError):
                        rpc.send(f"{type(error).__name__}: {error}")
                raise
            finally:
                env.close()
    finally:
        if rpc is not None:
            rpc.close()
        server.close()
        Path(cfg.socket_path).unlink(missing_ok=True)


def main() -> None:
    """Read launcher-owned configuration and run only the simulator process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = RunConfig(**json.loads(Path(args.config).read_text()))
    cfg.validate()
    try:
        serve(cfg)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
