# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the GR00T N1.7 apple policy on the apple pick-and-place task.

The policy runs in a GR00T policy server in its own environment. Start it first, from a
checkout of the GR00T repository (with its submodules) and with the downloaded checkpoint:

.. code-block:: bash

    uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT \\
        --model-path /path/to/GR00T-N1.7-AppleToPlate-cosine40k-checkpoint-15000 --port 5555

Then run the task, here on Newton, which the default install includes. No window opens unless
a visualizer is selected with ``--viz``. Add ``presets=cosmos`` to generate the head camera image
with Cosmos:

.. code-block:: bash

    uv run --with msgpack==1.2.3 --with pyzmq==27.2.0 \\
        python scripts/imitation_learning/gr00t/play_apple_pick_place.py \\
        --num_episodes 4 physics=newton_mjwarp renderer=newton_renderer
"""

from __future__ import annotations

import argparse
import io
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Run the GR00T N1.7 apple policy on the apple pick-and-place task.")
parser.add_argument("--task", type=str, default="IsaacContrib-PickPlace-Apple-G1-Dex3", help="Name of the task.")
parser.add_argument("--policy_host", type=str, default="127.0.0.1", help="Host running the GR00T policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the GR00T policy server.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")
parser.add_argument(
    "--save_frames", type=str, default=None, help="Directory to save the first environment's head camera images to."
)
parser.add_argument("--frame_interval", type=int, default=30, help="Steps between saved images.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
# the policy reads the head camera image
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

import os

import gymnasium as gym
import msgpack
import numpy as np
import torch
import zmq
from PIL import Image

import isaaclab_tasks  # noqa: F401

INSTRUCTION = "move the apple to the plate"
"""The task description the policy was trained with."""

ACTION_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
"""Policy output groups, in the joint order of the task's position and effort action terms."""


class PolicyClient:
    """Request action chunks from a GR00T N1.7 policy server."""

    def __init__(self, host: str, port: int, timeout_ms: int = 120000):
        # the first request also waits for the server to compile its model
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    def get_action(self, observation: dict) -> dict[str, np.ndarray]:
        """Return the action chunk for a batch of observations, each array shaped ``(num_envs, horizon, dim)``."""
        request = {"endpoint": "get_action", "data": {"observation": observation, "options": None}}
        self._socket.send(msgpack.packb(request, default=_encode))
        response = msgpack.unpackb(self._socket.recv(), object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T policy server: {response['error']}")
        return response[0]

    def close(self):
        self._socket.close()
        self._context.term()


def _encode(value: object) -> dict:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Unsupported observation type: {type(value).__name__}")
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}


def _decode(value: dict) -> dict | np.ndarray:
    if value.get("__ndarray_class__"):
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    return value


def policy_observation(obs: dict[str, torch.Tensor]) -> dict:
    """Convert the task's named policy observations to the GR00T observation format."""
    num_envs = obs["ego_view"].shape[0]
    return {
        "video": {"ego_view": obs["ego_view"].to(torch.uint8).cpu().numpy()[:, None]},
        "state": {name: value.float().cpu().numpy()[:, None] for name, value in obs.items() if name != "ego_view"},
        "language": {"annotation.human.task_description": [[INSTRUCTION] for _ in range(num_envs)]},
    }


def task_actions(chunk: dict[str, np.ndarray], device: str) -> torch.Tensor:
    """Concatenate the policy's joint targets and feedforward efforts into the task's action layout."""
    positions = [chunk[group] for group in ACTION_GROUPS]
    efforts = [chunk[f"effort_{group}"] for group in ACTION_GROUPS]
    return torch.as_tensor(np.concatenate(positions + efforts, axis=-1), dtype=torch.float32, device=device)


def save_frame(obs: dict[str, torch.Tensor], index: int):
    """Save the first environment's head camera image, the one the policy sees."""
    os.makedirs(args_cli.save_frames, exist_ok=True)
    image = obs["ego_view"][0].to(torch.uint8).cpu().numpy()
    Image.fromarray(image).save(os.path.join(args_cli.save_frames, f"frame_{index:05d}.png"))


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        policy = PolicyClient(args_cli.policy_host, args_cli.policy_port)
        obs, _ = env.reset()
        frames = 0
        if args_cli.save_frames:
            save_frame(obs["policy"], frames)
        actions, step = None, 0
        episodes = successes = 0
        while episodes < args_cli.num_episodes and env.unwrapped.sim.is_running():
            # play each predicted chunk to its end before asking for the next one
            if actions is None or step == actions.shape[1]:
                chunk = policy.get_action(policy_observation(obs["policy"]))
                actions, step = task_actions(chunk, env.unwrapped.device), 0
            with torch.inference_mode():
                obs, _, terminated, truncated, _ = env.step(actions[:, step])
            step += 1
            frames += 1
            if args_cli.save_frames and frames % args_cli.frame_interval == 0:
                save_frame(obs["policy"], frames)
            done = terminated | truncated
            if done.any():
                succeeded = env.unwrapped.termination_manager.get_term("success")[done]
                episodes += int(done.sum())
                successes += int(succeeded.sum())
                print(f"[INFO] Episodes: {episodes}, successes: {successes}")
                # the reset environments start from a new chunk
                actions = None
        policy.close()
        env.close()


if __name__ == "__main__":
    main()
