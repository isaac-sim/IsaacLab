# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the GR00T N1.7 apple policy on the apple pick-and-place task.

The policy runs in a GR00T policy server in its own environment. Start it first, from a
checkout of the GR00T repository (with its submodules) and with the downloaded checkpoint:

.. code-block:: bash

    uv run --frozen python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT \\
        --model-path /path/to/GR00T-N1.7-AppleToPlate-cosine40k-checkpoint-15000 --port 5555

Then run the task, here on Newton, which the default install includes. No window opens unless
a visualizer is selected with ``--viz``. Add ``--task IsaacContrib-PickPlace-Apple-G1-Dex3-Cosmos`` to
generate the head camera image with Cosmos:

.. code-block:: bash

    uv run --with msgpack==1.2.3 --with pyzmq==27.2.0 \\
        python scripts/imitation_learning/gr00t/play_apple_pick_place.py \\
        physics=newton_mjwarp renderer=newton_renderer
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Run the GR00T N1.7 apple policy on the apple pick-and-place task.")
parser.add_argument("--task", type=str, default="IsaacContrib-PickPlace-Apple-G1-Dex3", help="Name of the task.")
parser.add_argument("--policy_host", type=str, default="127.0.0.1", help="Host running the GR00T policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the GR00T policy server.")
parser.add_argument("--num_episodes", type=int, default=4, help="Number of episodes to run.")
parser.add_argument(
    "--save_frames",
    type=str,
    default=None,
    help="Directory to save the first environment's head camera image to, at the start and then once a second.",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
# the policy reads the head camera image
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab.cli import load_external_tasks

import isaaclab_tasks  # noqa: F401

# the Cosmos version of the task is registered by isaaclab_experimental
load_external_tasks()

# the GR00T client from the locomanipulation example, which does not import GR00T
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "locomanipulation_sdg", "gr00t"))
from policy_client import PolicyClient  # noqa: E402

INSTRUCTION = "move the apple to the plate"
"""The task description the policy was trained with."""

ACTION_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
"""Policy output groups, in the joint order of the task's position and effort action terms."""

FRAME_INTERVAL = 30
"""Steps between saved images: one second at the task's 30 Hz."""

POLICY_TIMEOUT_MS = 120000
"""The GR00T server compiles its model on the first request, so the first reply takes much longer than the others."""


def get_action_chunk(policy: PolicyClient, obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Return the action chunk for every environment, each array shaped ``(num_envs, horizon, dim)``."""
    # the N1.7 server takes the observation with options and returns the actions with an info dict
    return policy.get_action({"observation": policy_observation(obs), "options": None})[0]


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
        policy = PolicyClient(args_cli.policy_host, args_cli.policy_port, timeout_ms=POLICY_TIMEOUT_MS)
        obs, _ = env.reset()
        frames = 0
        if args_cli.save_frames:
            save_frame(obs["policy"], frames)
        actions, step = None, 0
        episodes = successes = 0
        while episodes < args_cli.num_episodes and env.unwrapped.sim.is_running():
            # play each predicted chunk to its end before asking for the next one
            if actions is None or step == actions.shape[1]:
                chunk = get_action_chunk(policy, obs["policy"])
                actions, step = task_actions(chunk, env.unwrapped.device), 0
            with torch.inference_mode():
                obs, _, terminated, truncated, _ = env.step(actions[:, step])
            step += 1
            frames += 1
            if args_cli.save_frames and frames % FRAME_INTERVAL == 0:
                save_frame(obs["policy"], frames)
            done = terminated | truncated
            if done.any():
                succeeded = env.unwrapped.termination_manager.get_term("success")[done]
                episodes += int(done.sum())
                successes += int(succeeded.sum())
                print(f"[INFO] Episodes: {episodes}, successes: {successes}")
                # The policy serves all environments in one batch, so the reset environments start a new chunk,
                # and the others take a new chunk from their current state too.
                actions = None
        policy.close()
        env.close()


if __name__ == "__main__":
    main()
