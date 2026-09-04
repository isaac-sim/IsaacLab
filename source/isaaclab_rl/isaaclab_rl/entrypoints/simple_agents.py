# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint-free playback workflows for Isaac Lab environments.

The zero and random agents are variations of playback that need no trained checkpoint:
the policy either infers finite zero or hold actions or samples uniform random actions.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.spaces import sample_space
from isaaclab.utils import math as math_utils

from isaaclab_rl.entrypoints.common import print_playback_ready

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    resolve_task_config,
    setup_preset_cli,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

PolicyName = Literal["zero", "random"]
"""Action policies supported by the checkpoint-free agents."""

_DESCRIPTIONS: dict[str, str] = {
    "zero": "Zero agent for Isaac Lab environments.",
    "random": "Random agent for Isaac Lab environments.",
}

_SEED = 42


def run(argv: list[str] | None = None, *, policy: PolicyName) -> None:
    """Run an Isaac Lab environment with a checkpoint-free policy.

    Args:
        argv: Command-line arguments excluding the executable name. Reads ``sys.argv`` when omitted.
        policy: Action policy to apply, either inferred zero actions or uniform random actions.

    Raises:
        ValueError: If the requested policy is not supported.
    """
    if policy not in _DESCRIPTIONS:
        raise ValueError(f"Unsupported policy {policy!r}. Expected one of: {sorted(_DESCRIPTIONS)}.")

    args_cli = _parse_args(argv, policy)

    torch.manual_seed(_SEED)

    # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    # override with CLI arguments and reject unsupported configurations before
    # launching Kit or initializing a native physics backend.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    try:
        env_cfg.validate()
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None

    with launch_simulation(env_cfg, args_cli):
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()
        zero_action_policy = _create_zero_action_policy(env) if policy == "zero" else None
        print_playback_ready(policy)
        # simulate environment
        # keep running while any visualizer is open, and until the step budget is exhausted
        sim = env.unwrapped.sim
        device = env.unwrapped.device
        step = 0
        while sim.is_headless_or_exist_active_visualizer():
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                break
            step += 1
            # run everything in inference mode
            with torch.inference_mode():
                if policy == "zero":
                    actions = zero_action_policy()
                else:
                    # sample actions from -1 to 1
                    actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
                # apply actions
                env.step(actions)
        # close the simulator
        env.close()


def _create_zero_action_policy(env: gym.Env) -> Callable[[], Any]:
    """Create a policy that emits finite actions for passive environment playback.

    Manager-based environments infer hold commands for absolute task-space action terms and use literal zeros for all
    other terms. Direct-workflow environments use zero-filled samples of their declared Gymnasium spaces, including
    composite and multi-agent spaces.
    """
    unwrapped = env.unwrapped
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is not None:
        return _create_manager_zero_action_policy(action_manager, unwrapped)

    if hasattr(unwrapped, "action_spaces"):
        actions = {
            agent: sample_space(space, unwrapped.device, batch_size=unwrapped.num_envs, fill_value=0)
            for agent, space in unwrapped.action_spaces.items()
        }
        return lambda: actions

    actions = sample_space(unwrapped.single_action_space, unwrapped.device, batch_size=unwrapped.num_envs, fill_value=0)
    return lambda: actions


def _create_manager_zero_action_policy(action_manager: Any, env: Any) -> Callable[[], torch.Tensor]:
    """Create a zero-action policy from the active action terms."""
    actions = torch.zeros_like(action_manager.action)
    term_policies = []
    index = 0
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        term_policy = _create_action_term_zero_policy(term, env)
        if term_policy is not None:
            term_policies.append((slice(index, index + term.action_dim), term_policy))
        index += term.action_dim

    def policy() -> torch.Tensor:
        actions.zero_()
        for action_slice, term_policy in term_policies:
            actions[:, action_slice] = term_policy()
        if not torch.isfinite(actions).all():
            raise RuntimeError("Zero agent inferred non-finite actions from the current environment state.")
        return actions

    return policy


def _create_action_term_zero_policy(term: Any, env: Any) -> Callable[[], torch.Tensor] | None:
    """Create the specialized zero-action policy required by an action term."""
    term_types = {cls.__name__ for cls in type(term).__mro__}

    if "PinkInverseKinematicsAction" in term_types:
        controlled_frame_ids, controlled_frame_names = term._asset.find_bodies(
            list(term.cfg.target_eef_link_names.values()), preserve_order=True
        )
        if len(controlled_frame_ids) != len(term.cfg.target_eef_link_names):
            raise ValueError(
                "Expected one controlled body for every Pink IK target. Resolved "
                f"{controlled_frame_names} from {list(term.cfg.target_eef_link_names.values())}."
            )
        if len(controlled_frame_ids) != term._num_frame_tasks:
            raise ValueError(
                f"Pink IK has {term._num_frame_tasks} variable frame tasks but "
                f"{len(controlled_frame_ids)} controlled bodies were configured."
            )

        def pink_policy() -> torch.Tensor:
            frame_poses = term._asset.data.body_link_pose_w.torch[:, controlled_frame_ids].clone()
            frame_poses[..., :3] -= env.scene.env_origins.unsqueeze(1)
            hand_joint_positions = term._asset.data.joint_pos.torch[:, term._hand_joint_ids]
            return torch.cat((frame_poses.flatten(start_dim=1), hand_joint_positions), dim=-1)

        return pink_policy

    if "DifferentialInverseKinematicsAction" in term_types and not term.cfg.controller.use_relative_mode:

        def differential_ik_policy() -> torch.Tensor:
            ee_pos, ee_quat = term._compute_frame_pose()
            command = ee_pos if term.cfg.controller.command_type == "position" else torch.cat((ee_pos, ee_quat), dim=-1)
            return _unscale_action(command, term._scale)

        return differential_ik_policy

    if "RMPFlowAction" in term_types and not term.cfg.use_relative_mode:

        def rmpflow_policy() -> torch.Tensor:
            ee_pos, ee_quat = term._compute_frame_pose()
            return _unscale_action(torch.cat((ee_pos, ee_quat), dim=-1), term._scale)

        return rmpflow_policy

    if "OperationalSpaceControllerAction" in term_types and term._pose_abs_idx is not None:
        term_actions = torch.zeros_like(term.raw_actions)

        def operational_space_policy() -> torch.Tensor:
            term_actions.zero_()
            term._compute_ee_pose()
            term._compute_task_frame_pose()
            if term._task_frame_pose_b is None:
                ee_pos_task = term._ee_pose_b[:, :3]
                ee_quat_task = term._ee_pose_b[:, 3:7]
            else:
                ee_pos_task, ee_quat_task = math_utils.subtract_frame_transforms(
                    term._task_frame_pose_b[:, :3],
                    term._task_frame_pose_b[:, 3:7],
                    term._ee_pose_b[:, :3],
                    term._ee_pose_b[:, 3:7],
                )
            term_actions[:, term._pose_abs_idx : term._pose_abs_idx + 3] = _unscale_action(
                ee_pos_task, term._position_scale
            )
            term_actions[:, term._pose_abs_idx + 3 : term._pose_abs_idx + 7] = _unscale_action(
                ee_quat_task, term._orientation_scale
            )
            return term_actions

        return operational_space_policy

    return None


def _unscale_action(command: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map a processed command back to policy-action coordinates without division by zero."""
    return torch.where(scale != 0.0, command / scale, torch.zeros_like(command))


def _parse_args(argv: list[str] | None, policy: PolicyName) -> argparse.Namespace:
    """Parse the command line of a checkpoint-free agent and hand the remainder to Hydra."""
    parser = argparse.ArgumentParser(description=_DESCRIPTIONS[policy])
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Number of environment steps to run. Runs unbounded when omitted."
    )
    # append AppLauncher cli args
    add_launcher_args(parser)
    # Keep checkpoint-free agents on the kitless default path.
    parser.set_defaults(visualizer=["newton_gl"])
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli
