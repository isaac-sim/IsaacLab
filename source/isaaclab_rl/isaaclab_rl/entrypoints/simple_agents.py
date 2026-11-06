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
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.envs.utils.spaces import sample_space
from isaaclab.utils import math as math_utils
from isaaclab.utils import validate

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from .common import (
    add_demo_recording_args,
    add_frontend_args,
    add_video_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    has_play_teleop,
    launch_playback,
    normalize_task_name,
    parse_step_limit,
    pre_launch_video_config,
    resolve_seed,
    video_playback_steps,
)

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
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
    args_cli.seed = resolve_seed(args_cli.seed)
    torch.manual_seed(_SEED if args_cli.seed is None else args_cli.seed)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    _run_environment(args_cli, env_cfg, policy=policy, enable_teleop=args_cli.teleop is True)


def run_teleop_if_configured(argv: list[str]) -> bool:
    """Run checkpoint-free playback when the resolved task declares teleoperation.

    Args:
        argv: Playback arguments without a policy backend or checkpoint selection.

    Returns:
        Whether teleoperation handled this playback request. Tasks without teleoperation
        continue through the existing policy backend dispatcher.
    """
    args_cli = _parse_args(argv, "zero")
    overrides = sys.argv[1:]
    # Backend-only flags and agent overrides belong to the policy dispatcher.
    if args_cli.teleop is not True and any(
        arg.startswith("-") or arg.lstrip("+~").partition("=")[0].partition(".")[0] == "agent" for arg in overrides
    ):
        return False
    try:
        env_cfg, _ = resolve_task_config(args_cli.task, "", play_mode=not args_cli.train_env_cfg, overrides=overrides)
    except ValueError:
        # An implicit probe cannot resolve presets declared only by the agent.
        # Let the policy backend compose the complete config and report errors.
        if args_cli.teleop is True:
            raise
        return False
    if not has_play_teleop(env_cfg):
        if args_cli.teleop is True:
            raise ValueError("--teleop requires a TeleopActionCfg in the task's actions.")
        return False
    args_cli.seed = resolve_seed(args_cli.seed)
    torch.manual_seed(_SEED if args_cli.seed is None else args_cli.seed)
    _run_environment(args_cli, env_cfg, policy="zero", enable_teleop=True)
    return True


def _run_environment(
    args_cli: argparse.Namespace, env_cfg: Any, *, policy: PolicyName, enable_teleop: bool = False
) -> None:
    """Run the resolved environment with a zero/random baseline and optional teleop overrides."""
    apply_env_overrides(args_cli, env_cfg)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    pre_launch_video_config(env_cfg, args_cli)
    # reject unsupported configurations before launching Kit or initializing a native physics backend
    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None

    with (
        launch_playback(env_cfg, args_cli, enable_teleop=enable_teleop) as playback_session,
        contextlib.ExitStack() as cleanup,
    ):
        log_dir = os.path.abspath(os.path.join("logs", f"{policy}_agent", normalize_task_name(args_cli.task)))
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")
        env = create_isaaclab_env(
            args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False, playback_session=playback_session
        )
        cleanup.callback(lambda: close_env(env))
        logger.info(f"Gym observation space: {env.observation_space}")
        logger.info(f"Gym action space: {env.action_space}")
        env.reset()
        if policy == "zero":
            action_policy = create_zero_action_policy(env)
        else:
            action_policy = create_random_action_policy(env)
        logger.info(f"{policy.capitalize()} agent is running, press Ctrl+C to exit...")

        budgets = [n for n in (args_cli.max_steps, video_playback_steps(args_cli, env_cfg)) if n is not None]
        max_steps = min(budgets, default=None)

        # keep running while any visualizer is open and the step budget is not exhausted
        sim = env.unwrapped.sim
        step = 0
        while sim.is_running():
            if playback_session is not None and playback_session.shutdown_requested:
                break
            if max_steps is not None and step >= max_steps:
                break
            step += 1
            started = time.perf_counter()
            with torch.inference_mode():
                env.step(action_policy())
            if args_cli.real_time:
                time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - started)))


def create_zero_action_policy(env: gym.Env) -> Callable[[], Any]:
    """Create a policy that emits finite actions for passive environment playback.

    Manager-based environments infer hold commands for absolute task-space action terms and use literal zeros for all
    other terms. Direct-workflow environments use zero-filled samples of their declared Gymnasium spaces, including
    composite and multi-agent spaces.
    """
    unwrapped = env.unwrapped
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is not None:
        return _create_manager_zero_action_policy(action_manager, unwrapped)

    if isinstance(unwrapped.cfg, DirectMARLEnvCfg):
        actions = {
            agent: sample_space(space, unwrapped.device, batch_size=unwrapped.num_envs, fill_value=0)
            for agent, space in unwrapped.action_spaces.items()
        }
        return lambda: actions

    actions = sample_space(unwrapped.single_action_space, unwrapped.device, batch_size=unwrapped.num_envs, fill_value=0)
    return lambda: actions


def create_random_action_policy(env: gym.Env) -> Callable[[], torch.Tensor | dict[str, torch.Tensor]]:
    """Create a policy that samples uniform random actions in ``[-1, 1]`` for each environment and agent."""
    unwrapped = env.unwrapped
    device = unwrapped.device
    if isinstance(unwrapped.cfg, DirectMARLEnvCfg):
        return lambda: {
            agent: 2 * torch.rand((unwrapped.num_envs, *space.shape), device=device) - 1
            for agent, space in unwrapped.action_spaces.items()
        }

    return lambda: 2 * torch.rand(env.action_space.shape, device=device) - 1


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
    from isaaclab.managers import ActionTerm, InteractiveActionTermCfg

    if isinstance(term, ActionTerm) and isinstance(term.cfg, InteractiveActionTermCfg):
        from isaaclab_teleop.action import TeleopAction

        if isinstance(term, TeleopAction):
            return _create_action_term_zero_policy(term.controller, env)
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
            return _unscale_action(command - term._offset, term._scale)

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
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--teleop", action=argparse.BooleanOptionalAction, default=None, help="Use the task's teleop configuration."
    )
    parser.add_argument("--real-time", action="store_true", help="Pace playback to the simulation timestep.")
    parser.add_argument("--train_env_cfg", action="store_true", help="Skip play-mode overrides during teleoperation.")
    parser.add_argument(
        "--max_steps",
        type=parse_step_limit,
        default=None,
        help="Number of environment steps to run. Runs unbounded when omitted.",
    )
    add_video_args(parser, action=f"the {policy} agent run")
    add_demo_recording_args(parser)
    add_frontend_args(parser)
    add_launcher_args(parser)
    # let task configs select the simulation device
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli
