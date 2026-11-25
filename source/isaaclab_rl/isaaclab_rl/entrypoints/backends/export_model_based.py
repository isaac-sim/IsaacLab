# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a proprioceptive motion actor and its stateful preprocessing to LEAPP."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# LEAPP captures the tensor operations in quaternion helpers.
torch.jit._state.disable()

from isaaclab.app import launch_simulation
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_inv, quat_mul

from .export_common import add_common_export_args, finalize_export_args, resolve_export_save_path
from .play_model_based import (
    GOAL_FRAME_STRIDE,
    NUM_GOAL_FRAMES,
    goal_window_indices,
    load_legacy_actor,
    load_legacy_task,
)
from .runner_rsl_rl import wrap_rsl_rl_env


class MotionPolicy(torch.nn.Module):
    """Blend moving and stationary goals through the unchanged RSL-RL actor."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        goal = obs["actor_goal"]
        width = self.actor.policy.goal_dim
        inputs = obs.copy()
        inputs["actor_goal"] = goal[..., :width]
        actions = self.actor(inputs)
        inputs["actor_goal"] = goal[..., width : 2 * width]
        standing = self.actor(inputs)
        return standing + goal[..., -1:].clamp(0, 1) * (actions - standing)


def _motion_input(env):
    """Expose the reference publisher's window as one deployment command."""
    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    command = env.command_manager.get_term("motion")
    # Share the exporter's per-observation cache with other consumers of this command.
    cache = env.scene._cache
    key = (id(command), "command:motion")
    if key in cache:
        return cache[key]
    motion = command.motion
    idx = goal_window_indices(motion, command.time_steps)
    value = torch.cat(
        (
            motion.joint_pos[idx],
            motion.joint_vel[idx],
            motion.root_quat_w[idx][..., [3, 0, 1, 2]],
            torch.ones_like(idx[..., None], dtype=torch.float32),
        ),
        -1,
    )
    cache[key] = annotate.input_tensors(
        str(env.unwrapped.spec.id),
        TensorSemantics(name="motion", ref=value, extra={"isaaclab_connection": "command:motion"}),
    )
    return cache[key]


def _deployment_goal(env, *, legacy):
    """Build moving and stationary goals from the externally supplied reference."""
    motion = _motion_input(env)
    if legacy:
        root = env.scene["robot"].data.root_quat_w.torch[:, None].expand(-1, NUM_GOAL_FRAMES, -1)
        relative = quat_mul(quat_inv(root), motion[..., [-4, -3, -2, -5]])
        orientation = matrix_from_quat(relative)[..., :2].flatten(-2)
        active = torch.cat((motion[..., :-5].flatten(1), orientation.flatten(1)), -1)
        joints = motion[:, :1, :29].expand(-1, NUM_GOAL_FRAMES, -1)
        stationary = torch.cat(
            (
                torch.cat((joints, torch.zeros_like(joints)), -1).flatten(1),
                orientation[:, :1].expand(-1, NUM_GOAL_FRAMES, -1).flatten(1),
            ),
            -1,
        )
    else:
        q, dq = motion[:, 0, :-5].chunk(2, -1)
        active = torch.cat((q, dq), -1)
        stationary = torch.cat((q, torch.zeros_like(dq)), -1)
    return torch.cat((active, stationary, motion[:, 0, -1:]), -1)


def _deployment_anchor(env, *, task):
    """Compute the tracked anchor orientation from deployable root and joint sensors."""
    data = env.scene["robot"].data
    q = data.joint_pos.torch
    root = data.root_quat_w.torch
    zero = torch.zeros_like(root[..., :3])
    motion = _motion_input(env)
    ref_q = motion[:, 0, : q.shape[-1]]
    ref_root = motion[:, 0, [-4, -3, -2, -5]]
    quat = task.kinematics.forward((zero, root, zero, zero, q, torch.zeros_like(q)))[1]
    ref_quat = task.kinematics.forward((zero, ref_root, zero, zero, ref_q, torch.zeros_like(ref_q)))[1]
    relative = quat_mul(quat_conjugate(quat[:, task.robot_anchor]), ref_quat[:, task.robot_anchor])
    return matrix_from_quat(relative)[..., :2].flatten(1)


def prepare_motion_export(env, actor, *, task=None):
    """Adapt reference inputs; the common LEAPP exporter captures observations, history and actions."""
    if actor.export_reference_action:
        raise ValueError("Motion deployment requires an absolute-action actor.")
    command = env.unwrapped.command_manager.get_term("motion")
    if command.motion.motion_ends.numel() != 1:
        raise ValueError("Select one reference clip for deployment.")
    manager = env.unwrapped.observation_manager
    goal = manager._group_obs_term_cfgs["actor_goal"][0]
    goal.func, goal.params = _deployment_goal, {"legacy": task is None}
    if task is not None:
        anchor = manager._group_obs_term_cfgs["policy"][manager.active_terms["policy"].index("motion_anchor_ori_b")]
        anchor.func, anchor.params = _deployment_anchor, {"task": task}
    return MotionPolicy(actor.cpu()).eval()


def save_motion(env, destination, node):
    """Save the resampled reference in the joint order used by the deployment graph."""
    owner = env.unwrapped
    command = owner.command_manager.get_term("motion")
    motion = command.motion
    np.savez(
        Path(destination) / node / "motion.npz",
        joint_pos=motion.joint_pos.cpu().numpy(),
        joint_vel=motion.joint_vel.cpu().numpy(),
        root_quat_w=motion.root_quat_w[:, [3, 0, 1, 2]].cpu().numpy(),
        joint_names=np.asarray(command.robot.joint_names),
        fps=np.float32(1 / owner.step_dt),
        goal_frames=NUM_GOAL_FRAMES,
        goal_stride=GOAL_FRAME_STRIDE,
    )


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_export_args(parser, agent_default="rsl_rl_cfg_entry_point")
    parser.add_argument("--motion_dir", help="Single reference clip, or a directory containing one clip.")
    parser.add_argument("--asset_path", help="Override the saved robot asset path.")
    args, overrides = finalize_export_args(parser, argv)
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if any(value != "physics=newton_mjwarp" for value in overrides):
        parser.error("Export restores the checkpoint's saved physical configuration.")
    cfg, agent, checkpoint, path = load_legacy_task(args)
    cfg.scene.num_envs = 1
    destination = Path(resolve_export_save_path(args, "model_based", str(Path(path).parent)))
    destination.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Motion-Tracking-G1", cfg=cfg)
        try:
            env = wrap_rsl_rl_env(env, agent)
            obs, _ = env.reset()
            actor = load_legacy_actor(checkpoint, obs, agent, env.unwrapped.device)
            with torch.inference_mode():
                from .export_rsl_rl import export_policy

                policy = prepare_motion_export(env, actor)
                export_policy(env, policy, args, cfg, str(destination), set(agent.obs_groups["actor"]))
                save_motion(env, destination, args.export_task_name or args.task)
        finally:
            env.close()
    return 0
