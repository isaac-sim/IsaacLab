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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tasks.core.motion_tracking.mdp.model_based.kinematics import (
    from_native,
    quat_inv,
    quat_mul,
    quat_to_rot6d,
)
from isaaclab_tasks.core.motion_tracking.mdp.model_based.representations import (
    GOAL_FRAME_STRIDE,
    NUM_GOAL_FRAMES,
    goal_window_indices,
)
from isaaclab_tasks.utils import resolve_task_config

from ..common import resolve_published_checkpoint
from .export_common import add_common_export_args, finalize_export_args, leapp_capture, resolve_export_save_path
from .play_model_based import REFERENCE_REVISION, _restore_environment, legacy_agent_cfg, load_legacy_actor
from .runner_rsl_rl import configure_rsl_rl_env_cfg, wrap_rsl_rl_env


class MotionPolicy(torch.nn.Module):
    """Use the trained motion encoder/decoder with explicit history feedback.

    Sensors use Isaac Lab's XYZW quaternion convention. Reference windows contain
    joint positions, joint velocities, WXYZ root quaternions and a motion blend
    weight at each frame. A zero weight tracks a stationary reference with the same actor.
    """

    def __init__(self, actor, default_joint_pos, action_scale, action_offset):
        super().__init__()
        if actor.export_reference_action:
            raise ValueError("Motion deployment requires an absolute-action actor.")
        self.actor = actor.policy
        self.register_buffer("default_joint_pos", default_joint_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_offset", action_offset)

    def forward(self, q, dq, quat, gyro, motion, history, previous_action, initialized):
        gravity = gyro.new_tensor((0.0, 0.0, -1.0)).expand_as(gyro)
        sensor = torch.cat(
            (quat_apply_inverse(quat, gravity), gyro, q - self.default_joint_pos, dq, previous_action), -1
        )
        history = torch.cat((history[:, 1:], sensor[:, None]), 1)
        history = torch.where(initialized.reshape(-1, 1, 1) > 0, history, sensor[:, None])
        # Keep the training layout: all joint targets, followed by all root orientations.
        relative = quat_mul(quat_inv(from_native(quat))[:, None].expand(-1, NUM_GOAL_FRAMES, -1), motion[..., -5:-1])
        goal = torch.cat((motion[..., :-5].flatten(1), quat_to_rot6d(relative).flatten(1)), -1)
        history_n = self.actor.obs_norm(history).flatten(1)

        def decode(goal):
            latent = torch.tanh(self.actor.encoder(self.actor.goal_norm(goal)))
            return self.actor.decoder(torch.cat((latent, history_n), -1))

        actions = decode(goal)
        # Keep a stationary reference active before and after the moving clip.
        # Both branches receive the history of the action actually applied below.
        joints = motion[:, :1, : q.shape[-1]].expand(-1, NUM_GOAL_FRAMES, -1)
        stationary = torch.cat((joints, torch.zeros_like(joints)), -1).flatten(1)
        orientation = quat_to_rot6d(relative[:, :1]).expand(-1, NUM_GOAL_FRAMES, -1).flatten(1)
        standing = decode(torch.cat((stationary, orientation), -1))
        blend = motion[:, 0, -1:].clamp(0, 1)
        applied_action = standing + blend * (actions - standing)
        target = applied_action * self.action_scale + self.action_offset
        return target, history, applied_action, torch.ones_like(initialized)


def export_policy(env, actor, args, cfg, destination):
    """Capture sensor inputs, history feedback and physical joint outputs."""
    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    from isaaclab.utils.leapp.export_annotator import _effective_joint_gains

    owner = env.unwrapped
    command = owner.command_manager.get_term("motion")
    robot = command.robot
    action = owner.action_manager.get_term("joint_pos")
    action_scale = torch.as_tensor(action._scale, device=owner.device).expand(1, robot.num_joints)
    action_offset = torch.as_tensor(action._offset, device=owner.device).expand(1, robot.num_joints)
    names = robot.joint_names
    if command.cfg.actor_obs != "proprio" or len(names) != 29:
        raise ValueError("G1 deployment requires a 29-joint proprioceptive actor.")
    if command.motion.motion_ends.numel() != 1:
        raise ValueError("Select one reference clip with --motion_dir for deployment.")
    policy = MotionPolicy(
        actor.cpu(),
        robot.data.default_joint_pos.torch[:1].cpu(),
        action_scale.cpu(),
        action_offset.cpu(),
    ).eval()
    kp, kd = (x[:1].cpu() for x in _effective_joint_gains(robot))
    history = torch.zeros(1, actor.policy.history_length, actor.policy.obs_dim)
    previous = torch.zeros(1, len(names))
    initialized = torch.zeros(1)
    node = args.export_task_name or args.task
    env.reset()
    with leapp_capture(args, save_path=str(destination), env_cfg=cfg) as steps:
        for _ in range(steps):
            data = robot.data
            idx = goal_window_indices(command.motion, command.time_steps)
            motion = torch.cat(
                (
                    command.motion.joint_pos[idx],
                    command.motion.joint_vel[idx],
                    command.motion.root_quat_w[idx],
                    torch.ones_like(idx[..., None], dtype=torch.float32),
                ),
                -1,
            ).cpu()
            values = [
                data.joint_pos.torch.cpu(),
                data.joint_vel.torch.cpu(),
                data.root_link_pose_w.torch[:, 3:].cpu(),
                quat_apply_inverse(data.root_link_pose_w.torch[:, 3:], data.root_link_vel_w.torch[:, 3:]).cpu(),
                motion,
            ]
            connections = [
                "state:robot:joint_pos",
                "state:robot:joint_vel",
                "state:robot:root_quat_w",
                "state:robot:root_ang_vel_b",
                "command:motion",
            ]
            elements = [names, names, ["qx", "qy", "qz", "qw"], ["wx", "wy", "wz"], None]
            inputs = [
                annotate.input_tensors(
                    node,
                    TensorSemantics(
                        name=key.split(":")[-1], ref=value, element_names=labels, extra={"isaaclab_connection": key}
                    ),
                )
                for key, value, labels in zip(connections, values, elements)
            ]
            history, previous, initialized = annotate.state_tensors(
                node, {"history": history, "previous_action": previous, "initialized": initialized}
            )
            target, history, actions, initialized = policy(*inputs, history, previous, initialized)
            history, previous, initialized = annotate.update_state(
                node, {"history": history, "previous_action": actions, "initialized": initialized}
            )
            outputs = [
                TensorSemantics(
                    name="joint_target",
                    ref=target,
                    element_names=names,
                    extra={"isaaclab_connection": "write:robot:set_joint_position_target_index"},
                )
            ]
            gains = [
                TensorSemantics(
                    name=name,
                    ref=value,
                    element_names=names,
                    extra={"isaaclab_connection": f"write:robot:write_joint_{kind}_to_sim_index"},
                )
                for name, kind, value in (("kp", "stiffness", kp), ("kd", "damping", kd))
            ]
            annotate.output_tensors(
                node, outputs, static_outputs=gains, export_with=args.export_method or "onnx-torchscript"
            )
            env.step(actions.detach().to(owner.device))
    # Export the already-resampled reference consumed by training, in native joint order.
    motion = command.motion
    np.savez(
        destination / node / "motion.npz",
        joint_pos=motion.joint_pos.cpu().numpy(),
        joint_vel=motion.joint_vel.cpu().numpy(),
        root_quat_w=motion.root_quat_w.cpu().numpy(),
        joint_names=np.asarray(names),
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
    path = args.checkpoint
    if path == "pretrained":
        if not args.task:
            parser.error("--task is required with --checkpoint pretrained")
        cfg, _ = resolve_task_config(args.task, args.agent, overrides=[])
        path = resolve_published_checkpoint("model_based", args.task, cfg)
        if path is None:
            return 1
    path = retrieve_file_path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["reference_revision"] != REFERENCE_REVISION:
        parser.error("Checkpoint reference revision does not match the installed algorithm")
    if args.task and args.task != checkpoint["task"]:
        parser.error("Task must match the checkpoint")
    args.task = checkpoint["task"]
    cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=[])
    agent = legacy_agent_cfg(checkpoint)
    _restore_environment(cfg, checkpoint["env_cfg"])
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.commands.motion.evaluation = True
    cfg.commands.motion.reset_noise = False
    cfg.episode_length_s = 1.0e9
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    motion_path = args.motion_dir or cfg.commands.motion.motion_dir
    cfg.commands.motion.motion_dir = motion_path if Path(motion_path).is_dir() else retrieve_file_path(motion_path)
    configure_rsl_rl_env_cfg(cfg, agent)
    destination = Path(resolve_export_save_path(args, "model_based", str(Path(path).parent)))
    destination.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            env = wrap_rsl_rl_env(env, agent)
            obs, _ = env.reset()
            actor = load_legacy_actor(checkpoint, obs, agent, env.unwrapped.device)
            with torch.inference_mode():
                export_policy(env, actor, args, cfg, destination)
        finally:
            env.close()
    return 0
