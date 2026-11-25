# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL LEAPP export backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

import gymnasium as gym
import numpy as np
from rsl_rl.models import MLPModel

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlDistillationRunnerCfg, RslRlSacRunnerCfg, RslRlSuperMimicRunnerCfg
from ..common import resolve_published_checkpoint, resolve_seed
from .export_common import (
    add_common_export_args,
    finalize_export_args,
    get_checkpoint_path,
    leapp_capture,
    prepare_export_env,
    resolve_export_save_path,
    run_export,
)
from .runner_rsl_rl import configure_rsl_rl_env_cfg, create_rsl_rl_runner, wrap_rsl_rl_env


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return the remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Export an RL agent with RSL-RL.")
    add_common_export_args(parser, agent_default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder used to locate checkpoints."
    )
    return finalize_export_args(parser, argv)


def get_actor_memory_module(policy: Any) -> Any | None:
    """Return the actor-side RNN module for supported RSL-RL recurrent policies."""
    return getattr(policy, "rnn", None)


def is_actor_recurrent_policy(policy: Any) -> bool:
    """Return whether the actor policy has a supported recurrent state container."""
    return bool(getattr(policy, "is_recurrent", False) and get_actor_memory_module(policy) is not None)


def get_actor_hidden_state(policy: Any) -> Any | None:
    """Return the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    if hasattr(policy, "get_hidden_state"):
        return policy.get_hidden_state()
    memory = get_actor_memory_module(policy)
    return None if memory is None else getattr(memory, "hidden_state", None)


def set_actor_hidden_state(policy: Any, actor_hidden: Any) -> None:
    """Assign the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    memory = get_actor_memory_module(policy)
    if memory is not None:
        memory.hidden_state = actor_hidden


def ensure_actor_hidden_state_initialized(policy: Any, batch_size: int, device: Any, dtype: torch.dtype) -> Any | None:
    """Initialize and return the actor hidden state when a recurrent policy has not created it yet."""
    actor_state = get_actor_hidden_state(policy)
    if actor_state is not None:
        return actor_state
    memory = get_actor_memory_module(policy)
    if memory is None or not hasattr(memory, "rnn"):
        return None
    zeros = torch.zeros(memory.rnn.num_layers, batch_size, memory.rnn.hidden_size, device=device, dtype=dtype)
    if isinstance(memory.rnn, torch.nn.LSTM):
        actor_state = (zeros.clone(), zeros.clone())
    else:
        actor_state = zeros
    set_actor_hidden_state(policy, actor_state)
    return actor_state


def state_dict_from_actor_hidden(actor_hidden: Any) -> dict[str, torch.Tensor]:
    """Convert the actor hidden state into the named tensor mapping expected by LEAPP state APIs."""
    if actor_hidden is None:
        return {}
    if isinstance(actor_hidden, tuple):
        return {f"actor_state_{idx}": tensor for idx, tensor in enumerate(actor_hidden)}
    return {"actor_state": actor_hidden}


def actor_hidden_from_registered(registered_state: Any, original_hidden: Any) -> Any:
    """Restore the registered LEAPP state to the hidden-state structure expected by the actor memory module."""
    if isinstance(original_hidden, tuple) and not isinstance(registered_state, tuple):
        return (registered_state,)
    return registered_state


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: Any, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to export, or None when no published checkpoint exists."""
    if args_cli.checkpoint == "pretrained":
        if isinstance(agent_cfg, (RslRlSacRunnerCfg, RslRlSuperMimicRunnerCfg)):
            raise ValueError("Published checkpoints are PPO policies and cannot be loaded by OffPolicyRunner agents.")
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def export_rsl_rl_agent(args_cli: argparse.Namespace, env_cfg: Any, agent_cfg: RslRlBaseRunnerCfg) -> bool:
    """Export an RSL-RL agent; returns whether a graph was written."""
    # the LEAPP runtime loads simulation modules, so import it only after the launch
    from leapp import annotate

    if args_cli.seed is not None:
        agent_cfg.seed = resolve_seed(args_cli.seed)
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg.device = args_cli.device

    env_cfg.scene.num_envs = 1
    # certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    logger.info(f"Loading checkpoint search path from directory: {log_root_path}")
    resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
    if not resume_path:
        logger.info(f"No checkpoint found for task: {args_cli.task} in directory: {log_root_path}")
        return False
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    configure_rsl_rl_env_cfg(env_cfg, agent_cfg, inference=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        env = wrap_rsl_rl_env(env, agent_cfg)
        logger.info(f"Loading model checkpoint from: {resume_path}")
        runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
        runner.load(resume_path, map_location=agent_cfg.device)
        # Capture portable operators instead of device-specific fused inference kernels.
        policy = runner.get_inference_policy(device="cpu")
        from rsl_rl.models import SuperMimicActor

        if isinstance(policy, SuperMimicActor):
            destination = Path(resolve_export_save_path(args_cli, "rsl_rl", log_dir))
            destination.mkdir(parents=True, exist_ok=True)
            task = type(runner.alg.task)(env=env, device="cpu")
            export_policy(env, policy, args_cli, env_cfg, destination, task=task)
            return True
        if not isinstance(policy, MLPModel):
            raise TypeError("LEAPP export requires an MLPModel or SuperMimicActor policy.")
        obs_set = "student" if isinstance(agent_cfg, RslRlDistillationRunnerCfg) else "actor"
        required_obs_groups = set(runner.cfg["obs_groups"][obs_set])
        policy_node_name = prepare_export_env(env, args_cli, required_obs_groups=required_obs_groups)
        recurrent = is_actor_recurrent_policy(policy)

        save_path = resolve_export_save_path(args_cli, "rsl_rl", log_dir)
        with leapp_capture(args_cli, save_path=save_path, env_cfg=env_cfg) as num_steps:
            obs = env.reset()[0]
            for _ in range(num_steps):
                with torch.inference_mode():
                    if recurrent:
                        actor_hidden = ensure_actor_hidden_state_initialized(
                            policy,
                            batch_size=env.num_envs,
                            device="cpu",
                            dtype=next(policy.parameters()).dtype,
                        )
                        registered_state = annotate.state_tensors(
                            policy_node_name, state_dict_from_actor_hidden(actor_hidden)
                        )
                        set_actor_hidden_state(policy, actor_hidden_from_registered(registered_state, actor_hidden))
                    actions = policy(obs.to("cpu"))
                    if recurrent:
                        annotate.update_state(
                            policy_node_name, state_dict_from_actor_hidden(get_actor_hidden_state(policy))
                        )
                    obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
    finally:
        env.close()
    return True


class MotionPolicy(torch.nn.Module):
    """Use the trained motion encoder/decoder with explicit history feedback.

    Sensors use Isaac Lab's XYZW quaternion convention. Reference windows contain
    joint positions, joint velocities, WXYZ root quaternions and a motion blend
    weight at each frame. A zero weight tracks a stationary reference with the same actor.
    """

    def __init__(self, actor, default_joint_pos, action_scale, action_offset, task=None):
        super().__init__()
        if actor.export_reference_action:
            raise ValueError("Motion deployment requires an absolute-action actor.")
        self.actor = actor.policy
        self.task = task
        self.register_buffer("default_joint_pos", default_joint_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_offset", action_offset)

    def forward(self, q, dq, quat, gyro, motion, history, previous_action, initialized):
        from .play_model_based import NUM_GOAL_FRAMES

        if self.task is None:
            gravity = gyro.new_tensor((0.0, 0.0, -1.0)).expand_as(gyro)
            sensor = torch.cat(
                (quat_apply_inverse(quat, gravity), gyro, q - self.default_joint_pos, dq, previous_action), -1
            )
        else:
            zero = torch.zeros_like(gyro)
            state = (zero, quat, zero, quat_apply(quat, gyro), q, dq)
            ref_q, ref_dq = motion[:, 0, : 2 * q.shape[-1]].chunk(2, -1)
            ref_quat = motion[:, 0, [-4, -3, -2, -5]]
            ref_state = (zero, ref_quat, zero, zero, ref_q, ref_dq)
            anchor = self.task.kinematics.forward(ref_state)[1][:, self.task.robot_anchor]
            observations = self.task.policy_observations(state, torch.cat((ref_q, ref_dq), -1), anchor, previous_action)
            sensor = observations["policy"]
        history = torch.cat((history[:, 1:], sensor[:, None]), 1)
        history = torch.where(initialized.reshape(-1, 1, 1) > 0, history, sensor[:, None])
        # Keep the training layout: all joint targets, followed by all root orientations.
        if self.task is None:
            relative = quat_mul(quat_inv(quat)[:, None].expand(-1, NUM_GOAL_FRAMES, -1), motion[..., [-4, -3, -2, -5]])
            goal = torch.cat((motion[..., :-5].flatten(1), matrix_from_quat(relative)[..., :2].flatten(1)), -1)
        else:
            goal = observations["actor_goal"]
        history_n = self.actor.obs_norm(history).flatten(1)

        def decode(goal):
            latent = torch.tanh(self.actor.encoder(self.actor.goal_norm(goal)))
            return self.actor.decoder(torch.cat((latent, history_n), -1))

        actions = decode(goal)
        # Keep a stationary reference active before and after the moving clip.
        # Both branches receive the history of the action actually applied below.
        if self.task is None:
            joints = motion[:, :1, : q.shape[-1]].expand(-1, NUM_GOAL_FRAMES, -1)
            stationary = torch.cat((joints, torch.zeros_like(joints)), -1).flatten(1)
            orientation = (
                matrix_from_quat(relative[:, :1])[..., :2].flatten(-2).expand(-1, NUM_GOAL_FRAMES, -1).flatten(1)
            )
            stationary = torch.cat((stationary, orientation), -1)
        else:
            stationary = torch.cat((ref_q, torch.zeros_like(ref_dq)), -1)
        standing = decode(stationary)
        blend = motion[:, 0, -1:].clamp(0, 1)
        applied_action = standing + blend * (actions - standing)
        target = applied_action * self.action_scale + self.action_offset
        return target, history, applied_action, torch.ones_like(initialized)


def export_policy(env, actor, args, cfg, destination, task=None):
    """Capture sensor inputs, history feedback and physical joint outputs."""
    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    from isaaclab.utils.leapp.export_annotator import _effective_joint_gains

    from .play_model_based import GOAL_FRAME_STRIDE, NUM_GOAL_FRAMES, goal_window_indices

    owner = env.unwrapped
    command = owner.command_manager.get_term("motion")
    robot = command.robot
    action = owner.action_manager.get_term("joint_pos")
    action_scale = torch.as_tensor(action._scale, device=owner.device).expand(1, robot.num_joints)
    action_offset = torch.as_tensor(action._offset, device=owner.device).expand(1, robot.num_joints)
    names = robot.joint_names
    if len(names) != 29 or (task is None and actor.policy.obs_dim != 93):
        raise ValueError("G1 deployment requires a 29-joint proprioceptive actor.")
    if command.motion.motion_ends.numel() != 1:
        raise ValueError("Select one reference clip with --motion_dir for deployment.")
    policy = MotionPolicy(
        actor.cpu(),
        robot.data.default_joint_pos.torch[:1].cpu(),
        action_scale.cpu(),
        action_offset.cpu(),
        task=task,
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
                    command.motion.root_quat_w[idx][..., [3, 0, 1, 2]],
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
        root_quat_w=motion.root_quat_w[:, [3, 0, 1, 2]].cpu().numpy(),
        joint_names=np.asarray(names),
        fps=np.float32(1 / owner.step_dt),
        goal_frames=NUM_GOAL_FRAMES,
        goal_stride=GOAL_FRAME_STRIDE,
    )


def _export_motion_checkpoint(argv: list[str] | None = None) -> int:
    """Export a motion checkpoint carrying its saved environment configuration."""
    from isaaclab.app import launch_simulation

    from .play_model_based import (
        REFERENCE_REVISION,
        _restore_environment,
        configure_legacy_observations,
        legacy_agent_cfg,
        load_legacy_actor,
    )

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
        parser.error("Pass a checkpoint file path or URL.")
    path = retrieve_file_path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["reference_revision"] != REFERENCE_REVISION:
        parser.error("Checkpoint reference revision does not match the installed algorithm")
    if args.task and args.task != checkpoint["task"]:
        parser.error("Task must match the checkpoint")
    args.task = checkpoint["task"]
    cfg, _ = resolve_task_config("Isaac-Motion-Tracking-G1", "rsl_rl_cfg_entry_point", overrides=[])
    agent = legacy_agent_cfg(checkpoint)
    _restore_environment(cfg, checkpoint["env_cfg"])
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.episode_length_s = 1.0e9
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    motion_path = args.motion_dir or cfg.commands.motion.motion_file
    cfg.commands.motion.motion_file = motion_path if Path(motion_path).is_dir() else retrieve_file_path(motion_path)
    configure_legacy_observations(cfg, agent)
    destination = Path(resolve_export_save_path(args, "model_based", str(Path(path).parent)))
    destination.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Motion-Tracking-G1", cfg=cfg)
        try:
            env = wrap_rsl_rl_env(env, agent)
            obs, _ = env.reset()
            actor = load_legacy_actor(checkpoint, obs, agent, env.unwrapped.device)
            with torch.inference_mode():
                export_policy(env, actor, args, cfg, destination)
        finally:
            env.close()
    return 0


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task")
    selector.add_argument("--motion_dir")
    selected, _ = selector.parse_known_args(argv)
    if selected.task == "Isaac-Motion-Tracking-GenCoord-G1" or selected.motion_dir is not None:
        return _export_motion_checkpoint(argv)
    args_cli, hydra_args = parse_export_args(argv)
    return run_export(args_cli, hydra_args, export_rsl_rl_agent)


if __name__ == "__main__":
    raise SystemExit(run())
