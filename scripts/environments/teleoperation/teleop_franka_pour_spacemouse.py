# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SpaceMouse teleoperation for the joint-position Franka Pour task."""

from __future__ import annotations

import argparse
import sys

import torch

DEFAULT_TASK = "Isaac-Pour-Franka-Teleop-v0"


def joint_targets_to_actions(
    *,
    joint_targets: torch.Tensor,
    action_offset: torch.Tensor,
    action_scale: float | torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    """Encode bounded joint targets in the task's normalized joint-action coordinates."""
    joint_targets = torch.clamp(joint_targets, min=lower_limits, max=upper_limits)
    scale = torch.as_tensor(action_scale, dtype=joint_targets.dtype, device=joint_targets.device)
    if torch.any(scale == 0.0):
        raise ValueError("Joint-position action scale must be nonzero.")
    return (joint_targets - action_offset) / scale


def compose_env_action(arm_action: torch.Tensor, gripper_command: torch.Tensor) -> torch.Tensor:
    """Append the normalized symmetric-gripper command to seven arm joint-position actions."""
    if gripper_command.ndim == 1:
        gripper_command = gripper_command.unsqueeze(-1)
    return torch.cat((arm_action, gripper_command), dim=-1)


def apply_tcp_offset_to_jacobian(
    jacobian: torch.Tensor,
    body_quat: torch.Tensor,
    offset_pos: torch.Tensor,
) -> torch.Tensor:
    """Move a root-frame geometric Jacobian to a hand-local tool-centre point."""
    from isaaclab.utils import math as math_utils

    result = jacobian.clone()
    offset_pos_root = math_utils.quat_apply(body_quat, offset_pos)
    result[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos_root), result[:, 3:, :])
    return result


def _base_frame_tcp_state_and_jacobian(
    robot,
    *,
    body_idx: int,
    joint_ids: list[int],
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TCP pose and Jacobian in the robot root frame."""
    from isaaclab.utils import math as math_utils

    body_pose_w = robot.data.body_link_pose_w.torch[:, body_idx]
    root_pose_w = robot.data.root_link_pose_w.torch
    body_pos_b, body_quat_b = math_utils.subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        body_pose_w[:, :3],
        body_pose_w[:, 3:7],
    )
    tcp_pos_b, tcp_quat_b = math_utils.combine_frame_transforms(
        body_pos_b,
        body_quat_b,
        offset_pos,
        offset_rot,
    )

    jacobian_idx = body_idx - 1 if robot.is_fixed_base else body_idx
    jacobian_joint_ids = [joint_id + robot.num_base_dofs for joint_id in joint_ids]
    jacobian_w = robot.data.body_link_jacobian_w.torch[:, jacobian_idx, :, jacobian_joint_ids]
    root_rot_w = math_utils.matrix_from_quat(math_utils.quat_inv(root_pose_w[:, 3:7]))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :3, :] = torch.bmm(root_rot_w, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_w, jacobian_b[:, 3:, :])
    jacobian_b = apply_tcp_offset_to_jacobian(jacobian_b, body_quat_b, offset_pos)
    return tcp_pos_b, tcp_quat_b, jacobian_b


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceMouse teleoperation for Franka Pour.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Registered Franka Pour task name.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments driven by one device.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after N steps; negative runs until close.")
    parser.add_argument("--mock", action="store_true", help="Use zero device commands for a hardware-free smoke test.")
    parser.add_argument("--pos_sensitivity", type=float, default=0.05, help="Translation sensitivity [m].")
    parser.add_argument("--rot_sensitivity", type=float, default=0.05, help="Rotation sensitivity [rad].")
    parser.add_argument("--ik_damping", type=float, default=0.05, help="Damped-least-squares regularization.")
    return parser


def main() -> None:
    """Launch the task and convert SpaceMouse Cartesian deltas to joint-position actions."""
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    parser = _build_arg_parser()
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        try:
            robot = env.scene["robot"]
            arm_cfg = env.cfg.actions.arm_action
            arm_term = env.action_manager.get_term("arm_action")
            joint_ids, joint_names = robot.find_joints(
                arm_cfg.joint_names,
                preserve_order=arm_cfg.preserve_order,
            )
            if len(joint_ids) != 7:
                raise RuntimeError(f"Expected seven Franka arm joints, found {joint_names}.")
            body_ids, body_names = robot.find_bodies(env.cfg.tcp_body_name)
            if len(body_ids) != 1:
                raise RuntimeError(f"Expected one TCP body, found {body_names}.")
            ik = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=True,
                    ik_method="dls",
                    ik_params={"lambda_val": args_cli.ik_damping},
                ),
                num_envs=env.num_envs,
                device=env.device,
            )
            spacemouse = None
            if not args_cli.mock:
                spacemouse = Se3SpaceMouse(
                    Se3SpaceMouseCfg(
                        pos_sensitivity=args_cli.pos_sensitivity,
                        rot_sensitivity=args_cli.rot_sensitivity,
                        gripper_term=True,
                        sim_device=env.device,
                    )
                )
            reset_requested = False

            def request_reset() -> None:
                nonlocal reset_requested
                reset_requested = True

            if spacemouse is not None:
                spacemouse.add_callback("R", request_reset)
            env.reset()
            ik.reset()
            if spacemouse is not None:
                spacemouse.reset()
                print(spacemouse)
                print("SpaceMouse drives the TCP through joint-position actions; R resets the environment.")

            offset_pos = torch.tensor(env.cfg.tcp_offset_pos, device=env.device).repeat(env.num_envs, 1)
            offset_rot = torch.tensor(env.cfg.tcp_offset_rot, device=env.device).repeat(env.num_envs, 1)

            step = 0
            with torch.inference_mode():
                while args_cli.max_steps < 0 or step < args_cli.max_steps:
                    if args_cli.max_steps < 0 and env.sim.visualizers:
                        if not any(v.is_running() and not v.is_closed for v in env.sim.visualizers):
                            break
                    if reset_requested:
                        env.reset()
                        if spacemouse is not None:
                            spacemouse.reset()
                        ik.reset()
                        reset_requested = False

                    command = (
                        spacemouse.advance()
                        if spacemouse is not None
                        else torch.zeros(7, device=env.device, dtype=torch.float32)
                    )
                    tcp_pos_b, tcp_quat_b, jacobian_b = _base_frame_tcp_state_and_jacobian(
                        robot,
                        body_idx=body_ids[0],
                        joint_ids=joint_ids,
                        offset_pos=offset_pos,
                        offset_rot=offset_rot,
                    )
                    ik.set_command(command[:6].repeat(env.num_envs, 1), tcp_pos_b, tcp_quat_b)
                    joint_pos = robot.data.joint_pos.torch[:, joint_ids]
                    joint_targets = ik.compute(tcp_pos_b, tcp_quat_b, jacobian_b, joint_pos)
                    joint_limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
                    arm_action = joint_targets_to_actions(
                        joint_targets=joint_targets,
                        action_offset=arm_term.action_offset,
                        action_scale=arm_term.action_scale,
                        lower_limits=joint_limits[..., 0],
                        upper_limits=joint_limits[..., 1],
                    )
                    gripper_command = command[6].repeat(env.num_envs)
                    env.step(compose_env_action(arm_action, gripper_command))
                    step += 1
        finally:
            env.close()


if __name__ == "__main__":
    main()
