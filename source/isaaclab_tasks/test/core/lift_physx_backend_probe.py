# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the canonical Franka Lift grasp under one PhysX runtime."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("isaacsim_physx", "ovphysx"), required=True)
    return parser.parse_args()


def _force_magnitude(sensor: Any) -> torch.Tensor:
    force = sensor.data.normal_force_matrix_w.torch
    return force.reshape(force.shape[0], -1, 3).norm(dim=-1).sum(dim=-1)


def main() -> None:
    args = _parse_args()
    simulation_app = None
    if args.backend == "isaacsim_physx":
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(headless=True).app

    import gymnasium as gym
    import torch

    import isaaclab.sim as sim_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    sim_utils.create_new_stage()
    cfg = load_cfg_from_registry("Isaac-Lift-Franka", "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=(args.backend, "cube"))
    cfg.seed = 42
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 2.0
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.curriculum = None

    # This probe owns the initial state. Removing randomization keeps the comparison about
    # articulation, contact, and clone behavior rather than independent random draws.
    for event_name in (
        "robot_physics_material",
        "object_physics_material",
        "object_physics_inertia",
        "joint_stiffness_and_damping",
        "joint_friction",
        "object_scale_mass",
        "variable_gravity",
        "gripper_closing_speed",
    ):
        setattr(cfg.events, event_name, None)
    reset_cfg = cfg.events.conditional_reset.params
    reset_cfg["buffer_size_per_group"] = 4
    reset_cfg["oversample_factor"] = 1.0
    reset_cfg["diversity_feature"] = None
    reset_cfg["success_monitor"] = None
    reset_cfg["terms"]["reset_object_to_target"].params["probability"] = 1.0
    cfg.terminations.time_out = None
    cfg.terminations.object_out_of_bound = None
    cfg.terminations.abnormal_robot = None
    cfg.rewards.early_termination = None

    env = gym.make("Isaac-Lift-Franka", cfg=cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        obj = unwrapped.scene["object"]
        action_term = unwrapped.action_manager.get_term("action")
        action_joint_names = list(action_term._joint_names)
        finger_action_ids = [action_joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
        finger_joint_ids = torch.tensor(
            [robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)], device=unwrapped.device
        )
        arm_joint_ids = torch.tensor(
            [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)], device=unwrapped.device
        )

        all_env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        grasp_env_ids = all_env_ids[::2]
        isolated_env_ids = all_env_ids[1::2]
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

        reset_finger_pos = robot.data.joint_pos.torch[:, finger_joint_ids].clone()
        left_sensor = unwrapped.scene.sensors["panda_leftfinger_object_s"]
        right_sensor = unwrapped.scene.sensors["panda_rightfinger_object_s"]
        reset_peak_force = torch.maximum(_force_magnitude(left_sensor), _force_magnitude(right_sensor))

        # Move alternating clones away after reset to prove that contacts stay clone-local.
        object_pose = obj.data.root_pose_w.torch[isolated_env_ids].clone()
        object_pose[:, :3] = unwrapped.scene.env_origins[isolated_env_ids] + torch.tensor(
            [0.0, 0.0, 1.5], device=unwrapped.device
        )
        obj.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=isolated_env_ids)
        obj.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(isolated_env_ids), 6), device=unwrapped.device), env_ids=isolated_env_ids
        )

        dual_contact = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
        contact_onset = torch.full((unwrapped.num_envs,), -1, dtype=torch.long, device=unwrapped.device)
        peak_left = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        peak_right = torch.zeros_like(peak_left)
        peak_arm_velocity = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        with torch.inference_mode():
            for step in range(12):
                actions.zero_()
                actions[grasp_env_ids[:, None], finger_action_ids] = -1.0
                env.step(actions)
                left_force = _force_magnitude(left_sensor)
                right_force = _force_magnitude(right_sensor)
                peak_left = torch.maximum(peak_left, left_force)
                peak_right = torch.maximum(peak_right, right_force)
                peak_arm_velocity = torch.maximum(
                    peak_arm_velocity, robot.data.joint_vel.torch[:, arm_joint_ids].abs().amax(dim=-1)
                )
                touching = (left_force > 0.01) & (right_force > 0.01)
                contact_onset[(contact_onset < 0) & touching] = step + 1
                dual_contact |= touching

        final_finger_pos = robot.data.joint_pos.torch[:, finger_joint_ids]
        mimic_error = (final_finger_pos[:, 0] - final_finger_pos[:, 1]).abs()
        result = {
            "backend": args.backend,
            "asset_path": cfg.scene.robot.spawn.usd_path,
            "asset_variants": cfg.scene.robot.spawn.variants,
            "object_spawner": type(cfg.scene.object.spawn).__name__,
            "action_type": type(action_term).__name__,
            "action_joint_names": action_joint_names,
            "reset_finger_position_mean": float(reset_finger_pos.mean().item()),
            "reset_peak_force_max": float(reset_peak_force.max().item()),
            "dual_contact_fraction": float(dual_contact[grasp_env_ids].float().mean().item()),
            "contact_onset_step_max": int(contact_onset[grasp_env_ids].max().item()),
            "grasp_peak_force_min": float(
                torch.minimum(peak_left[grasp_env_ids], peak_right[grasp_env_ids]).min().item()
            ),
            "isolated_peak_force_max": float(
                torch.maximum(peak_left[isolated_env_ids], peak_right[isolated_env_ids]).max().item()
            ),
            "grasp_peak_arm_velocity_max": float(peak_arm_velocity[grasp_env_ids].max().item()),
            "isolated_peak_arm_velocity_max": float(peak_arm_velocity[isolated_env_ids].max().item()),
            "grasp_finger_position_mean": float(final_finger_pos[grasp_env_ids].mean().item()),
            "isolated_finger_position_mean": float(final_finger_pos[isolated_env_ids].mean().item()),
            "mimic_error_max": float(mimic_error.max().item()),
        }
        print("LIFT_PHYSX_BACKEND_PROBE=" + json.dumps(result, sort_keys=True))
    finally:
        env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
