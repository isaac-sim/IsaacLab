# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class reset_to_start_states(ManagerTermBase):
    """Reset the robot, apple and plate to saved starts of real demonstrations.

    Environment ``i`` uses start ``i`` modulo the number of starts, so a run is repeatable and, with at least as many
    environments as starts, covers all of them. Poses are relative to the environment origin, as ``xyzw`` quaternions.

    The term's parameters are:

    - ``states_file``: JSON file with ``joint_names`` and a list of ``states``. Each state has ``robot_root_pose``,
      ``joint_positions``, ``apple_pose`` and ``plate_pose``.
    - ``robot_cfg``, ``apple_cfg``, ``plate_cfg``: the robot, apple and plate.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        with open(cfg.params["states_file"]) as f:
            data = json.load(f)
        self._joint_names = data["joint_names"]
        self._starts = {
            key: torch.tensor([s[key] for s in data["states"]], dtype=torch.float32, device=env.device)
            for key in ("robot_root_pose", "joint_positions", "apple_pose", "plate_pose")
        }

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        states_file: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
        plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    ):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        index = env_ids % self._starts["joint_positions"].shape[0]
        origins = env.scene.env_origins[env_ids]
        zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)

        for cfg, key in ((apple_cfg, "apple_pose"), (plate_cfg, "plate_pose")):
            pose = self._starts[key][index].clone()
            pose[:, :3] += origins
            asset = env.scene[cfg.name]
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

        robot = env.scene[robot_cfg.name]
        root_pose = self._starts["robot_root_pose"][index].clone()
        root_pose[:, :3] += origins
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
        joint_ids, _ = robot.find_joints(self._joint_names, preserve_order=True)
        joint_pos = self._starts["joint_positions"][index]
        robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids
        )
        # No action drives the legs, so their PD targets must hold the start pose; targets start at zero otherwise.
        command = robot.actuators.target_command
        command.set_position_index(value=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
        command.set_velocity_index(value=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids)
        command.set_effort_index(value=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids)


class reset_fixed_head_camera(ManagerTermBase):
    """Hold the head camera still at the pose the robot's head had in each environment's first reset.

    The real demonstrations were recorded from a camera that did not move while the robot did. The camera is not
    a child of the robot. At an environment's first reset this stores the world pose of the camera's offset from
    ``body_name``, and every reset restores it. Order this event after the one that sets the robot's start.

    The term's parameters are:

    - ``camera_cfg``: the camera, whose offset is relative to ``body_name``.
    - ``robot_cfg``: the robot.
    - ``body_name``: the robot body the camera's offset is relative to.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._quat = torch.zeros(env.num_envs, 4, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        camera_cfg: SceneEntityCfg = SceneEntityCfg("head_camera"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        body_name: str = "head_link",
    ):
        camera = env.scene[camera_cfg.name]
        robot = env.scene[robot_cfg.name]
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        first = env_ids[~self._placed[env_ids]]
        if len(first) > 0:
            # The robot's start was just written; without this its body poses still hold the previous state, and
            # the camera would be placed from there. It runs only at each environment's first reset.
            env.sim.forward()
            body_pose = robot.data.body_link_pose_w.torch[first, robot.body_names.index(body_name)]
            offset = camera.cfg.offset
            pos, quat = math_utils.combine_frame_transforms(
                body_pose[:, :3],
                body_pose[:, 3:7],
                body_pose.new_tensor(offset.pos).expand(len(first), -1),
                body_pose.new_tensor(offset.rot).expand(len(first), -1),
            )
            self._pos[first] = pos
            self._quat[first] = quat
            self._placed[first] = True
        camera.set_world_poses(
            self._pos[env_ids], self._quat[env_ids], env_ids=env_ids, convention=camera.cfg.offset.convention
        )
