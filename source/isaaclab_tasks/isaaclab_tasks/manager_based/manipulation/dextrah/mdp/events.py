# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import inspect
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class reset_end_effector_around_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        reach_asset_cfg: SceneEntityCfg = cfg.params.get("reach_asset_cfg")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.reach_asset: Articulation | RigidObject = env.scene[reach_asset_cfg.name]
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = None  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reach_asset_cfg: SceneEntityCfg,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        ik_iterations: int = 10,
    ) -> None:
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_tip_pos_w, fixed_tip_quat_w = self.reach_asset.data.root_pos_w, self.reach_asset.data.root_quat_w
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w[env_ids] + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids], self.robot.data.root_link_quat_w[env_ids], pos_w, quat_w
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
        n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        
        # Error Rate 75% ^ 10 = 0.05 (final error)
        for i in range(ik_iterations):
            env.sim.render()
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )
        self.robot.root_physx_view.get_jacobians()