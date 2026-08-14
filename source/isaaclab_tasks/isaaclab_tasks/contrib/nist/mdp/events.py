# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.nist.assembly_keypoints import NIST_BOARD_CFG
from isaaclab_tasks.contrib.nist.assembly_profile_cfg import AssemblyProfileCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    from isaaclab_tasks.contrib.nist.assembly_keypoints import Offset


def reset_fixed_asset_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_map: dict[str, str],
    pose_range: dict[str, tuple[float, float]],
):
    """Randomize the primary fixed asset uniformly about its nominal board pose.

    Samples ``pose_range`` about the pose the fixed asset would occupy on the board at its default
    placement, then writes it directly. Placing the fixed asset first (instead of deriving it from a
    yaw-randomized board) turns its distribution from a ring/donut into a filled uniform patch; the
    board is seated under it afterward by :func:`reset_board_under_fixed_asset`.

    Args:
        asset_map: Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name.
            The ``"fixed_asset"`` entry selects the board keypoint used as the nominal center.
        pose_range: Per-axis ``(min, max)`` sample ranges, keys ``x``/``y``/``z`` [m] and
            ``roll``/``pitch``/``yaw`` [rad]. Missing axes stay at the nominal value.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    fixed_asset: Articulation | RigidObject = env.scene["fixed_asset"]
    keypoint: Offset = getattr(NIST_BOARD_CFG, asset_map["fixed_asset"])

    board_default = nistboard.data.default_root_state.torch[env_ids]
    board_pos = board_default[:, 0:3] + env.scene.env_origins[env_ids]
    nominal_pos, nominal_quat = keypoint.combine(board_pos, board_default[:, 3:7])

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    new_pos = nominal_pos + samples[:, 0:3]
    new_quat = math_utils.quat_mul(
        nominal_quat, math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    )

    fixed_asset.write_root_pose_to_sim_index(root_pose=torch.cat([new_pos, new_quat], dim=1), env_ids=env_ids)
    fixed_asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(fixed_asset.data.root_vel_w.torch[env_ids]), env_ids=env_ids
    )


def reset_board_under_fixed_asset(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_map: dict[str, str]):
    """Seat the NIST board (and any extra board assets) under the already-placed fixed asset.

    Inverse of the board-first placement: solves the board root so ``board ∘ keypoint`` matches the
    fixed asset's current pose, then places every other asset in ``asset_map`` (e.g. non-held gears)
    at its own board keypoint. Run after :func:`reset_fixed_asset_uniform`.

    Args:
        asset_map: Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name.
            ``"fixed_asset"`` selects the keypoint solved against; the rest ride along on the board.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    fixed_asset: Articulation | RigidObject = env.scene["fixed_asset"]
    keypoint: Offset = getattr(NIST_BOARD_CFG, asset_map["fixed_asset"])

    fixed_pos = fixed_asset.data.root_pos_w.torch[env_ids]
    fixed_quat = fixed_asset.data.root_quat_w.torch[env_ids]
    board_pos, board_quat = keypoint.subtract(fixed_pos, fixed_quat)
    nistboard.write_root_pose_to_sim_index(root_pose=torch.cat([board_pos, board_quat], dim=1), env_ids=env_ids)
    nistboard.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(nistboard.data.root_vel_w.torch[env_ids]), env_ids=env_ids
    )

    for scene_key, keypoint_attr in asset_map.items():
        if scene_key == "fixed_asset":
            continue
        extra: Articulation | RigidObject = env.scene[scene_key]
        extra_pos, extra_quat = getattr(NIST_BOARD_CFG, keypoint_attr).combine(board_pos, board_quat)
        extra.write_root_pose_to_sim_index(root_pose=torch.cat([extra_pos, extra_quat], dim=1), env_ids=env_ids)
        extra.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(extra.data.root_vel_w.torch[env_ids]), env_ids=env_ids
        )


def _sweep_assembly_fraction(lo: float, hi: float, step: float = 0.001) -> Generator[tuple[float, float]]:
    """Yield ``(frac, frac)`` pairs that bounce between *lo* and *hi*."""
    frac = lo
    increasing = True
    while True:
        frac += step if increasing else -step
        if frac >= hi or frac <= lo:
            increasing = not increasing
        yield (frac, frac)


class reset_held_asset_on_fixed_asset(ManagerTermBase):
    """Sample assembly poses with a profile owned by this reset term."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        profile_cfg: AssemblyProfileCfg = cfg.params["assembly_profile"]
        self.profile = profile_cfg.class_type(profile_cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        assembly_profile: AssemblyProfileCfg,
        held_asset_align_offset: Offset,
        assembly_fraction_range: tuple[float, float],
        fixed_asset_cfg: SceneEntityCfg,
        held_asset_cfg: SceneEntityCfg,
        debug_term: bool = False,
    ):
        fixed_asset: RigidObject = env.scene[fixed_asset_cfg.name]
        held_asset: Articulation = env.scene[held_asset_cfg.name]

        fractions = (
            _sweep_assembly_fraction(*assembly_fraction_range) if debug_term else iter([assembly_fraction_range])
        )
        for frac_range in fractions:
            pos_offset, quat_offset = self.profile.sample(frac_range, len(env_ids), env.device)
            fixed_root_pos_w = fixed_asset.data.root_pos_w.torch
            fixed_root_quat_w = fixed_asset.data.root_quat_w.torch
            pos, quat = math_utils.combine_frame_transforms(
                fixed_root_pos_w[env_ids], fixed_root_quat_w[env_ids], pos_offset, quat_offset
            )
            pose = torch.cat(held_asset_align_offset.subtract(pos, quat), dim=1)
            vel = held_asset.data.default_root_state.torch[env_ids, 7:]
            held_asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            held_asset.write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
            if debug_term:
                env.sim.render()


def reset_held_asset_in_gripper(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    holding_body_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    held_asset_graspable_offset: Offset,
    held_asset_inhand_range: dict[str, tuple[float, float]],
    gripper_grasp_offset: Offset,
):
    robot: Articulation = env.scene[holding_body_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    end_effector_quat_w = robot.data.body_link_quat_w.torch[env_ids, holding_body_cfg.body_ids].view(-1, 4)
    end_effector_pos_w = robot.data.body_link_pos_w.torch[env_ids, holding_body_cfg.body_ids].view(-1, 3)
    grasp_quat = gripper_grasp_offset.quat_t(env.device).expand(len(env_ids), -1)

    # Randomize the grasp target (at the grasp point) BEFORE solving for the asset root, so the
    # pose noise pivots about the grasp point and the graspable frame stays coincident with the
    # gripper. Applying the noise to the root pose afterward rotates about the (offset) root, which
    # swings the graspable point off the gripper — the asset stops being held at the grasp point.
    range_list = [held_asset_inhand_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    grasp_pos_w = end_effector_pos_w + samples[:, 0:3]
    grasp_quat_w = math_utils.quat_mul(
        math_utils.quat_mul(end_effector_quat_w, grasp_quat),
        math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
    )

    new_pos_w, new_quat_w = held_asset_graspable_offset.subtract(grasp_pos_w, grasp_quat_w)

    held_asset.write_root_link_pose_to_sim_index(root_pose=torch.cat([new_pos_w, new_quat_w], dim=1), env_ids=env_ids)
    held_asset.write_root_com_velocity_to_sim_index(
        root_velocity=held_asset.data.default_root_state.torch[env_ids, 7:], env_ids=env_ids
    )


def grasp_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_diameter: float,
    flexible_angle: bool = True,
) -> None:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids][env_ids].clone()
    min_angle = held_asset_diameter / 2 * 1.15
    if flexible_angle:
        max_angle = robot.data.joint_pos_limits.torch[0, robot_cfg.joint_ids[0], 1]
        joint_pos[:] = (torch.rand((len(env_ids),), device=env.device) * (max_angle - min_angle) + min_angle).unsqueeze(
            1
        )
    else:
        joint_pos[:] = min_angle

    robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=robot_cfg.joint_ids, env_ids=env_ids)


class reset_end_effector_around_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.wrist_idx = 6
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
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
        self.grasp_angle_range = (0.3, 0.7)
        self.is_physx = "physx" in env.sim.physics_manager.__name__.lower()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        ik_iterations: tuple[int, int] = (5, 10),
    ) -> None:
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w[env_ids] + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])

        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w.torch[env_ids],
            self.robot.data.root_link_quat_w.torch[env_ids],
            pos_w,
            quat_w,
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Error Rate 75% ^ 10 = 0.05 (final error)
        lo, hi = ik_iterations
        k = int(torch.randint(low=lo, high=hi + 1, size=(1,)).item())
        # Clamp every write to the joint limits: unconstrained DLS steps walk joints past
        # their limits near workspace edges, and beyond-limit states written to sim produce
        # huge limit-constraint impulses (NaN on the mjwarp Newton solver at first step).
        limits = self.robot.data.joint_pos_limits.torch[env_ids][:, self.joint_ids]
        for _ in range(k):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (
                self.robot.data.joint_pos_target.torch[env_ids] - self.robot.data.joint_pos.torch[env_ids]
            )
            new_joint_pos = (delta_joint_pos + self.robot.data.joint_pos.torch[env_ids])[:, self.joint_ids]
            self.robot.write_joint_position_to_sim_index(
                position=torch.clamp(new_joint_pos, limits[..., 0], limits[..., 1]),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )
        if self.is_physx:
            self.robot.root_physx_view.get_jacobians()
