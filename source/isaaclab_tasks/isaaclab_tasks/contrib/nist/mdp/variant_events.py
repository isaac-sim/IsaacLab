# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset terms for the homogeneous Factory scene."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import instantiate
from isaaclab.utils import math as math_utils

from ..assembly_profile_cfg import UniformPoseNoiseCfg
from ..utils import reset_state
from .assembly_variants import assembly_variant_context

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    from ..assembly_keypoints import Offset


def reset_variant_fixed_asset_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    variant_context: str = "assembly_variants",
):
    """Randomize the primary fixed asset uniformly about its nominal board pose.

    Samples ``pose_range`` about the pose the fixed asset would occupy on the board at its default
    placement, then writes it directly. Placing the fixed asset first (instead of deriving it from a
    yaw-randomized board) turns its distribution from a ring/donut into a filled uniform patch; the
    board is seated under it afterward by :func:`reset_variant_board_under_fixed_asset`.

    Args:
        pose_range: Per-axis ``(min, max)`` sample ranges, keys ``x``/``y``/``z`` [m] and
            ``roll``/``pitch``/``yaw`` [rad]. Missing axes stay at the nominal value.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    fixed_asset: Articulation | RigidObject = env.scene["fixed_asset"]
    variants = assembly_variant_context(env, variant_context)

    board_default = nistboard.data.default_root_state.torch[env_ids]
    board_pos = board_default[:, 0:3] + env.scene.env_origins[env_ids]
    nominal_pos, nominal_quat = variants.combine("board", board_pos, board_default[:, 3:7], env_ids)

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


def reset_variant_board_under_fixed_asset(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, variant_context: str = "assembly_variants"
):
    """Seat the NIST board under the already-placed fixed asset.

    Inverse of the board-first placement: solves the board root so ``board ∘ keypoint`` matches the
    fixed asset's current pose. Run after :func:`reset_variant_fixed_asset_uniform`.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    fixed_asset: Articulation | RigidObject = env.scene["fixed_asset"]
    variants = assembly_variant_context(env, variant_context)

    fixed_pos = fixed_asset.data.root_pos_w.torch[env_ids]
    fixed_quat = fixed_asset.data.root_quat_w.torch[env_ids]
    board_pos, board_quat = variants.subtract("board", fixed_pos, fixed_quat, env_ids)
    nistboard.write_root_pose_to_sim_index(root_pose=torch.cat([board_pos, board_quat], dim=1), env_ids=env_ids)
    nistboard.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(nistboard.data.root_vel_w.torch[env_ids]), env_ids=env_ids
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


class reset_variant_held_asset_on_fixed_asset(ManagerTermBase):
    """Place the held asset along its selected assembly path with term-owned pose noise."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._variants = assembly_variant_context(env, cfg.params["variant_context"])
        self._fixed_asset: RigidObject = env.scene[cfg.params["fixed_asset_cfg"].name]
        self._held_asset: Articulation | RigidObject = env.scene[cfg.params["held_asset_cfg"].name]
        noise_cfg = cfg.params["pose_noise"]
        self._pose_noise = instantiate(noise_cfg) if noise_cfg is not None else None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        assembly_fraction_range: tuple[float, float],
        fixed_asset_cfg: SceneEntityCfg,
        held_asset_cfg: SceneEntityCfg,
        pose_noise: UniformPoseNoiseCfg | None = None,
        debug_term: bool = False,
        variant_context: str = "assembly_variants",
    ) -> None:
        """Seat the held asset's alignment frame on a sampled assembly pose.

        Args:
            assembly_fraction_range: Fraction interval; zero is assembled and one is at entry.
                Values above one extend beyond entry along the same path.
            fixed_asset_cfg: Asset against which the path is measured, bound at initialization.
            held_asset_cfg: Asset being placed, bound at initialization.
            pose_noise: Additional position [m] and Euler-angle [rad] noise in the fixed frame.
                The sampler is constructed once per term; None leaves the profile noise unchanged.
            debug_term: Sweep the fraction interval continuously for visual inspection.
            variant_context: Event term owning the selected assembly variants, bound at initialization.
        """
        fractions = (
            _sweep_assembly_fraction(*assembly_fraction_range) if debug_term else iter([assembly_fraction_range])
        )
        for frac_range in fractions:
            pos_offset, quat_offset = self._variants.sample_profile(frac_range, env_ids)
            if self._pose_noise is not None:
                # Displace the align point in the fixed frame, then tilt about that displaced point.
                noise_pos, noise_quat = self._pose_noise(len(env_ids), env.device)
                pos_offset = pos_offset + noise_pos
                quat_offset = math_utils.quat_mul(noise_quat, quat_offset)
            pos, quat = math_utils.combine_frame_transforms(
                self._fixed_asset.data.root_pos_w.torch[env_ids],
                self._fixed_asset.data.root_quat_w.torch[env_ids],
                pos_offset,
                quat_offset,
            )
            pose = torch.cat(self._variants.subtract("held_align", pos, quat, env_ids), dim=1)
            velocity = self._held_asset.data.default_root_state.torch[env_ids, 7:]
            self._held_asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            self._held_asset.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
            if debug_term:
                env.sim.render()


class sample_settled_asset_pose(ManagerTermBase):
    """Sample held-asset resting poses collected before reset-state generation."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._held_asset: RigidObject = env.scene[cfg.params["held_asset_cfg"].name]
        self._settled_poses: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        held_asset_cfg: SceneEntityCfg,
        drop_term: EventTermCfg,
        scene_assets: list[str],
        num_steps: int = 20,
    ) -> None:
        if env_ids.numel() == 0:
            return
        if self._settled_poses is None:
            self._precollect()

        variant_ids = self._held_asset.mesh_variant_ids.torch[env_ids].long()
        assert self._settled_poses is not None
        pose_ids = torch.randint(self._settled_poses.shape[1], (len(env_ids),), device=env.device)
        pose = self._settled_poses[variant_ids, pose_ids]
        pose[:, :3] += env.scene.env_origins[env_ids]
        self._held_asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._held_asset.write_root_com_velocity_to_sim_index(
            root_velocity=torch.zeros_like(self._held_asset.data.root_com_vel_w.torch[env_ids]), env_ids=env_ids
        )

    def _precollect(self) -> None:
        env = self._env
        drop_term: EventTermCfg = self.cfg.params["drop_term"]
        scene_assets: list[str] = self.cfg.params["scene_assets"]
        num_steps: int = self.cfg.params["num_steps"]
        num_variants = self._held_asset.num_mesh_variants
        poses_per_variant = max(env.num_envs // num_variants, 1)
        num_poses = poses_per_variant * num_variants
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        banked = sorted((set(env.scene._articulations) | set(env.scene._rigid_objects)) & set(scene_assets))
        snapshot = reset_state.get_reset_state(env, all_env_ids, banked, is_relative=True)
        settled_poses = torch.empty((num_poses, 7), device=env.device)

        # Leave only the table in the settling scene so the poses remain valid when the board and
        # fixed asset are randomized later.
        for name in banked:
            if name == self.cfg.params["held_asset_cfg"].name:
                continue
            asset = env.scene[name]
            pose = asset.data.root_link_pose_w.torch.clone()
            pose[:, 2] = env.scene.env_origins[:, 2] + 5.0
            asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=all_env_ids)
            asset.write_root_com_velocity_to_sim_index(
                root_velocity=torch.zeros_like(asset.data.root_com_vel_w.torch), env_ids=all_env_ids
            )

        substeps = 1 if env.sim.backend_manager.handles_decimation() else env.cfg.decimation
        for start in range(0, num_poses, env.num_envs):
            variant_ids = (torch.arange(env.num_envs, device=env.device) + start).remainder(num_variants)
            self._held_asset.write_mesh_variant_to_sim(variant_ids.to(torch.int32), all_env_ids)
            drop_term.func(env, all_env_ids, **drop_term.params)
            for _ in range(num_steps * substeps):
                env.sim.step(render=False)
            env.scene.update(dt=env.step_dt)

            count = min(env.num_envs, num_poses - start)
            pose = self._held_asset.data.root_link_pose_w.torch[:count].clone()
            pose[:, :3] -= env.scene.env_origins[:count]
            settled_poses[start : start + count] = pose

        reset_state.set_reset_state(env, snapshot, all_env_ids, banked, is_relative=True)

        self._settled_poses = settled_poses.view(poses_per_variant, num_variants, 7).transpose(0, 1).contiguous()


def reset_variant_held_asset_in_gripper(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    holding_body_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    held_asset_inhand_range: dict[str, tuple[float, float]],
    gripper_grasp_offset: Offset,
    variant_context: str = "assembly_variants",
):
    robot: Articulation = env.scene[holding_body_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    end_effector_quat_w = robot.data.body_link_quat_w.torch[env_ids, holding_body_cfg.body_ids].view(-1, 4)
    end_effector_pos_w = robot.data.body_link_pos_w.torch[env_ids, holding_body_cfg.body_ids].view(-1, 3)
    grasp_pos_w, grasp_quat_w = gripper_grasp_offset.combine(end_effector_pos_w, end_effector_quat_w)

    # Randomize the grasp target (at the grasp point) BEFORE solving for the asset root, so the
    # pose noise pivots about the grasp point and the graspable frame stays coincident with the
    # gripper. Applying the noise to the root pose afterward rotates about the (offset) root, which
    # swings the graspable point off the gripper — the asset stops being held at the grasp point.
    range_list = [held_asset_inhand_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    grasp_pos_w, grasp_quat_w = math_utils.combine_frame_transforms(
        grasp_pos_w,
        grasp_quat_w,
        samples[:, :3],
        math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
    )

    variants = assembly_variant_context(env, variant_context)
    new_pos_w, new_quat_w = variants.subtract("held_grasp_point", grasp_pos_w, grasp_quat_w, env_ids)

    held_asset.write_root_link_pose_to_sim_index(root_pose=torch.cat([new_pos_w, new_quat_w], dim=1), env_ids=env_ids)
    held_asset.write_root_com_velocity_to_sim_index(
        root_velocity=held_asset.data.default_root_state.torch[env_ids, 7:], env_ids=env_ids
    )


def grasp_variant_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    flexible_angle: bool = True,
    variant_context: str = "assembly_variants",
) -> None:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids][env_ids].clone()
    min_angle = assembly_variant_context(env, variant_context).grasp_diameter(env_ids) / 2 * 1.05
    if flexible_angle:
        max_angle = robot.data.joint_pos_limits.torch[0, robot_cfg.joint_ids[0], 1]
        joint_pos[:] = (torch.rand((len(env_ids),), device=env.device) * (max_angle - min_angle) + min_angle).unsqueeze(
            1
        )
    else:
        joint_pos[:] = min_angle.unsqueeze(1)

    robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=robot_cfg.joint_ids, env_ids=env_ids)


class reset_variant_end_effector_around_asset(ManagerTermBase):
    """Drive the gripper onto a sampled pose around an asset with a differential IK solve.

    Reports whether each solve actually landed on its target through :attr:`is_valid`, which
    :class:`~isaaclab_tasks.contrib.nist.utils.ChainedResetTerms` collects and the reset
    accumulator uses to reject the state instead of banking a missed grasp.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: str = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] | str = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        robot_ik_body_offset: Offset = cfg.params.get("robot_ik_body_offset")  # type: ignore

        self.variants = assembly_variant_context(env, cfg.params.get("variant_context", "assembly_variants"))
        range_list = (
            None
            if isinstance(pose_range_b, str)
            else [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        )
        self.wrist_idx = 6
        self.ranges = pose_range_b if range_list is None else torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.robot_ik_body_offset: Offset = robot_ik_body_offset
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
        self.is_physx = env.sim.backend_manager.backend_name == "physx"
        self.is_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        """Whether the last solve for each env reached its target within ``pose_tolerance``."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: str,
        pose_range_b: dict[str, tuple[float, float]] | str,
        robot_ik_cfg: SceneEntityCfg,
        robot_ik_body_offset: Offset,
        upright_gripper: bool = False,
        ik_iterations: tuple[int, int] = (5, 10),
        pose_tolerance: tuple[float, float] | None = None,
        variant_context: str = "assembly_variants",
    ) -> None:
        """Solve the gripper onto a pose sampled around ``fixed_asset_offset``.

        Args:
            fixed_asset_cfg: Asset the target pose is sampled around. Not necessarily the fixed
                asset — the strategies that grasp a part already on the board point this at the
                held asset.
            fixed_asset_offset: Keypoint on that asset the sample is centered on.
            pose_range_b: Per-axis ``(min, max)`` sample ranges about the keypoint, keys
                ``x``/``y``/``z`` [m] and ``roll``/``pitch``/``yaw`` [rad].
            robot_ik_cfg: The robot, its IK joints and the body being solved for.
            robot_ik_body_offset: Grasp frame of the gripper relative to that body.
            upright_gripper: Keep only the keypoint's yaw, so the sample is taken about a frame
                that is upright in the robot root frame rather than tilted with the asset.
            ik_iterations: ``(lo, hi)`` range the per-env iteration count is drawn from. Each
                iteration takes a quarter step, so the residual shrinks by ~0.75 per iteration.
            pose_tolerance: ``(position [m], orientation [rad])`` residual above which the solve is
                reported through :attr:`is_valid` as having missed. ``None`` reports every solve as
                valid, keeping whatever pose the iteration budget reached.
        """
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_keypoint_pos_w, fixed_keypoint_quat_w = self.variants.apply(
            self.fixed_asset_offset, self.fixed_asset, env_ids
        )
        ranges = self.variants.pose_range(self.ranges, env_ids) if isinstance(self.ranges, str) else self.ranges
        samples = math_utils.sample_uniform(ranges[..., 0], ranges[..., 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        robot_root_pos_w = self.robot.data.root_link_pos_w.torch[env_ids]
        robot_root_quat_w = self.robot.data.root_link_quat_w.torch[env_ids]
        grasp_reference_quat_w = fixed_keypoint_quat_w
        if upright_gripper:
            grasp_reference_quat_b = math_utils.quat_mul(math_utils.quat_inv(robot_root_quat_w), grasp_reference_quat_w)
            # The jaws close along the reference frame's y, so its x has to lie along the part's
            # long axis -- its local z for every variant here -- flattened into the horizontal
            # plane. Reading the frame's own yaw instead only agrees with that while the part
            # stands upright: once it is lying down the euler split puts the tilt in roll or pitch
            # depending on which way it fell, the yaw comes out zero either way, and the gripper
            # ends up closing along the part as often as across it. A part still standing on end
            # projects to nothing, leaving the yaw free, which is what it should be.
            long_axis_b = math_utils.quat_apply(
                grasp_reference_quat_b, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(len(env_ids), 3)
            )
            yaw = torch.atan2(long_axis_b[:, 1], long_axis_b[:, 0])
            zero = torch.zeros_like(yaw)
            grasp_reference_quat_w = math_utils.quat_mul(
                robot_root_quat_w, math_utils.quat_from_euler_xyz(zero, zero, yaw)
            )
        # for those non_reset_id, we will let ik solve for its current position
        grasp_pos_w, grasp_quat_w = math_utils.combine_frame_transforms(
            fixed_keypoint_pos_w,
            grasp_reference_quat_w,
            samples[:, :3],
            math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
        )
        body_pos_w, body_quat_w = self.robot_ik_body_offset.subtract(grasp_pos_w, grasp_quat_w)

        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            robot_root_pos_w,
            robot_root_quat_w,
            body_pos_w,
            body_quat_w,
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
                self.robot.actuators.target_command.position.torch[env_ids] - self.robot.data.joint_pos.torch[env_ids]
            )
            new_joint_pos = (delta_joint_pos + self.robot.data.joint_pos.torch[env_ids])[:, self.joint_ids]
            self.robot.write_joint_position_to_sim_index(
                position=torch.clamp(new_joint_pos, limits[..., 0], limits[..., 1]),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )

        # A target near a workspace edge, a singularity, or a joint limit is one the DLS steps
        # never reach however long they run, so the iteration budget alone does not guarantee the
        # gripper ended up where it was sent. Measure the residual and let the caller drop the env.
        if pose_tolerance is None:
            self.is_valid[env_ids] = True
        else:
            reached_pos_b, reached_quat_b = self.solver._compute_frame_pose()
            pos_error = torch.norm(reached_pos_b[env_ids] - pos_b[env_ids], dim=-1)
            rot_error = math_utils.quat_error_magnitude(reached_quat_b[env_ids], quat_b[env_ids])
            self.is_valid[env_ids] = (pos_error < pose_tolerance[0]) & (rot_error < pose_tolerance[1])

        # wrist_low  = self.robot.data.joint_pos_limits[env_ids, self.wrist_idx, 0]
        # wrist_high = self.robot.data.joint_pos_limits[env_ids, self.wrist_idx, 1]
        # wrist_pos = (wrist_low + (wrist_high - wrist_low) * torch.rand_like(wrist_low)).view(len(env_ids), -1)
        # self.robot.write_joint_position_to_sim(position=wrist_pos, joint_ids=self.wrist_idx, env_ids=env_ids)
        if self.is_physx:
            self.robot.root_physx_view.get_jacobians()


# @torch.jit.script
