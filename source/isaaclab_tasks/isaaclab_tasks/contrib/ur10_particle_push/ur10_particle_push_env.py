# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based UR10 task for sweeping granular media into a side-mounted bin."""

from __future__ import annotations

import math
from collections.abc import Sequence

import newton
import torch
import torch.nn.functional as F
import warp as wp
from isaaclab_newton.cloner import copy_newton_clone_source
from isaaclab_newton.ik import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
    NewtonIKSolver,
    NewtonIKSolverCfg,
)
from isaaclab_newton.physics import NewtonMPMManager

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply

from . import mdp
from .reset_randomization import (
    PushResetPoseBank,
    build_particle_reset,
    build_reset_targets,
)
from .ur10_particle_push_env_cfg import (
    HEIGHTMAP_VISUALIZATION_SHAPE,
    PUSH_ACTION_DIM,
    UR10ParticlePushEnvCfg,
    configure_sparse_mpm_capacities,
)


@wp.kernel(enable_backward=False)
def _mark_penetrating_reset_contacts(
    contact_count: wp.array(dtype=wp.int32),
    contact_max: int,
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_body: wp.array(dtype=wp.int32),
    shape_table: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_is_robot: wp.array(dtype=wp.int32),
    body_tests_table: wp.array(dtype=wp.int32),
    penetration_tolerance: float,
    colliding_worlds: wp.array(dtype=wp.int32),
):
    """Mark reset worlds with penetrating robot self-contact or distal table contact."""
    contact_index = wp.tid()
    if contact_index >= contact_max or contact_index >= contact_count[0]:
        return
    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    table0 = shape_table[shape0] != 0
    table1 = shape_table[shape1] != 0

    robot_body = -1
    if table0 and body1 >= 0:
        robot_body = body1
    elif table1 and body0 >= 0:
        robot_body = body0
    elif not table0 and not table1 and body0 >= 0 and body1 >= 0:
        # The cloned source contains the complete workcell, including touching table/bin
        # members. Only robot-robot pairs are self-collision candidates; scene-scene and
        # robot-scene pairs are outside this dedicated self/table reset screen.
        if body_is_robot[body0] == 0 or body_is_robot[body1] == 0:
            return
        world0 = body_world[body0]
        world1 = body_world[body1]
        if world0 < 0 or world0 != world1:
            return
        point0_w = wp.transform_point(body_q[body0], contact_point0[contact_index])
        point1_w = wp.transform_point(body_q[body1], contact_point1[contact_index])
        separation = wp.dot(contact_normal[contact_index], point1_w - point0_w)
        separation = separation - contact_margin0[contact_index] - contact_margin1[contact_index]
        if separation < -penetration_tolerance:
            wp.atomic_max(colliding_worlds, world0, 1)
        return
    else:
        return

    if body_tests_table[robot_body] == 0:
        return
    world = body_world[robot_body]
    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    if body0 >= 0:
        transform0 = body_q[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
    point0_w = wp.transform_point(transform0, contact_point0[contact_index])
    point1_w = wp.transform_point(transform1, contact_point1[contact_index])
    separation = wp.dot(contact_normal[contact_index], point1_w - point0_w)
    separation = separation - contact_margin0[contact_index] - contact_margin1[contact_index]
    if separation < -penetration_tolerance:
        wp.atomic_max(colliding_worlds, world, 1)


class UR10ParticlePushEnv(ManagerBasedRLEnv):
    """Sweep an initially randomized granular pile into an open side bin."""

    cfg: UR10ParticlePushEnvCfg

    def __init__(self, cfg: UR10ParticlePushEnvCfg, render_mode: str | None = None, **kwargs):
        # Re-resolve capacities after any CLI/Hydra num-environment override.
        configure_sparse_mpm_capacities(cfg)
        self._particle_max_velocity = float(cfg.particle_max_velocity)
        self._heightmap_visualizer: VisualizationMarkers | None = None
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.heightmap_visualizer_cfg is not None:
            self._setup_heightmap_visualizer()

    def load_managers(self) -> None:
        """Bind task state before constructing terms that infer their tensor shapes."""
        self._setup_task_state()
        super().load_managers()

    def _setup_task_state(self) -> None:
        """Resolve scene assets and allocate state shared by the MDP terms."""
        NewtonMPMManager.get_model().particle_max_velocity = self._particle_max_velocity

        self._robot = self.scene["robot"]
        self._media = self.scene["media"]

        joint_ids, joint_names = self._robot.find_joints(
            tuple(self.cfg.scene.robot.init_state.joint_pos),
            preserve_order=True,
            as_proxy=True,
        )
        if len(joint_ids) != PUSH_ACTION_DIM:
            raise RuntimeError(f"Expected six UR10 arm joints, resolved {joint_names}.")
        self._joint_ids = joint_ids.torch
        ee_ids, _ = self._robot.find_bodies(self.cfg.ee_body_name, preserve_order=True)
        if len(ee_ids) != 1:
            raise RuntimeError(f"Expected one {self.cfg.ee_body_name!r} body, found {len(ee_ids)}.")
        self._ee_body_id = ee_ids[0]

        self._joint_velocity_limits = self._robot.data.soft_joint_vel_limits.torch[:, self._joint_ids].clamp_min(1.0e-6)
        self._particle_workspace_lower = torch.tensor(
            self.cfg.particle_workspace_lower_bound,
            device=self.device,
        )
        self._particle_workspace_upper = torch.tensor(
            self.cfg.particle_workspace_upper_bound,
            device=self.device,
        )
        source_shape_profiles = torch.tensor(
            self.cfg.reset_source_shape_profiles,
            dtype=torch.float32,
            device=self.device,
        )
        self._reset_source_vertical_cell_count = source_shape_profiles[:, 0].long()
        self._reset_source_footprint_aspect_ratio = source_shape_profiles[:, 1]
        self._reset_source_shape_cycle_cursor = 0
        self._paddle_offset = torch.tensor(self.cfg.paddle_offset, device=self.device).expand(self.num_envs, -1)
        self._paddle_local_normal = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, -1)
        self._paddle_local_vertical = torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(self.num_envs, -1)
        self._bin_mouth = torch.tensor(
            (self.cfg.bin_inner_x_bounds[0], 0.0, 0.0),
            device=self.device,
        ).expand(self.num_envs, -1)
        self._paddle_position_center = torch.tensor((0.6, 0.0, 0.17), device=self.device)
        self._paddle_position_scale = torch.tensor((0.9, 0.55, 0.65), device=self.device)

        self._particle_reset_template_e = (
            self._media.data.default_particle_state_w.torch[..., :3] - self.scene.env_origins[:, None, :]
        ).clone()
        self.bin_fraction = torch.zeros(self.num_envs, device=self.device)
        self.spill_fraction = torch.zeros_like(self.bin_fraction)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transport_progress = torch.zeros_like(self.bin_fraction)
        self._rms_particle_speed = torch.zeros_like(self.bin_fraction)
        self.invalid_state = torch.zeros_like(self.success_this_step)
        self.escaped_workspace = torch.zeros_like(self.success_this_step)
        self.excessive_spill = torch.zeros_like(self.success_this_step)
        self._task_state_step = -1

        height, width = self.cfg.heightmap_shape
        history_length = self.cfg.heightmap_history_steps + 1
        self._heightmap_xy_offset = torch.zeros((self.num_envs, 2), device=self.device)
        self._heightmap_history = torch.zeros((self.num_envs, history_length, height, width), device=self.device)
        self._heightmap_history_pointer = -1
        self._heightmap_history_step = -1
        self._heightmap_history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._heightmap_history_reset_pending = True
        self._heightmap_goal_mask = mdp.build_bin_goal_mask(self.cfg, device=self.device).expand(self.num_envs, -1, -1)
        self._success_dwell_steps = max(1, math.ceil(self.cfg.success_dwell_time_s / self.step_dt))

        self._episode_start_centroid_x = torch.full(
            (self.num_envs,),
            self.cfg.pile_nominal_center[0],
            device=self.device,
        )

        paddle_position_e, paddle_quaternion, pile_center_xy, curriculum_level = build_reset_targets(
            self.cfg,
            device=self.device,
        )
        self._reset_pose_bank = PushResetPoseBank(
            joint_position=self._solve_reset_joint_positions(paddle_position_e, paddle_quaternion),
            pile_center_xy=pile_center_xy,
            curriculum_level=curriculum_level,
        )
        self._reset_pose_cycle_cursor = 0

    def _collision_free_reset_candidates(
        self,
        prototype_builder: newton.ModelBuilder,
        candidate_q: torch.Tensor,
    ) -> torch.Tensor:
        """Screen every IK endpoint against imported self-collision and the work surface."""
        if candidate_q.ndim != 3 or candidate_q.shape[2] != prototype_builder.joint_coord_count:
            raise ValueError(
                "Reset IK candidates must have shape (rows, candidates, prototype coordinates), "
                f"got {tuple(candidate_q.shape)}."
            )
        row_count, candidate_count, coordinate_count = candidate_q.shape
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=0.0,
            has_shape_collision=True,
            has_particle_collision=False,
            is_visible=False,
        )
        validation_builder = newton.ModelBuilder(up_axis=prototype_builder.up_axis)
        for row in range(row_count):
            validation_builder.begin_world(label=f"push_reset_{row}")
            validation_builder.add_builder(prototype_builder)
            validation_builder.add_shape_plane(
                -1,
                xform=wp.transform_identity(),
                width=0.0,
                length=0.0,
                cfg=shape_cfg,
                label="PushReset/WorkSurface",
            )
            validation_builder.end_world()
        validation_model = validation_builder.finalize(device=self.device)
        if validation_model.world_count != row_count:
            raise RuntimeError(f"Expected {row_count} reset-validation worlds, got {validation_model.world_count}.")
        validation_coordinate_count = validation_model.joint_coord_count // row_count
        if validation_coordinate_count != coordinate_count:
            raise RuntimeError(
                "Reset-validation coordinate count does not match the IK prototype: "
                f"{validation_coordinate_count} != {coordinate_count}."
            )

        body_names = [str(label).rsplit("/", 1)[-1] for label in validation_model.body_label]
        body_is_robot = torch.as_tensor(
            ["/Robot/" in str(label) for label in validation_model.body_label],
            device=self.device,
            dtype=torch.int32,
        )
        table_test_bodies = {
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            self.cfg.ee_body_name,
            "Paddle",
        }
        body_tests_table = torch.as_tensor(
            [name in table_test_bodies for name in body_names],
            device=self.device,
            dtype=torch.int32,
        )
        shape_table = torch.as_tensor(
            [str(label).endswith("PushReset/WorkSurface") for label in validation_model.shape_label],
            device=self.device,
            dtype=torch.int32,
        )
        if int(shape_table.sum()) != row_count:
            raise RuntimeError("Reset collision validation did not create one work surface per row.")

        pipeline = newton.CollisionPipeline(
            validation_model,
            broad_phase="explicit",
            include_static_kinematic_pairs=False,
            soft_contact_max=0,
            verify_buffers=True,
        )
        contacts = pipeline.contacts()
        state = validation_model.state()
        validation_q = wp.to_torch(validation_model.joint_q).reshape(row_count, coordinate_count)
        colliding_worlds = wp.zeros(row_count, dtype=wp.int32, device=validation_model.device)
        collision_free = torch.ones((row_count, candidate_count), dtype=torch.bool, device=self.device)
        for candidate_index in range(candidate_count):
            validation_q.copy_(candidate_q[:, candidate_index].contiguous())
            newton.eval_fk(
                validation_model,
                validation_model.joint_q,
                validation_model.joint_qd,
                state,
            )
            pipeline.collide(state, contacts)
            colliding_worlds.zero_()
            wp.launch(
                _mark_penetrating_reset_contacts,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_max,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    validation_model.shape_body,
                    wp.from_torch(shape_table, dtype=wp.int32),
                    validation_model.body_world,
                    state.body_q,
                    wp.from_torch(body_is_robot, dtype=wp.int32),
                    wp.from_torch(body_tests_table, dtype=wp.int32),
                    1.0e-4,
                ],
                outputs=[colliding_worlds],
                device=validation_model.device,
            )
            collision_free[:, candidate_index] = wp.to_torch(colliding_worlds) == 0
            wp.synchronize_device(validation_model.device)
            generated_contact_count = int(wp.to_torch(contacts.rigid_contact_count)[0])
            if generated_contact_count > contacts.rigid_contact_max:
                raise RuntimeError(
                    f"Reset validation generated {generated_contact_count} contacts for capacity "
                    f"{contacts.rigid_contact_max}."
                )
        return collision_free

    def _solve_reset_joint_positions(
        self,
        paddle_position_e: torch.Tensor,
        paddle_quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one small, canonical-branch, collision-screened reset pose bank."""
        pose_count = self.cfg.reset_pose_count
        if paddle_position_e.shape != (pose_count, 3):
            raise ValueError(
                f"Reset paddle positions have shape {tuple(paddle_position_e.shape)}; expected {(pose_count, 3)}."
            )
        if paddle_quaternion.shape != (pose_count, 4):
            raise ValueError(
                f"Reset paddle quaternions have shape {tuple(paddle_quaternion.shape)}; expected {(pose_count, 4)}."
            )
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, self._robot.cfg.prim_path) if plan is not None else None
        if resolved is None:
            raise RuntimeError(f"Could not resolve clone-plan source for {self._robot.cfg.prim_path!r}.")
        source_path = resolved[0]
        prototype_origin = -self.scene.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*prototype_origin.tolist()), wp.quat_identity())

        prototype_builder = copy_newton_clone_source(source_path, xform=prototype_xform)
        model = prototype_builder.finalize(device=self.device)

        ee_matches = [
            body_id
            for body_id, label in enumerate(model.body_label)
            if str(label).rsplit("/", 1)[-1] == self.cfg.ee_body_name
        ]
        if len(ee_matches) != 1:
            raise RuntimeError(f"Expected one {self.cfg.ee_body_name!r} body in the IK prototype, found {ee_matches}.")
        ee_body_id = ee_matches[0]
        joint_labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
        joint_q_start = wp.to_torch(model.joint_q_start).to(device=self.device, dtype=torch.long)

        def coordinate_id(joint_name: str) -> int:
            matches = [joint_id for joint_id, label in enumerate(joint_labels) if label == joint_name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in the IK prototype, found {matches}.")
            return int(joint_q_start[matches[0]].item())

        arm_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in self.cfg.scene.robot.init_state.joint_pos],
            device=self.device,
            dtype=torch.long,
        )
        home_arm_q = self._robot.data.default_joint_pos.torch[0, self._joint_ids]
        seed_q = wp.to_torch(model.joint_q).to(device=self.device, dtype=torch.float32).clone()
        seed_q[arm_coordinate_ids] = home_arm_q

        target_name = "reset_paddle"
        seed_count = self.cfg.reset_ik_seeds
        solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=seed_count,
                noise_std=self.cfg.reset_ik_noise_std,
                rng_seed=self.cfg.reset_seed,
                iterations=self.cfg.reset_ik_iterations,
                lambda_initial=0.1,
            ),
            model=model,
            num_envs=pose_count,
            device=str(model.device),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name=self.cfg.ee_body_name,
                    name=target_name,
                    body_offset_pos=self.cfg.paddle_offset,
                    position_weight=100.0,
                    rotation_weight=10.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ],
            link_resolver=lambda _body_name: ee_body_id,
        )
        pose_objective = solver.objectives_by_name[target_name]
        pose_objective.position_objective.set_target_positions(
            wp.from_torch(paddle_position_e.contiguous(), dtype=wp.vec3)
        )
        pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(paddle_quaternion.contiguous(), dtype=wp.vec4)
        )

        joint_limits = self._robot.data.soft_joint_pos_limits.torch[0, self._joint_ids]
        solver.solve(
            wp.from_torch(
                seed_q.expand(pose_count, -1).contiguous(),
                dtype=wp.float32,
            )
        )
        candidate_q = wp.to_torch(solver.joint_q).reshape(pose_count, seed_count, -1).clone()
        candidate_cost = wp.to_torch(solver.costs).reshape(pose_count, seed_count).clone()
        arm_q = candidate_q[..., arm_coordinate_ids]
        joint_margin = torch.minimum(
            arm_q - joint_limits[:, 0],
            joint_limits[:, 1] - arm_q,
        ).amin(dim=-1)
        valid = (
            torch.isfinite(arm_q).all(dim=-1)
            & torch.isfinite(candidate_cost)
            & (candidate_cost <= self.cfg.reset_ik_max_cost)
            & (joint_margin >= self.cfg.actions.arm_action.joint_limit_margin)
        )
        valid &= self._collision_free_reset_candidates(prototype_builder, candidate_q)
        missing = torch.nonzero(~valid.any(dim=1), as_tuple=False).flatten()
        if missing.numel() > 0:
            raise RuntimeError(f"Newton IK found no canonical collision-free reset pose for rows {missing.tolist()}.")

        # Anchor every target to the same nominal branch. Cost breaks ties between nearby seeds;
        # squared home distance prevents equivalent UR10 revolutions from entering the reset bank.
        home_distance = torch.square(arm_q - home_arm_q).sum(dim=-1)
        score = candidate_cost + 1.0e-4 * home_distance
        score = torch.where(valid, score, torch.full_like(score, torch.inf))
        selected = score.argmin(dim=1)
        row_ids = torch.arange(pose_count, device=self.device)
        joint_position = arm_q[row_ids, selected].contiguous()
        lower = joint_limits[:, 0] + self.cfg.actions.arm_action.joint_limit_margin
        upper = joint_limits[:, 1] - self.cfg.actions.arm_action.joint_limit_margin
        if bool(torch.any((joint_position < lower) | (joint_position > upper))):
            raise RuntimeError("Selected reset pose violates the configured joint-limit margin.")
        return joint_position

    def _sample_reset_pose_ids(self, curriculum_level: torch.Tensor) -> torch.Tensor:
        """Sample collision-screened robot starts from the requested curriculum levels."""
        count = curriculum_level.numel()
        level_count = len(self.cfg.reset_randomization_scales)
        poses_per_level = self._reset_pose_bank.row_count // level_count
        if bool(torch.any((curriculum_level < 0) | (curriculum_level >= level_count))):
            raise ValueError("Reset curriculum levels lie outside the pose bank.")
        if self.cfg.reset_cycle:
            pose_offset = (torch.arange(count, device=self.device) + self._reset_pose_cycle_cursor).remainder(
                poses_per_level
            )
            self._reset_pose_cycle_cursor = (self._reset_pose_cycle_cursor + count) % poses_per_level
        else:
            pose_offset = torch.randint(poses_per_level, (count,), device=self.device)
        pose_ids = curriculum_level * poses_per_level + pose_offset
        if not torch.equal(self._reset_pose_bank.curriculum_level[pose_ids], curriculum_level):
            raise RuntimeError("Reset pose bank is not grouped by curriculum level.")
        return pose_ids

    @property
    def arm_action(self) -> mdp.ClampedRelativeJointPositionAction:
        """Arm joint-delta action term."""
        return self.action_manager.get_term("arm_action")

    def update_task_state(self) -> None:
        """Update shared termination, reward, and critic metrics once per policy step."""
        if self._task_state_step == self.common_step_counter:
            return
        particle_position_e = self._particle_position_e()
        particle_velocity_e = self._media.data.particle_vel_w.torch
        joint_position = self._robot.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._robot.data.joint_vel.torch[:, self._joint_ids]
        ee_pose_w = self._robot.data.body_link_pose_w.torch[:, self._ee_body_id]
        ee_velocity_w = self._robot.data.body_link_vel_w.torch[:, self._ee_body_id]
        finite_particle_slots = torch.isfinite(particle_position_e).all(dim=2) & torch.isfinite(
            particle_velocity_e
        ).all(dim=2)
        finite_particles = finite_particle_slots.all(dim=1)
        finite_robot = (
            torch.isfinite(joint_position).all(dim=1)
            & torch.isfinite(joint_velocity).all(dim=1)
            & torch.isfinite(ee_pose_w).all(dim=1)
            & torch.isfinite(ee_velocity_w).all(dim=1)
        )
        safe_position_e = torch.nan_to_num(particle_position_e, nan=0.0, posinf=10.0, neginf=-10.0)
        safe_velocity_e = torch.nan_to_num(particle_velocity_e, nan=0.0, posinf=20.0, neginf=-20.0)
        particle_speed = torch.linalg.vector_norm(safe_velocity_e, dim=-1)
        self._rms_particle_speed = torch.sqrt(torch.square(particle_speed).mean(dim=1))

        self.bin_fraction, self.spill_fraction = mdp.compute_particle_metrics(safe_position_e, self.cfg)
        self.excessive_spill = self.spill_fraction > self.cfg.failure_max_spill_fraction
        self.transport_progress = mdp.compute_transport_progress(
            safe_position_e,
            self.cfg,
            self._episode_start_centroid_x,
        )

        escaped_particle_fraction = (
            ((safe_position_e < self._particle_workspace_lower) | (safe_position_e > self._particle_workspace_upper))
            .any(dim=2)
            .float()
            .mean(dim=1)
        )
        self.escaped_workspace = escaped_particle_fraction > self.cfg.max_escaped_particle_fraction
        safe_ee_pose_w = torch.nan_to_num(ee_pose_w, nan=0.0, posinf=10.0, neginf=-10.0)
        joint_limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        joint_position_in_bounds = (
            (joint_position >= joint_limits[..., 0] - self.cfg.state_bound_joint_position_margin)
            & (joint_position <= joint_limits[..., 1] + self.cfg.state_bound_joint_position_margin)
        ).all(dim=1)
        joint_velocity_in_bounds = (joint_velocity.abs() <= self.cfg.state_bound_max_joint_velocity).all(dim=1)
        ee_linear_velocity_in_bounds = (
            torch.linalg.vector_norm(ee_velocity_w[:, :3], dim=1) <= self.cfg.state_bound_max_ee_linear_velocity
        )
        ee_angular_velocity_in_bounds = (
            torch.linalg.vector_norm(ee_velocity_w[:, 3:], dim=1) <= self.cfg.state_bound_max_ee_angular_velocity
        )
        valid_quaternion = (torch.linalg.vector_norm(safe_ee_pose_w[:, 3:7], dim=1) - 1.0).abs() < 0.05
        bounded_robot = (
            joint_position_in_bounds
            & joint_velocity_in_bounds
            & ee_linear_velocity_in_bounds
            & ee_angular_velocity_in_bounds
        )
        self.invalid_state = ~(finite_particles & finite_robot & valid_quaternion & bounded_robot)
        self.invalid_state |= self.arm_action.invalid_actions
        self._success_streak, self.success_this_step = mdp.update_success_streak(
            self._success_streak,
            self.bin_fraction,
            self.spill_fraction,
            success_fraction=self.cfg.success_fraction,
            max_spill_fraction=self.cfg.success_max_spill_fraction,
            dwell_steps=self._success_dwell_steps,
        )
        # Terminate successful episodes or states that violate the numerical-safety envelope.
        unsafe = self.invalid_state | self.escaped_workspace | self.excessive_spill
        self._success_streak = torch.where(unsafe, torch.zeros_like(self._success_streak), self._success_streak)
        self.success_this_step &= ~unsafe
        self._task_state_step = self.common_step_counter

    def policy_observation(self) -> torch.Tensor:
        """Return the 30-value actor vector."""
        joint_position = torch.nan_to_num(
            self._robot.data.joint_pos.torch[:, self._joint_ids],
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        joint_velocity = torch.nan_to_num(
            self._robot.data.joint_vel.torch[:, self._joint_ids],
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        midpoint = 0.5 * (limits[..., 0] + limits[..., 1])
        half_range = (0.5 * (limits[..., 1] - limits[..., 0])).clamp_min(1.0e-6)
        normalized_joint_position = ((joint_position - midpoint) / half_range).clamp(-1.0, 1.0)
        normalized_joint_velocity = (joint_velocity / self._joint_velocity_limits).clamp(-1.0, 1.0)

        paddle_position_e, paddle_normal_e, paddle_vertical_e = self._paddle_pose_e()
        normalized_paddle_position = (
            (paddle_position_e - self._paddle_position_center) / self._paddle_position_scale
        ).clamp(-2.0, 2.0)
        normalized_paddle_to_bin = ((self._bin_mouth - paddle_position_e) / self._paddle_position_scale).clamp(
            -2.0, 2.0
        )

        policy = torch.cat(
            (
                normalized_joint_position,
                normalized_joint_velocity,
                self.arm_action.raw_actions,
                normalized_paddle_position,
                paddle_normal_e,
                paddle_vertical_e,
                normalized_paddle_to_bin,
            ),
            dim=-1,
        )
        return torch.nan_to_num(policy, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)

    def heightmap_observation(self) -> torch.Tensor:
        """Return the three-channel overhead heightmap."""
        particle_position_e = torch.nan_to_num(
            self._particle_position_e(),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        heightmap = self._heightmap_observation(particle_position_e)
        heightmap = torch.nan_to_num(heightmap, nan=0.0, posinf=1.0, neginf=0.0)
        self._update_heightmap_visualizer(heightmap)
        return heightmap

    def _setup_heightmap_visualizer(self) -> None:
        """Create the optional policy-heightmap markers."""
        rows, columns = HEIGHTMAP_VISUALIZATION_SHAPE
        x_lo, x_hi = self.cfg.heightmap_x_bounds
        y_lo, y_hi = self.cfg.heightmap_y_bounds
        x = torch.linspace(
            x_lo + 0.5 * (x_hi - x_lo) / columns,
            x_hi - 0.5 * (x_hi - x_lo) / columns,
            columns,
            device=self.device,
        )
        y = torch.linspace(
            y_lo + 0.5 * (y_hi - y_lo) / rows,
            y_hi - 0.5 * (y_hi - y_lo) / rows,
            rows,
            device=self.device,
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1).reshape(1, -1, 3)
        self._heightmap_visualizer_positions = positions.repeat(self.num_envs, 1, 1)
        self._heightmap_visualizer_positions.add_(self.scene.env_origins[:, None, :])
        self._heightmap_visualizer_scales = torch.zeros_like(self._heightmap_visualizer_positions)
        self._heightmap_visualizer_radius = 0.2 * min((x_hi - x_lo) / columns, (y_hi - y_lo) / rows)
        self._heightmap_visualizer = VisualizationMarkers(self.cfg.heightmap_visualizer_cfg)
        self._heightmap_visualizer.set_visibility(False)

    def _update_heightmap_visualizer(self, heightmap: torch.Tensor) -> None:
        """Update visible markers from the current policy-heightmap channel."""
        if self._heightmap_visualizer is None or not self._heightmap_visualizer.is_visible():
            return
        values = F.adaptive_max_pool2d(heightmap[:, :1], HEIGHTMAP_VISUALIZATION_SHAPE).flatten(1)
        self._heightmap_visualizer_positions[:, :, 2].copy_(
            self.scene.env_origins[:, 2, None]
            + self.cfg.heightmap_z_min
            + values * self.cfg.heightmap_z_range
            + self._heightmap_visualizer_radius
        )
        self._heightmap_visualizer_scales.copy_((values > 0.0)[..., None] * self._heightmap_visualizer_radius)
        self._heightmap_visualizer.visualize(
            translations=self._heightmap_visualizer_positions.flatten(0, 1),
            scales=self._heightmap_visualizer_scales.flatten(0, 1),
        )

    def critic_observation(self) -> torch.Tensor:
        """Return the 11-value privileged vector."""
        self.update_task_state()
        particle_position_e = torch.nan_to_num(
            self._particle_position_e(),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        particle_velocity_e = torch.nan_to_num(
            self._media.data.particle_vel_w.torch,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-self._particle_max_velocity, self._particle_max_velocity)
        # Clamp each point before reduction: a single large-but-finite escaped particle must not
        # dominate the privileged centroid and produce an unbounded critic target.
        bounded_particle_position_e = torch.maximum(
            torch.minimum(particle_position_e, self._particle_workspace_upper),
            self._particle_workspace_lower,
        )
        centroid_position = bounded_particle_position_e.mean(dim=1)
        centroid_velocity = particle_velocity_e.mean(dim=1)
        normalized_centroid = (centroid_position - torch.tensor((0.6, 0.0, 0.05), device=self.device)) / torch.tensor(
            (0.9, 0.55, 0.50), device=self.device
        )
        privileged = torch.cat(
            (
                self.bin_fraction[:, None],
                self.spill_fraction[:, None],
                normalized_centroid.clamp(-2.0, 2.0),
                (centroid_velocity / self._particle_max_velocity).clamp(-1.0, 1.0),
                (self._rms_particle_speed / self.cfg.success_max_rms_particle_speed).clamp(0.0, 4.0)[:, None],
                self.transport_progress[:, None],
                (self._success_streak.float() / self._success_dwell_steps).clamp(0.0, 1.0)[:, None],
            ),
            dim=-1,
        )
        return torch.nan_to_num(privileged, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)

    def randomize_push_scene(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Randomize a collision-screened robot start and its nearby single pile."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        reset_count = len(env_ids)
        curriculum = self.curriculum_manager.cfg.reset_randomization.func
        if not isinstance(curriculum, mdp.SinglePushCurriculum):
            raise TypeError("The push reset event requires SinglePushCurriculum.")
        curriculum_level = curriculum.levels[env_ids]
        pose_ids = self._sample_reset_pose_ids(curriculum_level)
        randomization_scale = torch.as_tensor(
            self.cfg.reset_randomization_scales,
            device=self.device,
        )[curriculum_level]
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_velocity = self._robot.data.default_root_vel.torch[env_ids].clone()
        joint_position = self._reset_pose_bank.joint_position[pose_ids]
        joint_velocity = torch.zeros_like(joint_position)

        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_velocity, env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(
            position=joint_position,
            velocity=joint_velocity,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=joint_position,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self.arm_action.set_reset_position(joint_position, env_ids=env_ids)
        # Consume FK invalidation before MPM snapshots the reset paddle transform.
        _ = self._robot.data.body_link_pose_w

        pile_yaw = (
            (2.0 * torch.rand(reset_count, device=self.device) - 1.0)
            * self.cfg.reset_particle_max_yaw
            * randomization_scale
        )
        shape_profile_count = self._reset_source_vertical_cell_count.numel()
        if self.cfg.reset_cycle:
            source_shape_profile = (
                torch.arange(reset_count, dtype=torch.long, device=self.device) + self._reset_source_shape_cycle_cursor
            ).remainder(shape_profile_count)
            self._reset_source_shape_cycle_cursor = (
                self._reset_source_shape_cycle_cursor + reset_count
            ) % shape_profile_count
        else:
            source_shape_profile = torch.randint(shape_profile_count, (reset_count,), device=self.device)
        particle_jitter = (
            2.0
            * torch.rand(
                (reset_count, self._media.particles_per_object, 3),
                device=self.device,
            )
            - 1.0
        ) * self.cfg.reset_particle_jitter
        particle_position_e = build_particle_reset(
            self._particle_reset_template_e[env_ids],
            self._reset_pose_bank.pile_center_xy[pose_ids],
            pile_yaw,
            particle_jitter,
            self.cfg,
            vertical_cell_count=self._reset_source_vertical_cell_count[source_shape_profile],
            footprint_aspect_ratio=self._reset_source_footprint_aspect_ratio[source_shape_profile],
        )
        particle_state = self._media.data.default_particle_state_w.torch[env_ids].clone()
        particle_state[..., :3] = particle_position_e + self.scene.env_origins[env_ids, None, :]
        particle_state[..., 3:] = 0.0
        self._media.write_particle_state_to_sim_index(particle_state, env_ids=env_ids)

        # Clear constitutive/contact/collider history for exactly the reset worlds.
        # Newton reset masks include one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[env_ids] = True
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )

        bin_fraction, spill_fraction = mdp.compute_particle_metrics(particle_position_e, self.cfg)
        start_centroid_x = particle_position_e[..., 0].mean(dim=1)
        transport_progress = mdp.compute_transport_progress(
            particle_position_e,
            self.cfg,
            start_centroid_x,
        )
        self.bin_fraction[env_ids] = bin_fraction
        self.spill_fraction[env_ids] = spill_fraction
        self._episode_start_centroid_x[env_ids] = start_centroid_x
        self.transport_progress[env_ids] = transport_progress

        self._success_streak[env_ids] = 0
        self.success_this_step[env_ids] = False
        self._rms_particle_speed[env_ids] = 0.0
        self.invalid_state[env_ids] = False
        self.escaped_workspace[env_ids] = False
        self.excessive_spill[env_ids] = False
        if self.cfg.heightmap_xy_noise_std > 0.0:
            offset = torch.randn((reset_count, 2), device=self.device) * self.cfg.heightmap_xy_noise_std
            self._heightmap_xy_offset[env_ids] = offset.clamp(
                -3.0 * self.cfg.heightmap_xy_noise_std,
                3.0 * self.cfg.heightmap_xy_noise_std,
            )
        else:
            self._heightmap_xy_offset[env_ids] = 0.0
        self._heightmap_history_reset[env_ids] = True
        self._heightmap_history_reset_pending = True
        self._task_state_step = self.common_step_counter

    def _particle_position_e(self) -> torch.Tensor:
        return self._media.data.particle_pos_w.torch - self.scene.env_origins[:, None, :]

    def _paddle_pose_e(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_pose_w = self._robot.data.body_link_pose_w.torch[:, self._ee_body_id]
        paddle_position_w = ee_pose_w[:, :3] + quat_apply(ee_pose_w[:, 3:7], self._paddle_offset)
        paddle_position_e = paddle_position_w - self.scene.env_origins
        paddle_normal_e = quat_apply(ee_pose_w[:, 3:7], self._paddle_local_normal)
        paddle_vertical_e = quat_apply(ee_pose_w[:, 3:7], self._paddle_local_vertical)
        return paddle_position_e, paddle_normal_e, paddle_vertical_e

    def _heightmap_observation(self, particle_position_e: torch.Tensor) -> torch.Tensor:
        """Return current/delayed particle surfaces and the calibrated bin mask."""
        new_step = self._heightmap_history_step != self.common_step_counter
        if new_step or self._heightmap_history_reset_pending:
            surface = self._segmented_overhead_heightmap(particle_position_e)
            if new_step:
                self._heightmap_history_pointer = (self._heightmap_history_pointer + 1) % self._heightmap_history.shape[
                    1
                ]
                self._heightmap_history[:, self._heightmap_history_pointer].copy_(surface)
                self._heightmap_history_step = self.common_step_counter
            if self._heightmap_history_reset_pending:
                reset_mask = self._heightmap_history_reset
                self._heightmap_history[reset_mask] = surface[reset_mask, None]
                reset_mask.zero_()
                self._heightmap_history_reset_pending = False

        current = self._heightmap_history[:, self._heightmap_history_pointer]
        delayed_index = (self._heightmap_history_pointer + 1) % self._heightmap_history.shape[1]
        delayed = self._heightmap_history[:, delayed_index]
        return torch.stack((current, delayed, self._heightmap_goal_mask), dim=1)

    def _segmented_overhead_heightmap(self, particle_position_e: torch.Tensor) -> torch.Tensor:
        height_cells, width_cells = self.cfg.heightmap_shape
        cell_count = height_cells * width_cells
        x_lo, x_hi = self.cfg.heightmap_x_bounds
        y_lo, y_hi = self.cfg.heightmap_y_bounds
        x, y, z = particle_position_e.unbind(dim=-1)
        if self.cfg.heightmap_xy_noise_std > 0.0:
            x = x + self._heightmap_xy_offset[:, None, 0]
            y = y + self._heightmap_xy_offset[:, None, 1]

        ix = torch.floor((x - x_lo) * width_cells / (x_hi - x_lo)).long()
        iy = torch.floor((y - y_lo) * height_cells / (y_hi - y_lo)).long()
        valid = (ix >= 0) & (ix < width_cells) & (iy >= 0) & (iy < height_cells) & torch.isfinite(z)
        flat_index = (iy.clamp(0, height_cells - 1) * width_cells + ix.clamp(0, width_cells - 1)).long()

        particle_surface_z = z + float(self.cfg.scene.media.spawn.radius)
        normalized_height = ((particle_surface_z - self.cfg.heightmap_z_min) / self.cfg.heightmap_z_range).clamp(
            0.0, 1.0
        )
        height = torch.zeros((self.num_envs, cell_count), device=self.device)
        height.scatter_reduce_(
            dim=1,
            index=flat_index,
            src=torch.where(valid, normalized_height, torch.zeros_like(normalized_height)),
            reduce="amax",
            include_self=True,
        )
        occupancy = (height > 0.0).to(dtype=height.dtype)

        if self.cfg.heightmap_depth_noise_std > 0.0:
            height = (
                height
                + occupancy
                * torch.randn_like(height)
                * (self.cfg.heightmap_depth_noise_std / self.cfg.heightmap_z_range)
            ).clamp(0.0, 1.0)
        if self.cfg.heightmap_dropout_probability > 0.0:
            keep = torch.rand_like(occupancy) >= self.cfg.heightmap_dropout_probability
            height *= keep

        return height.reshape(self.num_envs, height_cells, width_cells)
