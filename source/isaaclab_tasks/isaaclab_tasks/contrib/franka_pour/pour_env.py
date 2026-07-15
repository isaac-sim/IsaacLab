# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for a Franka pouring MPM media between two cups.

The visible dynamic source cup and kinematic receiving cup are scene-owned rigid objects. Their
USD bowl meshes are visual-only, while the source also owns an invisible rigid grasp proxy for
Newton-generated finger contacts. A narrow per-world Newton hook attaches cached hollow
particle-only colliders to both scene bodies and adds only one hidden solver object: a particle-only
spill floor.

A Newton :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` advances the robot and both cups
in the ``arm`` MJWarp entry and the particles and spill floor in the implicit ``media`` entry. Proxy
coupling makes both cups' particle colliders available to MPM without assigning one body to two
entries. The policy commands arm joint positions and a continuous symmetric finger target; all
observable and reset state flows through the scene assets' public APIs.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.cloner import copy_newton_source_builder, newton_builder_world_hook
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver import NewtonIKSolver
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import NewtonManager

import isaaclab.sim as sim_utils
from isaaclab.cloner import resolve_clone_plan_source
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from .cube_bowl_mesh import cube_bowl_inner_bounds, make_cube_bowl_mesh
from .cup_media import cup_cavity_lattice
from .mdp.terminations import (
    _delivered_particle_mask,
    _particles_in_workspace,
    _rigid_state_in_bounds,
    _spilled_particle_mask,
    _state_finite,
)
from .reset_dataset_generator import (
    GRASPING_CATEGORY,
    build_franka_pour_reset_task_contract,
    validate_production_reset_dataset,
    validate_reset_dataset,
)
from .reset_utils import (
    asymmetric_reset_offset_samples,
    balanced_cyclic_permutations,
    boolean_selection_mask,
    polar_workspace_cells,
    reset_rotation_vector_samples,
    sample_index_pools,
    scale_randomization_rows_by_extent,
    target_xy_behind_source,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab_newton.assets import MPMObject

    from .pour_env_cfg import FrankaPourEnvCfg

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
_RESET_PATH_SEGMENT_SUBDIVISIONS = 8


@wp.kernel(enable_backward=False)
def _mark_penetrating_self_contacts(
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
    body_world: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    penetration_tolerance: float,
    colliding_worlds: wp.array(dtype=wp.int32),
):
    """Mark replicated IK candidates containing a penetrating robot self-contact."""
    contact_index = wp.tid()
    if contact_index >= contact_max or contact_index >= contact_count[0]:
        return
    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    if body0 < 0 or body1 < 0:
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


class FrankaPourEnv(ManagerBasedRLEnv):
    """Franka grasping a dynamic cup of MPM media (Newton proxy-coupled MPM), pouring by tilting."""

    cfg: FrankaPourEnvCfg

    def __init__(self, cfg: FrankaPourEnvCfg, render_mode: str | None = None, **kwargs):
        resolved_cfg = cfg.finalize()
        self._prepare_newton_extras(resolved_cfg)
        with newton_builder_world_hook(self._add_pour_world_to_builder):
            super().__init__(resolved_cfg, render_mode, **kwargs)

    def load_managers(self) -> None:
        self._setup_after_physics()
        super().load_managers()

    # ------------------------------------------------------------------ build
    def _prepare_newton_extras(self, cfg: FrankaPourEnvCfg) -> None:
        """Bake task-local Newton collision geometry from the resolved scene config.

        Runs before ``super().__init__`` so the per-world builder hook has the
        geometry and contact values available while the scene is imported.
        """
        # Watertight cube-cup collision meshes. The source's outer extents exactly match the solid
        # grasp box, so its visible walls and the rigid finger contacts no longer disagree.
        self._cup_vertices, self._cup_indices = make_cube_bowl_mesh(
            inner_width=float(cfg.source_cup_inner_width),
            inner_depth=float(cfg.source_cup_inner_depth),
            wall_thickness=float(cfg.source_cup_wall_thickness),
            cavity_depth=float(cfg.source_cup_cavity_depth),
            bottom_thickness=float(cfg.source_cup_bottom_thickness),
        )
        self._target_vertices, self._target_indices = make_cube_bowl_mesh(
            inner_width=float(cfg.target_cup_inner_width),
            inner_depth=float(cfg.target_cup_inner_depth),
            wall_thickness=float(cfg.target_cup_wall_thickness),
            cavity_depth=float(cfg.target_cup_cavity_depth),
            bottom_thickness=float(cfg.target_cup_bottom_thickness),
        )
        self._source_collider_mesh = newton.Mesh(
            self._cup_vertices,
            self._cup_indices,
            compute_inertia=False,
            is_solid=False,
        )
        self._target_collider_mesh = newton.Mesh(
            self._target_vertices,
            self._target_indices,
            compute_inertia=False,
            is_solid=False,
        )
        self._source_inner_lo, self._source_inner_hi = cube_bowl_inner_bounds(
            cfg.source_cup_inner_width,
            cfg.source_cup_inner_depth,
            cfg.source_cup_cavity_depth,
            cfg.source_cup_bottom_thickness,
        )
        self._target_inner_lo, self._target_inner_hi = cube_bowl_inner_bounds(
            cfg.target_cup_inner_width,
            cfg.target_cup_inner_depth,
            cfg.target_cup_cavity_depth,
            cfg.target_cup_bottom_thickness,
        )

        # Cup-local media lattice (the env transforms it by the live cup pose every reset).
        self._media_local_points, _ = cup_cavity_lattice(cfg)

        # Cup reset pose (env frame): resting on the table, opening up (cup-local +z is world +z).
        self._cup_reset_pos = np.asarray(cfg.cup_reset_pos, dtype=np.float64)
        self._grasp_contact_ke = float(cfg.grasp_contact_ke)
        self._grasp_contact_kd = float(cfg.grasp_contact_kd)
        self._grasp_contact_kf = float(cfg.grasp_contact_kf)
        self._grasp_contact_mu = float(cfg.cup_grasp_box_friction)

        self._source_cup_friction = float(cfg.source_cup_friction)
        self._target_cup_friction = float(cfg.target_cup_friction)
        self._collider_margin = float(cfg.collider_margin)
        self._particle_max_velocity = float(cfg.particle_max_velocity)

    def _add_pour_world_to_builder(self, builder, env_id: int, position, quaternion) -> None:
        """Add only solver-specific collision representations to one imported scene world."""
        builder.particle_max_velocity = self._particle_max_velocity
        env_root = f"/World/envs/env_{env_id}"
        body_ids = self._current_world_range(builder, "body", env_id)
        shape_ids = self._current_world_range(builder, "shape", env_id)
        self._disable_robot_particle_collision(builder, body_ids, shape_ids)
        self._configure_finger_contact_material(builder, body_ids, shape_ids)

        source_body = self._find_world_body(builder, body_ids, env_id, "SourceCup")
        target_body = self._find_world_body(builder, body_ids, env_id, "TargetCup")
        self._add_kinematic_rigid_object_articulation(builder, target_body)
        grasp_proxy = self._find_world_shape(
            builder,
            shape_ids,
            env_id,
            "/SourceCup/geometry/grasp_proxy",
            body_id=source_body,
        )
        self._configure_grasp_proxy(builder, grasp_proxy)

        self._add_particle_collider(
            builder,
            body_id=source_body,
            mesh=self._source_collider_mesh,
            friction=self._source_cup_friction,
            label=f"{env_root}/SourceCup/ParticleCollider",
        )
        self._add_particle_collider(
            builder,
            body_id=target_body,
            mesh=self._target_collider_mesh,
            friction=self._target_cup_friction,
            label=f"{env_root}/TargetCup/ParticleCollider",
        )

        self._add_rigid_collider(
            builder,
            body_id=target_body,
            mesh=self._target_collider_mesh,
            friction=self._target_cup_friction,
            label=f"{env_root}/TargetCup/Collision",
        )

        world_xform = wp.transform(
            wp.vec3(*[float(value) for value in position]),
            wp.quat(*[float(value) for value in quaternion]),
        )
        spill_floor = builder.add_body(
            xform=world_xform,
            mass=0.0,
            inertia=wp.mat33(),
            is_kinematic=True,
            lock_inertia=True,
            label=f"{env_root}/SpillFloor",
        )
        spill_shape = builder.add_shape_plane(
            body=spill_floor,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=0.8,
                margin=self._collider_margin,
                has_shape_collision=False,
                has_particle_collision=True,
            ),
            color=(0.3, 0.3, 0.3),
            label=f"{env_root}/SpillFloor/Collision",
        )
        self._set_shape_roles(builder, spill_shape, rigid=False, particles=True, visible=False)

    @staticmethod
    def _current_world_range(builder, prefix: str, env_id: int) -> range:
        """Return the contiguous tail added for the currently open Newton world."""
        worlds = getattr(builder, f"{prefix}_world", None)
        if worlds is None:
            raise RuntimeError(f"Newton builder does not expose {prefix}_world assignments.")
        stop = len(worlds)
        start = stop
        while start > 0 and int(worlds[start - 1]) == env_id:
            start -= 1
        if start == stop:
            raise RuntimeError(f"Newton builder contains no {prefix} entries for open world {env_id}.")
        return range(start, stop)

    def _find_world_body(self, builder, body_ids: range, env_id: int, body_name: str) -> int:
        """Resolve exactly one imported body by Newton world and exact final path component."""
        matches = [body_id for body_id in body_ids if str(builder.body_label[body_id]).rsplit("/", 1)[-1] == body_name]
        if len(matches) != 1:
            labels = [str(builder.body_label[index]) for index in matches]
            raise RuntimeError(
                f"Expected exactly one {body_name!r} body in Newton world {env_id}, "
                f"found ids={matches}, labels={labels}."
            )
        return matches[0]

    @staticmethod
    def _add_kinematic_rigid_object_articulation(builder, body_id: int) -> None:
        """Expose an imported kinematic body through Newton's articulation-based rigid view."""
        body_label = str(builder.body_label[body_id])
        child_joints = [joint_id for _, joint_id in builder.joint_parents.get(body_id, ())]
        if not child_joints:
            joint_id = builder.add_joint_free(child=body_id, label=f"{body_label}/FreeJoint")
            builder.add_articulation([joint_id], label=body_label)
        elif len(child_joints) == 1:
            joint_id = child_joints[0]
            articulation_id = int(builder.joint_articulation[joint_id])
            if articulation_id < 0 or str(builder.articulation_label[articulation_id]) != body_label:
                raise RuntimeError(
                    f"Kinematic rigid body {body_label!r} has an unexpected joint/articulation association."
                )
        else:
            raise RuntimeError(
                f"Kinematic rigid body {body_label!r} must have at most one root joint, found {child_joints}."
            )

        builder.body_flags[body_id] = int(newton.BodyFlags.KINEMATIC)
        builder.body_mass[body_id] = 0.0
        builder.body_inv_mass[body_id] = 0.0
        builder.body_inertia[body_id] = wp.mat33()
        builder.body_inv_inertia[body_id] = wp.mat33()

    def _find_world_shape(self, builder, shape_ids: range, env_id: int, label_suffix: str, *, body_id: int) -> int:
        """Resolve exactly one imported shape by owning body and exact scene-relative path."""
        matches = [
            shape_id
            for shape_id in shape_ids
            if int(builder.shape_body[shape_id]) == body_id
            and str(builder.shape_label[shape_id]).endswith(label_suffix)
        ]
        if len(matches) != 1:
            labels = [str(builder.shape_label[index]) for index in matches]
            raise RuntimeError(
                f"Expected exactly one shape ending in {label_suffix!r} on body {body_id} "
                f"in Newton world {env_id}, found ids={matches}, labels={labels}."
            )
        return matches[0]

    def _configure_grasp_proxy(self, builder, shape_id: int) -> None:
        """Keep the imported grasp proxy rigid-only, invisible, and contact-tuned."""
        self._set_shape_roles(builder, shape_id, rigid=True, particles=False, visible=False)
        builder.shape_margin[shape_id] = self._collider_margin
        builder.shape_material_ke[shape_id] = self._grasp_contact_ke
        builder.shape_material_kd[shape_id] = self._grasp_contact_kd
        builder.shape_material_kf[shape_id] = self._grasp_contact_kf
        builder.shape_material_mu[shape_id] = self._grasp_contact_mu

    def _add_particle_collider(
        self,
        builder,
        *,
        body_id: int,
        mesh: newton.Mesh,
        friction: float,
        label: str,
    ) -> int:
        """Attach an invisible hollow particle-only collider to a scene-owned body."""
        shape_id = builder.add_shape_mesh(
            body_id,
            xform=wp.transform_identity(),
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=friction,
                density=0.0,
                margin=self._collider_margin,
                has_shape_collision=False,
                has_particle_collision=True,
                is_visible=False,
            ),
            label=label,
        )
        self._set_shape_roles(builder, shape_id, rigid=False, particles=True, visible=False)
        builder.shape_margin[shape_id] = self._collider_margin
        builder.shape_material_mu[shape_id] = friction
        return shape_id

    def _add_rigid_collider(
        self,
        builder,
        *,
        body_id: int,
        mesh: newton.Mesh,
        friction: float,
        label: str,
    ) -> int:
        """Attach an invisible hollow rigid-only collider to a solver-owned body."""
        shape_id = builder.add_shape_mesh(
            body_id,
            xform=wp.transform_identity(),
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=friction,
                density=0.0,
                ke=self._grasp_contact_ke,
                kd=self._grasp_contact_kd,
                kf=self._grasp_contact_kf,
                margin=self._collider_margin,
                has_shape_collision=True,
                has_particle_collision=False,
                is_visible=False,
            ),
            label=label,
        )
        self._set_shape_roles(builder, shape_id, rigid=True, particles=False, visible=False)
        builder.shape_margin[shape_id] = self._collider_margin
        builder.shape_material_mu[shape_id] = friction
        return shape_id

    @staticmethod
    def _set_shape_roles(builder, shape_id: int, *, rigid: bool, particles: bool, visible: bool) -> None:
        flags = int(builder.shape_flags[shape_id])
        assignments = (
            (newton.ShapeFlags.COLLIDE_SHAPES, rigid),
            (newton.ShapeFlags.COLLIDE_PARTICLES, particles),
            (newton.ShapeFlags.VISIBLE, visible),
        )
        for flag, enabled in assignments:
            if enabled:
                flags |= int(flag)
            else:
                flags &= ~int(flag)
        builder.shape_flags[shape_id] = flags

    def _disable_robot_particle_collision(self, builder, body_ids: range, shape_ids: range) -> None:
        """The robot shapes must not collide with MPM particles (only the cup cavity mesh does)."""
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in shape_ids:
            body_id = int(builder.shape_body[shape_id])
            if body_id not in body_ids:
                continue
            body_label = str(builder.body_label[body_id])
            if "/Robot/" in body_label or body_label.endswith("/Robot"):
                builder.shape_flags[shape_id] &= ~collide_particles

    def _configure_finger_contact_material(self, builder, body_ids: range, shape_ids: range) -> None:
        """Use a rigid contact material on the two finger collision shapes.

        Applying the same material to the fingers and cup keeps the intended pair response
        independent of import-side material defaults.
        """
        finger_suffixes = ("/panda_leftfinger", "/panda_rightfinger")
        for shape_id in shape_ids:
            body_id = int(builder.shape_body[shape_id])
            if body_id not in body_ids:
                continue
            if str(builder.body_label[body_id]).endswith(finger_suffixes):
                builder.shape_material_ke[shape_id] = self._grasp_contact_ke
                builder.shape_material_kd[shape_id] = self._grasp_contact_kd
                builder.shape_material_kf[shape_id] = self._grasp_contact_kf
                builder.shape_material_mu[shape_id] = self._grasp_contact_mu

    # ----------------------------------------------------------- post-physics
    def _setup_after_physics(self) -> None:
        dev = self.device
        self._robot = self.scene["robot"]
        self._source_cup = self.scene["source_cup"]
        self._target_cup = self.scene["target_cup"]
        self._media: MPMObject = self.scene["media"]

        self._arm_joint_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True)
        self._finger_joint_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True)
        self._joint_pos_limits_t = self._robot.data.joint_pos_limits.torch.clone()
        tcp_body_ids, _ = self._robot.find_bodies(self.cfg.tcp_body_name)
        if len(tcp_body_ids) != 1:
            raise RuntimeError(
                f"Expected one TCP parent body named {self.cfg.tcp_body_name!r}, found {len(tcp_body_ids)}."
            )
        self._tcp_body_idx = tcp_body_ids[0]
        self._tcp_offset_pos = torch.tensor(self.cfg.tcp_offset_pos, device=dev).repeat(self.num_envs, 1)
        self._tcp_offset_quat = torch.tensor(self.cfg.tcp_offset_rot, device=dev).repeat(self.num_envs, 1)
        approach_axis_c = torch.as_tensor(
            self.cfg.curriculum_randomized_reset_tcp_standoff,
            device=dev,
            dtype=torch.float32,
        )
        self._grasp_approach_axis_c = approach_axis_c / torch.linalg.vector_norm(approach_axis_c)
        grasp_tcp_quat_c = torch.as_tensor(
            self.cfg.cup_grasp_tcp_quat_c,
            device=dev,
            dtype=torch.float32,
        )
        # Keep the grasp frame explicit: tool +Z and jaw +Y remain parallel to the table, and the
        # same cup-local grasp is preserved under source-yaw randomization.
        self._desired_grasp_tcp_quat_c = math_utils.quat_unique(grasp_tcp_quat_c).repeat(self.num_envs, 1)

        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)
        self._num_particles = int(self._media.particles_per_object)
        self._media_local_points_t = torch.as_tensor(self._media_local_points, device=dev, dtype=torch.float32)
        self._particle_workspace_lower_t = torch.as_tensor(
            self.cfg.particle_workspace_lower_bound, device=dev, dtype=torch.float32
        )
        self._particle_workspace_upper_t = torch.as_tensor(
            self.cfg.particle_workspace_upper_bound, device=dev, dtype=torch.float32
        )
        grasp_stage_index = self.cfg.curriculum_stage_names.index("grasp")
        self._curriculum_arm_q_t = torch.as_tensor(
            (
                self.cfg.curriculum_drain_arm_q,
                self.cfg.curriculum_deep_tilt_arm_q,
                self.cfg.curriculum_tilt_arm_q,
                self.cfg.curriculum_pour_arm_q,
                *self.cfg._curriculum_transport_arm_configs(),
                self.cfg.curriculum_carry_arm_q,
                *(self.cfg.arm_home for _ in range(len(self.cfg.curriculum_stage_names) - grasp_stage_index)),
            ),
            device=dev,
            dtype=torch.float32,
        )
        self._curriculum_cup_quat_t = torch.zeros((len(self.cfg.curriculum_stage_names), 4), device=dev)
        self._curriculum_cup_quat_t[:, 3] = 1.0
        contact_position = float(self.cfg.cup_grasp_box_half[1])
        self._curriculum_finger_pos_t = torch.full(
            (len(self.cfg.curriculum_stage_names),),
            float(self.cfg.gripper_open_pos),
            device=dev,
        )
        self._curriculum_finger_pos_t[:grasp_stage_index] = contact_position
        # These waypoints seed and validate the reset bank only. The policy never receives or
        # follows them; every runtime arm command is a direct relative joint-position action.
        self._nominal_reference_waypoints_t = torch.as_tensor(
            (
                self.cfg.arm_home,
                self.cfg.arm_home,
                self.cfg.arm_home,
                self.cfg.arm_home,
                self.cfg.curriculum_carry_arm_q,
                self.cfg.curriculum_pour_arm_q,
                self.cfg.curriculum_pour_target_arm_q,
            ),
            device=dev,
            dtype=torch.float32,
        )
        self._grasp_stage_index = grasp_stage_index
        self._approach_stage_index = self.cfg.curriculum_stage_names.index("approach_1")
        self._full_stage_index = self.cfg.curriculum_stage_names.index("full")
        self._randomized_stage_index = self.cfg.curriculum_stage_names.index("randomized")
        self._uses_reset_dataset = getattr(self.cfg.curriculum, "reset_dataset", None) is not None
        reset_dataset_path = getattr(self.cfg, "reset_dataset_path", None)
        if self._uses_reset_dataset:
            if reset_dataset_path is None:
                raise ValueError("The reset-dataset task requires reset_dataset_path.")
            self._load_reset_dataset(reset_dataset_path)
        else:
            self._build_randomized_reset_bank()
            # The complete-path collision screen samples the midpoint-to-grasp joint segment at
            # eighths. Reuse those exact samples as progressively longer open-hand reset frontiers.
            approach_fractions = torch.as_tensor(
                self.cfg.curriculum_grasp_approach_fractions,
                device=dev,
                dtype=torch.float32,
            )
            self._curriculum_approach_arm_q_t = torch.lerp(
                self._reach_midgrasp_arm_q_bank_t[0].expand(approach_fractions.shape[0], -1),
                self._reach_grasp_arm_q_bank_t[0].expand(approach_fractions.shape[0], -1),
                approach_fractions.unsqueeze(-1),
            )
            self._build_independent_reset_fallbacks()
        self._last_source_bank_index = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
        self._last_arm_bank_index = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
        self._last_target_bank_index = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
        start_stage = int(self.cfg.curriculum_start_stage)
        start_randomization_level = int(self.cfg.curriculum_randomization_start_level)
        self.curriculum_stage = torch.full((self.num_envs,), start_stage, device=dev, dtype=torch.long)
        self.curriculum_randomization_level = torch.full(
            (self.num_envs,),
            start_randomization_level,
            device=dev,
            dtype=torch.long,
        )
        self.reset_dataset_row_id = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
        # Diagnostic tools may request exact raw dataset rows.  This override deliberately has
        # one meaning in both adaptive training and frozen playback; it is never relative to a
        # curriculum pool.
        self._forced_reset_dataset_row = torch.full_like(self.reset_dataset_row_id, -1)
        self.pour_target_frac = torch.full(
            (self.num_envs,),
            float(self.cfg.curriculum_target_frac[start_stage]),
            device=dev,
        )
        self.episode_succeeded = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self.ep_max_target_frac = torch.zeros(self.num_envs, device=dev)
        self._success_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lifted_grasp_seen = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._target_entry_seen = torch.zeros(
            (self.num_envs, self.num_particles),
            device=dev,
            dtype=torch.bool,
        )
        self._held_delivered = torch.zeros_like(self._target_entry_seen)
        self._held_delivery_tracker_step = -1
        self._source_inner_lo_t = torch.as_tensor(self._source_inner_lo, device=dev)
        self._source_inner_hi_t = torch.as_tensor(self._source_inner_hi, device=dev)
        self._target_inner_lo_t = torch.as_tensor(self._target_inner_lo, device=dev)
        self._target_inner_hi_t = torch.as_tensor(self._target_inner_hi, device=dev)
        self._particle_region_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._particle_region_cache_step = -1

    def _collision_free_ik_candidates(
        self,
        prototype_builder: newton.ModelBuilder,
        candidate_waypoints: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Validate top IK branches against Newton self-contact along the executed path.

        Each candidate occupies one isolated Newton world, so the ordinary collision pipeline can
        screen the exact imported Franka collision shapes in a compact batch. Sampling every
        joint-space segment catches folded branches that have valid endpoint IK costs but cannot be
        tracked by the physical articulation because non-adjacent links intersect.
        """
        if len(candidate_waypoints) < 2:
            raise ValueError("Collision validation requires at least two IK waypoints.")
        shape = candidate_waypoints[0].shape
        if len(shape) != 3:
            raise ValueError(f"IK candidate waypoints must have shape (rows, candidates, q), got {shape}.")
        if any(waypoint.shape != shape for waypoint in candidate_waypoints[1:]):
            raise ValueError("Every IK collision-validation waypoint must have the same shape.")
        row_count, candidate_count, coordinate_count = shape
        if coordinate_count != prototype_builder.joint_coord_count:
            raise ValueError(
                "IK candidate coordinate count does not match the clone prototype: "
                f"{coordinate_count} != {prototype_builder.joint_coord_count}."
            )

        validation_builder = newton.ModelBuilder(up_axis=prototype_builder.up_axis)
        validation_builder.replicate(prototype_builder, world_count=row_count)
        validation_model = validation_builder.finalize(device=self.device)
        if validation_model.world_count != row_count:
            raise RuntimeError(
                f"Expected {row_count} isolated IK collision worlds, got {validation_model.world_count}."
            )
        pipeline = newton.CollisionPipeline(
            validation_model,
            broad_phase="explicit",
            soft_contact_max=0,
            verify_buffers=True,
        )
        contacts = pipeline.contacts()
        state = validation_model.state()
        colliding_worlds = wp.zeros(row_count, dtype=wp.int32, device=validation_model.device)
        validation_coordinate_count = validation_model.joint_coord_count // row_count
        if validation_coordinate_count != coordinate_count:
            raise RuntimeError(
                "IK validation coordinate count must match the prototype; "
                f"got {validation_coordinate_count} coordinates after replicating {coordinate_count}."
            )
        robot_q = wp.to_torch(validation_model.joint_q).reshape(row_count, coordinate_count)
        collision_free = torch.ones((row_count, candidate_count), device=self.device, dtype=torch.bool)

        # The trajectory action uses a monotonic smoothstep along the same joint-space line. Nine
        # equally spaced samples per segment include both endpoints and detect the observed folded
        # elbow/hand intersections without turning startup validation into a simulation rollout.
        segment_fractions = tuple(
            sample / _RESET_PATH_SEGMENT_SUBDIVISIONS for sample in range(_RESET_PATH_SEGMENT_SUBDIVISIONS)
        )
        for candidate_index in range(candidate_count):
            colliding_worlds.zero_()
            for lower, upper in zip(candidate_waypoints[:-1], candidate_waypoints[1:], strict=True):
                lower_q = lower[:, candidate_index]
                upper_q = upper[:, candidate_index]
                for fraction in segment_fractions:
                    robot_q.copy_(torch.lerp(lower_q, upper_q, fraction).contiguous())
                    newton.eval_fk(
                        validation_model,
                        validation_model.joint_q,
                        validation_model.joint_qd,
                        state,
                    )
                    pipeline.collide(state, contacts)
                    wp.launch(
                        _mark_penetrating_self_contacts,
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
                            validation_model.body_world,
                            state.body_q,
                            1.0e-4,
                        ],
                        outputs=[colliding_worlds],
                        device=validation_model.device,
                    )
            robot_q.copy_(candidate_waypoints[-1][:, candidate_index].contiguous())
            newton.eval_fk(
                validation_model,
                validation_model.joint_q,
                validation_model.joint_qd,
                state,
            )
            pipeline.collide(state, contacts)
            wp.launch(
                _mark_penetrating_self_contacts,
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
                    validation_model.body_world,
                    state.body_q,
                    1.0e-4,
                ],
                outputs=[colliding_worlds],
                device=validation_model.device,
            )
            collision_free[:, candidate_index] = wp.to_torch(colliding_worlds) == 0

        return collision_free

    def _load_reset_dataset(self, configured_path: str) -> None:
        """Load and stage a validated direct-state dataset on the simulation device."""
        cache_path = Path(configured_path).expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Franka Pour reset dataset not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        self._validate_loaded_reset_dataset(payload)
        expected_hash = self.cfg.reset_dataset_content_sha256
        if expected_hash is not None and payload["content_sha256"] != expected_hash:
            raise RuntimeError(
                "Franka Pour reset dataset content hash does not match the configured hash: "
                f"{payload['content_sha256']} != {expected_hash}."
            )

        metadata = payload["metadata"]
        expected_joint_names = tuple(ARM_JOINTS + FINGER_JOINTS)
        if tuple(metadata["joint_names"]) != expected_joint_names:
            raise RuntimeError("Reset-dataset joint order does not match the Franka runtime joint order.")
        if metadata["frame"] != "environment" or metadata["quaternion_order"] != "xyzw":
            raise RuntimeError("Reset-dataset poses must use environment-frame XYZW representation.")
        layouts = payload["particle_layouts"]
        local_position = layouts["local_position"].to(device=self.device, dtype=torch.float32)
        local_velocity = layouts["local_velocity"].to(device=self.device, dtype=torch.float32)
        if local_position.shape[1:] != self._media_local_points_t.shape:
            raise RuntimeError(
                "Reset-dataset particle layout does not match the runtime media shape: "
                f"{tuple(local_position.shape[1:])} != {tuple(self._media_local_points_t.shape)}."
            )
        if not bool(torch.allclose(local_position[0], self._media_local_points_t, atol=1.0e-7, rtol=0.0)):
            raise RuntimeError("Reset-dataset particle layout does not match the current cup fill lattice.")

        self._reset_dataset_metadata = metadata
        self._reset_dataset_states = {
            name: value.to(device=self.device, non_blocking=True) for name, value in payload["states"].items()
        }
        self._reset_dataset_particle_local_position = local_position
        self._reset_dataset_particle_local_velocity = local_velocity
        logger.info(
            "Loaded reset dataset %s (%s) with %d rows.",
            cache_path,
            payload["content_sha256"],
            metadata["state_count"],
        )

    def _validate_loaded_reset_dataset(self, payload: dict) -> None:
        """Require dynamic-validation provenance for normal task execution."""
        validate_production_reset_dataset(
            payload,
            expected_task_contract=build_franka_pour_reset_task_contract(self),
        )

    def _build_randomized_reset_bank(self) -> None:
        """Build a small Newton-IK bank for collision-safe randomized pre-grasp resets."""
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = resolve_clone_plan_source(self._robot.cfg.prim_path, plan) if plan is not None else None
        if resolved is None:
            raise RuntimeError(f"Could not resolve clone-plan source for {self._robot.cfg.prim_path!r}.")
        source_path = resolved[0]
        prototype_origin = -self.env_origins[0]
        prototype_xform = wp.transform(
            wp.vec3(*prototype_origin.tolist()),
            wp.quat_identity(),
        )

        def local_prototype_builder() -> newton.ModelBuilder:
            """Copy the clone source into an environment-local IK frame."""
            source_builder = copy_newton_source_builder(source_path)
            local_builder = newton.ModelBuilder(up_axis=source_builder.up_axis)
            local_builder.add_builder(source_builder, xform=prototype_xform)
            return local_builder

        # Clone layouts are centered around the world origin, so the source environment can be
        # tens of metres away for large training batches. Solve IK and validate collisions in the
        # environment-local frame to make the reset bank independent of ``num_envs`` and avoid
        # losing millimetre-scale joint clearance to float32 cancellation.
        prototype_builder = local_prototype_builder()
        model = local_prototype_builder().finalize(device=self.device)

        hand_matches = [
            body_id
            for body_id, label in enumerate(model.body_label)
            if str(label).rsplit("/", 1)[-1] == self.cfg.tcp_body_name
        ]
        if len(hand_matches) != 1:
            raise RuntimeError(
                f"Expected one {self.cfg.tcp_body_name!r} body in the IK prototype, found {hand_matches}."
            )
        hand_id = hand_matches[0]
        joint_labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
        joint_q_start = wp.to_torch(model.joint_q_start).to(device=self.device, dtype=torch.long)

        def coordinate_id(joint_name: str) -> int:
            matches = [joint_id for joint_id, label in enumerate(joint_labels) if label == joint_name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in the IK prototype, found {matches}.")
            return int(joint_q_start[matches[0]].item())

        arm_coordinate_ids = torch.tensor(
            [coordinate_id(joint_name) for joint_name in ARM_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        finger_coordinate_ids = torch.tensor(
            [coordinate_id(joint_name) for joint_name in FINGER_JOINTS],
            device=self.device,
            dtype=torch.long,
        )

        def tcp_pose_for_arm_q(arm_q_values: tuple[float, ...]) -> torch.Tensor:
            """Evaluate one prototype arm configuration and return its TCP world pose."""
            joint_q = wp.to_torch(model.joint_q).to(device=self.device, dtype=torch.float32).clone()
            joint_q[arm_coordinate_ids] = torch.as_tensor(arm_q_values, device=self.device)
            joint_q[finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)
            state = model.state()
            newton.eval_fk(
                model,
                wp.from_torch(joint_q.contiguous(), dtype=wp.float32),
                model.joint_qd,
                state,
            )
            hand_pose = wp.to_torch(state.body_q)[hand_id : hand_id + 1]
            tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                hand_pose[:, :3],
                hand_pose[:, 3:7],
                self._tcp_offset_pos[0:1],
                self._tcp_offset_quat[0:1],
            )
            return torch.cat((tcp_pos, tcp_quat), dim=-1)[0].clone()

        nominal_carry_tcp_pose = tcp_pose_for_arm_q(self.cfg.curriculum_carry_arm_q)
        nominal_pour_tcp_pose = tcp_pose_for_arm_q(self.cfg.curriculum_pour_arm_q)
        nominal_tilt_tcp_pose = tcp_pose_for_arm_q(self.cfg.curriculum_pour_target_arm_q)

        grid_size = int(self.cfg.curriculum_randomized_reset_ik_grid_size)
        nominal_source = torch.as_tensor(self.cfg.cup_reset_pos, device=self.device)
        if self.cfg.curriculum_randomized_source_radius_range is None:
            source_range = torch.as_tensor(
                self.cfg.curriculum_randomized_source_position_range,
                device=self.device,
                dtype=torch.float32,
            )
            x_offsets = torch.linspace(-source_range[0], source_range[0], grid_size, device=self.device)
            y_offsets = torch.linspace(-source_range[1], source_range[1], grid_size, device=self.device)
            offset_x, offset_y = torch.meshgrid(x_offsets, y_offsets, indexing="ij")
            correlation = float(self.cfg.curriculum_randomized_source_xy_correlation)
            if float(source_range[0]) > 0.0:
                reachable_diagonal_y = offset_x / source_range[0] * source_range[1]
            else:
                reachable_diagonal_y = torch.zeros_like(offset_y)
            offset_y = correlation * reachable_diagonal_y + (1.0 - correlation) * offset_y
            offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1)
            base_source_positions = nominal_source.repeat(offsets.shape[0], 1)
            base_source_positions[:, :2] += offsets
            radial_facing = False
        else:
            base_source_positions = polar_workspace_cells(
                nominal_source,
                radius_range=self.cfg.curriculum_randomized_source_radius_range,
                azimuth_half_range=float(self.cfg.curriculum_randomized_source_azimuth_range),
                grid_size=grid_size,
            )
            radial_facing = True

        samples_per_source = int(self.cfg.curriculum_randomized_reset_ik_samples_per_source)
        pair_count = samples_per_source // 2
        if self.cfg.curriculum_randomized_reset_tcp_offset_lower is None:
            pair_index = torch.arange(pair_count, device=self.device, dtype=torch.float32) + 0.5
            pair_directions = (
                2.0
                * torch.stack(
                    (
                        torch.frac(pair_index * 0.754877666),
                        torch.frac(pair_index * 0.569840296),
                        torch.frac(pair_index * 0.438447187),
                    ),
                    dim=-1,
                )
                - 1.0
            )
            paired_jitter = pair_directions * torch.as_tensor(
                self.cfg.curriculum_randomized_reset_tcp_jitter,
                device=self.device,
            )
            jitter_parts = [paired_jitter, -paired_jitter]
            if samples_per_source % 2:
                jitter_parts.insert(0, torch.zeros((1, 3), device=self.device))
            tcp_jitter_samples = torch.cat(jitter_parts, dim=0)
        else:
            tcp_jitter_samples = asymmetric_reset_offset_samples(
                self.cfg.curriculum_randomized_reset_tcp_offset_lower,
                self.cfg.curriculum_randomized_reset_tcp_offset_upper,
                samples_per_source,
                device=self.device,
                dtype=torch.float32,
            )
        zero_jitter_slot = int(torch.argmin(torch.linalg.vector_norm(tcp_jitter_samples, dim=-1)).item())

        # Pair upright source yaw with the symmetric TCP samples. Keeping both signs in each source
        # cell avoids a directional reset bias while retaining the existing compact bank size.
        yaw_range = float(self.cfg.curriculum_randomized_source_yaw_range)
        pair_yaws = (
            (torch.arange(pair_count, device=self.device, dtype=torch.float32) + 1.0) / max(pair_count, 1) * yaw_range
        )
        yaw_parts = [pair_yaws, -pair_yaws]
        if samples_per_source % 2:
            yaw_parts.insert(0, torch.zeros(1, device=self.device))
        source_yaw_samples = torch.cat(yaw_parts, dim=0)

        source_cell_count = base_source_positions.shape[0]
        nominal_source_cell = int(
            torch.argmin(torch.linalg.vector_norm(base_source_positions - nominal_source, dim=-1)).item()
        )
        base_source_positions = base_source_positions.repeat_interleave(samples_per_source, dim=0)
        base_tcp_jitter = tcp_jitter_samples.repeat(source_cell_count, 1)
        reset_rotation_vectors = reset_rotation_vector_samples(
            self.cfg.curriculum_randomized_reset_tcp_rotation_angle_range,
            samples_per_source,
            device=self.device,
            dtype=torch.float32,
        )
        rotation_sample_ids = balanced_cyclic_permutations(
            torch.arange(samples_per_source, device=self.device),
            source_cell_count,
        )
        base_tcp_rotation_vectors = reset_rotation_vectors[rotation_sample_ids].reshape(-1, 3)
        # Rotate the yaw-to-jitter pairing in each source cell. Every cell retains identical yaw
        # and jitter/rotation marginals, while each fixed slot sees every perturbation globally.
        base_source_yaw_jitter = balanced_cyclic_permutations(source_yaw_samples, source_cell_count).reshape(-1)
        rows_per_extent = base_source_positions.shape[0]
        extent_levels = tuple(float(extent) for extent in self.cfg.curriculum_randomization_extent_levels)
        level_count = len(extent_levels)

        # Scale the same balanced source-cell/reset-offset design at every extent instead of
        # filtering one full-range bank. Equal-size level censuses preserve all marginals while
        # expanding the physical domain smoothly; the final extent is bit-for-bit full amplitude.
        base_source_offsets = base_source_positions - nominal_source
        source_positions = nominal_source + scale_randomization_rows_by_extent(
            base_source_offsets, extent_levels
        ).reshape(-1, 3)
        tcp_jitter = scale_randomization_rows_by_extent(base_tcp_jitter, extent_levels).reshape(-1, 3)
        tcp_rotation_vectors = scale_randomization_rows_by_extent(
            base_tcp_rotation_vectors,
            extent_levels,
        ).reshape(-1, 3)
        source_yaw_jitter = scale_randomization_rows_by_extent(
            base_source_yaw_jitter,
            extent_levels,
        ).reshape(-1)
        if radial_facing:
            source_yaws = torch.atan2(source_positions[:, 1], source_positions[:, 0]) + source_yaw_jitter
        else:
            source_yaws = source_yaw_jitter
        randomized_bank_size = rows_per_extent * level_count
        zero_bank_row = nominal_source_cell * samples_per_source + zero_jitter_slot

        # Stages two and three reuse safe varied arm starts around the nominal upright source.
        # Append dedicated solve-only rows so the randomized nominal-source rows keep all yaw values;
        # these rows are sliced out before constructing the stage-four extent banks.
        source_positions = torch.cat((source_positions, nominal_source.repeat(samples_per_source, 1)), dim=0)
        tcp_jitter = torch.cat((tcp_jitter, tcp_jitter_samples), dim=0)
        tcp_rotation_vectors = torch.cat(
            (tcp_rotation_vectors, torch.zeros((samples_per_source, 3), device=self.device)),
            dim=0,
        )
        source_yaws = torch.cat((source_yaws, torch.zeros(samples_per_source, device=self.device)), dim=0)
        bank_size = source_positions.shape[0]
        source_quaternions = torch.zeros((bank_size, 4), device=self.device)
        source_quaternions[:, 2] = torch.sin(0.5 * source_yaws)
        source_quaternions[:, 3] = torch.cos(0.5 * source_yaws)

        grasp_offset_c = torch.zeros((bank_size, 3), device=self.device)
        grasp_offset_c[:, 2] = float(self.cfg.cup_grasp_height)
        standoff_c = torch.as_tensor(
            self.cfg.curriculum_randomized_reset_tcp_standoff,
            device=self.device,
        ).expand(bank_size, -1)
        # Standoff and jitter are cup-local. Rotating both with source yaw keeps every reset on the
        # same side-approach ray instead of sweeping the horizontal fingers across a cup corner.
        tcp_offset_c = grasp_offset_c + standoff_c + tcp_jitter
        tcp_positions = source_positions + math_utils.quat_apply(source_quaternions, tcp_offset_c)

        target_positions_w = tcp_positions
        aligned_target_rotations_w = math_utils.quat_mul(
            source_quaternions,
            self._desired_grasp_tcp_quat_c[:1].expand(bank_size, -1),
        )
        rotation_angles = torch.linalg.vector_norm(tcp_rotation_vectors, dim=-1)
        rotation_axes = tcp_rotation_vectors / rotation_angles.clamp_min(1.0e-9).unsqueeze(-1)
        reset_rotation_quaternions = math_utils.quat_from_angle_axis(rotation_angles, rotation_axes)
        target_rotations_w = math_utils.quat_mul(
            aligned_target_rotations_w,
            reset_rotation_quaternions,
        ).contiguous()

        target_name = "reset_tcp"

        def ik_objectives() -> list:
            return [
                NewtonIKPoseObjectiveCfg(
                    body_name=self.cfg.tcp_body_name,
                    name=target_name,
                    body_offset_pos=self.cfg.tcp_offset_pos,
                    body_offset_rot=self.cfg.tcp_offset_rot,
                    # Grasp capture tolerates only millimetres of cross-track error. Weight
                    # position strongly enough that accepted IK cost cannot hide a bad insertion.
                    position_weight=100.0,
                    rotation_weight=5.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ]

        # The horizontal arm admits several kinematic branches. Sample only the randomized reset
        # pose so each row starts on an interior branch, then use a separate single-seed solver to
        # track every subsequent waypoint continuously without branch jumps.
        reset_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=64,
                noise_std=0.75,
                iterations=int(self.cfg.curriculum_randomized_reset_ik_iterations),
                lambda_initial=0.1,
            ),
            model=model,
            num_envs=bank_size,
            device=str(model.device),
            objectives=ik_objectives(),
            link_resolver=lambda body_name: hand_id,
        )
        pose_objective = reset_solver.objectives_by_name[target_name]
        pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_positions_w.contiguous(), dtype=wp.vec3)
        )
        pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_rotations_w.contiguous(), dtype=wp.vec4)
        )

        seed = wp.to_torch(model.joint_q).to(device=self.device, dtype=torch.float32).repeat(bank_size, 1)
        seed[:, arm_coordinate_ids] = torch.as_tensor(self.cfg.arm_home, device=self.device)
        seed[:, finger_coordinate_ids] = float(self.cfg.gripper_open_pos)
        reset_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))
        expanded_q = wp.to_torch(reset_solver.joint_q).reshape(
            bank_size,
            reset_solver.cfg.n_seeds,
            -1,
        )
        expanded_costs = wp.to_torch(reset_solver.costs).reshape(bank_size, reset_solver.cfg.n_seeds)
        expanded_arm_q = expanded_q[:, :, arm_coordinate_ids]
        arm_limits = self._joint_pos_limits_t[0, self._arm_joint_ids]
        expanded_margin = torch.minimum(
            expanded_arm_q - arm_limits[:, 0],
            arm_limits[:, 1] - expanded_arm_q,
        ).amin(dim=-1)
        candidate_valid = (
            torch.isfinite(expanded_arm_q).all(dim=-1)
            & torch.isfinite(expanded_costs)
            & (expanded_costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
            & (expanded_margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
        )
        candidate_rows = torch.arange(bank_size, device=self.device)
        seed_count = int(reset_solver.cfg.n_seeds)
        reset_has_candidate = candidate_valid.any(dim=-1)

        # A high-clearance reset solution can still run into a joint limit during the straight
        # lateral insertion. Preserve all feasible reset branches through pre-grasp and grasp,
        # then choose the trajectory with the largest minimum clearance over all three poses.
        # Invalid candidates are replaced with a finite row-local seed before launching IK so one
        # bad sample cannot poison the batched solver; their validity mask remains false.
        trajectory_valid = candidate_valid.clone()
        trajectory_margin = torch.where(
            trajectory_valid,
            expanded_margin,
            torch.full_like(expanded_margin, -torch.inf),
        )
        valid_fallback_indices = torch.argmax(trajectory_valid.to(dtype=torch.int32), dim=-1)
        margin_fallback_indices = torch.argmax(
            torch.nan_to_num(expanded_margin, nan=-torch.inf),
            dim=-1,
        )
        fallback_indices = torch.where(reset_has_candidate, valid_fallback_indices, margin_fallback_indices)
        fallback_q = expanded_q[candidate_rows, fallback_indices].unsqueeze(1)
        trajectory_q = torch.where(trajectory_valid.unsqueeze(-1), expanded_q, fallback_q).clone()

        continuation_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="none",
                n_seeds=1,
                iterations=int(self.cfg.curriculum_randomized_reset_ik_iterations),
                lambda_initial=0.1,
            ),
            model=model,
            num_envs=bank_size * seed_count,
            device=str(model.device),
            objectives=ik_objectives(),
            link_resolver=lambda body_name: hand_id,
        )
        continuation_objective = continuation_solver.objectives_by_name[target_name]

        def solve_candidate_waypoint(
            target_pose: torch.Tensor,
            initial_guess: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal trajectory_margin, trajectory_valid
            expanded_target = target_pose.repeat_interleave(seed_count, dim=0)
            continuation_objective.position_objective.set_target_positions(
                wp.from_torch(expanded_target[:, :3].contiguous(), dtype=wp.vec3)
            )
            continuation_objective.rotation_objective.set_target_rotations(
                wp.from_torch(expanded_target[:, 3:7].contiguous(), dtype=wp.vec4)
            )
            solved_full = (
                wp.to_torch(
                    continuation_solver.solve(
                        wp.from_torch(initial_guess.reshape(bank_size * seed_count, -1).contiguous(), dtype=wp.float32)
                    )
                )
                .reshape(bank_size, seed_count, -1)
                .clone()
            )
            solved_arm = solved_full[:, :, arm_coordinate_ids]
            solved_costs = wp.to_torch(continuation_solver.costs).reshape(bank_size, seed_count).clone()
            solved_margin = torch.minimum(
                solved_arm - arm_limits[:, 0],
                arm_limits[:, 1] - solved_arm,
            ).amin(dim=-1)
            waypoint_valid = (
                torch.isfinite(solved_arm).all(dim=-1)
                & torch.isfinite(solved_costs)
                & (solved_costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
                & (solved_margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
            )
            prior_valid = trajectory_valid.clone()
            trajectory_valid &= waypoint_valid
            trajectory_margin = torch.where(
                trajectory_valid,
                torch.minimum(trajectory_margin, solved_margin),
                torch.full_like(trajectory_margin, -torch.inf),
            )
            has_candidate = trajectory_valid.any(dim=-1)
            valid_fallback_indices = torch.argmax(trajectory_valid.to(dtype=torch.int32), dim=-1)
            prior_fallback_indices = torch.argmax(prior_valid.to(dtype=torch.int32), dim=-1)
            valid_fallback = solved_full[candidate_rows, valid_fallback_indices]
            prior_fallback = initial_guess[candidate_rows, prior_fallback_indices]
            fallback = torch.where(has_candidate.unsqueeze(-1), valid_fallback, prior_fallback).unsqueeze(1)
            sanitized = torch.where(trajectory_valid.unsqueeze(-1), solved_full, fallback)
            return sanitized, solved_arm, solved_costs, solved_margin

        # Route every reset through a centered lateral pre-grasp. The open fingers then insert
        # along hand +Z, parallel to the table, without sweeping across the glass.
        pregrasp_tcp_positions = source_positions + math_utils.quat_apply(
            source_quaternions,
            grasp_offset_c + standoff_c,
        )
        pregrasp_target_positions_w = pregrasp_tcp_positions
        pregrasp_target_pose = torch.cat(
            (pregrasp_target_positions_w, aligned_target_rotations_w),
            dim=-1,
        )
        pregrasp_candidates, pregrasp_arm_candidates, pregrasp_cost_candidates, pregrasp_margin_candidates = (
            solve_candidate_waypoint(pregrasp_target_pose, trajectory_q)
        )

        # Split the horizontal insertion into two straight Cartesian segments. Interpolating the
        # joint coordinates directly from a 12 cm standoff bowed the physical TCP below the glass
        # on some randomized IK branches; the midpoint keeps the open fingers on the side-approach
        # ray before the grasp gate is allowed to close.
        midgrasp_tcp_positions = source_positions + math_utils.quat_apply(
            source_quaternions,
            grasp_offset_c + 0.5 * standoff_c,
        )
        midgrasp_target_pose = torch.cat(
            (midgrasp_tcp_positions, aligned_target_rotations_w),
            dim=-1,
        )
        midgrasp_candidates, midgrasp_arm_candidates, midgrasp_cost_candidates, midgrasp_margin_candidates = (
            solve_candidate_waypoint(midgrasp_target_pose, pregrasp_candidates)
        )

        # Solve the paired grasp waypoint from the centered midpoint. Keeping every pose in the
        # same prevalidated bank avoids reset-time IK and preserves stationary action semantics.
        grasp_offset_c[:, 2] -= float(self.cfg.curriculum_grasp_descent_overshoot)
        grasp_tcp_positions = source_positions + math_utils.quat_apply(source_quaternions, grasp_offset_c)
        grasp_target_positions_w = grasp_tcp_positions
        grasp_target_pose = torch.cat(
            (grasp_target_positions_w, aligned_target_rotations_w),
            dim=-1,
        )
        grasp_candidates, grasp_arm_candidates, grasp_cost_candidates, grasp_margin_candidates = (
            solve_candidate_waypoint(grasp_target_pose, midgrasp_candidates)
        )

        # Keep all branches alive through the upright lift as well. This source-relative waypoint
        # is where a few edge-of-workspace branches that are safe at grasp first reach a limit.
        source_delta = source_positions - nominal_source
        carry_target_pose = nominal_carry_tcp_pose.repeat(bank_size, 1)
        carry_position_range = torch.as_tensor(
            (*self.cfg.curriculum_randomized_carry_position_range, 0.0),
            device=self.device,
        )
        carry_target_pose[:, :3] += torch.clamp(source_delta, min=-carry_position_range, max=carry_position_range)
        # Canonicalize yaw during the upright lift/carry. Keeping the radial grasp orientation all
        # the way to a fixed receiver-side pour forced broad-angle starts through folded wrist
        # branches. The cup remains upright while this collision-screened waypoint smoothly
        # unwinds source yaw before transport.
        carry_target_pose[:, 3:7] = self._desired_grasp_tcp_quat_c[:1]
        carry_candidates, carry_arm_candidates, carry_cost_candidates, carry_margin_candidates = (
            solve_candidate_waypoint(carry_target_pose, grasp_candidates)
        )

        # Keep receiver geometry fixed across the arm-start variants of one source cell. Two
        # low-discrepancy coordinates cover the receiver region while the geometric projection
        # below keeps it safely behind the source. Centering both sequences on the nominal source
        # cell makes the zero-amplitude frontier exactly reproduce the authored task.
        cell_index = torch.arange(source_cell_count, device=self.device, dtype=torch.float32)
        center_cell = float(nominal_source_cell)
        base_unit_samples = torch.stack(
            (
                torch.remainder((cell_index - center_cell) * 0.754877666 + 0.5, 1.0),
                torch.remainder((cell_index - center_cell) * 0.569840296 + 0.5, 1.0),
            ),
            dim=-1,
        ).repeat_interleave(samples_per_source, dim=0)
        source_outer_half_x = self.cfg.source_cup_inner_width / 2.0 + self.cfg.source_cup_wall_thickness
        source_outer_half_y = self.cfg.source_cup_inner_depth / 2.0 + self.cfg.source_cup_wall_thickness
        target_outer_half_y = self.cfg.target_cup_inner_depth / 2.0 + self.cfg.target_cup_wall_thickness
        minimum_y_separation = (
            source_outer_half_x * torch.abs(torch.sin(source_yaws[:randomized_bank_size]))
            + source_outer_half_y * torch.abs(torch.cos(source_yaws[:randomized_bank_size]))
            + target_outer_half_y
            + float(self.cfg.curriculum_randomized_cup_clearance)
        )
        target_range = torch.as_tensor(
            self.cfg.curriculum_randomized_target_position_range,
            device=self.device,
        )
        target_xy_parts = []
        for level, extent in enumerate(extent_levels):
            rows = slice(level * rows_per_extent, (level + 1) * rows_per_extent)
            target_xy_parts.append(
                target_xy_behind_source(
                    source_positions[rows, :2],
                    target_center=self.cfg.curriculum_randomized_target_center_xy,
                    target_half_range=target_range * extent,
                    minimum_y_separation=minimum_y_separation[rows],
                    unit_samples=base_unit_samples,
                )
            )
        target_xy = torch.cat(target_xy_parts, dim=0)
        target_positions = torch.as_tensor(self.cfg.target_cup_reset_pos, device=self.device).repeat(
            source_positions.shape[0], 1
        )
        target_positions[:randomized_bank_size, :2] = target_xy

        # Broad receiver positions need more than the single authored IK seed used by the narrow
        # task. Solve a compact set of branches for both upright-pour and deep-tilt endpoints, then
        # choose the shortest jointly valid pair. Source-side branch selection below still accounts
        # for the final carry-to-pour transition.
        receiver_seed_count = 16
        solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=receiver_seed_count,
                noise_std=0.5,
                iterations=int(self.cfg.curriculum_randomized_reset_ik_iterations),
                lambda_initial=0.1,
            ),
            model=model,
            num_envs=bank_size,
            device=str(model.device),
            objectives=ik_objectives(),
            link_resolver=lambda body_name: hand_id,
        )
        pose_objective = solver.objectives_by_name[target_name]

        def solve_reference_candidates(
            target_pose: torch.Tensor,
            initial_guess: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            pose_objective.position_objective.set_target_positions(
                wp.from_torch(target_pose[:, :3].contiguous(), dtype=wp.vec3)
            )
            pose_objective.rotation_objective.set_target_rotations(
                wp.from_torch(target_pose[:, 3:7].contiguous(), dtype=wp.vec4)
            )
            solver.solve(wp.from_torch(initial_guess.contiguous(), dtype=wp.float32))
            raw_solved_full = (
                wp.to_torch(solver.joint_q).reshape(bank_size, int(solver.cfg.n_seeds), model.joint_coord_count).clone()
            )
            solved_arm = raw_solved_full[:, :, arm_coordinate_ids]
            solved_costs = wp.to_torch(solver.costs).reshape(bank_size, receiver_seed_count).clone()
            solved_margin = torch.minimum(
                solved_arm - arm_limits[:, 0],
                arm_limits[:, 1] - solved_arm,
            ).amin(dim=-1)
            solved_valid = (
                torch.isfinite(solved_arm).all(dim=-1)
                & torch.isfinite(solved_costs)
                & (solved_costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
                & (solved_margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
            )
            return raw_solved_full, solved_arm, solved_costs, solved_margin, solved_valid

        nominal_target = torch.as_tensor(self.cfg.target_cup_reset_pos, device=self.device)
        target_delta = target_positions - nominal_target

        pour_target_pose = nominal_pour_tcp_pose.repeat(bank_size, 1)
        pour_target_pose[:, :3] += target_delta
        clearance_by_row = torch.zeros(bank_size, device=self.device)
        clearance_by_row[:randomized_bank_size] = torch.as_tensor(
            extent_levels,
            device=self.device,
        ).repeat_interleave(rows_per_extent) * float(self.cfg.curriculum_randomized_pour_clearance)
        pour_target_pose[:, 2] += clearance_by_row
        pour_target_pose[:, 3:7] = self._desired_grasp_tcp_quat_c[:1]
        pour_seed = seed.clone()
        pour_seed[:, arm_coordinate_ids] = torch.as_tensor(self.cfg.curriculum_pour_arm_q, device=self.device)
        pour_full_candidates, pour_arm_candidates, pour_cost_candidates, pour_margin_candidates, pour_valid = (
            solve_reference_candidates(
                pour_target_pose,
                pour_seed,
            )
        )

        tilt_target_pose = nominal_tilt_tcp_pose.repeat(bank_size, 1)
        tilt_target_pose[:, :3] += target_delta
        tilt_target_pose[:, 2] += clearance_by_row
        tilt_seed = seed.clone()
        tilt_seed[:, arm_coordinate_ids] = torch.as_tensor(
            self.cfg.curriculum_pour_target_arm_q,
            device=self.device,
        )
        tilt_full_candidates, tilt_arm_candidates, tilt_cost_candidates, tilt_margin_candidates, tilt_valid = (
            solve_reference_candidates(
                tilt_target_pose,
                tilt_seed,
            )
        )

        receiver_pair_valid = pour_valid.unsqueeze(-1) & tilt_valid.unsqueeze(-2)
        receiver_pair_transition = (
            (tilt_arm_candidates.unsqueeze(-3) - pour_arm_candidates.unsqueeze(-2)).square().sum(dim=-1)
        )
        pour_nominal = torch.as_tensor(self.cfg.curriculum_pour_arm_q, device=self.device)
        tilt_nominal = torch.as_tensor(self.cfg.curriculum_pour_target_arm_q, device=self.device)
        # Prefer the authored elbow/wrist branch whenever several receiver solutions have similar
        # path length. Without this tie-break, millimetric target changes can flip the complete
        # joint reference by more than a radian even though the task-space pose is continuous.
        nominal_branch_weight = 0.5
        receiver_pair_transition += nominal_branch_weight * (
            (pour_arm_candidates - pour_nominal).square().sum(dim=-1).unsqueeze(-1)
            + (tilt_arm_candidates - tilt_nominal).square().sum(dim=-1).unsqueeze(-2)
        )
        receiver_pair_score = torch.where(
            receiver_pair_valid,
            receiver_pair_transition,
            torch.full_like(receiver_pair_transition, torch.inf),
        )
        receiver_valid = receiver_pair_valid.flatten(start_dim=1).any(dim=-1)
        source_trajectory_valid = trajectory_valid.any(dim=-1)
        source_transition = (
            (pregrasp_arm_candidates - expanded_arm_q).square().sum(dim=-1)
            + (midgrasp_arm_candidates - pregrasp_arm_candidates).square().sum(dim=-1)
            + (grasp_arm_candidates - midgrasp_arm_candidates).square().sum(dim=-1)
            + (carry_arm_candidates - grasp_arm_candidates).square().sum(dim=-1)
        )
        source_nominal_deviation = (expanded_arm_q - self._nominal_reference_waypoints_t[0]).square().sum(dim=-1) + (
            carry_arm_candidates - self._nominal_reference_waypoints_t[4]
        ).square().sum(dim=-1)

        # Select complete reset->carry->pour->tilt paths jointly. Committing to the lowest-cost
        # receiver pair first can discard a different receiver branch that is much closer to the
        # source-relative carry branch, especially near a radial workspace boundary. Retain four
        # tilt alternatives per pour branch, rank the resulting source/pour/tilt paths, and send
        # the best complete candidates through the exact self-collision screen below.
        tilt_alternatives_per_pour = min(4, receiver_seed_count)
        fallback_tilt_score = receiver_pair_score.clone()
        fallback_tilt_score[:, :, 0] = torch.inf
        fallback_scores_by_pour, fallback_tilt_indices_by_pour = torch.topk(
            fallback_tilt_score,
            k=tilt_alternatives_per_pour - 1,
            dim=-1,
            largest=False,
            sorted=True,
        )
        receiver_scores_by_pour = torch.cat(
            (receiver_pair_score[:, :, :1], fallback_scores_by_pour),
            dim=-1,
        )
        zero_tilt_indices = torch.zeros_like(fallback_tilt_indices_by_pour[:, :, :1])
        receiver_tilt_indices_by_pour = torch.cat(
            (zero_tilt_indices, fallback_tilt_indices_by_pour),
            dim=-1,
        )
        carry_to_pour_transition = (
            (pour_arm_candidates.unsqueeze(1) - carry_arm_candidates.unsqueeze(2)).square().sum(dim=-1)
        )
        complete_path_score = (
            source_transition[:, :, None, None]
            + nominal_branch_weight * source_nominal_deviation[:, :, None, None]
            + carry_to_pour_transition[:, :, :, None]
            + receiver_scores_by_pour[:, None, :, :]
        )
        complete_path_valid = trajectory_valid[:, :, None, None] & torch.isfinite(
            receiver_scores_by_pour[:, None, :, :]
        )
        open_reset_branch = expanded_arm_q[:, :, 5] <= float(self.cfg.curriculum_randomized_reset_joint6_max)
        row_has_open_reset_branch = (complete_path_valid & open_reset_branch[:, :, None, None]).any(
            dim=(1, 2, 3),
            keepdim=True,
        )
        complete_path_valid &= ~row_has_open_reset_branch | open_reset_branch[:, :, None, None]
        complete_path_score = torch.where(
            complete_path_valid,
            complete_path_score,
            torch.full_like(complete_path_score, torch.inf),
        )
        # Broad radial starts expose multiple valid Franka IK branches. Reserve part of the exact
        # collision screen for the deterministic, unperturbed source seed; otherwise receiver-path
        # combinations can fill the global top-k and silently remove the only branch continuous
        # with the nominal grasp. The remaining slots retain the shortest complete paths globally.
        collision_candidate_count = min(64, complete_path_score[0].numel())
        receiver_paths_per_source = receiver_seed_count * tilt_alternatives_per_pour
        deterministic_source_candidate_count = min(8, receiver_paths_per_source, collision_candidate_count)
        flat_complete_path_score = complete_path_score.flatten(start_dim=1)
        flat_complete_path_valid = complete_path_valid.flatten(start_dim=1)
        deterministic_source_score = complete_path_score[:, 0].flatten(start_dim=1).clone()
        deterministic_source_score[:, 0] = torch.inf
        deterministic_source_fallback_indices = torch.topk(
            deterministic_source_score,
            k=deterministic_source_candidate_count - 1,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        deterministic_source_path_indices = torch.cat(
            (
                torch.zeros((bank_size, 1), device=self.device, dtype=torch.long),
                deterministic_source_fallback_indices,
            ),
            dim=-1,
        )
        global_path_indices = torch.topk(
            flat_complete_path_score,
            k=collision_candidate_count - deterministic_source_candidate_count,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        collision_path_indices = torch.cat((deterministic_source_path_indices, global_path_indices), dim=-1)
        collision_source_indices = torch.div(
            collision_path_indices,
            receiver_paths_per_source,
            rounding_mode="floor",
        )
        collision_receiver_indices = collision_path_indices.remainder(receiver_paths_per_source)
        collision_pour_indices = torch.div(
            collision_receiver_indices,
            tilt_alternatives_per_pour,
            rounding_mode="floor",
        )
        collision_tilt_slots = collision_receiver_indices.remainder(tilt_alternatives_per_pour)
        collision_tilt_indices = receiver_tilt_indices_by_pour[
            candidate_rows.unsqueeze(-1),
            collision_pour_indices,
            collision_tilt_slots,
        ]
        ranked_endpoint_valid = torch.gather(flat_complete_path_valid, 1, collision_path_indices)

        def gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return torch.gather(
                values,
                1,
                indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
            ).contiguous()

        def gather_source_waypoint(values: torch.Tensor, finger_position: float) -> torch.Tensor:
            gathered = gather_candidates(values, collision_source_indices)
            gathered = torch.where(ranked_endpoint_valid.unsqueeze(-1), gathered, seed.unsqueeze(1))
            gathered[:, :, finger_coordinate_ids] = finger_position
            return gathered

        pour_collision_candidates = gather_candidates(pour_full_candidates, collision_pour_indices)
        pour_collision_candidates = torch.where(
            ranked_endpoint_valid.unsqueeze(-1),
            pour_collision_candidates,
            pour_seed.unsqueeze(1),
        )
        pour_collision_candidates[:, :, finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)
        tilt_collision_candidates = gather_candidates(tilt_full_candidates, collision_tilt_indices)
        tilt_collision_candidates = torch.where(
            ranked_endpoint_valid.unsqueeze(-1),
            tilt_collision_candidates,
            tilt_seed.unsqueeze(1),
        )
        tilt_collision_candidates[:, :, finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)

        source_collision_candidates = (
            gather_source_waypoint(expanded_q, float(self.cfg.gripper_open_pos)),
            gather_source_waypoint(pregrasp_candidates, float(self.cfg.gripper_open_pos)),
            gather_source_waypoint(midgrasp_candidates, float(self.cfg.gripper_open_pos)),
            gather_source_waypoint(grasp_candidates, float(self.cfg.gripper_open_pos)),
            gather_source_waypoint(carry_candidates, float(self.cfg.gripper_preload_pos)),
        )
        collision_free = self._collision_free_ik_candidates(
            prototype_builder,
            (*source_collision_candidates, pour_collision_candidates, tilt_collision_candidates),
        )
        ranked_valid = ranked_endpoint_valid & collision_free
        row_has_collision_free_path = ranked_valid.any(dim=-1)
        ranked_complete_path_score = torch.gather(flat_complete_path_score, 1, collision_path_indices)

        # Resolve the redundant source-side IK branch against one exact nominal reach path. The
        # nominal row is solved in the same batched census and screened through the same complete
        # collision path, so this is a physical branch anchor rather than a hard-coded posture.
        # Lexicographic selection (nearest source branch, then shortest complete path) prevents a
        # millimetric geometry increment from changing elbow/wrist branches at curriculum promotion.
        nominal_reach_row = randomized_bank_size + zero_jitter_slot
        nominal_seed_valid = ranked_valid[nominal_reach_row] & (collision_source_indices[nominal_reach_row] == 0)
        if not bool(nominal_seed_valid.any()):
            raise RuntimeError(
                "The randomized curriculum requires a collision-free deterministic zero-jitter nominal reach path."
            )
        source_preliminary_score = source_transition + nominal_branch_weight * source_nominal_deviation
        screened_source_score = torch.gather(source_preliminary_score, 1, collision_source_indices)
        nominal_reach_score = torch.where(
            nominal_seed_valid,
            screened_source_score[nominal_reach_row],
            torch.full_like(screened_source_score[nominal_reach_row], torch.inf),
        )
        nominal_reach_slot = torch.argmin(nominal_reach_score)
        source_collision_arm_paths = torch.stack(
            tuple(candidate[:, :, arm_coordinate_ids] for candidate in source_collision_candidates),
            dim=2,
        )
        nominal_source_arm_path = source_collision_arm_paths[nominal_reach_row, nominal_reach_slot]
        source_branch_distance = (source_collision_arm_paths - nominal_source_arm_path).square().sum(dim=(-1, -2))
        valid_source_branch_distance = torch.where(
            ranked_valid,
            source_branch_distance,
            torch.full_like(source_branch_distance, torch.inf),
        )
        closest_source_branch = (
            valid_source_branch_distance <= valid_source_branch_distance.amin(dim=-1, keepdim=True) + 1.0e-6
        )
        deterministic_receiver_pair = (collision_pour_indices == 0) & (collision_tilt_indices == 0)
        deterministic_receiver_available = (ranked_valid & closest_source_branch & deterministic_receiver_pair).any(
            dim=-1, keepdim=True
        )
        preferred_receiver_pair = ~deterministic_receiver_available | deterministic_receiver_pair
        collision_selection_score = torch.where(
            ranked_valid & closest_source_branch & preferred_receiver_pair,
            ranked_complete_path_score,
            torch.full_like(ranked_complete_path_score, torch.inf),
        )
        # Some Cartesian samples near the requested workspace boundary admit endpoint IK but no
        # executable self-collision-free branch. Keep finite fallback data for diagnostics, then
        # exclude those rows from the reset pools below instead of weakening physical collision
        # constraints or biasing the controller with an impossible reference.
        collision_selection_score = torch.where(
            row_has_collision_free_path.unsqueeze(-1),
            collision_selection_score,
            ranked_complete_path_score,
        )
        best_collision_slots = torch.argmin(collision_selection_score, dim=-1)
        best_indices = collision_source_indices[candidate_rows, best_collision_slots]
        best_pour_indices = collision_pour_indices[candidate_rows, best_collision_slots]
        best_tilt_indices = collision_tilt_indices[candidate_rows, best_collision_slots]
        collision_free_count = ranked_valid.sum(dim=-1)
        arm_q = expanded_arm_q[candidate_rows, best_indices].clone()
        costs = expanded_costs[candidate_rows, best_indices].clone()
        margin = expanded_margin[candidate_rows, best_indices].clone()
        pregrasp_arm_q = pregrasp_arm_candidates[candidate_rows, best_indices].clone()
        pregrasp_costs = pregrasp_cost_candidates[candidate_rows, best_indices].clone()
        pregrasp_margin = pregrasp_margin_candidates[candidate_rows, best_indices].clone()
        midgrasp_arm_q = midgrasp_arm_candidates[candidate_rows, best_indices].clone()
        midgrasp_costs = midgrasp_cost_candidates[candidate_rows, best_indices].clone()
        midgrasp_margin = midgrasp_margin_candidates[candidate_rows, best_indices].clone()
        grasp_arm_q = grasp_arm_candidates[candidate_rows, best_indices].clone()
        grasp_costs = grasp_cost_candidates[candidate_rows, best_indices].clone()
        grasp_margin = grasp_margin_candidates[candidate_rows, best_indices].clone()
        carry_arm_q = carry_arm_candidates[candidate_rows, best_indices].clone()
        carry_costs = carry_cost_candidates[candidate_rows, best_indices].clone()
        carry_margin = carry_margin_candidates[candidate_rows, best_indices].clone()
        pour_arm_q = pour_arm_candidates[candidate_rows, best_pour_indices]
        pour_costs = pour_cost_candidates[candidate_rows, best_pour_indices]
        pour_margin = pour_margin_candidates[candidate_rows, best_pour_indices]
        tilt_arm_q = tilt_arm_candidates[candidate_rows, best_tilt_indices]
        tilt_costs = tilt_cost_candidates[candidate_rows, best_tilt_indices]
        tilt_margin = tilt_margin_candidates[candidate_rows, best_tilt_indices]

        # Raw census rows that fail the complete-path screen remain useful for coverage diagnostics,
        # but must never leave nonfinite joint data in the bank. Store an explicit invalid sentinel
        # in their scalar metadata and a finite nominal trajectory in their joint arrays; sampling
        # pools below contain only rows with a positive collision-free candidate count.
        selected_valid = row_has_collision_free_path.unsqueeze(-1)
        nominal_waypoints = self._nominal_reference_waypoints_t
        arm_q = torch.where(selected_valid, arm_q, nominal_waypoints[0])
        pregrasp_arm_q = torch.where(selected_valid, pregrasp_arm_q, nominal_waypoints[1])
        midgrasp_arm_q = torch.where(selected_valid, midgrasp_arm_q, nominal_waypoints[2])
        grasp_arm_q = torch.where(selected_valid, grasp_arm_q, nominal_waypoints[3])
        carry_arm_q = torch.where(selected_valid, carry_arm_q, nominal_waypoints[4])
        pour_arm_q = torch.where(selected_valid, pour_arm_q, nominal_waypoints[5])
        tilt_arm_q = torch.where(selected_valid, tilt_arm_q, nominal_waypoints[6])
        invalid_cost = torch.full_like(costs, torch.inf)
        invalid_margin = torch.full_like(margin, -torch.inf)
        costs = torch.where(row_has_collision_free_path, costs, invalid_cost)
        pregrasp_costs = torch.where(row_has_collision_free_path, pregrasp_costs, invalid_cost)
        midgrasp_costs = torch.where(row_has_collision_free_path, midgrasp_costs, invalid_cost)
        grasp_costs = torch.where(row_has_collision_free_path, grasp_costs, invalid_cost)
        carry_costs = torch.where(row_has_collision_free_path, carry_costs, invalid_cost)
        pour_costs = torch.where(row_has_collision_free_path, pour_costs, invalid_cost)
        tilt_costs = torch.where(row_has_collision_free_path, tilt_costs, invalid_cost)
        margin = torch.where(row_has_collision_free_path, margin, invalid_margin)
        pregrasp_margin = torch.where(row_has_collision_free_path, pregrasp_margin, invalid_margin)
        midgrasp_margin = torch.where(row_has_collision_free_path, midgrasp_margin, invalid_margin)
        grasp_margin = torch.where(row_has_collision_free_path, grasp_margin, invalid_margin)
        carry_margin = torch.where(row_has_collision_free_path, carry_margin, invalid_margin)
        pour_margin = torch.where(row_has_collision_free_path, pour_margin, invalid_margin)
        tilt_margin = torch.where(row_has_collision_free_path, tilt_margin, invalid_margin)

        # Level zero is an exact behavioral continuation of the mastered full task. Its first four
        # waypoints must come from the dedicated nominal-source reach bank; the authored waypoint
        # tensor intentionally repeats ``arm_home`` there and is only a placeholder until those
        # insertion poses are solved. Store the exact zero-jitter nominal reach path in one
        # canonical zero-bank row, then preserve the known-safe authored carry/pour/tilt branch.
        # Arm-start diversity is introduced continuously by the later nonzero offset extents.
        arm_q[zero_bank_row] = arm_q[nominal_reach_row]
        pregrasp_arm_q[zero_bank_row] = pregrasp_arm_q[nominal_reach_row]
        midgrasp_arm_q[zero_bank_row] = midgrasp_arm_q[nominal_reach_row]
        grasp_arm_q[zero_bank_row] = grasp_arm_q[nominal_reach_row]
        carry_arm_q[zero_bank_row] = nominal_waypoints[4]
        pour_arm_q[zero_bank_row] = nominal_waypoints[5]
        tilt_arm_q[zero_bank_row] = nominal_waypoints[6]
        costs[zero_bank_row] = costs[nominal_reach_row]
        pregrasp_costs[zero_bank_row] = pregrasp_costs[nominal_reach_row]
        midgrasp_costs[zero_bank_row] = midgrasp_costs[nominal_reach_row]
        grasp_costs[zero_bank_row] = grasp_costs[nominal_reach_row]
        carry_costs[zero_bank_row] = carry_costs[nominal_reach_row]
        pour_costs[zero_bank_row] = pour_costs[nominal_reach_row]
        tilt_costs[zero_bank_row] = tilt_costs[nominal_reach_row]
        margin[zero_bank_row] = margin[nominal_reach_row]
        pregrasp_margin[zero_bank_row] = pregrasp_margin[nominal_reach_row]
        midgrasp_margin[zero_bank_row] = midgrasp_margin[nominal_reach_row]
        grasp_margin[zero_bank_row] = grasp_margin[nominal_reach_row]
        carry_margin[zero_bank_row] = carry_margin[nominal_reach_row]
        pour_margin[zero_bank_row] = pour_margin[nominal_reach_row]
        tilt_margin[zero_bank_row] = tilt_margin[nominal_reach_row]
        collision_free_count[zero_bank_row] = collision_free_count[nominal_reach_row]
        row_has_collision_free_path[zero_bank_row] = True

        randomized_rows = slice(0, randomized_bank_size)
        reach_rows = slice(randomized_bank_size, bank_size)
        self._randomized_source_pos_bank_t = source_positions[randomized_rows]
        self._randomized_source_yaw_bank_t = source_yaws[randomized_rows]
        self._randomized_source_quat_bank_t = source_quaternions[randomized_rows]
        self._randomized_target_pos_bank_t = target_positions[randomized_rows]
        self._randomized_tcp_pos_bank_t = tcp_positions[randomized_rows]
        self._randomized_tcp_quat_bank_t = target_rotations_w[randomized_rows]
        self._randomized_tcp_rotation_vector_bank_t = tcp_rotation_vectors[randomized_rows]
        self._randomized_arm_q_bank_t = arm_q[randomized_rows]
        self._randomized_pregrasp_arm_q_bank_t = pregrasp_arm_q[randomized_rows]
        self._randomized_midgrasp_arm_q_bank_t = midgrasp_arm_q[randomized_rows]
        self._randomized_grasp_arm_q_bank_t = grasp_arm_q[randomized_rows]
        self._randomized_carry_arm_q_bank_t = carry_arm_q[randomized_rows]
        self._randomized_pour_arm_q_bank_t = pour_arm_q[randomized_rows]
        self._randomized_tilt_arm_q_bank_t = tilt_arm_q[randomized_rows]
        self._randomized_reset_ik_cost_t = costs[randomized_rows]
        self._randomized_reset_ik_margin_t = margin[randomized_rows]
        self._randomized_collision_free_candidate_count_t = collision_free_count[randomized_rows]
        self._randomized_pregrasp_ik_cost_t = pregrasp_costs[randomized_rows]
        self._randomized_pregrasp_ik_margin_t = pregrasp_margin[randomized_rows]
        self._randomized_midgrasp_ik_cost_t = midgrasp_costs[randomized_rows]
        self._randomized_midgrasp_ik_margin_t = midgrasp_margin[randomized_rows]
        self._randomized_grasp_ik_cost_t = grasp_costs[randomized_rows]
        self._randomized_grasp_ik_margin_t = grasp_margin[randomized_rows]
        self._randomized_carry_ik_cost_t = carry_costs[randomized_rows]
        self._randomized_carry_ik_margin_t = carry_margin[randomized_rows]
        self._randomized_pour_ik_cost_t = pour_costs[randomized_rows]
        self._randomized_pour_ik_margin_t = pour_margin[randomized_rows]
        self._randomized_tilt_ik_cost_t = tilt_costs[randomized_rows]
        self._randomized_tilt_ik_margin_t = tilt_margin[randomized_rows]
        extent_bank_indices = torch.arange(randomized_bank_size, device=self.device).reshape(
            level_count,
            rows_per_extent,
        )
        randomized_collision_free = (
            row_has_collision_free_path[randomized_rows]
            & (arm_q[randomized_rows, 5] <= float(self.cfg.curriculum_randomized_reset_joint6_max))
        ).reshape(level_count, rows_per_extent)
        randomized_reference_path = torch.stack(
            (
                arm_q[randomized_rows],
                pregrasp_arm_q[randomized_rows],
                midgrasp_arm_q[randomized_rows],
                grasp_arm_q[randomized_rows],
                carry_arm_q[randomized_rows],
                pour_arm_q[randomized_rows],
                tilt_arm_q[randomized_rows],
            ),
            dim=1,
        )
        reference_branch_delta = torch.abs(randomized_reference_path - randomized_reference_path[zero_bank_row]).amax(
            dim=(-1, -2)
        )
        extent_by_row = reference_branch_delta.new_tensor(extent_levels).repeat_interleave(rows_per_extent)
        # Early frontiers should expand task geometry, not introduce unrelated redundant IK
        # branches. Permit joint displacement proportional to physical extent, then remove this
        # continuity filter once the policy has mastered half of the randomization amplitude.
        branch_continuity_limit = 0.04 + 3.0 * extent_by_row
        branch_continuity_limit = torch.where(
            extent_by_row >= 0.5,
            torch.full_like(branch_continuity_limit, torch.inf),
            branch_continuity_limit,
        )
        randomized_collision_free &= (reference_branch_delta <= branch_continuity_limit).reshape(
            level_count, rows_per_extent
        )
        self._randomized_reference_branch_delta_t = reference_branch_delta
        self._randomized_reference_branch_limit_t = branch_continuity_limit
        source_cell_ids = torch.arange(rows_per_extent, device=self.device) // samples_per_source
        # The exact zero-amplitude anchor is one canonical task, not 539 duplicate randomization
        # rows. Later levels retain equal source-cell weighting while expanding arm-start diversity
        # with the physical extent. Exposing every redundant IK/reset variant at the first one-
        # percent geometry step creates a discrete task jump even though the cup poses barely move.
        index_pools = [torch.as_tensor((zero_bank_row,), device=self.device)]
        weight_pools = [torch.ones(1, device=self.device)]
        source_cell_counts = [1]
        minimum_variant_counts = [1]
        minimum_variants = int(self.cfg.curriculum_randomized_min_reset_variants_per_source)
        for extent, level_indices, level_valid in zip(
            extent_levels[1:],
            extent_bank_indices[1:].unbind(dim=0),
            randomized_collision_free[1:].unbind(dim=0),
            strict=True,
        ):
            variants_per_cell = torch.bincount(source_cell_ids[level_valid], minlength=source_cell_count)
            level_valid = level_valid & (variants_per_cell >= minimum_variants)[source_cell_ids]

            # Keep the safest, most continuous paths first and grow toward the complete validated
            # arm-start marginal. The final extent selects all sample slots, while early levels
            # isolate learning the new Cartesian geometry from a simultaneous redundant-IK jump.
            variant_limit = max(1, math.ceil(extent * samples_per_source))
            score_by_cell = torch.where(
                level_valid,
                reference_branch_delta[level_indices],
                torch.full_like(reference_branch_delta[level_indices], torch.inf),
            ).reshape(source_cell_count, samples_per_source)
            ordered_slots = torch.argsort(score_by_cell, dim=1, stable=True)
            selected_by_cell = torch.zeros_like(score_by_cell, dtype=torch.bool)
            selected_by_cell.scatter_(1, ordered_slots[:, :variant_limit], True)
            level_valid &= selected_by_cell.reshape(-1)

            index_pool = level_indices[level_valid]
            _, inverse_cell_ids, rows_per_cell = torch.unique(
                source_cell_ids[level_valid],
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            # Give every feasible source XY cell equal probability, then sample its validated
            # yaw/jitter branches uniformly. Flat row sampling would overrepresent central cells,
            # where more IK seeds survive the posture and self-collision screens.
            weight_pool = rows_per_cell[inverse_cell_ids].to(dtype=torch.float32).reciprocal()
            index_pools.append(index_pool)
            weight_pools.append(weight_pool)
            source_cell_counts.append(int(rows_per_cell.numel()))
            minimum_variant_counts.append(int(rows_per_cell.min().item()) if rows_per_cell.numel() else 0)
        self._randomized_extent_index_pools = tuple(index_pools)
        self._randomized_extent_index_weights = tuple(weight_pools)
        self._randomized_extent_source_cell_counts = tuple(source_cell_counts)
        self._randomized_extent_minimum_variant_counts = tuple(minimum_variant_counts)
        if any(pool.numel() == 0 for pool in self._randomized_extent_index_pools):
            raise RuntimeError("Every randomization extent must contain a collision-free reset pose.")
        minimum_source_cells = math.ceil(
            grid_size * grid_size * float(self.cfg.curriculum_randomized_min_source_cell_fraction)
        )
        if any(count < minimum_source_cells for count in self._randomized_extent_source_cell_counts[1:]):
            source_valid_by_level = source_trajectory_valid[randomized_rows].reshape(level_count, rows_per_extent)
            receiver_valid_by_level = receiver_valid[randomized_rows].reshape(level_count, rows_per_extent)

            def covered_cell_counts(valid_by_level: torch.Tensor) -> tuple[int, ...]:
                return tuple(
                    int(torch.unique(source_cell_ids[level_valid]).numel())
                    for level_valid in valid_by_level.unbind(dim=0)
                )

            raise RuntimeError(
                "Randomized reset posture screening retained too little source workspace coverage: "
                f"required at least {minimum_source_cells}/{grid_size * grid_size} source XY cells per extent, "
                f"got {self._randomized_extent_source_cell_counts}; source-waypoint IK covered "
                f"{covered_cell_counts(source_valid_by_level)}, receiver-waypoint IK covered "
                f"{covered_cell_counts(receiver_valid_by_level)}."
            )
        required_minimum_variant_counts = tuple(
            1 if extent == 0.0 else min(minimum_variants, max(1, math.ceil(extent * samples_per_source)))
            for extent in extent_levels
        )
        if any(
            actual < required
            for actual, required in zip(
                self._randomized_extent_minimum_variant_counts,
                required_minimum_variant_counts,
                strict=True,
            )
        ):

            def variant_histograms(valid_rows: torch.Tensor) -> tuple[tuple[int, ...], ...]:
                return tuple(
                    tuple(
                        int((torch.bincount(source_cell_ids[level_valid], minlength=source_cell_count) == count).sum())
                        for count in range(samples_per_source + 1)
                    )
                    for level_valid in valid_rows.unbind(dim=0)
                )

            source_valid_by_level = source_trajectory_valid[randomized_rows].reshape(level_count, rows_per_extent)
            receiver_valid_by_level = receiver_valid[randomized_rows].reshape(level_count, rows_per_extent)
            slot_survival_counts = tuple(
                tuple(int(count) for count in level_valid.reshape(source_cell_count, samples_per_source).sum(dim=0))
                for level_valid in randomized_collision_free.unbind(dim=0)
            )
            raise RuntimeError(
                "Randomized reset posture screening retained too little arm-start diversity: "
                f"required per-level minima {required_minimum_variant_counts}, "
                f"got {self._randomized_extent_minimum_variant_counts}. "
                "Per-level source-cell histograms indexed by retained-variant count are "
                f"source-IK={variant_histograms(source_valid_by_level)}, "
                f"receiver-IK={variant_histograms(receiver_valid_by_level)}, "
                f"collision-free={variant_histograms(randomized_collision_free)}. "
                f"Collision-free sample-slot survival counts are {slot_survival_counts}."
            )
        if self.cfg.curriculum_randomized_source_radius_range is not None:
            final_level_offset = (level_count - 1) * rows_per_extent
            final_source_cells = torch.unique(
                (self._randomized_extent_index_pools[-1] - final_level_offset) // samples_per_source,
                sorted=True,
            )
            final_radial_rings = torch.unique(
                torch.div(final_source_cells, grid_size, rounding_mode="floor"),
                sorted=True,
            )
            final_azimuth_slots = torch.unique(final_source_cells.remainder(grid_size), sorted=True)
            required_radial_rings = torch.arange(grid_size, device=self.device)
            center_azimuth_slot = grid_size // 2
            has_both_azimuth_sides = bool(
                (final_azimuth_slots[0] < center_azimuth_slot) & (final_azimuth_slots[-1] > center_azimuth_slot)
            )
            has_broad_azimuth_span = bool(final_azimuth_slots[-1] - final_azimuth_slots[0] >= center_azimuth_slot)
            if (
                not torch.equal(final_radial_rings, required_radial_rings)
                or not has_both_azimuth_sides
                or not has_broad_azimuth_span
            ):
                raise RuntimeError(
                    "Randomized reset posture screening did not retain the configured polar-workspace "
                    "coverage: required every radial ring and broad coverage on both azimuth sides, got radial "
                    f"rings {final_radial_rings.tolist()} and azimuth slots {final_azimuth_slots.tolist()}."
                )
        if source_positions[reach_rows].shape[0] != samples_per_source:
            raise RuntimeError(
                f"Expected {samples_per_source} dedicated nominal-source reach poses, "
                f"found {source_positions[reach_rows].shape[0]}."
            )
        reach_indices = torch.arange(randomized_bank_size, bank_size, device=self.device)
        reach_indices = reach_indices[row_has_collision_free_path[reach_rows]]
        if reach_indices.numel() < minimum_variants:
            raise RuntimeError(
                "Nominal-source reach posture screening retained too little arm-start diversity: "
                f"required at least {minimum_variants} variants, got {reach_indices.numel()}."
            )
        self._reach_tcp_pos_bank_t = tcp_positions[reach_indices]
        self._reach_source_yaw_bank_t = source_yaws[reach_indices]
        self._reach_arm_q_bank_t = arm_q[reach_indices]
        self._reach_pregrasp_arm_q_bank_t = pregrasp_arm_q[reach_indices]
        self._reach_midgrasp_arm_q_bank_t = midgrasp_arm_q[reach_indices]
        self._reach_grasp_arm_q_bank_t = grasp_arm_q[reach_indices]
        self._reach_reset_ik_cost_t = costs[reach_indices]
        self._reach_reset_ik_margin_t = margin[reach_indices]
        self._reach_collision_free_candidate_count_t = collision_free_count[reach_indices]
        # Newton IK and the zero-copy Warp/Torch views above run asynchronously. The temporary
        # solver and prototype are released when this method returns, so complete every gather
        # before their backing allocations can be reclaimed. This is a one-time startup barrier.
        wp.synchronize_device(model.device)

    def _build_independent_reset_fallbacks(self) -> None:
        """Precompute guaranteed-clearance fallbacks for independently mixed reset rows."""
        bank_size = self._randomized_source_pos_bank_t.shape[0]
        arm_fallback = torch.full((bank_size,), -1, device=self.device, dtype=torch.long)
        target_fallback = torch.full_like(arm_fallback, -1)
        minimum_arm_distance = float(self.cfg.curriculum_independent_arm_min_tcp_distance)

        for level, pool in enumerate(self._randomized_extent_index_pools):
            source_position = self._randomized_source_pos_bank_t[pool]
            if self.cfg.curriculum_independent_arm_fraction_levels[level] > 0.0:
                arm_distance = torch.cdist(source_position, self._randomized_tcp_pos_bank_t[pool])
                farthest_arm_slot = torch.argmax(arm_distance, dim=1)
                farthest_arm_distance = arm_distance.gather(1, farthest_arm_slot.unsqueeze(-1)).squeeze(-1)
                if bool(torch.any(farthest_arm_distance < minimum_arm_distance)):
                    raise RuntimeError(
                        f"Randomization level {level} has no independent arm reset with the required "
                        f"{minimum_arm_distance:.3f} m TCP/source clearance."
                    )
                arm_fallback[pool] = pool[farthest_arm_slot]
            else:
                arm_fallback[pool] = pool

            if self.cfg.curriculum_independent_target_fraction_levels[level] > 0.0:
                candidate_targets = pool.unsqueeze(0).expand(pool.numel(), -1)
                target_clearance = self._independent_target_clearance(
                    pool,
                    candidate_targets,
                )
                farthest_target_slot = torch.argmax(target_clearance, dim=1)
                farthest_target_clearance = target_clearance.gather(
                    1,
                    farthest_target_slot.unsqueeze(-1),
                ).squeeze(-1)
                if bool(torch.any(farthest_target_clearance < 0.0)):
                    raise RuntimeError(
                        f"Randomization level {level} has no independent receiver reset with the configured "
                        "rectangular cup clearance."
                    )
                target_fallback[pool] = pool[farthest_target_slot]
            else:
                target_fallback[pool] = pool

        self._independent_arm_fallback_index_t = arm_fallback
        self._independent_target_fallback_index_t = target_fallback

    def _independent_target_clearance(
        self,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return conservative rectangular source/receiver separation margins [m]."""
        if source_indices.ndim != 1 or target_indices.ndim != 2 or target_indices.shape[0] != source_indices.shape[0]:
            raise ValueError("Independent reset indices must have shapes (N,) and (N, K).")
        source_position = self._randomized_source_pos_bank_t[source_indices]
        source_yaw = self._randomized_source_yaw_bank_t[source_indices]
        target_position = self._randomized_target_pos_bank_t[target_indices]
        source_half_x = 0.5 * float(self.cfg.source_cup_inner_width) + float(self.cfg.source_cup_wall_thickness)
        source_half_y = 0.5 * float(self.cfg.source_cup_inner_depth) + float(self.cfg.source_cup_wall_thickness)
        target_half_x = 0.5 * float(self.cfg.target_cup_inner_width) + float(self.cfg.target_cup_wall_thickness)
        target_half_y = 0.5 * float(self.cfg.target_cup_inner_depth) + float(self.cfg.target_cup_wall_thickness)
        clearance = float(self.cfg.curriculum_randomized_cup_clearance)
        source_aabb_half_x = source_half_x * torch.abs(torch.cos(source_yaw)) + source_half_y * torch.abs(
            torch.sin(source_yaw)
        )
        source_aabb_half_y = source_half_x * torch.abs(torch.sin(source_yaw)) + source_half_y * torch.abs(
            torch.cos(source_yaw)
        )
        position_delta = torch.abs(target_position[:, :, :2] - source_position[:, None, :2])
        clearance_x = position_delta[:, :, 0] - (source_aabb_half_x[:, None] + target_half_x + clearance)
        clearance_y = position_delta[:, :, 1] - (source_aabb_half_y[:, None] + target_half_y + clearance)
        # A positive margin on either world axis is sufficient to separate the two rectangles.
        return torch.maximum(clearance_x, clearance_y)

    # ----------------------------------------------------------- poses / obs
    def _pose_w_to_e(self, pose_w: torch.Tensor) -> torch.Tensor:
        """Convert a public world-frame pose view to a finite environment-frame pose."""
        pos = torch.nan_to_num(pose_w[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        raw_quat = pose_w[:, 3:7]
        quat = torch.nan_to_num(raw_quat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        ident = torch.zeros_like(raw_quat)
        ident[:, 3] = 1.0
        valid = torch.isfinite(raw_quat).all(dim=-1, keepdim=True) & (norm > 1.0e-6)
        quat = torch.where(valid, quat / torch.clamp(norm, min=1.0e-6), ident)
        return torch.cat((pos, quat), dim=-1)

    def ee_pose_e(self) -> torch.Tensor:
        """End-effector (panda_hand) pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx])

    def cup_pose_e(self) -> torch.Tensor:
        """Cup body pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._source_cup.data.root_link_pose_w.torch)

    def cup_velocity_w(self) -> torch.Tensor:
        """Source-cup linear and angular velocity in the world frame [m/s, rad/s]."""
        return self._source_cup.data.root_link_vel_w.torch

    def target_pose_e(self) -> torch.Tensor:
        """Receiving-cup pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._target_cup.data.root_link_pose_w.torch)

    def tcp_pose_e(self) -> torch.Tensor:
        """Tool-centre pose in the robot-root/environment frame."""
        body_pose_w = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        root_pose_w = self._robot.data.root_link_pose_w.torch
        pos, quat = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], body_pose_w[:, :3], body_pose_w[:, 3:7]
        )
        pos, quat = math_utils.combine_frame_transforms(pos, quat, self._tcp_offset_pos, self._tcp_offset_quat)
        return torch.cat((torch.nan_to_num(pos), torch.nan_to_num(quat)), dim=-1)

    def tcp_pos_e(self) -> torch.Tensor:
        return self.tcp_pose_e()[:, :3]

    def grasp_approach_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return signed axial and unsigned cross-track TCP errors from the grasp point [m]."""
        cup_pose = self.cup_pose_e()
        error_e = self.tcp_pos_e() - self.cup_grasp_point_e()
        error_c = math_utils.quat_apply_inverse(cup_pose[:, 3:7], error_e)
        axial = torch.sum(error_c * self._grasp_approach_axis_c, dim=-1)
        cross_track = torch.linalg.vector_norm(
            error_c - axial.unsqueeze(-1) * self._grasp_approach_axis_c,
            dim=-1,
        )
        return axial, cross_track

    def cup_grasp_point_e(self) -> torch.Tensor:
        """World-facing grasp point at the middle of the source cup walls, in env coordinates."""
        pose = self.cup_pose_e()
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = float(self.cfg.cup_grasp_height)
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], offset)

    def gripper_width(self) -> torch.Tensor:
        """Distance represented by the two symmetric Panda finger joint positions [m]."""
        finger_pos = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        width = finger_pos.sum(dim=-1)
        valid = torch.isfinite(finger_pos).all(dim=-1) & torch.isfinite(width)
        return torch.where(valid, width, torch.full_like(width, float(self.gripper_open_width)))

    def finger_joint_pos(self) -> torch.Tensor:
        """Individual policy-controlled finger joint positions [m]."""
        return self._robot.data.joint_pos.torch[:, self._finger_joint_ids]

    def finger_joint_vel(self) -> torch.Tensor:
        """Individual policy-controlled finger joint velocities [m/s]."""
        return self._robot.data.joint_vel.torch[:, self._finger_joint_ids]

    def desired_grasp_tcp_quat_c(self) -> torch.Tensor:
        """Desired TCP orientation in the source-cup frame as canonical XYZW quaternions."""
        return self._desired_grasp_tcp_quat_c

    def arm_joint_pos(self) -> torch.Tensor:
        """Current policy-controlled arm joint positions [rad]."""
        return self._robot.data.joint_pos.torch[:, self._arm_joint_ids]

    @property
    def gripper_open_width(self) -> float:
        return 2.0 * float(self.cfg.gripper_open_pos)

    @property
    def gripper_grasp_width(self) -> float:
        return 2.0 * float(self.cfg.cup_grasp_box_half[1])

    @property
    def cup_reset_height(self) -> float:
        return float(self.cfg.cup_reset_pos[2])

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def set_curriculum_stage(
        self,
        env_ids: list[int] | torch.Tensor | slice,
        stage: int,
    ) -> None:
        """Assign a curriculum stage and success threshold to selected environments."""
        if stage < 0 or stage >= len(self.cfg.curriculum_stage_names):
            raise ValueError(f"Curriculum stage {stage} is out of range.")
        self.curriculum_stage[env_ids] = stage
        self.pour_target_frac[env_ids] = float(self.cfg.curriculum_target_frac[stage])

    def set_curriculum_randomization_level(
        self,
        env_ids: list[int] | torch.Tensor | slice,
        level: int,
    ) -> None:
        """Assign one prevalidated source-randomization extent to selected environments."""
        if level < 0 or level >= len(self._randomized_extent_index_pools):
            raise ValueError(f"Curriculum randomization level {level} is out of range.")
        self.curriculum_randomization_level[env_ids] = level

    def particle_pos_e(self) -> torch.Tensor:
        """Per-env MPM particle positions in env coordinates, shape ``(N, P, 3)``."""
        return self._media.data.particle_pos_w.torch - self.env_origins[:, None, :]

    def particle_vel_e(self) -> torch.Tensor:
        """Per-env MPM particle velocities in environment axes, shape ``(N, P, 3)``."""
        # Environments differ only by translation, so world and environment velocity axes coincide.
        return self._media.data.particle_vel_w.torch

    def _points_inside_cup(
        self, points_e: torch.Tensor, pose_e: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        rel = points_e - pose_e[:, None, :3]
        quat = pose_e[:, None, 3:7].expand(-1, points_e.shape[1], -1)
        local = math_utils.quat_apply_inverse(quat, rel)
        margin = float(self.cfg.particle_count_margin)
        return ((local >= lo - margin) & (local <= hi + margin)).all(dim=-1)

    def _particle_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Boolean ``(source, target, spilled)`` masks, cached within one manager step."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._particle_region_cache is not None and self._particle_region_cache_step == step:
            return self._particle_region_cache
        points = self.particle_pos_e()
        source = self._points_inside_cup(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t)
        target_region = self._points_inside_cup(
            points,
            self.target_pose_e(),
            self._target_inner_lo_t,
            self._target_inner_hi_t,
        )
        # Geometric overlap is not delivery: nesting the source cup inside the receiver must not
        # score particles that remain physically contained by the source cup.
        target = _delivered_particle_mask(source, target_region)
        spill_height = float(self.cfg.spill_table_height) + float(self.cfg.particle_count_margin)
        spilled = _spilled_particle_mask(points, source, target, max_height=spill_height)
        self._particle_region_cache = (source, target, spilled)
        self._particle_region_cache_step = step
        return source, target, spilled

    def particles_in_target_mask(self) -> torch.Tensor:
        """Particles inside the target cup and no longer inside the source cup."""
        return self._particle_region_masks()[1]

    def particle_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached source, source-exclusive target, and irreversible-spill masks."""
        return self._particle_region_masks()

    def update_held_delivery_tracker(self, held_pour: torch.Tensor) -> None:
        """Record particles whose target-entry edge occurs during a held pour.

        The tracker is idempotent within one manager step because success termination and reward
        evaluation consume the same state. An unheld entry does not permanently disqualify a
        particle: after it leaves the target, a later valid held re-entry can still qualify.

        Args:
            held_pour: Per-environment mask for a preloaded, lifted source grasp.
        """
        if held_pour.shape != (self.num_envs,):
            raise ValueError(f"held_pour must have shape ({self.num_envs},), got {tuple(held_pour.shape)}.")
        step = int(self.common_step_counter)
        if self._held_delivery_tracker_step == step:
            return
        in_target = self.particles_in_target_mask()
        target_entry = in_target & ~self._target_entry_seen
        self._held_delivered |= target_entry & held_pour.unsqueeze(-1)
        self._target_entry_seen.copy_(in_target)
        self._held_delivery_tracker_step = step

    def held_delivered_mask(self) -> torch.Tensor:
        """Particles with at least one target-entry edge during a held pour."""
        return self._held_delivered

    def current_held_delivered_mask(self) -> torch.Tensor:
        """Validly delivered particles that remain inside the receiving cup."""
        return self.particles_in_target_mask() & self._held_delivered

    def particles_spilled_mask(self) -> torch.Tensor:
        """Per-particle irreversible-spill membership used by one-time penalties."""
        return self._particle_region_masks()[2]

    def count_in_source(self) -> torch.Tensor:
        return self._particle_region_masks()[0].sum(dim=1).float()

    def count_in_target(self) -> torch.Tensor:
        return self._particle_region_masks()[1].sum(dim=1).float()

    def count_spilled(self) -> torch.Tensor:
        return self._particle_region_masks()[2].sum(dim=1).float()

    def spilled_fraction(self) -> torch.Tensor:
        return self.count_spilled() / max(self.num_particles, 1)

    def state_finite(self) -> torch.Tensor:
        """Per-env instability guard over robot, source cup, and MPM media state."""
        cup_velocity = self._source_cup.data.root_link_vel_w.torch
        return _state_finite(
            self._robot.data.joint_pos.torch,
            self._robot.data.joint_vel.torch,
            self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx],
            self._source_cup.data.root_link_pose_w.torch,
            cup_velocity[:, :3],
            cup_velocity[:, 3:],
            self._media.data.particle_pos_w.torch,
        )

    def rigid_state_in_bounds(self) -> torch.Tensor:
        """Return whether finite rigid state remains within task-safe observation bounds."""
        cup_velocity = self._source_cup.data.root_link_vel_w.torch
        hand_pose_w = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        tcp_position_w, tcp_quaternion_w = math_utils.combine_frame_transforms(
            hand_pose_w[:, :3],
            hand_pose_w[:, 3:7],
            self._tcp_offset_pos,
            self._tcp_offset_quat,
        )
        tcp_pose_w = torch.cat((tcp_position_w, tcp_quaternion_w), dim=-1)
        return _rigid_state_in_bounds(
            self._robot.data.joint_pos.torch,
            self._robot.data.joint_vel.torch,
            self._joint_pos_limits_t,
            tcp_pose_w,
            self._source_cup.data.root_link_pose_w.torch,
            cup_velocity[:, :3],
            cup_velocity[:, 3:],
            self.env_origins,
            self._particle_workspace_lower_t,
            self._particle_workspace_upper_t,
            joint_position_margin=self.cfg.state_bound_joint_position_margin,
            max_joint_velocity=self.cfg.state_bound_max_joint_velocity,
            max_cup_linear_velocity=self.cfg.state_bound_max_cup_linear_velocity,
            max_cup_angular_velocity=self.cfg.state_bound_max_cup_angular_velocity,
        )

    def particles_in_workspace(self) -> torch.Tensor:
        """Return a per-environment mask for media inside the configured local workspace."""
        return _particles_in_workspace(
            self.particle_pos_e(),
            self._particle_workspace_lower_t,
            self._particle_workspace_upper_t,
        )

    @staticmethod
    def _select_first_safe_candidate(
        candidates: torch.Tensor,
        safe: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Select the first safe sampled index, or a prevalidated fallback."""
        first_safe = torch.argmax(safe.to(dtype=torch.int32), dim=1)
        selected = candidates[torch.arange(candidates.shape[0], device=candidates.device), first_safe]
        return torch.where(safe.any(dim=1), selected, fallback)

    def _sample_independent_reset_indices(
        self,
        source_indices: torch.Tensor,
        levels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample collision-screened arm and receiver rows independently from the source row."""
        attempts = int(self.cfg.curriculum_independent_sample_attempts)
        candidate_columns = [
            sample_index_pools(
                self._randomized_extent_index_pools,
                levels,
                weights=self._randomized_extent_index_weights,
            )
            for _ in range(attempts)
        ]
        candidates = torch.stack(candidate_columns, dim=1)
        rows = torch.arange(source_indices.shape[0], device=self.device)
        source_position = self._randomized_source_pos_bank_t[source_indices]

        candidate_tcp = self._randomized_tcp_pos_bank_t[candidates]
        tcp_distance = torch.linalg.vector_norm(candidate_tcp - source_position[:, None, :], dim=-1)
        arm_safe = tcp_distance >= float(self.cfg.curriculum_independent_arm_min_tcp_distance)
        independent_arm = self._select_first_safe_candidate(
            candidates,
            arm_safe,
            self._independent_arm_fallback_index_t[source_indices],
        )

        target_clearance = self._independent_target_clearance(source_indices, candidates)
        target_safe = target_clearance >= 0.0
        independent_target = self._select_first_safe_candidate(
            candidates,
            target_safe,
            self._independent_target_fallback_index_t[source_indices],
        )

        arm_fraction = source_position.new_tensor(self.cfg.curriculum_independent_arm_fraction_levels)[levels]
        target_fraction = source_position.new_tensor(self.cfg.curriculum_independent_target_fraction_levels)[levels]
        arm_indices = torch.where(torch.rand_like(arm_fraction) < arm_fraction, independent_arm, source_indices)
        target_indices = torch.where(
            torch.rand_like(target_fraction) < target_fraction,
            independent_target,
            source_indices,
        )
        # Keep this local variable explicit: it makes shape errors in future pool changes fail near
        # sampling rather than later during indexed assignment.
        if arm_indices.shape != rows.shape or target_indices.shape != rows.shape:
            raise RuntimeError("Independent reset sampling returned an invalid index shape.")
        return arm_indices, target_indices

    # ----------------------------------------------------------- reset
    def _reset_from_dataset(self, env_ids: torch.Tensor, world_mask: torch.Tensor) -> None:
        """Restore exact dataset rows and clear all per-world solver history."""
        rows = self.reset_dataset_row_id[env_ids]
        if bool(torch.any((rows < 0) | (rows >= self._reset_dataset_states["category"].numel()))):
            raise RuntimeError("The reset-dataset curriculum must assign every environment a valid row.")
        states = self._reset_dataset_states
        arm_q = states["arm_joint_position"][rows]
        arm_qd = states["arm_joint_velocity"][rows]
        finger_q = states["finger_joint_position"][rows]
        finger_qd = states["finger_joint_velocity"][rows]
        finger_target = states["finger_joint_target"][rows]

        self._robot.write_joint_position_to_sim_index(
            position=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=arm_qd,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_position_to_sim_index(
            position=finger_q,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=finger_qd,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=finger_target,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self.action_manager.get_term("gripper_action").set_reset_position(
            finger_target[:, :1],
            env_ids=env_ids,
        )
        # Consume the public FK invalidation before rigid proxies and observations access bodies.
        _ = self._robot.data.body_link_pose_w

        source_pose = states["source_root_pose"][rows].clone()
        source_pose[:, :3] += self.env_origins[env_ids]
        target_pose = states["target_root_pose"][rows].clone()
        target_pose[:, :3] += self.env_origins[env_ids]
        self._source_cup.write_root_pose_to_sim_index(root_pose=source_pose, env_ids=env_ids)
        self._source_cup.write_root_velocity_to_sim_index(
            root_velocity=states["source_root_velocity"][rows],
            env_ids=env_ids,
        )
        self._target_cup.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self._target_cup.write_root_velocity_to_sim_index(
            root_velocity=states["target_root_velocity"][rows],
            env_ids=env_ids,
        )

        layout_ids = states["particle_layout_id"][rows].long()
        local_position = self._reset_dataset_particle_local_position[layout_ids]
        local_velocity = self._reset_dataset_particle_local_velocity[layout_ids]
        particle_count = local_position.shape[1]
        source_quat = source_pose[:, None, 3:7].expand(-1, particle_count, -1)
        particle_position = math_utils.quat_apply(source_quat, local_position) + source_pose[:, None, :3]
        particle_velocity = math_utils.quat_apply(source_quat, local_velocity)
        self._media.write_particle_pos_to_sim_index(particle_position, env_ids=env_ids)
        self._media.write_particle_velocity_to_sim_index(particle_velocity, env_ids=env_ids)

        # Public particle writers restore both Newton state buffers. This masked solver reset then
        # clears MPM stress/deformation and every private contact/collider history for the worlds.
        NewtonManager.reset_solver_state(
            world_mask=None if self.num_envs == 1 else wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )
        self._last_source_bank_index[env_ids] = -1
        self._last_arm_bank_index[env_ids] = -1
        self._last_target_bank_index[env_ids] = -1
        self._particle_region_cache = None
        self._particle_region_cache_step = -1
        self.episode_succeeded[env_ids] = False
        self.ep_max_target_frac[env_ids] = 0.0
        self._success_dwell_count[env_ids] = 0
        self._lost_grasp_dwell_count[env_ids] = 0
        # A grasping cache row is an offline-validated demonstrated grasp. Seed the latch from the
        # row category so opening the hand immediately after reset cannot evade dropped-cup
        # termination before the first runtime grasp observation. Non-grasping rows remain
        # unlatched and may freely approach the cup.
        self._lifted_grasp_seen[env_ids] = states["category"][rows] == GRASPING_CATEGORY
        self._target_entry_seen[env_ids] = False
        self._held_delivered[env_ids] = False
        self._held_delivery_tracker_step = -1

    def reset_pour_scene(self, env_ids: torch.Tensor) -> None:
        """Reset the arm, source cup, and particles through their public asset APIs."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        world_mask = boolean_selection_mask(self.num_envs, env_ids)
        if self._uses_reset_dataset:
            self._reset_from_dataset(env_ids, world_mask)
            return
        n = env_ids.numel()
        stage = self.curriculum_stage[env_ids]
        arm_q = self._curriculum_arm_q_t[stage].clone()
        cup_pos_e = torch.as_tensor(self.cfg.cup_reset_pos, device=self.device).repeat(n, 1)
        target_pos_e = torch.as_tensor(self.cfg.target_cup_reset_pos, device=self.device).repeat(n, 1)
        source_quat = self._curriculum_cup_quat_t[stage].clone()
        target_quat = self._curriculum_cup_quat_t[stage].clone()
        self._last_source_bank_index[env_ids] = -1
        self._last_arm_bank_index[env_ids] = -1
        self._last_target_bank_index[env_ids] = -1
        grasp_rows = torch.nonzero(stage == self._grasp_stage_index, as_tuple=False).flatten()
        if grasp_rows.numel() > 0:
            arm_q[grasp_rows] = self._reach_grasp_arm_q_bank_t[0]
        approach_rows = torch.nonzero(
            (stage >= self._approach_stage_index) & (stage < self._full_stage_index),
            as_tuple=False,
        ).flatten()
        if approach_rows.numel() > 0:
            approach_indices = stage[approach_rows] - self._approach_stage_index
            arm_q[approach_rows] = self._curriculum_approach_arm_q_t[approach_indices]
        full_rows = torch.nonzero(stage == self._full_stage_index, as_tuple=False).flatten()
        if full_rows.numel() > 0:
            arm_q[full_rows] = self._reach_pregrasp_arm_q_bank_t[0]
        randomized_rows = torch.nonzero(stage == self._randomized_stage_index, as_tuple=False).flatten()
        if randomized_rows.numel() > 0:
            randomization_levels = self.curriculum_randomization_level[env_ids[randomized_rows]]
            source_indices = sample_index_pools(
                self._randomized_extent_index_pools,
                randomization_levels,
                weights=self._randomized_extent_index_weights,
            )
            arm_indices, target_indices = self._sample_independent_reset_indices(
                source_indices,
                randomization_levels,
            )
            arm_q[randomized_rows] = self._randomized_arm_q_bank_t[arm_indices]
            cup_pos_e[randomized_rows] = self._randomized_source_pos_bank_t[source_indices]
            source_quat[randomized_rows] = self._randomized_source_quat_bank_t[source_indices]
            target_pos_e[randomized_rows] = self._randomized_target_pos_bank_t[target_indices]
            randomized_env_ids = env_ids[randomized_rows]
            self._last_source_bank_index[randomized_env_ids] = source_indices
            self._last_arm_bank_index[randomized_env_ids] = arm_indices
            self._last_target_bank_index[randomized_env_ids] = target_indices

        zero_arm_velocity = torch.zeros_like(arm_q)
        self._robot.write_joint_position_to_sim_index(
            position=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=zero_arm_velocity,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        # The operator-only absolute joint controller uses the reset pose as its zero-command
        # origin. The training controller is a standard relative action and has no such method.
        arm_action = self.action_manager.get_term("arm_action")
        if hasattr(arm_action, "set_action_offset"):
            arm_action.set_action_offset(arm_q, env_ids=env_ids)
        finger_position = self._curriculum_finger_pos_t[stage].unsqueeze(-1).expand(-1, len(FINGER_JOINTS)).clone()
        finger_drive_target = finger_position.clone()
        self._robot.write_joint_position_to_sim_index(
            position=finger_position,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(finger_position),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=finger_drive_target,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        gripper_target = torch.where(
            (stage < self._grasp_stage_index).unsqueeze(-1),
            torch.full((n, 1), float(self.cfg.gripper_preload_pos), device=self.device),
            torch.full((n, 1), float(self.cfg.gripper_open_pos), device=self.device),
        )
        self.action_manager.get_term("gripper_action").set_reset_position(
            gripper_target,
            env_ids=env_ids,
        )

        # Public root/joint writers invalidate FK. Reading a public body-pose view consumes
        # the accumulated articulation mask, making all dirtied body poses authoritative
        # before solver caches and source-cup proxy transforms are refreshed. Priming the
        # robot view also prevents its next observation from issuing a redundant FK launch.
        _ = self._robot.data.body_link_pose_w

        tcp_pose_e = self.tcp_pose_e()[env_ids]
        lifted_stage = stage < self._grasp_stage_index
        held_source_quat = math_utils.quat_mul(
            tcp_pose_e[:, 3:7],
            math_utils.quat_conjugate(self._desired_grasp_tcp_quat_c[env_ids]),
        )
        source_quat = torch.where(lifted_stage.unsqueeze(-1), held_source_quat, source_quat)
        grasp_offset = torch.zeros((n, 3), device=self.device)
        grasp_offset[:, 2] = float(self.cfg.cup_grasp_height)
        tcp_cup_pos_e = tcp_pose_e[:, :3] - math_utils.quat_apply(source_quat, grasp_offset)
        # Lifted stages follow their solved TCP. Full and randomized stages use authored or
        # IK-paired table positions.
        cup_pos_e = torch.where(lifted_stage.unsqueeze(-1), tcp_cup_pos_e, cup_pos_e)
        cup_world = cup_pos_e + self.env_origins[env_ids]
        cup_pose = torch.cat((cup_world, source_quat), dim=-1)
        self._source_cup.write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
        self._source_cup.write_root_velocity_to_sim_index(
            root_velocity=cup_pose.new_zeros((n, 6)),
            env_ids=env_ids,
        )
        target_world = target_pos_e + self.env_origins[env_ids]
        target_pose = torch.cat((target_world, target_quat), dim=-1)
        self._target_cup.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self._target_cup.write_root_velocity_to_sim_index(
            root_velocity=target_pose.new_zeros((n, 6)),
            env_ids=env_ids,
        )

        new_p = self._sample_cup_media(cup_world, source_quat)
        self._media.write_particle_pos_to_sim_index(new_p, env_ids=env_ids)
        self._media.write_particle_velocity_to_sim_index(torch.zeros_like(new_p), env_ids=env_ids)
        # Particle writers restore q/qd in both state buffers. Reset constitutive and collider
        # history for the same worlds so a captured replay cannot retain the previous episode.
        NewtonManager.reset_solver_state(
            world_mask=None if self.num_envs == 1 else wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )
        self._particle_region_cache = None
        self._particle_region_cache_step = -1
        self.episode_succeeded[env_ids] = False
        self.ep_max_target_frac[env_ids] = 0.0
        self._success_dwell_count[env_ids] = 0
        self._lost_grasp_dwell_count[env_ids] = 0
        self._lifted_grasp_seen[env_ids] = False
        self._target_entry_seen[env_ids] = False
        self._held_delivered[env_ids] = False
        # Selective resets can occur after this step's termination/reward pass. Invalidating the
        # scalar cache is cheap and keeps direct reset/replay workflows correct as well.
        self._held_delivery_tracker_step = -1

    def _sample_cup_media(self, cup_pos: torch.Tensor, cup_quat: torch.Tensor) -> torch.Tensor:
        """Transform the local media lattice into selected cup poses on the simulation device."""
        particle_count = self._media_local_points_t.shape[0]
        local_points = self._media_local_points_t.unsqueeze(0).expand(cup_pos.shape[0], -1, -1)
        quaternions = cup_quat.unsqueeze(1).expand(-1, particle_count, -1)
        return math_utils.quat_apply(quaternions, local_points) + cup_pos.unsqueeze(1)


class FrankaPourResetDatasetValidationEnv(FrankaPourEnv):
    """Offline replay environment that accepts schema-valid candidate datasets."""

    def _validate_loaded_reset_dataset(self, payload: dict) -> None:
        """Validate candidate structure without requiring output provenance yet."""
        validate_reset_dataset(
            payload,
            expected_task_contract=build_franka_pour_reset_task_contract(self),
        )


class FrankaPourResetSamplerEnv(FrankaPourEnv):
    """One-world offline scene that skips procedural reset banks and RL managers."""

    def load_managers(self) -> None:
        """Resolve only scene data consumed by the reset-dataset generator."""
        dev = self.device
        self._robot = self.scene["robot"]
        self._source_cup = self.scene["source_cup"]
        self._target_cup = self.scene["target_cup"]
        self._media: MPMObject = self.scene["media"]
        self._arm_joint_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True)
        self._finger_joint_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True)
        self._joint_pos_limits_t = self._robot.data.joint_pos_limits.torch.clone()
        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)
        self._num_particles = int(self._media.particles_per_object)
        self._media_local_points_t = torch.as_tensor(self._media_local_points, device=dev, dtype=torch.float32)

        # ManagerBasedRLEnv.close() deletes these attributes unconditionally.  No manager is
        # constructed because this scene is never reset or stepped through the Gym API.
        self.command_manager = None
        self.reward_manager = None
        self.termination_manager = None
        self.curriculum_manager = None
        self.recorder_manager = None
        self.action_manager = None
        self.observation_manager = None
