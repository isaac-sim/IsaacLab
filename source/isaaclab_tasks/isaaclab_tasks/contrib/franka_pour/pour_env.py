# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for a Franka pouring MPM media between two cups.

The visible dynamic source cup and kinematic receiving cup are scene-owned rigid objects loaded
from one authored USD asset. A narrow per-world Newton hook assigns the authored bowl and proxy
shapes their solver-specific collision roles and adds only one hidden solver object: a particle-only
spill floor.

A Newton :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` advances the robot and both cups
in the ``arm`` MJWarp entry and the particles and spill floor in the implicit ``media`` entry. Proxy
coupling makes both cups' particle colliders available to MPM without assigning one body to two
entries. The policy commands arm joint positions and a continuous symmetric finger target; all
observable and reset state flows through the scene assets' public APIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonMPMManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from .geometry import (
    CUP_GRASP_HEIGHT,
    CUP_GRASP_TCP_QUAT_C,
    MPM_COLLIDER_MARGIN,
    SOURCE_CUP_GEOMETRY,
    TARGET_CUP_GEOMETRY,
    points_inside_box,
)
from .pour_env_cfg import _configure_mpm_capacities
from .reset_dataset_io import RESET_DATASET_STATE_NAMES, reset_dataset_validate_runtime

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RESET_DATASET_GENERATOR_COMMAND = "uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0"

if TYPE_CHECKING:
    from isaaclab_newton.assets import MPMObject

    from .pour_env_cfg import FrankaPourResetDatasetEnvCfg

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
GRASPING_CATEGORY = 1
# The SeattleLab tabletop collision cube ends 3 mm below the task's z=0 support plane.
_SEATTLELAB_TABLETOP_COLLISION_INSET = 0.003

# Newton exposes force-space shape gains through the numeric ``mujoco:solref_mode``
# builder attribute but does not currently publish the corresponding enum.
_MJWARP_SOLREF_MODE_FORCE_SPACE = 0
_SOURCE_CUP_PARTICLE_FRICTION = 0.9


def _resolve_reset_dataset_path(configured_path: str) -> Path:
    """Resolve a reset dataset from the repository root and require it to exist."""
    cache_path = Path(configured_path).expanduser()
    if not cache_path.is_absolute():
        cache_path = _REPO_ROOT / cache_path
    cache_path = cache_path.resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Franka Pour reset dataset not found: {cache_path}. Generate it from the Isaac Lab repository root "
            f"with: {_RESET_DATASET_GENERATOR_COMMAND}"
        )
    return cache_path


def _set_mjwarp_force_space_solref_mode(builder: newton.ModelBuilder, shape_id: int) -> None:
    """Configure one task-owned MJWarp shape to use Newton force-space gains."""
    solref_mode = builder.custom_attributes.get("mujoco:solref_mode")
    if solref_mode is None:
        raise RuntimeError("MJWarp did not register its per-shape mujoco:solref_mode attribute.")
    if solref_mode.values is None:
        solref_mode.values = {}
    if not isinstance(solref_mode.values, dict):
        raise RuntimeError("MJWarp's per-shape mujoco:solref_mode storage must be sparse.")
    solref_mode.values[shape_id] = _MJWARP_SOLREF_MODE_FORCE_SPACE


class FrankaPourEnv(ManagerBasedRLEnv):
    """Franka grasping a dynamic cup of MPM media (Newton proxy-coupled MPM), pouring by tilting."""

    cfg: FrankaPourResetDatasetEnvCfg

    def __init__(self, cfg: FrankaPourResetDatasetEnvCfg, render_mode: str | None = None, **kwargs):
        _configure_mpm_capacities(cfg)
        self._prepare_newton_extras(cfg)
        with newton_builder_world_hook(self._add_pour_world_to_builder):
            super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self) -> None:
        self._setup_after_physics()
        super().load_managers()

    def _prepare_newton_extras(self, cfg: FrankaPourResetDatasetEnvCfg) -> None:
        """Cache task geometry bounds and Newton contact values from the resolved config.

        Runs before ``super().__init__`` so the per-world builder hook has the
        contact values available while the authored scene is imported.
        """
        self._source_inner_lo, self._source_inner_hi = SOURCE_CUP_GEOMETRY.inner_bounds
        self._target_inner_lo, self._target_inner_hi = TARGET_CUP_GEOMETRY.inner_bounds

        self._grasp_contact_ke = float(cfg.grasp_contact_ke)
        self._grasp_contact_kd = float(cfg.grasp_contact_kd)
        self._grasp_contact_kf = float(cfg.grasp_contact_kf)
        self._grasp_contact_mu = float(cfg.scene.source_cup.spawn.physics_material.static_friction)

        self._source_cup_friction = _SOURCE_CUP_PARTICLE_FRICTION
        self._target_cup_friction = float(cfg.scene.target_cup.spawn.physics_material.static_friction)
        self._rigid_contact_margin = float(cfg.collider_margin)
        self._mpm_collider_margin = MPM_COLLIDER_MARGIN
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
        source_bowl = self._find_world_shape(
            builder,
            shape_ids,
            env_id,
            "/SourceCup/geometry/bowl",
            body_id=source_body,
        )
        target_bowl = self._find_world_shape(
            builder,
            shape_ids,
            env_id,
            "/TargetCup/geometry/bowl",
            body_id=target_body,
        )
        target_rigid_collider = self._find_world_shape(
            builder,
            shape_ids,
            env_id,
            "/TargetCup/geometry/rigid_collider",
            body_id=target_body,
        )
        self._configure_particle_collider(
            builder,
            source_bowl,
            friction=self._source_cup_friction,
        )
        self._configure_particle_collider(
            builder,
            target_bowl,
            friction=self._target_cup_friction,
        )
        self._configure_rigid_collider(
            builder,
            target_rigid_collider,
            friction=self._target_cup_friction,
        )
        table_colliders = [
            shape_id
            for shape_id in shape_ids
            if str(builder.shape_label[shape_id]).endswith("/Table/Collisions/Cube")
            and int(builder.shape_flags[shape_id]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        if len(table_colliders) != 1:
            labels = [str(builder.shape_label[shape_id]) for shape_id in table_colliders]
            raise RuntimeError(
                f"Expected exactly one rigid SeattleLab tabletop collider in Newton world {env_id}, "
                f"found ids={table_colliders}, labels={labels}."
            )
        table_collider = table_colliders[0]
        table_xform = builder.shape_transform[table_collider]
        table_position = wp.transform_get_translation(table_xform)
        builder.shape_transform[table_collider] = wp.transform(
            wp.vec3(
                table_position[0],
                table_position[1],
                table_position[2] + _SEATTLELAB_TABLETOP_COLLISION_INSET,
            ),
            wp.transform_get_rotation(table_xform),
        )
        builder.shape_margin[table_collider] = 0.0
        self._configure_mjwarp_force_space_shape(builder, table_collider)

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
                margin=self._mpm_collider_margin,
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
        # The source cup starts in physical contact with the aligned tabletop. A contact margin
        # here would make its resting support margin-only and destabilize the contact manifold.
        builder.shape_margin[shape_id] = 0.0
        self._configure_mjwarp_force_space_shape(builder, shape_id)
        builder.shape_material_kf[shape_id] = self._grasp_contact_kf
        builder.shape_material_mu[shape_id] = self._grasp_contact_mu

    def _configure_mjwarp_force_space_shape(self, builder, shape_id: int) -> None:
        """Make MJWarp consume the configured Newton contact gains for one imported shape."""
        builder.shape_material_ke[shape_id] = self._grasp_contact_ke
        builder.shape_material_kd[shape_id] = self._grasp_contact_kd
        _set_mjwarp_force_space_solref_mode(builder, shape_id)

    def _configure_particle_collider(
        self,
        builder,
        shape_id: int,
        *,
        friction: float,
    ) -> None:
        """Configure an authored visible bowl mesh as a particle-only collider."""
        self._set_shape_roles(builder, shape_id, rigid=False, particles=True, visible=True)
        builder.shape_margin[shape_id] = self._mpm_collider_margin
        builder.shape_material_mu[shape_id] = friction

    def _configure_rigid_collider(
        self,
        builder,
        shape_id: int,
        *,
        friction: float,
    ) -> None:
        """Configure the authored receiver proxy as an invisible rigid-only collider."""
        self._set_shape_roles(builder, shape_id, rigid=True, particles=False, visible=False)
        builder.shape_margin[shape_id] = self._rigid_contact_margin
        builder.shape_material_ke[shape_id] = self._grasp_contact_ke
        builder.shape_material_kd[shape_id] = self._grasp_contact_kd
        builder.shape_material_kf[shape_id] = self._grasp_contact_kf
        builder.shape_material_mu[shape_id] = friction

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
        """Set explicit contact gains and friction on the two finger shapes.

        This preserves the imported MJWarp ``solref`` interpretation mode. The
        grasp proxy is separately configured for force-space interpretation.
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

    def _bind_scene_assets(self) -> None:
        """Resolve the scene assets and indices shared by runtime and offline sampling."""
        self._robot = self.scene["robot"]
        self._source_cup = self.scene["source_cup"]
        self._target_cup = self.scene["target_cup"]
        self._media: MPMObject = self.scene["media"]
        arm_joint_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True, as_proxy=True)
        finger_joint_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True, as_proxy=True)
        self._arm_joint_ids = arm_joint_ids.torch
        self._finger_joint_ids = finger_joint_ids.torch
        self._joint_pos_limits_t = self._robot.data.joint_pos_limits.torch.clone()
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)
        self._num_particles = int(self._media.particles_per_object)

    def _setup_after_physics(self) -> None:
        """Bind runtime state and load the externally generated reset dataset."""
        self._bind_scene_assets()
        dev = self.device
        tcp_body_ids, _ = self._robot.find_bodies(self.cfg.tcp_body_name)
        if len(tcp_body_ids) != 1:
            raise RuntimeError(
                f"Expected one TCP parent body named {self.cfg.tcp_body_name!r}, found {len(tcp_body_ids)}."
            )
        self._tcp_body_idx = tcp_body_ids[0]
        self._tcp_offset_pos = torch.tensor(self.cfg.tcp_offset_pos, device=dev).repeat(self.num_envs, 1)
        self._tcp_offset_quat = torch.tensor(self.cfg.tcp_offset_rot, device=dev).repeat(self.num_envs, 1)
        grasp_tcp_quat_c = torch.as_tensor(CUP_GRASP_TCP_QUAT_C, device=dev, dtype=torch.float32)
        self._desired_grasp_tcp_quat_c = math_utils.quat_unique(grasp_tcp_quat_c).repeat(self.num_envs, 1)
        self._gripper_open_width = 2.0 * float(self.cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"])
        self._gripper_grasp_width = 2.0 * SOURCE_CUP_GEOMETRY.outer_half_extents[1]
        self._cup_reset_height = float(self.cfg.scene.source_cup.init_state.pos[2])
        self._source_outer_half_t = torch.tensor(SOURCE_CUP_GEOMETRY.outer_half_extents, device=dev)
        self._target_outer_half_t = torch.tensor(TARGET_CUP_GEOMETRY.outer_half_extents, device=dev)

        self._particle_workspace_lower_t = torch.as_tensor(
            self.cfg.particle_workspace_lower_bound, device=dev, dtype=torch.float32
        )
        self._particle_workspace_upper_t = torch.as_tensor(
            self.cfg.particle_workspace_upper_bound, device=dev, dtype=torch.float32
        )
        if not self.cfg.reset_dataset_path:
            raise ValueError("The Pour task requires reset_dataset_path.")
        self._load_reset_dataset(self.cfg.reset_dataset_path)

        self.reset_dataset_row_id = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
        self.pour_target_frac = torch.full((self.num_envs,), float(self.cfg.pour_target_frac), device=dev)
        self.episode_succeeded = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._success_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lifted_grasp_seen = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._source_inner_lo_t = torch.as_tensor(self._source_inner_lo, device=dev)
        self._source_inner_hi_t = torch.as_tensor(self._source_inner_hi, device=dev)
        self._target_inner_lo_t = torch.as_tensor(self._target_inner_lo, device=dev)
        self._target_inner_hi_t = torch.as_tensor(self._target_inner_hi, device=dev)
        self._particle_region_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._particle_region_cache_step = -1
        self._particle_pos_e_cache: torch.Tensor | None = None
        self._particle_pos_e_cache_step = -1

    def _load_reset_dataset(self, configured_path: str) -> None:
        """Load and stage a validated direct-state dataset on the simulation device."""
        from .pour_env_cfg import _reset_dataset_task_contract

        cache_path = _resolve_reset_dataset_path(configured_path)
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        _, states, layouts = reset_dataset_validate_runtime(
            payload,
            expected_content_sha256=self.cfg.reset_dataset_content_sha256,
            expected_task_contract=_reset_dataset_task_contract(self.cfg),
        )

        local_position = layouts["local_position"].to(device=self.device, dtype=torch.float32)
        local_velocity = layouts["local_velocity"].to(device=self.device, dtype=torch.float32)
        expected_particle_shape = (self._num_particles, 3)
        if local_position.shape[1:] != expected_particle_shape:
            raise RuntimeError(
                "Reset-dataset particle layout does not match the runtime media shape: "
                f"{tuple(local_position.shape[1:])} != {expected_particle_shape}."
            )

        self._reset_dataset_states = {
            name: states[name].to(device=self.device, non_blocking=True) for name in RESET_DATASET_STATE_NAMES
        }
        self._reset_dataset_particle_local_position = local_position
        self._reset_dataset_particle_local_velocity = local_velocity
        logger.info(
            "Loaded reset dataset %s (%s) with %d rows.",
            cache_path,
            payload["content_sha256"],
            len(states["category"]),
        )

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

    def cup_pose_e(self) -> torch.Tensor:
        """Cup body pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._source_cup.data.root_link_pose_w.torch)

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

    def cup_grasp_point_e(self) -> torch.Tensor:
        """World-facing grasp point at the middle of the source cup walls, in env coordinates."""
        pose = self.cup_pose_e()
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = CUP_GRASP_HEIGHT
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], offset)

    def gripper_width(self) -> torch.Tensor:
        """Distance represented by the two symmetric Panda finger joint positions [m]."""
        finger_pos = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        width = finger_pos.sum(dim=-1)
        valid = torch.isfinite(finger_pos).all(dim=-1) & torch.isfinite(width)
        return torch.where(valid, width, torch.full_like(width, self._gripper_open_width))

    def particle_pos_e(self) -> torch.Tensor:
        """Per-env MPM particle positions in env coordinates, shape ``(N, P, 3)``."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._particle_pos_e_cache is None or self._particle_pos_e_cache_step != step:
            self._particle_pos_e_cache = self._media.data.particle_pos_w.torch - self.env_origins[:, None, :]
            self._particle_pos_e_cache_step = step
        return self._particle_pos_e_cache

    def _particle_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Boolean ``(source, target, spilled)`` masks, cached within one manager step."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._particle_region_cache is not None and self._particle_region_cache_step == step:
            return self._particle_region_cache
        points = self.particle_pos_e()
        margin = float(self.cfg.particle_count_margin)
        source = points_inside_box(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t, margin)
        target_region = points_inside_box(
            points,
            self.target_pose_e(),
            self._target_inner_lo_t,
            self._target_inner_hi_t,
            margin,
        )
        # Geometric overlap is not delivery: nesting the source cup inside the receiver must not
        # score particles that remain physically contained by the source cup.
        target = target_region & ~source
        spill_margin = max(margin, MPM_COLLIDER_MARGIN)
        spill_height = float(self.cfg.spill_table_height) + spill_margin
        spilled = ~(source | target) & (points[..., 2] <= spill_height)
        self._particle_region_cache = (source, target, spilled)
        self._particle_region_cache_step = step
        return source, target, spilled

    def _reset_from_dataset(self, env_ids: torch.Tensor, world_mask: torch.Tensor) -> None:
        """Restore exact dataset rows and clear all per-world solver history."""
        rows = self.reset_dataset_row_id[env_ids]
        # The curriculum value-validates forced overrides before assigning rows; random and
        # adaptive assignments are constructed from the loaded dataset's exact row range.
        states = self._reset_dataset_states
        arm_q = states["arm_joint_position"][rows]
        arm_qd = states["arm_joint_velocity"][rows]
        finger_q = states["finger_joint_position"][rows]
        finger_qd = states["finger_joint_velocity"][rows]
        finger_target = states["finger_joint_target"][rows]

        self._robot.write_joint_state_to_sim_index(
            position=arm_q,
            velocity=arm_qd,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_state_to_sim_index(
            position=finger_q,
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
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )
        self._particle_region_cache = None
        self._particle_region_cache_step = -1
        self._particle_pos_e_cache = None
        self._particle_pos_e_cache_step = -1
        self.episode_succeeded[env_ids] = False
        self._success_dwell_count[env_ids] = 0
        self._lost_grasp_dwell_count[env_ids] = 0
        # Seed the dropped-grasp latch for validated grasp rows; non-grasp rows start clear.
        self._lifted_grasp_seen[env_ids] = states["category"][rows] == GRASPING_CATEGORY

    def reset_pour_scene(self, env_ids: torch.Tensor) -> None:
        """Restore selected environments from reset-dataset rows."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return
        # Newton reset masks include one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, device=self.device, dtype=torch.bool)
        world_mask[env_ids] = True
        self._reset_from_dataset(env_ids, world_mask)
