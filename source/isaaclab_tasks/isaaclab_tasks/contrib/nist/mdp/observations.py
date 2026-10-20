# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory observations for static and reset-selectable assemblies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab import cloner
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils import resolve_matching_prims_from_source

from ..utils.pose_offset import Offset
from .assembly_variants import assembly_variant_context

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


_IDENTITY_OFFSET = Offset()


@wp.kernel
def _scene_point_cloud_in_root_frame(
    fixed_points: wp.array2d(dtype=wp.vec3f),
    held_points: wp.array2d(dtype=wp.vec3f),
    robot_points: wp.array(dtype=wp.vec3f),
    robot_body_ids: wp.array(dtype=wp.int32),
    variant_ids: wp.array(dtype=wp.int32),
    fixed_poses: wp.array2d(dtype=wp.transformf),
    held_poses: wp.array2d(dtype=wp.transformf),
    robot_poses: wp.array2d(dtype=wp.transformf),
    root_poses: wp.array(dtype=wp.transformf),
    point_counts: wp.vec3i,
    output: wp.array2d(dtype=wp.vec3f),
):
    env_id, point_id = wp.tid()
    fixed_end = point_counts[0]
    held_end = fixed_end + point_counts[1]

    if point_id < fixed_end:
        local_point = fixed_points[variant_ids[env_id], point_id]
        point_w = wp.transform_point(fixed_poses[env_id, 0], local_point)
    elif point_id < held_end:
        held_point_id = point_id - fixed_end
        local_point = held_points[variant_ids[env_id], held_point_id]
        point_w = wp.transform_point(held_poses[env_id, 0], local_point)
    else:
        robot_point_id = point_id - held_end
        local_point = robot_points[robot_point_id]
        point_w = wp.transform_point(robot_poses[env_id, robot_body_ids[robot_point_id]], local_point)

    root_pose = root_poses[env_id]
    root_pos = wp.transform_get_translation(root_pose)
    root_quat = wp.transform_get_rotation(root_pose)
    output[env_id, point_id] = wp.quat_rotate_inv(root_quat, point_w - root_pos)


@wp.kernel
def _relative_asset_pose(
    target_poses: wp.array2d(dtype=wp.transformf),
    root_poses: wp.array2d(dtype=wp.transformf),
    offset_ids: wp.array(dtype=wp.int32),
    target_offsets: wp.array(dtype=wp.transformf),
    root_offsets: wp.array(dtype=wp.transformf),
    body_indices: wp.vec2i,
    output: wp.array(dtype=wp.transformf),
):
    env_id = wp.tid()
    offset_id = offset_ids[env_id]
    target_offset = target_offsets[offset_id]
    root_offset = root_offsets[offset_id]
    target_pose = target_poses[env_id, body_indices[0]]
    root_pose = root_poses[env_id, body_indices[1]]
    target_quat = wp.transform_get_rotation(target_pose)
    root_quat = wp.transform_get_rotation(root_pose)
    relative_pos = wp.transform_get_translation(target_pose) - wp.transform_get_translation(root_pose)
    relative_pos += wp.quat_rotate(target_quat, wp.transform_get_translation(target_offset))
    relative_pos -= wp.quat_rotate(root_quat, wp.transform_get_translation(root_offset))
    target_quat = target_quat * wp.transform_get_rotation(target_offset)
    root_quat = root_quat * wp.transform_get_rotation(root_offset)

    root_quat_inv = wp.quat_inverse(root_quat)
    output[env_id] = wp.transformf(wp.quat_rotate(root_quat_inv, relative_pos), root_quat_inv * target_quat)


@wp.kernel
def _asset_link_velocity_in_root_frame(
    target_velocities: wp.array2d(dtype=wp.spatial_vectorf),
    target_poses: wp.array2d(dtype=wp.transformf),
    root_poses: wp.array(dtype=wp.transformf),
    target_body_idx: int,
    has_offset: bool,
    offset: wp.vec3f,
    output: wp.array(dtype=wp.spatial_vectorf),
):
    env_id = wp.tid()
    velocity = target_velocities[env_id, target_body_idx]
    linear_velocity = wp.spatial_top(velocity)
    angular_velocity = wp.spatial_bottom(velocity)
    if has_offset:
        target_quat = wp.transform_get_rotation(target_poses[env_id, target_body_idx])
        linear_velocity += wp.cross(angular_velocity, wp.quat_rotate(target_quat, offset))

    root_quat = wp.transform_get_rotation(root_poses[env_id])
    output[env_id] = wp.spatial_vector(
        wp.quat_rotate_inv(root_quat, linear_velocity), wp.quat_rotate_inv(root_quat, angular_velocity)
    )


def _body_index(asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        return 0
    if len(body_ids) != 1:
        raise ValueError(f"{asset_cfg.name!r} must select exactly one body.")
    return body_ids[0]


def _sample_variant_points(
    env: ManagerBasedEnv, asset: RigidObject, num_points: int, source_indices: tuple[int, ...], seed: int
) -> torch.Tensor:
    from ..utils import mesh_ops
    from ..utils.rigid_object_hasher import RigidObjectHasher

    plan = env.sim.get_clone_plan()
    paths = cloner.path.get_asset_prototype_paths(plan)
    prototypes = cloner.path.get_asset_prototypes(plan, asset.cfg.prim_path)
    hasher = RigidObjectHasher(
        env.num_envs,
        asset.cfg.prim_path,
        device=env.device,
        rigid_body_root=True,
        compact_sources=True,
        source_paths=tuple(paths[prototypes[index]] for index in source_indices),
    )
    if hasher.num_root != asset.num_mesh_variants:
        raise ValueError(
            f"Expected {asset.num_mesh_variants} collision sources below {asset.cfg.prim_path!r}, found"
            f" {hasher.num_root}."
        )
    points = mesh_ops.sample_object_point_cloud(
        asset.num_mesh_variants,
        num_points,
        asset.cfg.prim_path,
        env.device,
        rigid_object_hasher=hasher,
        seed=seed,
    )
    if points is None:
        raise ValueError(f"No collision geometry found below {asset.cfg.prim_path!r}.")
    # Sampling order belongs to the checkpoint; kernel rows follow the live mesh-variant IDs.
    inverse = torch.tensor(source_indices, device=env.device).argsort()
    return points[inverse].contiguous()


def _sample_articulation_points(
    env: ManagerBasedEnv,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    num_points: int,
    seed: int,
    geometry_usd_path: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils import get_all_matching_child_prims
    from isaaclab.utils.assets import retrieve_file_path

    from ..utils import mesh_ops
    from ..utils.rigid_object_hasher import RigidObjectHasher

    if isinstance(asset_cfg.body_ids, slice):
        body_ids = list(range(asset.num_bodies))
    else:
        body_ids = asset_cfg.body_ids
    body_names = [asset.body_names[body_id] for body_id in body_ids]
    points_per_body, remainder = divmod(num_points, len(body_ids))
    body_counts = [points_per_body + (body_index < remainder) for body_index in range(len(body_ids))]

    point_clouds = []
    point_body_ids = []
    # A checkpoint may retain an observation mesh after the physical asset is updated.
    geometry_stage = Usd.Stage.Open(retrieve_file_path(geometry_usd_path)) if geometry_usd_path else None
    for body_id, body_name, body_count in zip(body_ids, body_names, body_counts, strict=True):
        if body_count == 0:
            continue
        if geometry_stage is None:
            source_prim, _ = resolve_matching_prims_from_source(
                asset.cfg.prim_path,
                predicate=lambda candidate, name=body_name: (
                    candidate.GetName() == name and candidate.HasAPI(UsdPhysics.RigidBodyAPI)
                ),
                expected_num_matches=1,
            )[0]
            prim_path = source_prim.GetPath().pathString
        else:
            matches = get_all_matching_child_prims(
                "/",
                predicate=lambda candidate, name=body_name: (
                    candidate.GetName() == name and candidate.HasAPI(UsdPhysics.RigidBodyAPI)
                ),
                stage=geometry_stage,
                traverse_instance_prims=True,
            )
            if len(matches) != 1:
                raise ValueError(f"Observation geometry must contain exactly one rigid body named {body_name!r}.")
            prim_path = matches[0].GetPath().pathString
        hasher = RigidObjectHasher(
            env.num_envs,
            prim_path,
            device=env.device,
            compact_sources=True,
            source_paths=(prim_path,),
            stage=geometry_stage,
        )
        if hasher.num_root != 1:
            raise ValueError(f"Robot body {prim_path!r} must have one clone source.")
        points = mesh_ops.sample_object_point_cloud(
            1, body_count, prim_path, env.device, rigid_object_hasher=hasher, seed=seed + body_id
        )
        if points is None:
            raise ValueError(f"No collision geometry found below {prim_path!r}.")
        point_clouds.append(points[0])
        point_body_ids.extend([body_id] * body_count)

    if not point_clouds:
        raise ValueError(f"No collision geometry found below {asset.cfg.prim_path!r}.")
    return (
        torch.cat(point_clouds).contiguous(),
        torch.tensor(point_body_ids, dtype=torch.int32, device=env.device),
    )


class scene_point_cloud_b(ManagerTermBase):
    """Fixed-asset, held-asset, and robot surface points in the robot root frame.

    ``sampling_variant_order`` is an optional full permutation of the assembly names. It
    preserves sampling streams when a scene's catalog order changes; output rows still follow
    scene variant IDs. ``sampling_seed`` defaults to 42. ``robot_geometry_usd_path`` optionally
    pins the robot's observation geometry independently of its physical asset. It must retain
    the selected rigid-body names and local frames. No reference points are loaded from disk.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg", SceneEntityCfg("fixed_asset"))
        held_asset_cfg: SceneEntityCfg = cfg.params.get("held_asset_cfg", SceneEntityCfg("held_asset"))
        robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        fixed_asset: RigidObject = env.scene[fixed_asset_cfg.name]
        held_asset: RigidObject = env.scene[held_asset_cfg.name]
        robot_asset: Articulation = env.scene[robot_asset_cfg.name]

        point_counts = (
            cfg.params.get("fixed_num_points", 256),
            cfg.params.get("held_num_points", 256),
            cfg.params.get("robot_num_points", 256),
        )
        if any(count <= 0 for count in point_counts):
            raise ValueError(f"Point counts must be positive, got {point_counts}.")

        variant_names = assembly_variant_context(env).variant_names
        sampling_order = cfg.params.get("sampling_variant_order")
        if sampling_order is None:
            sampling_order = variant_names
        if len(sampling_order) != len(variant_names) or set(sampling_order) != set(variant_names):
            raise ValueError("sampling_variant_order must be a full permutation of the scene's assembly names.")
        source_indices = tuple(variant_names.index(name) for name in sampling_order)
        seed = cfg.params.get("sampling_seed", 42)
        fixed_points = _sample_variant_points(env, fixed_asset, point_counts[0], source_indices, seed)
        held_points = _sample_variant_points(env, held_asset, point_counts[1], source_indices, seed)
        robot_points, robot_body_ids = _sample_articulation_points(
            env, robot_asset, robot_asset_cfg, point_counts[2], seed, cfg.params.get("robot_geometry_usd_path")
        )
        self._kernel_inputs = (
            wp.from_torch(fixed_points, dtype=wp.vec3f),
            wp.from_torch(held_points, dtype=wp.vec3f),
            wp.from_torch(robot_points, dtype=wp.vec3f),
            wp.from_torch(robot_body_ids, dtype=wp.int32),
            fixed_asset.mesh_variant_ids.warp,
            fixed_asset.data.body_link_pose_w.warp,
            held_asset.data.body_link_pose_w.warp,
            robot_asset.data.body_link_pose_w.warp,
            robot_asset.data.root_link_pose_w.warp,
            wp.vec3i(*point_counts),
        )

        output = wp.empty((env.num_envs, sum(point_counts)), dtype=wp.vec3f, device=env.device)
        self._kernel_outputs = (output,)
        output_torch = wp.to_torch(output)
        self._output_torch = output_torch.flatten(1) if cfg.params.get("flatten", True) else output_torch

    def __call__(
        self,
        env: ManagerBasedEnv,
        fixed_asset_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
        held_asset_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        fixed_num_points: int = 256,
        held_num_points: int = 256,
        robot_num_points: int = 256,
        flatten: bool = True,
        sampling_variant_order: tuple[str, ...] | None = None,
        sampling_seed: int = 42,
        robot_geometry_usd_path: str | None = None,
    ) -> torch.Tensor:
        """Return ordered surface points [m], with fixed, held, then robot points."""
        wp.launch(
            _scene_point_cloud_in_root_frame,
            dim=(self.num_envs, fixed_num_points + held_num_points + robot_num_points),
            inputs=self._kernel_inputs,
            outputs=self._kernel_outputs,
            device=self.device,
        )
        return self._output_torch


class target_asset_pose_in_root_asset_frame(ManagerTermBase):
    """Pose of one asset frame expressed in another, with static or variant offsets."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        target_asset_cfg: SceneEntityCfg = cfg.params["target_asset_cfg"]
        root_asset_cfg: SceneEntityCfg = cfg.params.get("root_asset_cfg", SceneEntityCfg("robot"))
        target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
        root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]
        offset_specs = (
            cfg.params.get("target_asset_offset", _IDENTITY_OFFSET),
            cfg.params.get("root_asset_offset", _IDENTITY_OFFSET),
        )
        if any(isinstance(offset, str) for offset in offset_specs):
            variants = assembly_variant_context(env)
            offset_ids = variants.variant_ids_warp
            offset_tables = tuple(variants.offset_warp(offset) for offset in offset_specs)
        else:
            offset_ids = wp.zeros(env.num_envs, dtype=wp.int32, device=env.device)
            offset_tables = tuple(
                wp.array([offset.pose], dtype=wp.transformf, device=env.device) for offset in offset_specs
            )
        target_poses = target_asset.data.body_link_pose_w.warp
        root_poses = target_poses if target_asset is root_asset else root_asset.data.body_link_pose_w.warp
        self._kernel_inputs = (
            target_poses,
            root_poses,
            offset_ids,
            *offset_tables,
            wp.vec2i(_body_index(target_asset_cfg), _body_index(root_asset_cfg)),
        )

        output = wp.empty(env.num_envs, dtype=wp.transformf, device=env.device)
        self._kernel_outputs = (output,)
        self._output_torch = wp.to_torch(output)

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_asset_cfg: SceneEntityCfg,
        root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_asset_offset: Offset | str = _IDENTITY_OFFSET,
        root_asset_offset: Offset | str = _IDENTITY_OFFSET,
    ) -> torch.Tensor:
        """Return position [m] and quaternion (x, y, z, w), shape ``(num_envs, 7)``."""
        wp.launch(
            _relative_asset_pose,
            dim=self.num_envs,
            inputs=self._kernel_inputs,
            outputs=self._kernel_outputs,
            device=self.device,
        )
        return self._output_torch


class asset_link_velocity_in_root_asset_frame(ManagerTermBase):
    """Linear and angular velocity of a target keypoint in an asset root frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        target_asset_cfg: SceneEntityCfg = cfg.params["target_asset_cfg"]
        root_asset_cfg: SceneEntityCfg = cfg.params.get("root_asset_cfg", SceneEntityCfg("robot"))
        self._target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
        self._root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

        self._target_body_idx = _body_index(target_asset_cfg)

        target_asset_offset: Offset | None = cfg.params.get("target_asset_offset")
        self._has_offset = target_asset_offset is not None
        self._offset = wp.vec3f(*(target_asset_offset.pos if target_asset_offset else (0.0, 0.0, 0.0)))
        self._output = wp.empty(env.num_envs, dtype=wp.spatial_vectorf, device=env.device)
        self._output_torch = wp.to_torch(self._output)

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_asset_cfg: SceneEntityCfg,
        root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_asset_offset: Offset = Offset(),
    ) -> torch.Tensor:
        """Return linear velocity [m/s] and angular velocity [rad/s], shape ``(num_envs, 6)``."""
        wp.launch(
            _asset_link_velocity_in_root_frame,
            dim=self.num_envs,
            inputs=[
                self._target_asset.data.body_com_vel_w.warp,
                self._target_asset.data.body_link_pose_w.warp,
                self._root_asset.data.root_link_pose_w.warp,
                self._target_body_idx,
                self._has_offset,
                self._offset,
            ],
            outputs=[self._output],
            device=self.device,
        )
        return self._output_torch
