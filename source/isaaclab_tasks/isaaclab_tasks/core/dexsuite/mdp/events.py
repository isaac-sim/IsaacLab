# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the dexsuite tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from tqdm import tqdm

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.managers import EventTermCfg, ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, random_orientation, sample_uniform

from .utils import (
    collect_body_collision_meshes,
    collect_collision_meshes,
    get_reset_state,
    sample_object_point_cloud,
    set_reset_state,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .events_cfg import MeshClearanceCfg, SlabClearanceCfg


def reset_joints_shared_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Reset joints with a single shared offset per environment.

    Like :func:`isaaclab.envs.mdp.reset_joints_by_offset`, but all configured joints of an
    environment receive the SAME offset draw. For mechanically coupled joints (e.g. a
    two-finger gripper driven by one motor through an equality constraint), independent
    per-joint draws write constraint-violating states that the solver snaps shut at episode
    birth; a shared draw keeps the pair consistent while still randomizing the width.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        position_range: Uniform offset range around the default joint positions
            [m or rad, depending on joint type].
        asset_cfg: The asset and coupled joints to reset.
    """
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
    offset = sample_uniform(position_range[0], position_range[1], (len(env_ids), 1), device=default.device)
    positions = (default + offset).clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=positions, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(positions), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )


def reset_to_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    probability: float,
    target_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset a fraction of the assets onto a target body with a uniformly random orientation.

    Each environment is selected with :paramref:`probability`; selected assets are placed at
    the target body's current position plus a position offset sampled from
    :paramref:`pose_range` (``x``/``y``/``z`` keys; orientation is sampled uniformly from
    SO(3), so ``roll``/``pitch``/``yaw`` keys are ignored). Non-selected environments are left
    untouched, so this term composes with a preceding uniform reset as a spawn-in-hand
    curriculum: a small share of episodes starts with the object at the gripper, keeping
    contact-rich starts in the distribution. Interpenetrating draws are expected to be
    rejected by the criteria of the wrapping :class:`conditional_reset`.

    The target body pose is read from the live simulation state, so the term must run after
    the reset terms that pose the target asset (dict order inside the wrapper).

    Args:
        env: The environment.
        env_ids: Environments being reset.
        pose_range: Position offset ranges in the target body frame [m], keys ``x``/``y``/``z``.
        velocity_range: Velocity ranges, keys ``x``/``y``/``z``/``roll``/``pitch``/``yaw``
            [m/s, rad/s].
        probability: Per-environment selection probability.
        target_cfg: Target body (e.g. the gripper palm) to spawn the asset at.
        asset_cfg: The asset to reset.
    """
    picked = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    if len(picked) == 0:
        return
    asset = env.scene[asset_cfg.name]
    target = env.scene[target_cfg.name]
    target_pos = target.data.body_pos_w.torch[picked][:, target_cfg.body_ids, :].reshape(len(picked), -1)[:, :3]
    target_quat = target.data.body_quat_w.torch[picked][:, target_cfg.body_ids, :].reshape(len(picked), -1)[:, :4]

    keys = ("x", "y", "z")
    offsets = torch.tensor([tuple(pose_range.get(key, (0.0, 0.0))) for key in keys], device=asset.device)
    # offsets are expressed in the target body frame, so e.g. a +z range places the object
    # along the gripper approach axis (between the fingertips) at any hand orientation
    local_offsets = sample_uniform(offsets[:, 0], offsets[:, 1], (len(picked), 3), device=asset.device)
    positions = target_pos + quat_apply(target_quat, local_offsets)
    orientations = random_orientation(len(picked), device=asset.device)

    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    vel_ranges = torch.tensor([tuple(velocity_range.get(key, (0.0, 0.0))) for key in keys], device=asset.device)
    velocities = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(picked), 6), device=asset.device)

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=picked)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=picked)


class conditional_reset(ManagerTermBase):
    """Run wrapped reset terms and guarantee the resulting states satisfy a criterion.

    Wraps a dict of ordinary reset event terms. The nested :class:`EventTermCfg` objects are
    resolved by the event manager's own at-play pass (nested term configs inside ``params``
    are processed recursively), so this term only *calls* them and never resolves functions,
    scene entities, or class terms itself.

    On the first reset, the wrapped terms are re-rolled and the states satisfying
    :paramref:`valid_criteria` are harvested into a buffer of :paramref:`buffer_size_per_group`
    samples per group (rejection sampling, amortized once). The prefill ignores ``env_ids``
    and rolls every environment — a partial first reset could otherwise never fill the groups
    it does not cover — and since the rolls perturb all environments, the first reset then
    restores a banked sample to every environment. Every subsequent reset restores a random
    banked sample to the requested environments only — the wrapped terms and criteria never
    run again.

    The captured state is the reset surface of the scene (see :func:`get_reset_state`):
    root pose/velocity plus joint positions/velocities of every articulation, and the root
    pose/velocity of every rigid object, buffered relative to the environment origins so a
    sample harvested in one environment can be replayed in another.

    With heterogeneous cloning (e.g. multi-asset spawned objects), environments are only
    interchangeable within the same unique asset combination: a state harvested in a cube
    environment is not a valid state for a capsule environment. The buffer is therefore
    partitioned by the scene's clone plan — an environment's column of the plan's clone mask
    is its asset-combination signature — as ``[num_groups * buffer_size_per_group]`` rows,
    and failing environments are only patched from their own group's partition.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prefilled = False
        self._buffer = torch.empty(0, 0, device=env.device)
        self._reset_assets = list(env.scene.articulations) + list(env.scene.rigid_objects)
        self._group = torch.empty(0, dtype=torch.long, device=env.device)
        self._fill = torch.empty(0, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        terms: dict[str, EventTermCfg],
        valid_criteria: dict[str, ManagerTermBaseCfg],
        buffer_size_per_group: int = 20,
        max_prefill_iters: int | None = None,
    ):
        """Restore banked criterion-valid states, prefilling the bank on the first reset.

        Args:
            env: The environment.
            env_ids: Environments being reset. The prefill phase ignores this and rolls all
                environments so every asset-combination group can bank states; the first
                reset therefore restores a banked state to every environment.
            terms: Reset event terms to wrap, applied in insertion order during prefill.
                Resolved by the event manager before the first call.
            valid_criteria: Criteria as term configs (e.g. :class:`SlabClearanceCfg`,
                :class:`MeshClearanceCfg`), each evaluated as
                ``func(env, env_ids, **params) -> BoolTensor`` over the freshly reset
                environments and combined with logical AND. Resolved by the event manager
                like any nested term config.
            buffer_size_per_group: Required number of valid states to bank per unique asset
                combination during the prefill phase.
            max_prefill_iters: Optional re-roll budget for the prefill phase. If ``None``,
                prefill continues until every group reaches :paramref:`buffer_size_per_group`.
        """

        def roll_once(roll_ids: torch.Tensor) -> torch.Tensor:
            for term in terms.values():
                term.func(env, roll_ids, **term.params)
            # no explicit refresh needed: state writes invalidate the FK timestamps and the
            # criteria's kinematic reads recompute on demand
            ok = torch.ones(len(roll_ids), dtype=torch.bool, device=roll_ids.device)
            for criterion in valid_criteria.values():
                ok &= criterion.func(env, roll_ids, **criterion.params)
            return ok

        if not self._prefilled:
            # envs sharing a clone-mask column are clones of the same unique asset combination
            mask = env.scene.clone_plan.clone_mask.to(device=env.device, dtype=torch.uint8)
            self._group = torch.unique(mask.T, dim=0, return_inverse=True)[1]
            num_groups = int(self._group.max().item()) + 1
            self._fill = torch.zeros(num_groups, dtype=torch.long, device=env.device)
            iteration = 0

            # prefill ignores env_ids and rolls every environment: a partial first reset may
            # not cover all groups, and a group with no rolled envs could never fill
            all_ids = torch.arange(env.num_envs, device=env.device)

            with tqdm(
                total=num_groups * buffer_size_per_group,
                desc="Prefilling reset buffer",
                unit="state",
                dynamic_ncols=True,
            ) as progress:
                while not bool((self._fill >= buffer_size_per_group).all()):
                    if max_prefill_iters is not None and iteration >= max_prefill_iters:
                        short = torch.nonzero(self._fill < buffer_size_per_group).view(-1)
                        counts = {int(group): int(self._fill[group]) for group in short}
                        raise RuntimeError(
                            "conditional_reset: could not fill the reset-state buffer for every asset-combination "
                            f"group. Required {buffer_size_per_group} valid states per group, got {counts} after "
                            f"{max_prefill_iters} prefill iterations."
                        )
                    iteration += 1
                    valid_ids = all_ids[roll_once(all_ids)]
                    for group in torch.unique(self._group[valid_ids]).tolist():
                        filled = int(self._fill[group])
                        remaining = buffer_size_per_group - filled
                        if remaining <= 0:
                            continue
                        take = valid_ids[self._group[valid_ids] == group][:remaining]
                        if len(take) == 0:
                            continue
                        state = get_reset_state(env, take, self._reset_assets, is_relative=True)
                        if self._buffer.numel() == 0:
                            capacity = num_groups * buffer_size_per_group
                            self._buffer = torch.empty(
                                capacity, state.shape[-1], device=state.device, dtype=state.dtype
                            )
                        row = group * buffer_size_per_group + filled
                        self._buffer[row : row + len(take)] = state
                        self._fill[group] += len(take)
                        progress.update(len(take))
            self._prefilled = True
            # drop the prefill-only terms/criteria so their device memory is freed
            terms.clear()
            valid_criteria.clear()
            # the rolls above perturbed every environment, so the first reset restores a
            # banked state to all of them, not just the requested subset
            env_ids = all_ids

        # ``rand * fill`` floors to a uniform draw in ``[0, fill)``.
        groups = self._group[env_ids]
        donor = (torch.rand(len(env_ids), device=env_ids.device) * self._fill[groups]).long()
        rows = groups * buffer_size_per_group + donor
        set_reset_state(env, self._buffer[rows], env_ids, self._reset_assets, is_relative=True)


@wp.func
def _slab_signed_dist(
    point_env: wp.vec3,
    slab_top: wp.array(dtype=wp.float32),
    slab_x: wp.array2d(dtype=wp.float32),
    slab_y: wp.array2d(dtype=wp.float32),
    slab_has_x: wp.array(dtype=wp.int32),
    slab_has_y: wp.array(dtype=wp.int32),
    margin: float,
    current: float,
) -> float:
    result = current
    for s in range(slab_top.shape[0]):
        inside = True
        if slab_has_x[s] != 0 and (point_env[0] < slab_x[s, 0] - margin or point_env[0] > slab_x[s, 1] + margin):
            inside = False
        if slab_has_y[s] != 0 and (point_env[1] < slab_y[s, 0] - margin or point_env[1] > slab_y[s, 1] + margin):
            inside = False
        if inside:
            result = wp.min(result, point_env[2] - slab_top[s])
    return result


@wp.kernel
def _object_points_slab_min(
    env_ids: wp.array(dtype=wp.int32),
    obj_points: wp.array2d(dtype=wp.vec3),
    obj_pose: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3),
    slab_top: wp.array(dtype=wp.float32),
    slab_x: wp.array2d(dtype=wp.float32),
    slab_y: wp.array2d(dtype=wp.float32),
    slab_has_x: wp.array(dtype=wp.int32),
    slab_has_y: wp.array(dtype=wp.int32),
    margin: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    point_env = wp.transform_point(obj_pose[env], obj_points[env, k]) - env_origins[env]
    dist = _slab_signed_dist(point_env, slab_top, slab_x, slab_y, slab_has_x, slab_has_y, margin, out_min[i])
    wp.atomic_min(out_min, i, dist)


@wp.kernel
def _robot_vertices_slab_min(
    env_ids: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    vertex_body: wp.array(dtype=wp.int32),
    body_pose: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3),
    slab_top: wp.array(dtype=wp.float32),
    slab_x: wp.array2d(dtype=wp.float32),
    slab_y: wp.array2d(dtype=wp.float32),
    slab_has_x: wp.array(dtype=wp.int32),
    slab_has_y: wp.array(dtype=wp.int32),
    margin: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    point_env = wp.transform_point(body_pose[env, vertex_body[k]], vertices[k]) - env_origins[env]
    dist = _slab_signed_dist(point_env, slab_top, slab_x, slab_y, slab_has_x, slab_has_y, margin, out_min[i])
    wp.atomic_min(out_min, i, dist)


@wp.kernel
def _object_points_mesh_min(
    env_ids: wp.array(dtype=wp.int32),
    obj_points: wp.array2d(dtype=wp.vec3),
    obj_pose: wp.array(dtype=wp.transformf),
    body_pose: wp.array2d(dtype=wp.transformf),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_body: wp.array(dtype=wp.int32),
    mesh_center: wp.array(dtype=wp.vec3),
    mesh_radius: wp.array(dtype=wp.float32),
    min_clearance: float,
    max_dist: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    point_world = wp.transform_point(obj_pose[env], obj_points[env, k])
    dist = out_min[i]
    for m in range(mesh_ids.shape[0]):
        point_local = wp.transform_point(wp.transform_inverse(body_pose[env, mesh_body[m]]), point_world)
        # bounding-sphere prefilter: |p - center| - radius lower-bounds the signed distance,
        # so skipped pairs cannot flip ``out_min >= min_clearance``
        lower_bound = wp.length(point_local - mesh_center[m]) - mesh_radius[m]
        if lower_bound > min_clearance and lower_bound > 0.0:
            continue
        query = wp.mesh_query_point_sign_winding_number(mesh_ids[m], point_local, max_dist)
        if query.result:
            mesh_dist = wp.length(point_local - wp.mesh_eval_position(mesh_ids[m], query.face, query.u, query.v))
            if query.sign < 0.0:
                mesh_dist = -mesh_dist
            dist = wp.min(dist, mesh_dist)
    wp.atomic_min(out_min, i, dist)


@wp.kernel
def _robot_vertices_object_mesh_min(
    env_ids: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    vertex_body: wp.array(dtype=wp.int32),
    body_pose: wp.array2d(dtype=wp.transformf),
    obj_pose: wp.array(dtype=wp.transformf),
    env_object_mesh: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_center: wp.array(dtype=wp.vec3),
    mesh_radius: wp.array(dtype=wp.float32),
    min_clearance: float,
    max_dist: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    mesh_index = env_object_mesh[env]
    point_world = wp.transform_point(body_pose[env, vertex_body[k]], vertices[k])
    point_local = wp.transform_point(wp.transform_inverse(obj_pose[env]), point_world)
    dist = out_min[i]

    lower_bound = wp.length(point_local - mesh_center[mesh_index]) - mesh_radius[mesh_index]
    if lower_bound <= min_clearance or lower_bound <= 0.0:
        query = wp.mesh_query_point_sign_winding_number(mesh_ids[mesh_index], point_local, max_dist)
        if query.result:
            mesh_dist = wp.length(
                point_local - wp.mesh_eval_position(mesh_ids[mesh_index], query.face, query.u, query.v)
            )
            if query.sign < 0.0:
                mesh_dist = -mesh_dist
            dist = wp.min(dist, mesh_dist)

    wp.atomic_min(out_min, i, dist)


class mesh_clearance(ManagerTermBase):
    """Valid when the object and robot collision meshes clear each other.

    Reset draws can place the object overlapping the arm; the solver resolves the overlap
    ballistically at episode birth. Checks both the object's surface point cloud against the
    robot's collision meshes and the robot's collision vertices against the object's collision
    mesh with Warp signed-distance queries — the winding-number sign catches full containment.

    The object point cloud comes from the same sampler as the point-cloud observation (per
    clone-plan prototype, geometry-keyed cache), so with the default count the cloud is
    shared, not recomputed. Robot collision meshes are extracted once from the USD collision
    prims and baked into each body's frame.

    Configured with :class:`MeshClearanceCfg`; called as ``(env, env_ids) -> BoolTensor``,
    ``True`` where the state is valid.
    """

    cfg: MeshClearanceCfg

    def __init__(self, cfg: MeshClearanceCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        device = env.device
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._object = env.scene[cfg.object_name]
        points = sample_object_point_cloud(
            env.num_envs, cfg.num_object_points, self._object.cfg.prim_path, device=device
        )
        self._obj_points = wp.from_torch(points.contiguous(), dtype=wp.vec3)

        body_meshes, _ = collect_body_collision_meshes(self._robot, cfg.body_names)
        self._meshes = []
        mesh_body = []
        mesh_center = []
        mesh_radius = []
        for body_id, mesh in body_meshes.items():
            self._meshes.append(
                wp.Mesh(
                    points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=device),
                    support_winding_number=True,
                )
            )
            mesh_body.append(body_id)
            # bounding sphere (body frame) for the kernel's query prefilter
            center = mesh.vertices.mean(axis=0)
            mesh_center.append(center)
            mesh_radius.append(float(np.linalg.norm(mesh.vertices - center, axis=1).max()))
        self._mesh_ids = wp.array([mesh.id for mesh in self._meshes], dtype=wp.uint64, device=device)
        self._mesh_body = wp.array(mesh_body, dtype=wp.int32, device=device)
        self._mesh_center = wp.array(np.stack(mesh_center), dtype=wp.vec3, device=device)
        self._mesh_radius = wp.array(mesh_radius, dtype=wp.float32, device=device)

        vertices, vertex_body = [], []
        for body_id, mesh in body_meshes.items():
            vertices.append(mesh.vertices)
            vertex_body.extend([body_id] * len(mesh.vertices))
        self._vertices = wp.array(np.concatenate(vertices), dtype=wp.vec3, device=device)
        self._vertex_body = wp.array(vertex_body, dtype=wp.int32, device=device)

        object_meshes = []
        env_object_mesh = np.zeros(env.num_envs, dtype=np.int32)
        mesh_by_path: dict[str, int] = {}
        clone_plan = sim_utils.SimulationContext.instance().get_clone_plan()
        for _, _, source_path, env_ids in iter_clone_plan_matches(clone_plan, self._object.cfg.prim_path):
            if source_path not in mesh_by_path:
                object_prim = sim_utils.get_current_stage().GetPrimAtPath(source_path)
                object_mesh_by_id = collect_collision_meshes(object_prim, lambda prim: (0, object_prim))
                if not object_mesh_by_id:
                    raise RuntimeError(f"no collision meshes found under '{source_path}'.")
                object_scale = np.asarray(sim_utils.resolve_prim_scale(object_prim), dtype=np.float32)
                object_mesh = object_mesh_by_id[0]
                object_mesh.apply_scale(object_scale)
                mesh_by_path[source_path] = len(object_meshes)
                object_meshes.append(object_mesh)
            env_object_mesh[np.asarray(env_ids, dtype=np.int64)] = mesh_by_path[source_path]

        self._object_meshes = []
        object_mesh_center = []
        object_mesh_radius = []
        for mesh in object_meshes:
            self._object_meshes.append(
                wp.Mesh(
                    points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=device),
                    support_winding_number=True,
                )
            )
            center = mesh.vertices.mean(axis=0)
            object_mesh_center.append(center)
            object_mesh_radius.append(float(np.linalg.norm(mesh.vertices - center, axis=1).max()))
        self._object_mesh_ids = wp.array([mesh.id for mesh in self._object_meshes], dtype=wp.uint64, device=device)
        self._object_mesh_center = wp.array(np.stack(object_mesh_center), dtype=wp.vec3, device=device)
        self._object_mesh_radius = wp.array(object_mesh_radius, dtype=wp.float32, device=device)
        self._env_object_mesh = wp.array(env_object_mesh, dtype=wp.int32, device=device)

        # query horizon: must exceed both the clearance and plausible penetration depths so
        # contained points still resolve a (negative) signed distance
        self._max_dist = max(4.0 * cfg.min_clearance, 0.15)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        num = len(env_ids)
        out_min = wp.full(num, 1.0e6, dtype=wp.float32, device=env.device)
        wp.launch(
            _object_points_mesh_min,
            dim=(num, self.cfg.num_object_points),
            inputs=[
                wp.from_torch(env_ids.to(torch.int32).contiguous()),
                self._obj_points,
                self._object.data.root_link_pose_w.warp,
                self._robot.data.body_link_pose_w.warp,
                self._mesh_ids,
                self._mesh_body,
                self._mesh_center,
                self._mesh_radius,
                self.cfg.min_clearance,
                self._max_dist,
                out_min,
            ],
            device=env.device,
        )
        wp.launch(
            _robot_vertices_object_mesh_min,
            dim=(num, self._vertices.shape[0]),
            inputs=[
                wp.from_torch(env_ids.to(torch.int32).contiguous()),
                self._vertices,
                self._vertex_body,
                self._robot.data.body_link_pose_w.warp,
                self._object.data.root_link_pose_w.warp,
                self._env_object_mesh,
                self._object_mesh_ids,
                self._object_mesh_center,
                self._object_mesh_radius,
                self.cfg.min_clearance,
                self._max_dist,
                out_min,
            ],
            device=env.device,
        )
        return wp.to_torch(out_min) >= self.cfg.min_clearance


class slab_clearance(ManagerTermBase):
    """Valid when the object's surface and the robot's collision geometry clear the slabs.

    Reset draws can pose the arm into the table (depenetration slams joints to several times
    their velocity limit within steps) and spawn long shapes with random orientation
    intersecting the tabletop. Checks the object's surface point cloud and the robot's
    collision-mesh vertices against horizontal obstacle slabs in the environment frame.

    Configured with :class:`SlabClearanceCfg`; called as ``(env, env_ids) -> BoolTensor``,
    ``True`` where the state is valid.
    """

    cfg: SlabClearanceCfg

    def __init__(self, cfg: SlabClearanceCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        device = env.device
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._object = env.scene[cfg.object_name]
        points = sample_object_point_cloud(
            env.num_envs, cfg.num_object_points, self._object.cfg.prim_path, device=device
        )
        self._obj_points = wp.from_torch(points.contiguous(), dtype=wp.vec3)

        body_meshes, _ = collect_body_collision_meshes(self._robot, cfg.body_names)
        vertices, vertex_body = [], []
        for body_id, mesh in body_meshes.items():
            vertices.append(mesh.vertices)
            vertex_body.extend([body_id] * len(mesh.vertices))
        self._vertices = wp.array(np.concatenate(vertices), dtype=wp.vec3, device=device)
        self._vertex_body = wp.array(vertex_body, dtype=wp.int32, device=device)

        slabs = cfg.obstacle_slabs
        self._slab_args = [
            wp.array(np.array([top_z for _, _, top_z in slabs], dtype=np.float32), device=device),
            wp.array(np.array([x or (0.0, 0.0) for x, _, _ in slabs], dtype=np.float32).reshape(-1, 2), device=device),
            wp.array(np.array([y or (0.0, 0.0) for _, y, _ in slabs], dtype=np.float32).reshape(-1, 2), device=device),
            wp.array(np.array([x is not None for x, _, _ in slabs], dtype=np.int32), device=device),
            wp.array(np.array([y is not None for _, y, _ in slabs], dtype=np.int32), device=device),
        ]
        self._env_origins = wp.from_torch(env.scene.env_origins.contiguous(), dtype=wp.vec3)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        num = len(env_ids)
        out_min = wp.full(num, 1.0e6, dtype=wp.float32, device=env.device)
        ids = wp.from_torch(env_ids.to(torch.int32).contiguous())
        wp.launch(
            _object_points_slab_min,
            dim=(num, self.cfg.num_object_points),
            inputs=[
                ids,
                self._obj_points,
                self._object.data.root_link_pose_w.warp,
                self._env_origins,
                *self._slab_args,
                self.cfg.min_clearance,
                out_min,
            ],
            device=env.device,
        )
        wp.launch(
            _robot_vertices_slab_min,
            dim=(num, len(self._vertices)),
            inputs=[
                ids,
                self._vertices,
                self._vertex_body,
                self._robot.data.body_link_pose_w.warp,
                self._env_origins,
                *self._slab_args,
                self.cfg.min_clearance,
                out_min,
            ],
            device=env.device,
        )
        return wp.to_torch(out_min) >= self.cfg.min_clearance
