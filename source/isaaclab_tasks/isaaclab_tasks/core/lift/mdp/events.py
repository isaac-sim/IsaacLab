# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the lift environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from tqdm import tqdm

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.envs.reset import ResetStateBuffer, ResetStateSampler, ResetStateSamplerCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils import replace
from isaaclab.utils.math import quat_apply, quat_mul, random_orientation, sample_uniform, sample_uniform_from_ranges
from isaaclab.utils.warp.memory import resolve_1d_mask

from .utils import (
    collect_body_collision_meshes,
    collect_collision_meshes,
    farthest_point_sampling,
    get_reset_state,
    sample_object_point_cloud,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation, CableObject, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv

    from .events_cfg import GraspTravelDistanceCfg, MeshClearanceCfg, SlabClearanceCfg


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
    offset = sample_uniform(position_range[0], position_range[1], (default.shape[0], 1), device=default.device)
    positions = (default + offset).clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=positions, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(positions), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )


def reset_cable_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> None:
    """Reset a cable to its default shape with a uniformly sampled translation [m]."""
    asset: CableObject = env.scene[asset_cfg.name]
    segment_pose = asset.data.default_segment_pose_w.torch[env_ids].clone()
    segment_velocity = asset.data.default_segment_velocity_w.torch[env_ids].clone()
    offset = sample_uniform_from_ranges(position_range, ("x", "y", "z"), segment_pose.shape[0], device=asset.device)
    segment_pose[..., :3] += offset.unsqueeze(1)
    asset.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
    asset.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity, env_ids=env_ids)


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
    env_ids = env.scene._ALL_INDICES[env_ids]
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


class reset_to_grasp(ManagerTermBase):
    """Place selected object variants in aligned parallel-gripper pre-grasps."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        gripper_cfg: SceneEntityCfg = cfg.params["gripper_cfg"]
        target_cfg: SceneEntityCfg = cfg.params["target_cfg"]
        self._asset = env.scene[asset_cfg.name]
        self._gripper = env.scene[gripper_cfg.name]
        self._target = env.scene[target_cfg.name]
        self._target_body_ids = target_cfg.body_ids

        object_cfg = getattr(env.cfg.scene, asset_cfg.name)
        plan = env.scene.clone_plan
        object_prototypes = cloner.path.get_asset_prototypes(plan, object_cfg.prim_path)
        if len(object_prototypes) == 0:
            raise ValueError(f"Could not find clone-plan prototypes for asset '{asset_cfg.name}'.")

        grasp_configs = cfg.params["grasp_configs"]
        grasp_by_geometry = {_grasp_geometry(shape): index for index, (shape, _, _) in enumerate(grasp_configs)}
        if len(grasp_by_geometry) != len(grasp_configs):
            raise ValueError("reset_to_grasp requires one unambiguous pre-grasp per object geometry.")

        variant_ids = np.full(env.num_envs, -1, dtype=np.int64)
        for prototype_id in object_prototypes:
            spawn = plan.asset_cfgs[prototype_id].spawn
            shape = spawn.assets_cfg[0] if isinstance(spawn, sim_utils.MultiAssetSpawnerCfg) else spawn
            geometry = _grasp_geometry(shape)
            if geometry not in grasp_by_geometry:
                raise ValueError(f"reset_to_grasp has no configured pre-grasp for object geometry {geometry}.")

            world_ids, _ = cloner.query.get_asset_prototype_unique_world_index(plan.topology, int(prototype_id))
            world_ids = world_ids[world_ids >= 0]
            if (variant_ids[world_ids] >= 0).any():
                raise ValueError(f"Multiple object variants are assigned to asset '{asset_cfg.name}' in one world.")
            variant_ids[world_ids] = grasp_by_geometry[geometry]
        if (variant_ids < 0).any():
            raise ValueError(f"No object variant is assigned to asset '{asset_cfg.name}' in some worlds.")

        self._variant_ids = torch.as_tensor(variant_ids, device=env.device)
        self._gripper_joint_ids = self._gripper.find_joints(gripper_cfg.joint_names)[0]
        self._gripper_joint_positions = torch.tensor([opening for _, opening, _ in grasp_configs], device=env.device)
        self._asset_orientations = torch.tensor([orientation for _, _, orientation in grasp_configs], device=env.device)
        self._offset_ranges = torch.tensor(
            [cfg.params["pose_range"].get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device=env.device
        )
        self._zero_joint_velocities = torch.zeros((env.num_envs, len(self._gripper_joint_ids)), device=env.device)
        self._zero_root_velocities = torch.zeros((env.num_envs, 6), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        probability: float,
        target_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        grasp_configs: list[tuple[sim_utils.SpawnerCfg, float, tuple[float, float, float, float]]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> None:
        """Reset a fraction of environments to configured pre-grasps.

        Args:
            env: The environment.
            env_ids: Environments to reset.
            pose_range: Object-position offsets in the target frame [m].
            probability: Per-environment probability of applying the pre-grasp.
            target_cfg: Body defining the pre-grasp frame.
            gripper_cfg: Parallel gripper joints to pose.
            grasp_configs: Shape configs paired with a finger opening [m or rad, depending on joint type]
                and object orientation in the target frame in ``(x, y, z, w)`` order. Geometry, not list order,
                associates each pre-grasp with its cloned object. Static values are cached at initialization.
            asset_cfg: Object asset to reset.
        """
        env_ids = env.scene._ALL_INDICES[env_ids]
        picked = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
        if len(picked) == 0:
            return

        variant_ids = self._variant_ids[picked]

        target_pos = self._target.data.body_pos_w.torch[picked][:, self._target_body_ids, :].flatten(1)[:, :3]
        target_quat = self._target.data.body_quat_w.torch[picked][:, self._target_body_ids, :].flatten(1)[:, :4]
        local_offsets = sample_uniform(
            self._offset_ranges[:, 0], self._offset_ranges[:, 1], (len(picked), 3), device=env.device
        )
        positions = target_pos + quat_apply(target_quat, local_offsets)
        local_orientations = self._asset_orientations[variant_ids]
        orientations = quat_mul(target_quat, local_orientations)

        joint_positions = self._gripper_joint_positions[variant_ids, None].expand(-1, len(self._gripper_joint_ids))
        self._gripper.write_joint_position_to_sim_index(
            position=joint_positions, joint_ids=self._gripper_joint_ids, env_ids=picked
        )
        self._gripper.write_joint_velocity_to_sim_index(
            velocity=self._zero_joint_velocities[: len(picked)], joint_ids=self._gripper_joint_ids, env_ids=picked
        )
        self._gripper.set_joint_position_target_index(
            target=joint_positions, joint_ids=self._gripper_joint_ids, env_ids=picked
        )
        self._asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=picked)
        self._asset.write_root_velocity_to_sim_index(
            root_velocity=self._zero_root_velocities[: len(picked)], env_ids=picked
        )


def _grasp_geometry(shape: sim_utils.SpawnerCfg) -> tuple:
    """Identify parallel-gripper geometry independently of mesh authoring and prototype order."""
    if isinstance(shape, (sim_utils.CuboidCfg, sim_utils.MeshCuboidCfg)):
        return ("cuboid", *shape.size)
    if isinstance(shape, (sim_utils.SphereCfg, sim_utils.MeshSphereCfg)):
        return ("sphere", shape.radius)
    if isinstance(shape, (sim_utils.CapsuleCfg, sim_utils.MeshCapsuleCfg)):
        return ("capsule", shape.radius, shape.height, shape.axis)
    raise ValueError(f"reset_to_grasp does not support shape {type(shape).__name__}.")


@wp.kernel
def _restore_root_origin(
    mask: wp.array(dtype=wp.bool),
    origins: wp.array(dtype=wp.vec3f),
    pose: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    if mask[i]:
        pose[i] = wp.transformf(wp.transform_get_translation(pose[i]) + origins[i], wp.transform_get_rotation(pose[i]))


@wp.kernel
def _requested_restore_mask(
    selected: wp.array(dtype=wp.bool), invalid: wp.array(dtype=wp.bool), restored: wp.array(dtype=wp.bool)
):
    i = wp.tid()
    restored[i] = selected[i] and not invalid[i]


class conditional_reset(ManagerTermBase):
    """Run wrapped reset terms and guarantee the resulting states satisfy a criterion.

    Wraps a dict of ordinary reset event terms. The nested :class:`EventTermCfg` objects are
    resolved by the event manager's own at-play pass (nested term configs inside ``params``
    are processed recursively), so this term only *calls* them and never resolves functions,
    scene entities, or class terms itself.

    On the first reset the wrapped terms are re-rolled and the states satisfying
    :paramref:`valid_criteria` are harvested into a buffer of :paramref:`buffer_size_per_group`
    samples per group by rejection sampling. The prefill ignores ``env_ids`` and rolls every
    environment, since a partial first reset could otherwise never fill the groups it does not
    cover. The bank is then frozen: the wrapped terms and criteria never run again, and every
    reset from that point restores a banked sample to the requested environments.

    Given a :paramref:`diversity_feature` the prefill keeps the most spread-out states rather
    than the first valid ones, and given a :paramref:`reset_sampler` the restore is drawn by
    each state's measured success rate rather than uniformly.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sampler: ResetStateSampler | None = None
        self._bank: ResetStateBuffer | None = None
        self._success: wp.array | None = None
        self._all_mask = wp.ones(env.num_envs, dtype=wp.bool, device=env.device)
        self._mask = wp.empty_like(self._all_mask)
        self._restore_mask = wp.empty_like(self._all_mask)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Log how well the policy does across the bank, once a monitor is tracking it.

        Reported from here rather than from the reward term because the bank is what the number is
        about: it covers every state the policy is being restarted from, not just the episodes that
        happen to be ending now.
        """
        if self._success is not None:
            measured = self._sampler.episode_counts.torch > 0
            rates = self._sampler.success_rate.torch
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                rates * measured
            ).sum() / measured.sum().clamp_min(1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        terms: dict[str, EventTermCfg],
        valid_criteria: dict[str, ManagerTermBaseCfg],
        buffer_size_per_group: int = 20,
        max_prefill_iters: int | None = None,
        diversity_feature: ManagerTermBaseCfg | None = None,
        oversample_factor: float = 2.0,
        reset_sampler: ResetStateSamplerCfg | None = None,
        success_reward_term: str = "success",
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
                prefill continues until every group reaches its harvest target.
            diversity_feature: Optional descriptor as a term config (e.g.
                :class:`GraspTravelDistanceCfg`), evaluated as
                ``func(env, env_ids, **params) -> Tensor`` of shape [len(env_ids), feature_dim]
                over the freshly reset environments and resolved by the event manager like the
                criteria. When given, the bank is farthest-point sampled over this descriptor
                instead of taking the valid states first-come. Each column is rescaled to unit
                span before sampling, so the axes count equally however they are scaled.
            oversample_factor: Candidates to harvest per banked state. Higher values give the
                farthest-point pass more to choose from at a proportional prefill cost. Ignored
                without :paramref:`diversity_feature`.
            reset_sampler: Optional shared reset sampler configuration. When given, banked states are
                drawn from the configured scores and completed outcomes are recorded against their starts.
                None draws uniformly without monitoring outcomes.
            success_reward_term: Reward term supplying the episode outcome, which must keep a sticky
                per-environment ``succeeded`` buffer like :class:`success_reward` does. Only read
                with :paramref:`reset_sampler`.
        """

        if self._bank is None:
            env_ids = self._prefill(
                env,
                terms=terms,
                valid_criteria=valid_criteria,
                buffer_size_per_group=buffer_size_per_group,
                max_prefill_iters=max_prefill_iters,
                diversity_feature=diversity_feature,
                oversample_factor=oversample_factor,
                reset_sampler=reset_sampler,
                success_reward_term=success_reward_term,
            )

        mask = resolve_1d_mask(ids=env_ids, all_mask=self._all_mask, scratch_mask=self._mask)
        # The viewer marks manual resets as terminated, but leaves this natural-outcome mask unchanged.
        completed = wp.from_torch(env.reset_buf & (env.episode_length_buf > 0)) if self._success is not None else None
        self.reset_mask(mask, completed=completed)

    def _prefill(
        self,
        env: ManagerBasedEnv,
        *,
        terms: dict[str, EventTermCfg],
        valid_criteria: dict[str, ManagerTermBaseCfg],
        buffer_size_per_group: int,
        max_prefill_iters: int | None,
        diversity_feature: ManagerTermBaseCfg | None,
        oversample_factor: float,
        reset_sampler: ResetStateSamplerCfg | None,
        success_reward_term: str,
    ) -> torch.Tensor:
        """Collect each asset group's valid starts and bind native replay once.

        Generation perturbs every environment, so return all environment IDs for the
        first restore, including when the caller requested only a partial reset.
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

        # The plan already identifies the world prototype selected by each environment.
        prototypes, group_ids = np.unique(env.scene.clone_plan.topology.world_prototype_layout, return_inverse=True)
        groups = torch.as_tensor(group_ids, device=env.device)
        num_groups = len(prototypes)
        # without a descriptor there is nothing to spread over, so harvesting extra is waste
        harvest_size = buffer_size_per_group
        if diversity_feature is not None:
            harvest_size = round(buffer_size_per_group * oversample_factor)
        if buffer_size_per_group < 1 or harvest_size < buffer_size_per_group:
            raise ValueError("Each group must harvest at least as many states as it retains, with at least one kept.")
        capacity = num_groups * harvest_size
        filled = [0] * num_groups
        buffer = None
        descriptor = None
        reset_assets = list(env.scene.articulations) + list(env.scene.rigid_objects)
        iteration = 0

        # prefill ignores env_ids and rolls every environment: a partial first reset may
        # not cover all groups, and a group with no rolled envs could never fill
        all_ids = torch.arange(env.num_envs, device=env.device)

        with tqdm(
            total=capacity,
            desc="Prefilling reset buffer",
            unit="state",
            dynamic_ncols=True,
        ) as progress:
            while min(filled) < harvest_size:
                if max_prefill_iters is not None and iteration >= max_prefill_iters:
                    counts = {group: count for group, count in enumerate(filled) if count < harvest_size}
                    raise RuntimeError(
                        "conditional_reset: could not fill the reset-state buffer for every asset-combination "
                        f"group. Required {harvest_size} valid states per group, got {counts} after "
                        f"{max_prefill_iters} prefill iterations."
                    )
                iteration += 1
                valid_ids = all_ids[roll_once(all_ids)]
                valid_groups = groups[valid_ids]
                for group in valid_groups.unique().tolist():
                    remaining = harvest_size - filled[group]
                    if remaining == 0:
                        continue
                    take = valid_ids[valid_groups == group][:remaining]
                    state = get_reset_state(env, take, reset_assets, is_relative=True)
                    if buffer is None:
                        buffer = state.new_empty((capacity, state.shape[-1]))
                    row = group * harvest_size + filled[group]
                    buffer[row : row + len(take)] = state
                    if diversity_feature is not None:
                        # measured now: the next roll overwrites the states these describe
                        feature = diversity_feature.func(env, take, **diversity_feature.params)
                        if descriptor is None:
                            descriptor = feature.new_empty((capacity, feature.shape[-1]))
                        descriptor[row : row + len(take)] = feature
                    filled[group] += len(take)
                    progress.update(len(take))
        if descriptor is not None:
            buffer = self._keep_most_spread(buffer, descriptor, num_groups, harvest_size, buffer_size_per_group)
        self._initialize_replay(buffer, groups, num_groups, reset_sampler, success_reward_term)
        # drop the prefill-only terms/criteria so their device memory is freed
        terms.clear()
        valid_criteria.clear()
        # the rolls above perturbed every environment, so the first reset restores a
        # banked state to all of them, not just the requested subset
        return all_ids

    @staticmethod
    def _keep_most_spread(
        buffer: torch.Tensor, descriptor: torch.Tensor, num_groups: int, harvest_size: int, keep_size: int
    ) -> torch.Tensor:
        """Thin each group's harvest down to the ``keep_size`` states most spread over the descriptor.

        The returned buffer has a ``keep_size`` stride so each group stays contiguous.
        """
        kept = buffer.new_empty((num_groups * keep_size, buffer.shape[-1]))
        for group in range(num_groups):
            start = group * harvest_size
            feature = descriptor[start : start + harvest_size]
            # rescale to unit span per axis, otherwise the axis with the widest numbers decides the
            # spread on its own and the others are covered only incidentally
            feature = feature / (feature.amax(dim=0) - feature.amin(dim=0)).clamp(min=1e-6)
            picked = start + farthest_point_sampling(feature, keep_size)
            kept[group * keep_size : (group + 1) * keep_size] = buffer[picked]
        return kept

    def _initialize_replay(
        self,
        buffer: torch.Tensor,
        groups: torch.Tensor,
        num_groups: int,
        cfg: ResetStateSamplerCfg | None,
        success_reward_term: str,
    ) -> None:
        """Bind typed snapshots and full-environment scratch arrays before graph capture."""
        env = self._env
        success = None
        if cfg is None:
            cfg = ResetStateSamplerCfg(history_length=1, scores=[])
        else:
            success_term = env.reward_manager.get_term_cfg(success_reward_term).func
            success = wp.from_torch(success_term.succeeded)
        cfg = replace(cfg, seed=(cfg.seed + (env.cfg.seed or 0)) % 2**32)
        self._sampler = ResetStateSampler(
            cfg, env.num_envs, buffer.shape[0], env.device, group_ids=groups, num_groups=num_groups
        )
        self._origins = wp.from_torch(env.scene.env_origins, dtype=wp.vec3f)
        self._restore_states: dict[str, dict[str, wp.array]] = {}
        self._articulations = []
        self._rigid_objects = []
        states: dict[str, dict[str, wp.array]] = {}
        offset = 0
        for assets, articulation in ((env.scene.articulations, True), (env.scene.rigid_objects, False)):
            for name, asset in assets.items():
                fields = {
                    "root_pose": wp.from_torch(buffer[:, offset : offset + 7].contiguous(), dtype=wp.transformf),
                    "root_velocity": wp.from_torch(
                        buffer[:, offset + 7 : offset + 13].contiguous(), dtype=wp.spatial_vectorf
                    ),
                }
                offset += 13
                if articulation:
                    fields["joint_position"] = wp.from_torch(buffer[:, offset : offset + asset.num_joints].contiguous())
                    offset += asset.num_joints
                    fields["joint_velocity"] = wp.from_torch(buffer[:, offset : offset + asset.num_joints].contiguous())
                    offset += asset.num_joints
                states[name] = fields
                output = {
                    field: wp.empty((env.num_envs, *value.shape[1:]), dtype=value.dtype, device=env.device)
                    for field, value in fields.items()
                }
                self._restore_states[name] = output
                if articulation:
                    zero_velocity = wp.zeros((env.num_envs, asset.num_joints), dtype=wp.float32, device=env.device)
                    self._articulations.append((asset, output, zero_velocity))
                else:
                    self._rigid_objects.append((asset, output))
        self._bank = ResetStateBuffer(states)
        self._success = success

    def reset_mask(
        self,
        mask: wp.array | None = None,
        *,
        state_ids: wp.array | None = None,
        completed: wp.array | None = None,
    ) -> None:
        """Restore selected environments through the backend's mask writers.

        Complete prefill through the ordinary event call first. Newton/MJWarp supports CUDA
        graph capture of this method. Run it outside capture with OvPhysX or Kit-based PhysX.
        The full-environment mask and optional explicit catalog IDs may change between graph
        replays. Supply the full-environment ``completed`` mask for natural episode outcomes
        before the task's reward buffer is cleared. Exclude zero-step and unfinished episodes
        from this mask. None records no episode outcomes.
        """
        if self._bank is None:
            raise RuntimeError("Initialize conditional_reset through its event call before capturing reset_mask.")
        mask = resolve_1d_mask(mask=mask, all_mask=self._all_mask, scratch_mask=self._mask)
        if self._success is not None and completed is not None:
            self._sampler.reset(mask, success=self._success, valid=completed, state_ids=state_ids)
        else:
            self._sampler.reset(mask, state_ids=state_ids)
        if state_ids is not None:
            wp.launch(
                _requested_restore_mask,
                self._env.num_envs,
                [mask, self._sampler.invalid_requests.warp, self._restore_mask],
                device=self._env.device,
            )
            mask = self._restore_mask
        self._bank.get_state(self._sampler.indices, out=self._restore_states, mask=mask)
        for asset, fields, zero_velocity in self._articulations:
            wp.launch(
                _restore_root_origin,
                self._env.num_envs,
                [mask, self._origins, fields["root_pose"]],
                device=self._env.device,
            )
            asset.write_root_link_pose_to_sim_mask(root_pose=fields["root_pose"], env_mask=mask)
            asset.write_root_com_velocity_to_sim_mask(root_velocity=fields["root_velocity"], env_mask=mask)
            asset.write_joint_position_to_sim_mask(position=fields["joint_position"], env_mask=mask)
            asset.write_joint_velocity_to_sim_mask(velocity=fields["joint_velocity"], env_mask=mask)
            asset.actuators.target_command.set_position_mask(value=fields["joint_position"], env_mask=mask)
            asset.actuators.target_command.set_velocity_mask(value=zero_velocity, env_mask=mask)
        for asset, fields in self._rigid_objects:
            wp.launch(
                _restore_root_origin,
                self._env.num_envs,
                [mask, self._origins, fields["root_pose"]],
                device=self._env.device,
            )
            asset.write_root_link_pose_to_sim_mask(root_pose=fields["root_pose"], env_mask=mask)
            asset.write_root_com_velocity_to_sim_mask(root_velocity=fields["root_velocity"], env_mask=mask)


class grasp_travel_distance(ManagerTermBase):
    """How far the hand has to close on the object, and how far the object then has to travel.

    Two-axis spread descriptor for :class:`conditional_reset`, measuring the distance from the
    object to whichever measured robot body is farthest from it and the distance from the object to
    the middle of the goal region. Together they are what makes one reset state a different learning
    problem from another: a start with the object already between the fingertips and next to the
    goal asks for a different policy than one with the hand an arm's length away and the goal across
    the workspace. Spreading over both keeps the easy and hard corners that rejection sampling alone
    would leave rare, rather than spreading over one axis and letting the other fall where it may.

    The goal is redrawn from :attr:`GraspTravelDistanceCfg.command_name`'s ranges on every reset and
    is not stored with a banked state, so the second axis measures the middle of that range: the
    distance the object has to travel up to the per-episode spread of the goal itself.

    Configured with :class:`GraspTravelDistanceCfg`; called as ``(env, env_ids) -> Tensor`` of shape
    [len(env_ids), 2], columns ``[0]`` hand-to-object and ``[1]`` object-to-goal, in metres or in
    log-metres depending on :attr:`GraspTravelDistanceCfg.log_scale`.
    """

    # distances below this are treated as this far apart, keeping the logarithm finite
    MIN_DISTANCE = 1e-3

    cfg: GraspTravelDistanceCfg

    def __init__(self, cfg: GraspTravelDistanceCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._object = env.scene[cfg.object_name]
        self._body_ids = self._robot.find_bodies(cfg.body_names)[0]
        # taken from the config, not the command manager: descriptors are built when physics starts
        # playing, before any manager exists. Kept in the robot frame, where commands are sampled.
        command_cfg = getattr(env.cfg.commands, cfg.command_name, None)
        if command_cfg is None:
            raise ValueError(
                f"grasp_travel_distance: no command term named '{cfg.command_name}' to read the goal region from."
            )
        ranges = command_cfg.ranges
        self._goal_center_b = torch.tensor(
            [sum(ranges.pos_x) / 2, sum(ranges.pos_y) / 2, sum(ranges.pos_z) / 2], device=env.device
        )

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        object_pos = self._object.data.root_pos_w.torch[env_ids]
        body_pos = self._robot.data.body_pos_w.torch[env_ids][:, self._body_ids]
        grasp = torch.linalg.norm(body_pos - object_pos[:, None, :], dim=-1).amax(dim=-1, keepdim=True)
        goal_pos = self._robot.data.root_pos_w.torch[env_ids] + quat_apply(
            self._robot.data.root_quat_w.torch[env_ids], self._goal_center_b.expand(object_pos.shape[0], 3)
        )
        travel = torch.linalg.norm(goal_pos - object_pos, dim=-1, keepdim=True)
        feature = torch.cat([grasp, travel], dim=-1)
        return feature.clamp_min(self.MIN_DISTANCE).log() if self.cfg.log_scale else feature


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
    mesh with Warp signed-distance queries; the winding-number sign catches full containment.

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
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        source_paths = cloner.path.get_asset_prototype_paths(plan)
        for index in cloner.path.get_asset_prototypes(plan, self._object.cfg.prim_path):
            env_ids, _ = cloner.query.get_asset_prototype_unique_world_index(plan.topology, index)
            if not len(env_ids):
                continue
            source_path = source_paths[index]
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
        env_ids = env.scene._ALL_INDICES[env_ids]
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
        env_ids = env.scene._ALL_INDICES[env_ids]
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


def reset_deformable_over_support(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    clear_gap_range: tuple[float, float],
    support_cfg: tuple[SceneEntityCfg, SceneEntityCfg],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> None:
    """Reset a deformable object and move its supports with it.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Deformable displacement bounds [m] keyed by ``x``, ``y``, ``z``. The planar
            displacement is shared with the supports.
        clear_gap_range: Clear gap bounds [m] for a support pair.
        support_cfg: Negative-Y and positive-Y support pair.
        asset_cfg: Scene entity of the deformable object to reset.
    """
    deformable: DeformableObject = env.scene[asset_cfg.name]
    supports: tuple[RigidObject, RigidObject] = (env.scene[support_cfg[0].name], env.scene[support_cfg[1].name])

    num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
    offset = sample_uniform_from_ranges(position_range, ("x", "y", "z"), num_envs, device=deformable.device)

    nodal_state = deformable.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] += offset.unsqueeze(1)
    deformable.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    root_poses = [support.data.default_root_pose.torch[env_ids].clone() for support in supports]
    for root_pose in root_poses:
        root_pose[:, :3] += env.scene.env_origins[env_ids]
        root_pose[:, :2] += offset[:, :2]

    gap = sample_uniform(*clear_gap_range, (num_envs,), device=supports[0].device)
    center_y = 0.5 * (root_poses[0][:, 1] + root_poses[1][:, 1])
    thickness_neg = supports[0].cfg.spawn.size[1]
    thickness_pos = supports[1].cfg.spawn.size[1]
    root_poses[0][:, 1] = center_y - 0.5 * (gap + thickness_neg)
    root_poses[1][:, 1] = center_y + 0.5 * (gap + thickness_pos)

    for support, root_pose in zip(supports, root_poses):
        support.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        support.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(support.data.default_root_vel.torch[env_ids]), env_ids=env_ids
        )
