# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from tqdm import tqdm

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.managers import EventTermCfg, ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, random_orientation, sample_uniform

from .utils import (
    collect_body_collision_meshes,
    collect_collision_meshes,
    farthest_point_sampling,
    get_reset_state,
    sample_object_point_cloud,
    set_reset_state,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .events_cfg import GraspTravelDistanceCfg, MeshClearanceCfg, SlabClearanceCfg, SuccessMonitorCfg


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


class SuccessMonitor:
    """Per-slot success-rate table whose draws favor a target success rate.

    Each monitored slot — for the reset bank, one banked state — keeps a ring buffer of its most
    recent episode outcomes. Sampling weights peak at :attr:`SuccessMonitorCfg.target_success_rate`,
    so with the default target of one half the draw concentrates on the states the policy solves
    about half the time and spends little on the ones it always solves or never solves.

    Slots are laid out as :paramref:`num_partitions` contiguous blocks of :paramref:`partition_size`
    and every draw stays inside one partition. The reset bank needs this because a state harvested
    in one asset-combination group cannot be replayed in another; callers with no such constraint
    pass a single partition.
    """

    def __init__(self, cfg: SuccessMonitorCfg, num_partitions: int, partition_size: int, device: str):
        """Allocate the outcome table.

        Args:
            cfg: Monitor configuration.
            num_partitions: Number of independently sampled blocks of slots.
            partition_size: Slots per partition.
            device: Device holding the table; slot ids passed in must live on it too.
        """
        self.cfg = cfg
        self.num_partitions = num_partitions
        self.partition_size = partition_size
        self.device = device

        num_slots = num_partitions * partition_size
        self.success_buf = torch.zeros((num_slots, cfg.monitored_history_len), device=device)
        self.success_rate = torch.zeros(num_slots, device=device)
        self.success_pointer = torch.zeros(num_slots, device=device, dtype=torch.long)
        self.success_size = torch.zeros(num_slots, device=device, dtype=torch.long)

    def get_success_rate(self) -> torch.Tensor:
        """Return a copy of every slot's measured success rate, shape [num_slots]."""
        return self.success_rate.clone()

    def get_mean_success_rate(self) -> float:
        """Return the success rate over the whole table, averaged across the slots that have episodes.

        Slots with no episodes yet are left out rather than counted as failures, so the number
        reports how well the policy does on the states it has actually been given and does not dip
        just because the table is still filling.
        """
        measured = self.success_size > 0
        if not bool(measured.any()):
            return 0.0
        return float(self.success_rate[measured].mean())

    def success_update(self, slot_ids: torch.Tensor, success: torch.Tensor):
        """Append episode outcomes to their slots' ring buffers and refresh the success rates.

        Slots may repeat within one call, which happens whenever several environments replayed the
        same banked state; the outcomes are appended in order and only the newest
        :attr:`SuccessMonitorCfg.monitored_history_len` of them survive.

        Args:
            slot_ids: Slot each outcome belongs to, shape [num_outcomes].
            success: Whether each episode succeeded, shape [num_outcomes].
        """
        if len(slot_ids) == 0:
            return
        history = self.cfg.monitored_history_len
        # group the outcomes by slot, stably so the newest stay last within each slot
        order = torch.argsort(slot_ids, stable=True)
        ordered_slots = slot_ids[order]
        unique_slots, counts = torch.unique_consecutive(ordered_slots, return_counts=True)

        # position of each outcome within its slot's append run, with anything beyond the ring's
        # capacity given a negative position and dropped
        starts = counts.cumsum(0) - counts
        offset = torch.arange(len(ordered_slots), device=self.device) - starts.repeat_interleave(counts)
        offset = offset - (counts - history).clamp(min=0).repeat_interleave(counts)
        kept = offset >= 0

        slots = ordered_slots[kept]
        positions = (self.success_pointer[slots] + offset[kept]) % history
        self.success_buf[slots, positions] = success[order][kept].to(dtype=self.success_buf.dtype)

        written = counts.clamp(max=history)
        self.success_pointer[unique_slots] = (self.success_pointer[unique_slots] + written) % history
        self.success_size[unique_slots] = (self.success_size[unique_slots] + written).clamp(max=history)
        # unwritten ring entries are zero, so the row sum is the number of successes remembered
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)

    def target_weights(self) -> torch.Tensor:
        """Return each slot's sampling weight, peaking at the target success rate.

        The weight is the Beta density shape ``p^(a-1) * (1-p)^(b-1)`` with ``a = 1 + kappa * target``
        and ``b = 1 + kappa * (1 - target)``, which places the mode at the target and flattens as
        ``kappa`` shrinks.

        Returns:
            Weights, shape [num_slots]. Unnormalized; :func:`torch.multinomial` normalizes them.
        """
        target = min(max(self.cfg.target_success_rate, 0.0), 1.0)
        kappa = max(self.cfg.kappa, 0.0)
        a = 1.0 + kappa * target
        b = 1.0 + kappa * (1.0 - target)

        # The offset avoids ``0 ** 0`` and, more importantly, floors the weight at the extremes:
        # slots that always succeed, always fail, or have not been drawn at all stay in circulation
        # instead of being starved for good, so their rates can still be revised.
        eps = 1e-4
        rate = self.success_rate
        weights = ((rate + eps).pow(a - 1.0) * (1.0 - rate + eps).pow(b - 1.0)).clamp_min(eps)
        # ``w ** (1/T)`` equals ``softmax(log(w) / T)`` up to the normalization the draw applies anyway
        return weights.pow(1.0 / max(self.cfg.temperature, 1.0))

    def sample_by_target_rate(self, partition_ids: torch.Tensor) -> torch.Tensor:
        """Draw one slot per requested partition, preferring slots near the target success rate.

        Args:
            partition_ids: Partition to draw from for each sample, shape [num_samples].

        Returns:
            Slot ids, shape [num_samples].
        """
        weights = self.target_weights().view(self.num_partitions, self.partition_size)
        slots = torch.multinomial(weights[partition_ids], 1).view(-1)
        return partition_ids * self.partition_size + slots


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
    than the first valid ones, and given a :paramref:`success_monitor` the restore is drawn by
    each state's measured success rate rather than uniformly.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prefilled = False
        self._buffer = torch.empty(0, 0, device=env.device)
        self._descriptor = torch.empty(0, 0, device=env.device)
        self._reset_assets = list(env.scene.articulations) + list(env.scene.rigid_objects)
        self._group = torch.empty(0, dtype=torch.long, device=env.device)
        self._fill = torch.empty(0, dtype=torch.long, device=env.device)
        self._monitor: SuccessMonitor | None = None
        self._success_term = None
        # bank row each environment is currently playing; -1 until its first restore
        self._playing_row = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Log how well the policy does across the bank, once a monitor is tracking it.

        Reported from here rather than from the reward term because the bank is what the number is
        about: it covers every state the policy is being restarted from, not just the episodes that
        happen to be ending now.
        """
        if self._monitor is not None:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._monitor.get_mean_success_rate()

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
        success_monitor: SuccessMonitorCfg | None = None,
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
            success_monitor: Optional :class:`SuccessMonitorCfg`. When given, banked states are drawn
                by success rate instead of uniformly, and each state's outcomes are recorded as the
                episodes started from it end.
            success_reward_term: Reward term supplying the episode outcome, which must keep a sticky
                per-environment ``succeeded`` buffer like :class:`success_reward` does. Only read
                with :paramref:`success_monitor`.
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
            # without a descriptor there is nothing to spread over, so harvesting extra is waste
            harvest_size = buffer_size_per_group
            if diversity_feature is not None:
                harvest_size = round(buffer_size_per_group * oversample_factor)
            self._fill = torch.zeros(num_groups, dtype=torch.long, device=env.device)
            iteration = 0

            # prefill ignores env_ids and rolls every environment: a partial first reset may
            # not cover all groups, and a group with no rolled envs could never fill
            all_ids = torch.arange(env.num_envs, device=env.device)

            with tqdm(
                total=num_groups * harvest_size,
                desc="Prefilling reset buffer",
                unit="state",
                dynamic_ncols=True,
            ) as progress:
                while not bool((self._fill >= harvest_size).all()):
                    if max_prefill_iters is not None and iteration >= max_prefill_iters:
                        short = torch.nonzero(self._fill < harvest_size).view(-1)
                        counts = {int(group): int(self._fill[group]) for group in short}
                        raise RuntimeError(
                            "conditional_reset: could not fill the reset-state buffer for every asset-combination "
                            f"group. Required {harvest_size} valid states per group, got {counts} after "
                            f"{max_prefill_iters} prefill iterations."
                        )
                    iteration += 1
                    valid_ids = all_ids[roll_once(all_ids)]
                    for group in torch.unique(self._group[valid_ids]).tolist():
                        filled = int(self._fill[group])
                        remaining = harvest_size - filled
                        if remaining <= 0:
                            continue
                        take = valid_ids[self._group[valid_ids] == group][:remaining]
                        if len(take) == 0:
                            continue
                        state = get_reset_state(env, take, self._reset_assets, is_relative=True)
                        if self._buffer.numel() == 0:
                            capacity = num_groups * harvest_size
                            self._buffer = torch.empty(
                                capacity, state.shape[-1], device=state.device, dtype=state.dtype
                            )
                        row = group * harvest_size + filled
                        self._buffer[row : row + len(take)] = state
                        if diversity_feature is not None:
                            # measured now: the next roll overwrites the states these describe
                            feature = diversity_feature.func(env, take, **diversity_feature.params)
                            if self._descriptor.numel() == 0:
                                self._descriptor = torch.empty(
                                    num_groups * harvest_size,
                                    feature.shape[-1],
                                    device=feature.device,
                                    dtype=feature.dtype,
                                )
                            self._descriptor[row : row + len(take)] = feature
                        self._fill[group] += len(take)
                        progress.update(len(take))
            if diversity_feature is not None:
                self._keep_most_spread(num_groups, harvest_size, buffer_size_per_group)
            if success_monitor is not None:
                # one monitored slot per banked state, partitioned exactly like the bank
                self._monitor = success_monitor.class_type(
                    success_monitor, num_groups, buffer_size_per_group, env.device
                )
            self._prefilled = True
            # drop the prefill-only terms/criteria so their device memory is freed
            terms.clear()
            valid_criteria.clear()
            # the rolls above perturbed every environment, so the first reset restores a
            # banked state to all of them, not just the requested subset
            env_ids = all_ids

        groups = self._group[env_ids]
        if self._monitor is None:
            # ``rand * fill`` floors to a uniform draw in ``[0, fill)``.
            donor = (torch.rand(len(env_ids), device=env_ids.device) * self._fill[groups]).long()
            rows = groups * buffer_size_per_group + donor
        else:
            # credit before drawing: the outcome belongs to the row these environments were playing
            self._credit_episodes(env, env_ids, self._monitor, success_reward_term)
            rows = self._monitor.sample_by_target_rate(groups)
            self._playing_row[env_ids] = rows
        set_reset_state(env, self._buffer[rows], env_ids, self._reset_assets, is_relative=True)

    def _keep_most_spread(self, num_groups: int, harvest_size: int, keep_size: int):
        """Thin each group's harvest down to the ``keep_size`` states most spread over the descriptor.

        Rewrites the buffer with a ``keep_size`` stride so the draw at the end of :meth:`__call__`
        addresses it unchanged, and frees the descriptors, which are prefill-only.
        """
        kept = torch.empty(
            num_groups * keep_size, self._buffer.shape[-1], device=self._buffer.device, dtype=self._buffer.dtype
        )
        for group in range(num_groups):
            start = group * harvest_size
            feature = self._descriptor[start : start + harvest_size]
            # rescale to unit span per axis, otherwise the axis with the widest numbers decides the
            # spread on its own and the others are covered only incidentally
            feature = feature / (feature.amax(dim=0) - feature.amin(dim=0)).clamp(min=1e-6)
            picked = start + farthest_point_sampling(feature, keep_size)
            kept[group * keep_size : (group + 1) * keep_size] = self._buffer[picked]
        self._buffer = kept
        self._fill.fill_(keep_size)
        self._descriptor = torch.empty(0, 0, device=self._descriptor.device)

    def _credit_episodes(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        monitor: SuccessMonitor,
        success_reward_term: str,
    ):
        """Record the outcome of the episode each environment just finished against the row it played.

        Reads the success flag of the reward term rather than recomputing it, and must therefore run
        while the flag still describes the finished episode: reset events fire before the reward
        manager resets its terms.
        """
        if self._success_term is None:
            self._success_term = env.reward_manager.get_term_cfg(success_reward_term).func
            if not hasattr(self._success_term, "succeeded"):
                raise TypeError(
                    f"conditional_reset: reward term '{success_reward_term}' keeps no 'succeeded' buffer, so"
                    " episode outcomes cannot be credited to the banked states. Point 'success_reward_term'"
                    " at a term that keeps one, such as 'success_reward'."
                )
        played = self._playing_row[env_ids]
        # an environment that has not been restored yet has no episode to credit
        started = played >= 0
        monitor.success_update(played[started], self._success_term.succeeded[env_ids][started])


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
            self._robot.data.root_quat_w.torch[env_ids], self._goal_center_b.expand(len(env_ids), 3)
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
        for _, _, source_path, env_ids in cloner.query.iter_sources(clone_plan, self._object.cfg.prim_path):
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
