# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command term for the SO101 keyboard letter-typing task."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers import CommandTerm, ManagerTermBase
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    skew_symmetric_matrix,
)

from isaaclab_tasks.core.lift.mdp.events import SuccessMonitor
from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg
from isaaclab_tasks.core.lift.mdp.utils import get_reset_state, set_reset_state

from . import typing_vis

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .typing_commands_cfg import LetterTypingCommandCfg


@wp.kernel
def _resample_reset_kernel(
    env_ids: wp.array(dtype=wp.int32),
    typeable: wp.array(dtype=wp.int64),
    lo: wp.int32,
    hi: wp.int32,
    seed: wp.int32,
    match_prob: wp.float32,
    target: wp.array2d(dtype=wp.int64),
    typed: wp.array2d(dtype=wp.int64),
    target_len: wp.array(dtype=wp.int64),
    typed_len: wp.array(dtype=wp.int64),
    prefix_len: wp.array(dtype=wp.int64),
    distance: wp.array(dtype=wp.float32),
    max_prefix: wp.array(dtype=wp.int64),
    min_prefix: wp.array(dtype=wp.int64),
):
    """Per-thread typing-state sampler: draws a target word and a partially-typed start buffer whose keys match
    the target with prob ``match_prob`` (else a mistake), guards against an instant success, and seeds the
    metrics + progress water marks. Used both per-env on the normal reset and, launched over a large pool, to
    oversample candidates for the bucket-downsampled reset curriculum. The ragged fill is a plain per-thread
    loop, so no padded-matrix masking and no ``sum(t)`` host sync.
    """
    i = wp.tid()
    e = env_ids[i]
    rng = wp.rand_init(seed, e)
    m = int(typeable.shape[0])
    width = int(target.shape[1])

    n = wp.randi(rng, lo, hi + 1)  # target_len in [lo, hi]
    # typed_len over the FULL buffer width [0, width] (not [0, target_len]) so a start can OVERSHOOT the word
    # (extra keys past a fully-correct prefix -> "backspace N times to success"), not just diverge mid-word.
    t = wp.randi(rng, 0, width + 1)

    # target word: n random typeable keys, then -1 padding.
    for j in range(width):
        if j < n:
            target[e, j] = typeable[wp.randi(rng, 0, m)]
        else:
            target[e, j] = wp.int64(-1)

    # start buffer: each pre-typed key matches the target key with prob match_prob (a correct prefix), else a
    # random key (a mistake); positions past the target (j >= n) are always random extras. This spans the whole
    # spectrum {empty, correct prefix, wrong-middle, overshoot}; the build's bucket-downsampling balances
    # coverage across them (see LetterTypingCommand._sample_diverse_states), so there is no explicit prefill knob.
    for j in range(width):
        if j < t:
            if j < n and wp.randf(rng) < match_prob:
                typed[e, j] = target[e, j]
            else:
                typed[e, j] = typeable[wp.randi(rng, 0, m)]
        else:
            typed[e, j] = wp.int64(-1)

    # instant-success guard: a full (t == target_len) buffer must differ from the target in its last slot,
    # else the episode would start already complete. typeable slot ids are unique, so a distinct typeable
    # index guarantees a distinct slot.
    if t == n and n > 0:
        last = n - 1
        r = wp.randi(rng, 0, m)
        if typeable[r] == target[e, last]:
            r = (r + 1) % m
        typed[e, last] = typeable[r]

    # correct-prefix length = longest common prefix of typed and target (bounded loop avoids an
    # out-of-bounds read and does not rely on short-circuit evaluation).
    p = int(0)
    while p < t:
        if typed[e, p] != target[e, p]:
            break
        p = p + 1

    target_len[e] = wp.int64(n)
    typed_len[e] = wp.int64(t)
    prefix_len[e] = wp.int64(p)
    distance[e] = wp.float32(n + t - 2 * p)
    max_prefix[e] = wp.int64(p)
    min_prefix[e] = wp.int64(p)


class LetterTypingCommand(CommandTerm):
    """Letter-typing command for the procedural SO101 keyboard.

    Each episode samples a fixed-length target sequence of key slots. The agent "types" by fully
    pressing then releasing a key joint: a keystroke is registered once on each press edge, so
    holding a key down counts as a single key. Every key pressed on a given step registers (a
    real-keyboard model), so mashing several keys at once types them all - the extras land as mistakes.
    Pressing the backspace key deletes the last typed key. The typed buffer cannot grow past its width
    (``max_len``), so once it is full the agent must backspace before it can type again.

    Command (absolute / ``letter_full``): concatenation of the normalized target and typed slot ids,
    shape ``(num_envs, 2 * max_len)``. ``letter_left`` instead returns only the target slots not yet
    correctly typed, shape ``(num_envs, max_len)``.
    """

    cfg: LetterTypingCommandCfg

    def __init__(self, cfg: LetterTypingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.keyboard: Articulation = env.scene[cfg.object_name]
        # Buffer/observation width. Decoupled from letter_length so the obs size (and a trained policy's
        # input layer) stays fixed when letter_length is varied for evaluation; defaults to letter_length[1].
        self.max_len = int(cfg.max_len) if cfg.max_len is not None else int(cfg.letter_length[1])
        if self.max_len < int(cfg.letter_length[1]):
            raise ValueError(
                f"max_len ({self.max_len}) must be >= letter_length[1] ({int(cfg.letter_length[1])}) so a"
                " sampled word always fits the buffer."
            )

        # Gather maps from the articulation's (num_instances, dof | num_bodies) data into per-env GLOBAL
        # key-slot order (slot = part * dof + local), so reads are plain advanced indexing (no hidden
        # copy): joint_pos[inst_idx, col_idx] and body_pos_w[inst_idx, body_col_idx]. Backend-agnostic
        # (PhysX prim-path order vs Newton regular grid) and agnostic to single vs fixed_dof partitioning.
        view = self.keyboard.root_view
        dof = len(self.keyboard.joint_names)
        local_of_col = []
        for name in self.keyboard.joint_names:
            match = re.search(r"key_([0-9]+)_joint", name)
            assert match is not None, f"{name!r} is not a keyboard key joint"
            local_of_col.append(int(match.group(1)))
        body_col_of_local = {local: self.keyboard.body_names.index(f"key_{local:03d}") for local in local_of_col}

        prim_paths = getattr(view, "prim_paths", None)
        if prim_paths is not None:
            num_instances = len(prim_paths)
            env_part = []
            for path in prim_paths:
                env_match = re.search(r"/env_([0-9]+)", path)
                part_match = re.search(r"/part_([0-9]+)", path)
                assert env_match is not None, f"{path!r} is not inside a numbered environment"
                env_part.append((int(env_match.group(1)), int(part_match.group(1)) if part_match else 0))
        else:
            num_instances = int(view.count)
            per_world = max(num_instances // self.num_envs, 1)
            env_part = [(inst // per_world, inst % per_world) for inst in range(num_instances)]

        self.num_keys = (num_instances // self.num_envs) * dof
        inst = np.zeros((self.num_envs, self.num_keys), dtype=np.int64)
        joint_col = np.zeros((self.num_envs, self.num_keys), dtype=np.int64)
        body_col = np.zeros((self.num_envs, self.num_keys), dtype=np.int64)
        for instance, (env_id, part) in enumerate(env_part):
            for col, local in enumerate(local_of_col):
                slot = part * dof + local
                inst[env_id, slot] = instance
                joint_col[env_id, slot] = col
                body_col[env_id, slot] = body_col_of_local[local]
        self._inst_idx = torch.as_tensor(inst, device=self.device)
        self._col_idx = torch.as_tensor(joint_col, device=self.device)
        self._body_col_idx = torch.as_tensor(body_col, device=self.device)

        # key roles in global slot space, resolved once at the config layer (see so101_env_cfg)
        self._typeable = torch.tensor(cfg.typeable_slots, device=self.device)
        self._backspace = cfg.backspace_slot

        # per-key press threshold (filled lazily once joint limits are available)
        self._press_level: torch.Tensor | None = None

        # buffers: -1 marks an empty / padding slot
        n, length = self.num_envs, self.max_len
        self.target = torch.full((n, length), -1, dtype=torch.long, device=self.device)
        self.typed = torch.full((n, length), -1, dtype=torch.long, device=self.device)
        self.target_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self.typed_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._prev_pressed = torch.zeros((n, self.num_keys), dtype=torch.bool, device=self.device)
        # Set on reset so the first post-reset update adopts the current key state as the baseline
        # (keys already depressed at reset must be released before they count as keystrokes).
        self._just_reset = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.distance = torch.zeros(n, device=self.device)
        self.prefix_len = torch.zeros(n, dtype=torch.long, device=self.device)
        # Per-episode STARTING distance-to-success (target_len + typed_len - 2*prefix at reset), captured at the
        # end of reset() and read at the NEXT reset to bucket the terminal success by how far the episode began
        # from done (the ``distance<K`` split metrics). Cumulative bands up to the max distance ``2*max_len``.
        self._start_distance = torch.zeros(n, dtype=torch.long, device=self.device)
        self._distance_bands = tuple(range(2, 2 * self.max_len + 1, 2))

        # High-water-mark progress: per-episode extremes of the correct-prefix length and the per-step
        # new-record flags the ratchet reward reads (see :meth:`_update_metrics` and
        # :func:`~...mdp.rewards.letter_typing_progress`). Seeded from the (possibly pre-filled) reset buffer.
        self.max_prefix = torch.zeros(n, dtype=torch.long, device=self.device)
        self.min_prefix = torch.zeros(n, dtype=torch.long, device=self.device)
        self.new_high = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.new_low = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Monotonic per-reset RNG seed for the Warp reset sampler; combined with the env id inside the
        # kernel so each env gets a decorrelated, non-repeating random start on every reset.
        self._resample_seed = 0

        self.metrics["distance"] = torch.zeros(n, device=self.device)
        # Mean start-distance over the whole reset-curriculum buffer (a fixed characterization of the buffer's
        # difficulty, computed once in _log_buffer_stats). When set, it REPLACES the per-episode terminal
        # "distance" metric at reset; stays None (falls back to terminal) when the buffer is disabled.
        self._buf_avg_distance: float | None = None

        # Reset-pose IK: a differential-IK controller (built once) snaps the arm so its jaw tip hovers
        # above the first key on every reset. See :meth:`_solve_reset_pose`.
        self._reset_ik: DifferentialIKController | None = None
        if cfg.reset.ik is not None:
            ik_cfg = cfg.reset.ik
            self.robot: Articulation = env.scene[cfg.asset_name]
            self._ik_joint_ids, _ = self.robot.find_joints(list(ik_cfg.joint_names))
            body_ids, _ = self.robot.find_bodies(ik_cfg.body_name)
            self._ik_body_idx = body_ids[0]
            # Jacobian rows/cols drop the (fixed) base, mirroring DifferentialInverseKinematicsAction.
            self._ik_jacobi_body_idx = self._ik_body_idx - 1 if self.robot.is_fixed_base else self._ik_body_idx
            self._ik_jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self._ik_joint_ids]
            offset_pos = ik_cfg.body_offset.pos if ik_cfg.body_offset is not None else (0.0, 0.0, 0.0)
            offset = torch.tensor(offset_pos, device=self.device)
            self._ik_offset = offset.expand(self.num_envs, 3)
            # finger axis = direction from the link origin to the tip (used to aim the approach pitch).
            axis_norm = float(torch.linalg.norm(offset))
            finger_axis = (
                offset / axis_norm if axis_norm > 1.0e-6 else torch.tensor((0.0, 0.0, -1.0), device=self.device)
            )
            self._ik_finger_axis = finger_axis.expand(self.num_envs, 3)
            roll_deg, pitch_deg, yaw_deg = cfg.reset.ik_rpy_deg
            self._ik_roll = math.radians(roll_deg)
            self._ik_pitch = math.radians(pitch_deg)
            self._ik_yaw = math.radians(yaw_deg)
            self._ik_hover = torch.tensor((0.0, 0.0, cfg.reset.ik_hover_height), device=self.device)
            # Iteration budget: an int is a fixed count; a (min, max) tuple is sampled per env each solve so
            # poses land at varying convergence (a reach-difficulty gradient). Normalize to (min, max).
            it = cfg.reset.ik_iters
            self._ik_iters = (int(it), int(it)) if isinstance(it, int) else (int(it[0]), int(it[1]))
            self._reset_ik = DifferentialIKController(ik_cfg.controller, num_envs=self.num_envs, device=self.device)

        # Success-conditioned reset curriculum (buffer built lazily on the first reset; see _build_buffer).
        # Needs reset-IK to synthesize the snapshot poses, so it stays off when reset.ik is unset.
        cur = cfg.reset
        self._cur_enabled = bool(cur.enabled) and self._reset_ik is not None
        if cur.enabled and self._reset_ik is None:
            print(
                "[typing] WARNING: reset.enabled=True but reset.ik is None - the replay curriculum needs the"
                " reset-IK to synthesize snapshot poses, so it is disabled (0 buffer). Set reset.ik to use it."
            )
        self._buffer_built = False
        # True only while inside reset() (episode reset). The base class also calls _resample_command on the
        # mid-episode resampling timer; the curriculum snapshot-restore (a physical teleport) must fire on
        # episode reset only, never mid-word - so the timer path always takes the normal random resample.
        self._episode_reset = False
        # Last observed value of each uniform/buffer split success metric, carried forward on empty batches
        # so the logged curves have no NaN gaps (see reset()).
        self._split_last: dict[str, float] = {}
        if self._cur_enabled:
            cap = int(cur.buffer_size)
            self._cur_buffer_size = cap
            self._cur_reset_assets = (
                tuple(cur.reset_assets) if cur.reset_assets is not None else (cfg.asset_name, cfg.object_name)
            )
            self._cur_normal_weight = max(float(cur.normal_weight), 0.0)
            self._cur_sample_eps = max(float(cur.sample_eps), 1e-12)  # strictly positive: guards multinomial
            # Coverage-sampling knobs for the lazy buffer build (see _sample_diverse_states / _build_buffer).
            self._cur_oversample = max(int(cur.oversample_factor), 1)
            self._cur_feat_w = (
                float(cur.w_target),
                float(cur.w_next_key),
                float(cur.w_unmatched),
                float(cur.w_prefix_complete),
                float(cur.w_remaining),
            )
            target = min(max(float(cur.beta_target), 0.0), 1.0)
            kappa = max(float(cur.beta_kappa), 0.0)
            monitor_cfg = SuccessMonitorCfg(
                monitored_history_len=int(cur.history_len),
                target_success_rate=target,
                kappa=kappa,
            )
            self.success_monitor = SuccessMonitor(
                monitor_cfg, num_partitions=1, partition_size=cap, device=str(self.device)
            )
            self.success_monitor.success_rate.fill_(target)
            # Snapshot buffers: physical state (allocated lazily once its width is known) + typing command.
            self._buf_state: torch.Tensor | None = None
            self._buf_target = torch.full((cap, self.max_len), -1, dtype=torch.long, device=self.device)
            self._buf_typed = torch.full((cap, self.max_len), -1, dtype=torch.long, device=self.device)
            self._buf_target_len = torch.zeros(cap, dtype=torch.long, device=self.device)
            self._buf_typed_len = torch.zeros(cap, dtype=torch.long, device=self.device)
            # Per-snapshot reach residual [m] (tip -> target key) at build time; reported in the build stats.
            self._buf_reach = torch.zeros(cap, dtype=torch.float32, device=self.device)
            # Snapshot id that seeded each env this episode (-1 == normal reset), for outcome attribution.
            self._env_source = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    def __str__(self) -> str:
        return f"LetterTypingCommand(keys={self.num_keys}, max_len={self.max_len}, mode={self.cfg.command_mode})"

    @property
    def command(self) -> torch.Tensor:
        scale = max(self.num_keys - 1, 1)
        target_n = torch.where(self.target >= 0, self.target.float() / scale, -1.0)
        if self.cfg.command_mode == "letter_left":
            # hide the already-correct prefix; show only target slots still to be (correctly) typed
            pos = torch.arange(self.max_len, device=self.device)
            in_prefix = pos < self._prefix_len()[:, None]
            return torch.where(in_prefix, -1.0, target_n)
        typed_n = torch.where(self.typed >= 0, self.typed.float() / scale, -1.0)
        return torch.cat([target_n, typed_n], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset the typing command for ``env_ids``.

        Without the curriculum this is just the normal random reset. With it, the snapshot buffer is built
        once (lazily, on the first reset) and each env either takes the normal random reset or is restored to
        a buffered frontier snapshot, chosen per env by the Beta sampler (see :meth:`_sample_sources`). The
        source picked for each env is recorded so the ending episode's success can be attributed to the right
        snapshot in :meth:`reset`.
        """
        if not self._cur_enabled or not self._episode_reset:
            # No curriculum, or the mid-episode resampling timer: just re-draw the word/buffer, no teleport.
            self._resample_normal(env_ids)
            return
        if not self._buffer_built:
            self._build_buffer()
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        k = int(env_ids_t.numel())
        if k == 0:
            return
        source = self._sample_sources(k)  # (k,): -1 normal reset, else snapshot index in [0, cap)
        self._env_source[env_ids_t] = source
        normal_mask = source < 0
        normal_ids = env_ids_t[normal_mask]
        if normal_ids.numel() > 0:
            self._resample_normal(normal_ids)
        buf_mask = ~normal_mask
        if bool(buf_mask.any()):
            self._restore_snapshot(env_ids_t[buf_mask], source[buf_mask])

    def _sample_sources(self, k: int) -> torch.Tensor:
        """Pick a reset source per env: ``-1`` for the normal reset, else a snapshot index in ``[0, cap)``.

        Each snapshot's weight is the Beta kernel at its success rate (peaked at the target). The normal path's
        weight is scaled by ``cap`` so it competes with the MEAN snapshot score, not the sum:
        ``p(normal) = normal_weight / (normal_weight + mean_score)``. This is independent of ``buffer_size`` and
        still rises toward 1 as snapshots resolve to 0/1 and their weights approach the exploration floor.
        At initialization, all snapshots have the target rate, so ``mean_score = peak``.
        Therefore, ``p(normal) = normal_weight / (normal_weight + peak)``.
        """
        scores = self.success_monitor.target_weights()
        # Scale the normal weight by cap so it competes with the mean (not summed) snapshot score, making
        # p(normal) buffer-size-invariant while still handing sampling to the normal path as the buffer masters.
        normal_w = self._cur_normal_weight * self._cur_buffer_size
        weights = torch.cat([scores.new_full((1,), normal_w), scores])  # (cap + 1,)
        # Floor every weight so the distribution can never sum to 0 (which makes torch.multinomial assert
        # "sum of probabilities <= 0"). That degenerate case arises when normal_weight == 0 and every snapshot
        # weight has reached its exploration floor; the clamp then falls back to roughly uniform replay.
        weights = weights.clamp(min=self._cur_sample_eps)
        idx = torch.multinomial(weights, k, replacement=True)  # (k,) in [0, cap]
        return idx - 1  # 0 -> -1 (normal); j -> snapshot j - 1

    def _oversample(self, m_candidates: int):
        """Draw ``m_candidates`` random typing states via the Warp sampler (whole pool in one launch).

        Returns ``(target, typed, target_len, typed_len, prefix_len)`` as torch tensors (``target``/``typed``
        shaped ``[M, max_len]``). One-time build helper for :meth:`_sample_diverse_states`.
        """
        lo, hi = self.cfg.letter_length
        self._resample_seed += 1
        w = self.max_len
        target = torch.full((m_candidates, w), -1, dtype=torch.long, device=self.device)
        typed = torch.full((m_candidates, w), -1, dtype=torch.long, device=self.device)
        target_len = torch.zeros(m_candidates, dtype=torch.long, device=self.device)
        typed_len = torch.zeros(m_candidates, dtype=torch.long, device=self.device)
        prefix_len = torch.zeros(m_candidates, dtype=torch.long, device=self.device)
        distance = torch.zeros(m_candidates, dtype=torch.float32, device=self.device)
        scratch1 = torch.zeros(m_candidates, dtype=torch.long, device=self.device)  # max_prefix (unused here)
        scratch2 = torch.zeros(m_candidates, dtype=torch.long, device=self.device)  # min_prefix (unused here)
        env_ids = torch.arange(m_candidates, device=self.device, dtype=torch.int32)
        wp.launch(
            _resample_reset_kernel,
            dim=m_candidates,
            inputs=[
                wp.from_torch(env_ids, dtype=wp.int32),
                wp.from_torch(self._typeable.contiguous(), dtype=wp.int64),
                int(lo),
                int(hi),
                int(self._resample_seed),
                float(self.cfg.reset.match_prob),
            ],
            outputs=[
                wp.from_torch(target, dtype=wp.int64),
                wp.from_torch(typed, dtype=wp.int64),
                wp.from_torch(target_len, dtype=wp.int64),
                wp.from_torch(typed_len, dtype=wp.int64),
                wp.from_torch(prefix_len, dtype=wp.int64),
                wp.from_torch(distance, dtype=wp.float32),
                wp.from_torch(scratch1, dtype=wp.int64),
                wp.from_torch(scratch2, dtype=wp.int64),
            ],
            device=str(self.device),
        )
        wp.synchronize()
        return target, typed, target_len, typed_len, prefix_len

    def _sample_diverse_states(self, cap: int):
        """Oversample candidate states and grid-bucket-downsample to ``cap`` with uniform coverage.

        The feature is the (per-axis weighted) ``(target, next_key, wrongness, prefix_complete, remaining)``
        descriptor, with every axis normalized to ~[0, 1] so the ``w_*`` weights are direct per-axis grid
        resolutions (not swamped by the raw key-index range). ``wrongness = (typed_len - prefix) / target_len``
        is the length-normalized backspace depth; ``remaining = (target_len - prefix) / max_len`` is the
        *absolute* keys-still-to-type (the forward half of the distance-to-success). Together the survivors
        uniformly cover the cases the agent faces - type each key, backspace at each depth, overshoot-to-success,
        and every distance-from-success - rather than the raw sampling distribution (which skews near-done).
        Returns the chosen ``(target, typed, target_len, typed_len)`` (each ``cap`` rows).
        """
        m_candidates = cap * self._cur_oversample
        target, typed, target_len, typed_len, prefix = self._oversample(m_candidates)
        rows = torch.arange(m_candidates, device=self.device)
        nxt = torch.minimum(prefix, (target_len - 1).clamp(min=0))
        needs_bs = typed_len > prefix
        next_key = torch.where(needs_bs, torch.full_like(prefix, self._backspace), target[rows, nxt]).clamp(min=0)
        # Backspace depth as a FRACTION of the word length, so "fully wrong" costs the same regardless of length
        # (len-1 unmatched=1 and len-3 unmatched=3 both map to 1.0). This removes the only structural length
        # bias in the feature; the natural word-diversity bias (~108**len distinct words) is kept via `target`.
        wrongness = (typed_len - prefix).float() / target_len.float()
        prefix_complete = (prefix == target_len).long()
        # Keys still to type, normalized by the FIXED max_len (not target_len) so difficulty stays absolute:
        # a len-3 word with 1 key left and a len-1 word from scratch both map to 1/max_len. This is the forward
        # half of the edit distance to success that the other axes were missing (see w_remaining).
        remaining = (target_len - prefix).float() / float(self.max_len)
        distance = target_len + typed_len - 2 * prefix
        valid = (distance > 0).nonzero(as_tuple=False).squeeze(-1)  # drop the (guarded, rare) instant successes
        # Normalize the key-index axes to [0, 1] (padding -1 -> -1/scale, a distinct "empty slot" coordinate)
        # so all axes share a scale and the weights below set each axis's grid resolution directly.
        wt, wn, wu, wc, wr = self._cur_feat_w
        key_scale = float(max(self.num_keys - 1, 1))
        feat = torch.cat(
            [
                (wt / key_scale) * target.float(),
                (wn / key_scale * next_key.float())[:, None],
                (wu * wrongness)[:, None],
                (wc * prefix_complete.float())[:, None],
                (wr * remaining)[:, None],
            ],
            dim=1,
        )
        points = feat[valid]
        n, dims = points.shape
        if n <= cap:
            chosen = valid
        else:
            mins = points.amin(dim=0)
            extents = (points.amax(dim=0) - mins).clamp(min=1e-6)
            target_cells = min(n, max(int(cap * 3.0), cap + 1))
            cell_side = (extents.prod() / target_cells) ** (1.0 / dims)
            bucket_indices = ((points - mins) / cell_side).long()

            shape = bucket_indices.amax(dim=0) + 1
            strides = torch.empty(dims, dtype=torch.int64, device=self.device)
            strides[-1] = 1
            for index in range(dims - 2, -1, -1):
                strides[index] = strides[index + 1] * shape[index + 1]
            bucket_ids = (bucket_indices.to(torch.int64) * strides).sum(dim=-1)

            priority = torch.randint(0, 1 << 30, (n,), dtype=torch.int64, device=self.device)
            permutation = torch.argsort(bucket_ids * (1 << 30) + priority)
            sorted_bucket_ids = bucket_ids[permutation]
            first = torch.ones_like(sorted_bucket_ids, dtype=torch.bool)
            first[1:] = sorted_bucket_ids[1:] != sorted_bucket_ids[:-1]
            bucket_survivors = permutation[first]

            if bucket_survivors.numel() > cap:
                subset = torch.randperm(bucket_survivors.numel(), device=self.device)[:cap]
                bucket_survivors = bucket_survivors[subset]
            elif bucket_survivors.numel() < cap:
                survivor_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
                survivor_mask[bucket_survivors] = True
                remaining_indices = (~survivor_mask).nonzero(as_tuple=False).squeeze(-1)
                needed = cap - bucket_survivors.numel()
                subset = torch.randperm(remaining_indices.numel(), device=self.device)[:needed]
                bucket_survivors = torch.cat([bucket_survivors, remaining_indices[subset]])
            chosen = valid[bucket_survivors]
        return target[chosen], typed[chosen], target_len[chosen], typed_len[chosen]

    def _build_buffer(self):
        """Build the snapshot buffer once, on the first reset.

        Coverage-samples ``cap`` diverse ``(target, typed)`` states (see :meth:`_sample_diverse_states`), then
        runs the reset-IK once per survivor (in ``num_envs``-sized batches) to snapshot its physical state.
        The live env state is scrambled by the last IK batch, but the enclosing reset overwrites it.
        """
        from tqdm import tqdm

        cap = self._cur_buffer_size
        tgt, typd, tlen, typlen = self._sample_diverse_states(cap)
        self._buf_target[:] = tgt
        self._buf_typed[:] = typd
        self._buf_target_len[:] = tlen
        self._buf_typed_len[:] = typlen

        all_ids = torch.arange(self.num_envs, device=self.device)
        with tqdm(total=cap, desc="[typing] building reset-curriculum buffer (IK)", unit="snap") as pbar:
            for start in range(0, cap, self.num_envs):
                n = min(self.num_envs, cap - start)
                ids = all_ids[:n]
                # Load this batch's cached command into the live state so the reset-IK aims at the right key.
                self.target[:n] = tgt[start : start + n]
                self.typed[:n] = typd[start : start + n]
                self.target_len[:n] = tlen[start : start + n]
                self.typed_len[:n] = typlen[start : start + n]
                self.prefix_len[:n] = self._prefix_len()[:n]
                self._solve_reset_pose(ids)
                state = get_reset_state(self._env, ids, self._cur_reset_assets, is_relative=True)
                if self._buf_state is None:
                    self._buf_state = torch.zeros((cap, state.shape[1]), dtype=state.dtype, device=self.device)
                self._buf_state[start : start + n] = state
                # tip -> target-key residual of the (variably converged) pose, for the closeness stats.
                ee_pos = self.robot.data.body_pos_w.torch[:, self._ik_body_idx]
                ee_quat = self.robot.data.body_quat_w.torch[:, self._ik_body_idx]
                tip = ee_pos + quat_apply(ee_quat, self._ik_offset)
                reach = torch.linalg.norm(tip - (self.target_key_pos_w() + self._ik_hover), dim=-1)
                self._buf_reach[start : start + n] = reach[:n]
                pbar.update(n)
        self._buffer_built = True
        self._log_buffer_stats()

    def _log_buffer_stats(self):
        """Print one-time coverage/distribution stats for the freshly built snapshot buffer."""
        cap = self._cur_buffer_size
        labels = self.cfg.slot_labels
        typeable = self._typeable.long()
        total = int(typeable.numel())

        # Mirror target_key_slot() on the buffered commands: the key each snapshot's arm was posed over.
        pos = torch.arange(self.max_len, device=self.device)
        comparable = (pos < self._buf_typed_len[:, None]) & (pos < self._buf_target_len[:, None])
        mismatch = comparable & (self._buf_typed != self._buf_target)
        first_mismatch = torch.where(mismatch, pos, self.max_len).min(dim=1).values
        prefix = torch.minimum(first_mismatch, torch.minimum(self._buf_typed_len, self._buf_target_len))
        needs_bs = self._buf_typed_len > prefix
        # Fixed mean start-distance of the buffer (mirrors self.distance = target_len + typed_len - 2*prefix);
        # logged every reset as Metrics/typing/distance, so the curve reads as the buffer's average difficulty.
        self._buf_avg_distance = float((self._buf_target_len + self._buf_typed_len - 2 * prefix).float().mean())
        nxt = torch.minimum(prefix, (self._buf_target_len - 1).clamp(min=0))
        rows = torch.arange(cap, device=self.device)
        reach_counts = torch.bincount(self._buf_target[rows, nxt][~needs_bs], minlength=self.num_keys)[typeable]
        word_counts = torch.bincount(self._buf_target[self._buf_target >= 0], minlength=self.num_keys)[typeable]

        def _missing(counts: torch.Tensor) -> str:
            miss = typeable[counts == 0].tolist()
            if not miss:
                return "none"
            names = [labels[s] if s < len(labels) and labels[s] else str(s) for s in miss]
            return f"{len(miss)} keys [{', '.join(names[:15])}{', ...' if len(miss) > 15 else ''}]"

        print(f"[typing] reset-curriculum buffer built: {cap} snapshots over {total} typeable keys (+ backspace)")
        print(
            f"[typing]   reach target (key posed over): covered {int((reach_counts > 0).sum())}/{total}"
            f" | count/key min/mean/max={int(reach_counts.min())}/{reach_counts.float().mean():.1f}"
            f"/{int(reach_counts.max())} | missing: {_missing(reach_counts)}"
        )
        print(
            f"[typing]   target-word keys (union): covered {int((word_counts > 0).sum())}/{total}"
            f" | missing: {_missing(word_counts)}"
        )
        empty = int((self._buf_typed_len == 0).sum())
        correct = int(((self._buf_typed_len > 0) & ~needs_bs).sum())
        overshoot = int((needs_bs & (prefix == self._buf_target_len)).sum())
        wrong = int((needs_bs & (prefix < self._buf_target_len)).sum())
        print(
            f"[typing]   start-state mix: empty={empty} ({100.0 * empty / cap:.0f}%),"
            f" correct-prefix={correct} ({100.0 * correct / cap:.0f}%),"
            f" wrong-key-backspace={wrong} ({100.0 * wrong / cap:.0f}%),"
            f" overshoot-backspace={overshoot} ({100.0 * overshoot / cap:.0f}%)"
        )
        print(f"[typing]   mean start-distance (logged as Metrics/typing/distance): {self._buf_avg_distance:.3f}")
        lo, hi = int(self.cfg.letter_length[0]), int(self.cfg.letter_length[1])
        len_counts = torch.bincount(self._buf_target_len, minlength=hi + 1).tolist()
        len_str = ", ".join(f"len{n}={len_counts[n]} ({100.0 * len_counts[n] / cap:.0f}%)" for n in range(lo, hi + 1))
        print(f"[typing]   target-length mix: {len_str}")
        # Per-length distance-to-success histogram (distance = keys to backspace + keys still to type): the axis
        # the coverage feature now balances. A flat spread per length means hard (high-distance) states are as
        # present as easy near-done ones, so the buffer/curriculum is not biased toward trivial states.
        buf_distance = self._buf_target_len + self._buf_typed_len - 2 * prefix
        dmax = int(buf_distance.max())
        print("[typing]   distance-to-success mix (per target-length):")
        for n in range(lo, hi + 1):
            sel = self._buf_target_len == n
            cnt_n = int(sel.sum())
            if cnt_n == 0:
                continue
            dh = torch.bincount(buf_distance[sel], minlength=dmax + 1).tolist()
            parts = ", ".join(f"d{d}={dh[d]} ({100.0 * dh[d] / cnt_n:.0f}%)" for d in range(dmax + 1) if dh[d] > 0)
            print(f"[typing]     len{n} (n={cnt_n}): {parts}")
        reach_cm = self._buf_reach * 100.0
        med, p90 = torch.quantile(reach_cm, torch.tensor([0.5, 0.9], device=self.device)).tolist()
        lo_cm, mean_cm, hi_cm = float(reach_cm.min()), float(reach_cm.mean()), float(reach_cm.max())
        print(
            f"[typing]   reach closeness (tip->key, iters={self._ik_iters}): min/median/mean/p90/max ="
            f" {lo_cm:.1f}/{med:.1f}/{mean_cm:.1f}/{p90:.1f}/{hi_cm:.1f} cm"
        )

    def _restore_snapshot(self, env_ids: torch.Tensor, snap: torch.Tensor):
        """Restore the physical state + typing command of snapshots ``snap`` into ``env_ids``."""
        assert self._buf_state is not None  # allocated in _build_buffer, which always runs first
        set_reset_state(self._env, self._buf_state[snap], env_ids, self._cur_reset_assets, is_relative=True)
        self.target[env_ids] = self._buf_target[snap]
        self.typed[env_ids] = self._buf_typed[snap]
        self.target_len[env_ids] = self._buf_target_len[snap]
        self.typed_len[env_ids] = self._buf_typed_len[snap]
        self._prev_pressed[env_ids] = False
        self._just_reset[env_ids] = True
        prefix = self._prefix_len()[env_ids]
        self.prefix_len[env_ids] = prefix
        self.distance[env_ids] = (self.target_len[env_ids] + self.typed_len[env_ids] - 2 * prefix).float()
        self.max_prefix[env_ids] = prefix
        self.min_prefix[env_ids] = prefix
        self.new_high[env_ids] = False
        self.new_low[env_ids] = False

    def _resample_normal(self, env_ids: Sequence[int] | torch.Tensor):
        # Normal (random) reset path: draw the target word + a match_prob-typed start buffer, guard against an
        # instant success, and seed the typing metrics + progress water marks - one Warp thread per resetting
        # env, so the ragged fill reads as a per-thread loop instead of padded-matrix masking, with no sum(t)
        # host sync (see :func:`_resample_reset_kernel`). Non-ragged per-env bookkeeping stays in torch.
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        k = int(env_ids_t.numel())
        if k == 0:
            return
        lo, hi = self.cfg.letter_length
        self._resample_seed += 1
        wp.launch(
            _resample_reset_kernel,
            dim=k,
            inputs=[
                wp.from_torch(env_ids_t.to(torch.int32).contiguous(), dtype=wp.int32),
                wp.from_torch(self._typeable.contiguous(), dtype=wp.int64),
                int(lo),
                int(hi),
                int(self._resample_seed),
                float(self.cfg.reset.match_prob),
            ],
            outputs=[
                wp.from_torch(self.target, dtype=wp.int64),
                wp.from_torch(self.typed, dtype=wp.int64),
                wp.from_torch(self.target_len, dtype=wp.int64),
                wp.from_torch(self.typed_len, dtype=wp.int64),
                wp.from_torch(self.prefix_len, dtype=wp.int64),
                wp.from_torch(self.distance, dtype=wp.float32),
                wp.from_torch(self.max_prefix, dtype=wp.int64),
                wp.from_torch(self.min_prefix, dtype=wp.int64),
            ],
            device=str(self.device),
        )
        self._prev_pressed[env_ids_t] = False
        self._just_reset[env_ids_t] = True
        self.new_high[env_ids_t] = False
        self.new_low[env_ids_t] = False

    def _update_command(self):
        if self._press_level is None:
            # Convention (keyboard_schema.KEY_ACTUATION_FRACTION): rest at upper_limit (~0), bottom at
            # lower_limit (-travel); a key actuates once depressed past a fraction of its travel.
            limits = self.keyboard.data.joint_pos_limits.torch[self._inst_idx, self._col_idx]  # (N, K, 2)
            lower, upper = limits[..., 0], limits[..., 1]
            self._press_level = upper - self.cfg.actuation_fraction * (upper - lower)

        pos = self.keyboard.data.joint_pos.torch[self._inst_idx, self._col_idx]  # (N, K)
        pressed = pos < self._press_level
        # First control step after a reset: adopt the carried-over pressed keys as the baseline so a key
        # already held at reset is not counted as a fresh keystroke. Applied unconditionally via a masked
        # write, so there is no per-step ``.any()`` device->host sync.
        self._prev_pressed = torch.where(self._just_reset[:, None], pressed, self._prev_pressed)
        self._just_reset.zero_()
        new_press = pressed & ~self._prev_pressed
        self._prev_pressed = pressed

        # Real-keyboard model: EVERY key newly pressed this step registers as its own keystroke (no single
        # "winning" key), so mashing types them all - the extras land in the buffer as mistakes to backspace.
        # The whole update is vectorized, static-shape and sync-free (no ``nonzero`` / boolean-index gather),
        # so it stays cheap at large num_envs.
        n = new_press.shape[0]
        rows = torch.arange(n, device=self.device)
        cols = torch.arange(self.num_keys, device=self.device)

        # Backspace (edge-triggered): delete the last typed key for envs whose backspace key was newly
        # pressed. Advanced-index write over every env - a no-op where backspace was not pressed.
        bs_mask = new_press[:, self._backspace]  # (N,)
        del_len = (self.typed_len - 1).clamp(min=0)  # (N,)
        at_del = self.typed[rows, del_len]  # (N,) current value at the delete position
        self.typed[rows, del_len] = torch.where(bs_mask, torch.full_like(at_del, -1), at_del)
        self.typed_len = torch.where(bs_mask, del_len, self.typed_len)

        # Type: append every newly pressed non-backspace key, in slot (column) order, after the current
        # buffer position and up to max_len. ``rank`` is each press's 0-indexed position among this env's
        # presses, so its write column is ``typed_len + rank``. A spare "trash" column (index max_len)
        # absorbs the non-press / overflow writes, so one masked ``scatter_`` places everything without any
        # variable-length index gather; ``scatter_`` collisions only happen on the trash column (all -1).
        type_press = new_press & (cols != self._backspace)  # (N, K)
        rank = type_press.long().cumsum(dim=1) - 1  # (N, K)
        dest = self.typed_len[:, None] + rank  # (N, K)
        fits = type_press & (dest < self.max_len)  # (N, K)
        buf = torch.full((n, self.max_len + 1), -1, dtype=self.typed.dtype, device=self.device)
        buf[:, : self.max_len] = self.typed
        buf.scatter_(1, torch.where(fits, dest, self.max_len), torch.where(fits, cols.expand(n, -1), -1))
        self.typed.copy_(buf[:, : self.max_len])
        self.typed_len = self.typed_len + fits.sum(dim=1)

    def _update_metrics(self):
        # Sequential typing: only the contiguous correct prefix counts. Anything typed past the first
        # mismatch is "non-prefix" and must be backspaced even if it coincidentally matches later.
        prefix = self._prefix_len()
        self.prefix_len = prefix  # exposed for the reward term
        # distance = non_prefix (= typed_len - prefix) + missing (= target_len - prefix); 0 at success.
        self.distance = (self.target_len + self.typed_len - 2 * prefix).float()
        # High-water-mark progress signal for the ratchet reward: flag a new episode max / min prefix this
        # step, then advance the marks. Because a re-reached length is neither a new max nor a new min, a
        # keystroke can only ever be rewarded once, so backspace/retype loops cannot farm reward.
        self.new_high = prefix > self.max_prefix
        self.new_low = prefix < self.min_prefix
        self.max_prefix = torch.maximum(self.max_prefix, prefix)
        self.min_prefix = torch.minimum(self.min_prefix, prefix)
        self.metrics["distance"] = self.distance

    def _prefix_len(self) -> torch.Tensor:
        """Length of the longest common prefix of ``typed`` and ``target`` per env, shape ``(N,)``."""
        pos = torch.arange(self.max_len, device=self.device)
        comparable = (pos < self.typed_len[:, None]) & (pos < self.target_len[:, None])
        mismatch = comparable & (self.typed != self.target)
        first_mismatch = torch.where(mismatch, pos, self.max_len).min(dim=1).values
        return torch.minimum(first_mismatch, torch.minimum(self.typed_len, self.target_len))

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Resample the target word and snap the arm above its first key (see :meth:`_solve_reset_pose`).

        Runs only on episode reset (not on the mid-episode resampling timer), so the IK snap never
        teleports the arm while the agent is mid-word.
        """
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)

        # Terminal success (read BEFORE super().reset() resamples the word) of the ending episodes, plus the
        # STARTING distance-to-success each began at (captured at its previous reset).
        succeeded = (self.distance[ids] == 0).float()
        start_d = self._start_distance[ids]

        # Split the success stats by the reset source that SEEDED each ending episode (still in _env_source,
        # before super().reset() overwrites it): "buffer" = restored snapshot (curriculum-biased), "uniform" =
        # normal random reset (unbiased). Each split reports overall success plus success over episodes that
        # STARTED within a distance band (``distance<K``: began < K keystrokes from done), so difficulty is read
        # off the episode's start, not its (fixed-length) word. Empty subsets carry the last value forward (0
        # before the first sample) so the curves have no NaN gaps. Without the curriculum every env is uniform.
        if self._cur_enabled and self._buffer_built:
            from_buffer = self._env_source[ids] >= 0
        else:
            from_buffer = torch.zeros_like(succeeded, dtype=torch.bool)
        split_metrics: dict[str, float] = {}
        for tag, mask in (("uniform", ~from_buffer), ("buffer", from_buffer)):
            s = succeeded[mask]
            sd = start_d[mask]
            # (key, successes, count); count 0 -> carry the last value instead of logging NaN / a false 0.
            entries = [(f"{tag}/success_rate", float(s.sum()), int(s.numel()))]
            for thr in self._distance_bands:
                in_band = sd < thr
                entries.append((f"{tag}/distance<{thr}", float(s[in_band].sum()), int(in_band.sum())))
            for key, succ, cnt in entries:
                if cnt > 0:
                    self._split_last[key] = succ / cnt
                split_metrics[key] = self._split_last.get(key, 0.0)

        # Curriculum: attribute the ending episode's success to the snapshot that seeded each env. Uses the
        # sources assigned at the PREVIOUS reset (still in _env_source) with the terminal distance read above,
        # before super().reset() -> _resample_command overwrites both.
        if self._cur_enabled and self._buffer_built:
            src = self._env_source[ids]
            valid = src >= 0
            if bool(valid.any()):
                self.success_monitor.success_update(src[valid], succeeded[valid] > 0.5)
                unmeasured = self.success_monitor.success_size == 0
                self.success_monitor.success_rate[unmeasured] = self.success_monitor.cfg.target_success_rate

        # Mark the episode-reset window so _resample_command takes the curriculum path (snapshot restore)
        # here, while the mid-episode resampling timer stays on the normal random resample.
        self._episode_reset = True
        extras = super().reset(env_ids)
        self._episode_reset = False
        # Record the starting distance of the freshly-seeded episodes (both reset paths set self.distance to the
        # start value; the IK snap below only moves the arm) for the distance<K success split at their next reset.
        self._start_distance[ids] = self.distance[ids].long()
        extras.update(split_metrics)
        # Report "distance" as the buffer's fixed mean start-distance (its average difficulty) rather than the
        # noisy, curriculum-biased terminal distance of the episodes that happened to end this batch. Falls back
        # to the terminal value (set by super().reset() from metrics["distance"]) when there is no buffer.
        if self._buf_avg_distance is not None:
            extras["distance"] = self._buf_avg_distance

        if self._reset_ik is not None:
            # Snap above the key the agent must press next for the (possibly pre-filled) buffer. The solve
            # reads target_key_slot(), which already accounts for the correct prefix and any pending
            # backspace, so no need to assume an empty buffer here.
            snap_ids = ids
            if self._cur_enabled and self._buffer_built:
                # Buffer-restored envs already sit at their snapshot pose; only re-solve the normal-path envs.
                snap_ids = ids[self._env_source[ids] < 0]
            self._solve_reset_pose(snap_ids)
        return extras

    def _solve_reset_pose(self, env_ids: torch.Tensor):
        """Snap the arm so the moving-jaw tip hovers above the first key, angled to press, without stepping.

        Seeds the arm from its default pose (optionally jittered by ``reset.ik_seed_joint_noise``), then runs
        damped-least-squares differential IK on the
        moving-jaw tip. Position is the primary task (always driven to zero); the ``~40%`` of iterations
        after the base pan settles also drive the approach orientation from ``reset.ik_rpy_deg`` (see
        :meth:`_approach_target_quat`), but only within the null space of position so the tip never leaves
        the key. Each iteration writes the joints and calls ``sim.forward()``; the lazy data layer
        recomputes the tip pose and Jacobian from the written joints, so the loop converges with no
        physics (dynamics) step. The solve runs over
        all envs (the controller is sized to ``num_envs``) but only ``env_ids`` are written, leaving
        mid-episode envs untouched.
        """
        if self._reset_ik is None or len(env_ids) == 0:
            return
        # _resample_command filled typed/target and the derived typed_len/prefix_len via a Warp kernel
        # launched moments earlier in super().reset(). target_key_pos_w() below reads those tensors back
        # in torch within this same call (no env step in between), so synchronize Warp first: otherwise
        # the read can race the kernel and see stale lengths, making needs_backspace False and snapping the
        # arm to the next key instead of the backspace key for a pre-filled wrong buffer.
        wp.synchronize()
        sim = self._env.sim
        # Apply the pre-solve reset (e.g. keyboard-pose randomization) to these envs BEFORE reading key
        # positions below, so the arm is posed to this reset's keyboard. Runs per build batch (so the buffer
        # captures diverse keyboard poses) and for normal-path reset envs; buffer-restored envs never reach
        # here and keep their snapshot's keyboard. The root write is flushed by the sim.forward() after the
        # arm seed just below (reset_root_state_uniform only writes, it does not step).
        pre = self.cfg.reset.pre_solve_reset
        if pre is not None:
            # Mirror EventManager term resolution: class-based terms (ManagerTermBase subclasses)
            # are instantiated once with (cfg, env) and the callable instance replaces ``func``.
            if inspect.isclass(pre.func) and issubclass(pre.func, ManagerTermBase):
                pre.func = pre.func(cfg=pre, env=self._env)
            pre.func(self._env, env_ids, **pre.params)
        # Seed the arm from the default pose, optionally biased by a uniform joint offset (mirrors the
        # reset_joints_by_offset event it replaces) so the reset start-states carry the same joint diversity as
        # the no-IK case; zero velocity. On low IK-iteration snapshots the offset survives into the pose, while
        # fully-converged solves reach the same hover regardless of the seed. noise == 0 -> exact default.
        default_q = self.robot.data.default_joint_pos.torch[env_ids]
        noise = self.cfg.reset.ik_seed_joint_noise
        if noise > 0.0:
            seed_q = default_q + (torch.rand_like(default_q) * 2.0 - 1.0) * noise
            limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
            seed_q = torch.clamp(seed_q, limits[..., 0], limits[..., 1])
        else:
            seed_q = default_q
        self.robot.write_joint_position_to_sim_index(position=seed_q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_q), env_ids=env_ids)
        sim.forward()
        # world-frame hover target above each env's first key (all envs; only env_ids are written back).
        target_w = self.target_key_pos_w() + self._ik_hover
        lo, hi = self._ik_iters
        n_position = max(1, int(round(0.6 * hi)))
        # Per-env iteration budget: after an env hits its sampled count it stops refining (pose frozen), so
        # snapshots land at varying convergence - different tip-to-key distances - a reach-difficulty gradient.
        iters_env = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        max_step = 0.2  # rad/iter cap so a near-singular DLS solve can't flip the arm on a hard key
        lambda_sq = 0.05**2  # damped-least-squares damping (squared), for the pseudo-inverses below
        eye_task = torch.eye(3, device=self.device)
        eye_joint = torch.eye(len(self._ik_joint_ids), device=self.device)
        quat_des: torch.Tensor | None = None
        for i in range(hi):
            ee_pos_w = self.robot.data.body_pos_w.torch[:, self._ik_body_idx]
            ee_quat_w = self.robot.data.body_quat_w.torch[:, self._ik_body_idx]
            # shift the parent-body pose/Jacobian to the jaw tip (world frame).
            lever_w = quat_apply(ee_quat_w, self._ik_offset)  # tip offset expressed in world
            tip_pos_w = ee_pos_w + lever_w
            jac = self.robot.data.body_link_jacobian_w.torch[:, self._ik_jacobi_body_idx, :, self._ik_jacobi_joint_ids]
            jac = jac.clone()
            jac[:, 0:3, :] = jac[:, 0:3, :] + torch.bmm(-skew_symmetric_matrix(lever_w), jac[:, 3:, :])
            q_arm = self.robot.data.joint_pos.torch[:, self._ik_joint_ids]

            # Task-priority damped least squares. The 5-DoF arm cannot reach an arbitrary key position AND
            # a downward pitch, so full-pose DLS trades away position (measured ~6 cm off). Instead make
            # POSITION the primary task (always driven to zero) and add the pitch as a SECONDARY task
            # projected into the null space of position, so orientation can never move the tip off the key.
            j_pos = jac[:, 0:3, :]  # (N, 3, n_joints)
            e_pos = target_w - tip_pos_w  # (N, 3)
            jt_pos = j_pos.transpose(1, 2)
            pos_pinv = jt_pos @ torch.linalg.inv(j_pos @ jt_pos + lambda_sq * eye_task)  # (N, n_joints, 3)
            dq = pos_pinv @ e_pos.unsqueeze(-1)  # (N, n_joints, 1)

            # Phase 2 (once position has settled): pitch the finger down, but only within position's null
            # space. quat_des is computed once from the heading the arm settled into during phase 1.
            if i >= n_position:
                if quat_des is None:
                    quat_des = self._approach_target_quat(ee_quat_w)
                e_rot = axis_angle_from_quat(quat_mul(quat_des, quat_conjugate(ee_quat_w)))  # (N, 3) world
                j_rot = jac[:, 3:6, :]
                jt_rot = j_rot.transpose(1, 2)
                rot_pinv = jt_rot @ torch.linalg.inv(j_rot @ jt_rot + lambda_sq * eye_task)
                null_proj = eye_joint - pos_pinv @ j_pos  # (N, n_joints, n_joints)
                dq = dq + null_proj @ (rot_pinv @ e_rot.unsqueeze(-1))

            q_des = q_arm + torch.clamp(dq.squeeze(-1), -max_step, max_step)
            limits = self.robot.data.soft_joint_pos_limits.torch[:, self._ik_joint_ids]
            q_des = torch.clamp(q_des, limits[..., 0], limits[..., 1])
            # Freeze envs that have spent their sampled iteration budget (write back their current pose).
            q_des = torch.where((i < iters_env)[:, None], q_des, q_arm)
            self.robot.write_joint_position_to_sim_index(
                position=q_des[env_ids], joint_ids=self._ik_joint_ids, env_ids=env_ids
            )
            sim.forward()

    def _approach_target_quat(self, ee_quat_w: torch.Tensor) -> torch.Tensor:
        """Desired approach orientation from ``reset.ik_rpy_deg = (roll, pitch, yaw)``.

        Tilts the finger axis ``pitch`` below horizontal along the arm's settled heading (rotated by
        ``yaw`` about world ``+Z``), then rolls the jaw ``roll`` about that approach (finger) axis. The
        pitch is realized as the minimal rotation of ``ee_quat_w`` onto the desired finger axis. This is
        the *desired* pose only: the null-space solve tracks it best-effort and never sacrifices the tip
        position, so it need not be exactly reachable on the 5-DoF arm.

        Args:
            ee_quat_w: Current moving-jaw link orientation (w, x, y, z), shape ``(num_envs, 4)``.
        """
        finger = quat_apply(ee_quat_w, self._ik_finger_axis)  # current finger axis in world
        heading = finger[:, :2] / torch.linalg.norm(finger[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-6)
        # yaw: rotate the horizontal heading about world +Z.
        if self._ik_yaw != 0.0:
            cos_y, sin_y = math.cos(self._ik_yaw), math.sin(self._ik_yaw)
            hx, hy = heading[:, 0].clone(), heading[:, 1].clone()
            heading = torch.stack([cos_y * hx - sin_y * hy, sin_y * hx + cos_y * hy], dim=-1)
        cos_p, sin_p = math.cos(self._ik_pitch), math.sin(self._ik_pitch)
        down = torch.full((finger.shape[0], 1), -sin_p, device=self.device)
        target_axis = torch.cat([cos_p * heading, down], dim=-1)  # unit: finger pitched below horizontal
        # pitch: minimal rotation tilting the current finger axis onto the desired one.
        rot_axis = torch.linalg.cross(finger, target_axis)
        rot_axis = rot_axis / torch.linalg.norm(rot_axis, dim=-1, keepdim=True).clamp_min(1.0e-6)
        angle = torch.acos((finger * target_axis).sum(-1).clamp(-1.0, 1.0))
        quat = quat_mul(quat_from_angle_axis(angle, rot_axis), ee_quat_w)
        # roll: rotate the jaw about its approach (finger) axis.
        if self._ik_roll != 0.0:
            roll = torch.full((finger.shape[0],), self._ik_roll, device=self.device)
            quat = quat_mul(quat_from_angle_axis(roll, target_axis), quat)
        return quat

    """
    Key geometry (world-frame key positions in global slot order).
    """

    def key_pos_w(self) -> torch.Tensor:
        """World positions of every key in global slot order, shape ``(num_envs, num_keys, 3)`` [m]."""
        return self.keyboard.data.body_pos_w.torch[self._inst_idx, self._body_col_idx]

    def target_key_slot(self) -> torch.Tensor:
        """Global key slot the agent should press next, shape ``(num_envs,)``.

        Returns the backspace slot whenever a wrong/extra key sits past the correct prefix
        (``typed_len > prefix_len``) - deleting it is the only way to make progress, since typing more
        just fills the buffer without extending the prefix. Otherwise returns the next correct key of the
        target word; when the target is already fully typed, the last target key (a benign, in-reach goal).
        Shared by the reach reward (:meth:`target_key_pos_w`) and the debug halo so they stay in sync.
        """
        env_idx = torch.arange(self.num_envs, device=self.device)
        nxt = torch.minimum(self.prefix_len, (self.target_len - 1).clamp(min=0))
        # a wrong/extra key past the prefix must be backspaced before any new key can help; guide there.
        needs_backspace = self.typed_len > self.prefix_len
        return torch.where(needs_backspace, self._backspace, self.target[env_idx, nxt]).clamp(min=0)

    def target_key_pos_w(self) -> torch.Tensor:
        """World position of the next key to press, shape ``(num_envs, 3)`` [m] (see :meth:`target_key_slot`)."""
        env_idx = torch.arange(self.num_envs, device=self.device)
        slot = self.target_key_slot()
        return self.keyboard.data.body_pos_w.torch[self._inst_idx[env_idx, slot], self._body_col_idx[env_idx, slot]]

    """
    Debug visualization: LED dot-matrix banner of the target and typed words plus a next-key halo.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_typing_visualizer"):
                from isaaclab.markers import VisualizationMarkers

                self._typing_visualizer = VisualizationMarkers(
                    typing_vis.make_typing_visualizer_cfg(
                        self.cfg.visualizer_prim_path,
                        self.cfg.viz_pixel_size,
                        self.cfg.viz_key_marker_radius,
                        self.cfg.viz_key_marker_height,
                    )
                )
                # keyboard-dependent buffers are built lazily on the first callback (the keyboard is
                # not yet assigned while the base class enables debug vis during ``__init__``).
                self._viz_ready = False
            self._typing_visualizer.set_visibility(True)
        elif hasattr(self, "_typing_visualizer"):
            self._typing_visualizer.set_visibility(False)

    def _build_viz_resources(self):
        device = self.device
        # Markers are drawn only for these envs (cfg.viz_env_ids; None -> all). Playback keeps this as None so
        # every visible keyboard has its own typing banner and next-key marker.
        # Auto-clamp to the envs that exist: viz_env_ids may exceed num_envs (e.g. the cfg default [0, 1]
        # while running --num_envs=1), and indexing a non-existent env trips a CUDA out-of-bounds assert.
        requested = (
            torch.arange(self.num_envs, device=device)
            if self.cfg.viz_env_ids is None
            else torch.tensor(self.cfg.viz_env_ids, dtype=torch.long, device=device)
        )
        self._viz_env_ids = requested[requested < self.num_envs]
        # Floating banner axes (world frame): letters spread along +Y, dot rows stack along +Z.
        self._viz_right = torch.tensor([0.0, 1.0, 0.0], device=device)
        self._viz_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._glyph_lit, char_to_index = typing_vis.build_glyph_table(device)
        self._slot_glyph = typing_vis.slot_glyph_indices(self.cfg.slot_labels, char_to_index, device)
        self._pixel_offset = typing_vis.pixel_offsets(self.cfg.viz_pixel_size, self._viz_right, self._viz_up)
        self._cell_offset = typing_vis.cell_offsets(
            self.max_len,
            self.cfg.viz_pixel_size,
            self.cfg.viz_letter_gap,
            self.cfg.viz_row_gap,
            self._viz_right,
            self._viz_up,
        )

    def _debug_vis_callback(self, event):
        if not self.keyboard.is_initialized:
            return
        if not getattr(self, "_viz_ready", False):
            self._build_viz_resources()
            self._viz_ready = True

        # Only draw markers for the selected envs (cfg.viz_env_ids) so many-env runs stay readable.
        env_sel = self._viz_env_ids
        if env_sel.numel() == 0:
            return
        n, length = env_sel.shape[0], self.max_len
        arange_n = torch.arange(n, device=self.device)
        jpos = torch.arange(length, device=self.device)[None, :]

        # Per-env typing state, restricted to the visualized envs.
        target = self.target[env_sel]
        typed = self.typed[env_sel]
        prefix_len = self.prefix_len[env_sel]
        typed_len = self.typed_len[env_sel]
        target_len = self.target_len[env_sel]

        # Key world positions in global slot order -> banner anchor (key centroid) and per-key lookup.
        key_w = self.key_pos_w()[env_sel]  # (n, num_keys, 3)
        anchor = key_w.mean(dim=1) + self._viz_up * self.cfg.viz_banner_height  # (n, 3)

        # Glyph index per letter cell (-1 marks an empty cell).
        empty = torch.full_like(target, -1)
        target_glyph = torch.where(target >= 0, self._slot_glyph[target.clamp(min=0)], empty)
        typed_glyph = torch.where(typed >= 0, self._slot_glyph[typed.clamp(min=0)], empty)
        rows_glyph = torch.stack([target_glyph, typed_glyph], dim=1)  # (n, 2, L)

        # Color (marker prototype) per letter cell.
        prefix = prefix_len[:, None]
        target_color = torch.full((n, length), typing_vis.PIX_PENDING, dtype=torch.long, device=self.device)
        target_color = torch.where(jpos < prefix, torch.full_like(target_color, typing_vis.PIX_CORRECT), target_color)
        target_color = torch.where(jpos == prefix, torch.full_like(target_color, typing_vis.PIX_NEXT), target_color)
        typed_color = torch.where(
            jpos < prefix,
            torch.full((n, length), typing_vis.PIX_CORRECT, dtype=torch.long, device=self.device),
            torch.full((n, length), typing_vis.PIX_WRONG, dtype=torch.long, device=self.device),
        )
        rows_color = torch.stack([target_color, typed_color], dim=1)  # (n, 2, L)

        cell_origin = anchor[:, None, None, :] + self._cell_offset[None]  # (n, 2, L, 3)
        translations, indices = typing_vis.compute_letter_markers(
            rows_glyph, rows_color, self._glyph_lit, cell_origin, self._pixel_offset
        )

        # Next-key halo: follow the actual reach target - the next correct key, or the backspace key
        # while a wrong/extra key still needs deleting (typed_len > prefix_len). Hidden once the word is
        # complete. Uses the same slot as the reach reward so the marker and the reward never disagree.
        target_slot = self.target_key_slot()[env_sel]  # (n,)
        has_ring = (typed_len > prefix_len) | (prefix_len < target_len)
        ring_pos = key_w[arange_n, target_slot] + self._viz_up * self.cfg.viz_key_marker_lift
        ring_pos = ring_pos[has_ring]
        if ring_pos.shape[0] > 0:
            ring_idx = torch.full((ring_pos.shape[0],), typing_vis.KEY_NEXT, dtype=torch.long, device=self.device)
            translations = torch.cat([translations, ring_pos], dim=0)
            indices = torch.cat([indices, ring_idx], dim=0)

        if translations.shape[0] == 0:
            return
        self._typing_visualizer.visualize(translations=translations, marker_indices=indices)
