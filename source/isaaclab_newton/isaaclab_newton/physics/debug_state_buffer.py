# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug state buffer for NaN incident replay in Newton physics."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from newton import Model, State

logger = logging.getLogger(__name__)

_MAX_BUFFER_SIZE = 2000


class DebugStateBuffer:
    """Rolling buffer of Newton state snapshots with GPU-side NaN detection.

    On every step the current state is copied into a ring buffer (GPU-GPU via
    ``State.assign``).  NaN detection runs entirely on GPU using a single fused
    ``torch.isnan`` + ``any`` per array -- no data ever leaves the device on the
    hot path.  Only when a NaN is found does the buffer dump to CPU for export.

    When the model has multiple worlds (replicated envs), the buffer identifies
    which env(s) contain NaN, exports only those slices, and suppresses future
    detection of already-exported env_ids (NaN is sticky).

    After :attr:`max_exports` exports, :attr:`nan_halt` is set and subsequent
    ``step`` calls are no-ops.  The caller (Newton manager) should check
    ``nan_halt`` and raise to stop simulation.

    Diagnostics (solver convergence, forces, accelerations, mass matrix
    condition, contact penetration) are recorded alongside state snapshots
    when provided via the ``diagnostics`` argument to :meth:`step`.

    When ``solver_data`` is provided, the buffer exports the complete MuJoCo
    Warp solver state on NaN for exact single-step reproduction.  Solver
    divergence history is retrospectively analyzed from the diagnostics ring
    during export -- no extra GPU->CPU syncs on the hot path.
    """

    def __init__(
        self,
        model: Model,
        buffer_size: int,
        export_path: str = ".",
        max_exports: int = 1,
        scene_exporter: Callable[[str, list[int]], None] | None = None,
    ) -> None:
        """Initialize the debug state buffer.

        Args:
            model: Finalized Newton model (used to allocate state clones and read
                world layout).
            buffer_size: Number of state snapshots to keep. Capped at
                :data:`_MAX_BUFFER_SIZE`.
            export_path: Directory for npz export.
            max_exports: Maximum number of NaN export events before halting. Each
                event exports a distinct set of newly-NaN env_ids.
            scene_exporter: Optional callable ``(usd_path, env_ids) -> None`` that
                exports USD prim subtrees for the given env_ids. Called once per
                export event. If *env_ids* is empty the exporter should export the
                whole scene (single-env case).
        """
        size = min(max(int(buffer_size), 1), _MAX_BUFFER_SIZE)
        self._ring: list[State] = [model.state() for _ in range(size)]
        self._diag_ring: list[dict[str, torch.Tensor]] = [{} for _ in range(size)]
        self._size: int = size
        self._write_idx: int = 0
        self._export_path: str = export_path
        self._max_exports: int = max(int(max_exports), 1)
        self._scene_exporter = scene_exporter

        self._export_count: int = 0
        self._nan_halt: bool = False
        self._exported_envs: set[int] = set()

        # Per-env layout (populated if model has worlds)
        self._world_count: int = 0
        self._body_starts: np.ndarray | None = None
        self._joint_coord_starts: np.ndarray | None = None
        self._joint_dof_starts: np.ndarray | None = None

        # Solver data reference (updated every step, used on export)
        self._last_solver_data: dict[str, Any] | None = None

        # Rolling mjw_data shadow: 2 slots, GPU-GPU copy each step.
        # On NaN, the previous slot has the pre-NaN solver input state.
        self._mjw_shadow: list[dict[str, torch.Tensor]] = [{}, {}]
        self._mjw_shadow_idx: int = 0

        # Per-env episode step tracking (set externally via :attr:`episode_length_buf`)
        self._episode_length_buf: torch.Tensor | None = None

        # Pre-step state snapshot: captures state *before* the solver step so
        # that on NaN export we have the exact solver input (post-reset state
        # when episode_step=0).
        self._pre_step_state: State = model.state()

        self._read_world_layout(model)

    def _read_world_layout(self, model: Model) -> None:
        world_count = getattr(model, "world_count", 0) or 0
        if world_count <= 1:
            return
        body_ws = getattr(model, "body_world_start", None)
        if body_ws is None:
            return
        self._world_count = int(world_count)
        self._body_starts = body_ws.numpy() if hasattr(body_ws, "numpy") else np.asarray(body_ws)
        jc = getattr(model, "joint_coord_world_start", None)
        jd = getattr(model, "joint_dof_world_start", None)
        self._joint_coord_starts = jc.numpy() if jc is not None and hasattr(jc, "numpy") else (np.asarray(jc) if jc is not None else None)
        self._joint_dof_starts = jd.numpy() if jd is not None and hasattr(jd, "numpy") else (np.asarray(jd) if jd is not None else None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of state snapshots in the buffer."""
        return self._size

    @property
    def nan_halt(self) -> bool:
        """True after :attr:`max_exports` NaN exports have occurred."""
        return self._nan_halt

    @property
    def episode_length_buf(self) -> torch.Tensor | None:
        """Per-env episode step counter. Set by the RL env so the buffer can
        report how far into each episode a NaN occurs."""
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, buf: torch.Tensor) -> None:
        self._episode_length_buf = buf

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def snapshot_pre_step(self, current_state: State) -> None:
        """Capture state before the solver step.

        Call this right before ``NewtonManager._simulate()``.  On NaN, the
        export will include this snapshot as ``pre_step_*`` arrays — the exact
        solver input state (which, for ``episode_step=0``, is the post-reset
        configuration).
        """
        if self._nan_halt:
            return
        self._pre_step_state.assign(current_state)

    def step(
        self,
        current_state: State,
        sim_time: float,
        diagnostics: dict | None = None,
        solver_data: dict[str, Any] | None = None,
    ) -> None:
        """Copy state into ring, advance index, check for NaN, export if found.

        Everything except the final export stays on GPU.  No-op when
        :attr:`nan_halt` is True or the ring is empty.

        Args:
            current_state: Current Newton state (GPU).
            sim_time: Simulation time [s] at this step.
            diagnostics: Optional dict of per-step diagnostic tensors (GPU).
                Supported keys: ``solver_niter``, ``qfrc_constraint``,
                ``qfrc_actuator``, ``qacc``, ``qM_diag_min``,
                ``contact_dist_min``.  Values are copied to CPU numpy arrays
                and stored in the diagnostics ring buffer.
            solver_data: Optional dict with MuJoCo Warp solver references for
                divergence detection and NaN snapshot export.  Expected keys:
                ``mjw_data`` (the solver's Data object), ``max_iterations``
                (int), ``nv`` (int, actual DOFs).
        """
        if self._nan_halt or not self._ring:
            return

        self._last_solver_data = solver_data

        idx = self._write_idx
        self._ring[idx].assign(current_state)

        if diagnostics:
            snap = self._diag_ring[idx]
            for key, val in diagnostics.items():
                if val is None:
                    continue
                src = val if isinstance(val, torch.Tensor) else wp.to_torch(val)
                if key in snap and snap[key].shape == src.shape:
                    snap[key].copy_(src)
                else:
                    snap[key] = src.detach().clone()
            self._diag_ring[idx] = snap

        self._write_idx = (idx + 1) % self._size

        # Shadow-copy key mjw_data arrays (GPU-GPU) for pre-NaN snapshot
        if solver_data is not None:
            self._snapshot_mjw_data(solver_data)

        nan_detected, bad_envs = self._detect_nan(current_state)
        if nan_detected:
            self._export(sim_time, bad_envs)

    _MJW_SNAPSHOT_KEYS = (
        "qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_applied", "qfrc_bias",
        "qfrc_passive", "qfrc_constraint", "qfrc_actuator", "qfrc_smooth", "qM",
        "solver_niter",
    )
    _MJW_EFC_SNAPSHOT_KEYS = ("force",)
    _MJW_CONTACT_SNAPSHOT_KEYS = ("dist", "pos", "frame", "friction", "dim", "geom", "worldid")
    _MJW_SCALAR_SNAPSHOT_KEYS = ("nacon",)

    def _snapshot_mjw_data(self, solver_data: dict[str, Any]) -> None:
        """GPU-GPU copy of key mjw_data arrays into a rolling 2-slot shadow."""
        mjd = solver_data.get("mjw_data")
        if mjd is None:
            return
        slot = self._mjw_shadow[self._mjw_shadow_idx]
        for name in self._MJW_SNAPSHOT_KEYS:
            arr = getattr(mjd, name, None)
            if arr is None:
                continue
            src = wp.to_torch(arr) if isinstance(arr, wp.array) else arr
            if not isinstance(src, torch.Tensor):
                continue
            if name in slot and slot[name].shape == src.shape:
                slot[name].copy_(src)
            else:
                slot[name] = src.detach().clone()
        for sub_name, keys in (("efc", self._MJW_EFC_SNAPSHOT_KEYS),
                                ("contact", self._MJW_CONTACT_SNAPSHOT_KEYS)):
            sub = getattr(mjd, sub_name, None)
            if sub is None:
                continue
            for name in keys:
                arr = getattr(sub, name, None)
                if arr is None:
                    continue
                src = wp.to_torch(arr) if isinstance(arr, wp.array) else arr
                if not isinstance(src, torch.Tensor):
                    continue
                key = f"{sub_name}_{name}"
                if key in slot and slot[key].shape == src.shape:
                    slot[key].copy_(src)
                else:
                    slot[key] = src.detach().clone()
        for name in self._MJW_SCALAR_SNAPSHOT_KEYS:
            arr = getattr(mjd, name, None)
            if arr is None:
                continue
            src = wp.to_torch(arr) if isinstance(arr, wp.array) else arr
            if not isinstance(src, torch.Tensor):
                continue
            if name in slot and slot[name].shape == src.shape:
                slot[name].copy_(src)
            else:
                slot[name] = src.detach().clone()
        self._mjw_shadow_idx = 1 - self._mjw_shadow_idx

    def _detect_nan(self, state: State) -> tuple[bool, list[int]]:
        """GPU-side NaN check.  Returns (has_nan, list_of_bad_env_ids).

        When world layout is unavailable, bad_env_ids is empty.
        Already-exported env_ids are excluded from results.

        Multi-world detection is fully vectorized: per-world NaN flags are
        computed with a single ``reshape`` + ``any(dim=1)`` per array, avoiding
        the per-world ``.item()`` sync loop that would otherwise serialize
        GPU and CPU (O(world_count) syncs per step).
        """
        arrays: list[torch.Tensor] = []
        if state.joint_qd is not None:
            arrays.append(wp.to_torch(state.joint_qd))
        if state.joint_q is not None:
            arrays.append(wp.to_torch(state.joint_q))
        if state.body_q is not None:
            arrays.append(wp.to_torch(state.body_q))
        if state.body_qd is not None:
            arrays.append(wp.to_torch(state.body_qd))
        if not arrays:
            return False, []

        if self._world_count <= 1 or self._body_starts is None:
            combined = torch.cat([a.flatten() for a in arrays])
            if not torch.isnan(combined).any().item():
                return False, []
            return True, []

        # Vectorized per-world NaN check (1-2 GPU syncs total, not O(world_count))
        wc = self._world_count
        device = arrays[0].device
        per_world_nan = torch.zeros(wc, dtype=torch.bool, device=device)
        for arr in arrays:
            flat = arr.flatten()
            if flat.shape[0] % wc == 0:
                per_world_nan |= torch.isnan(flat.reshape(wc, -1)).any(dim=1)
            elif torch.isnan(flat).any().item():
                per_world_nan[:] = True

        if self._exported_envs:
            exported = torch.tensor(sorted(self._exported_envs), dtype=torch.long, device=device)
            per_world_nan[exported] = False

        if not per_world_nan.any().item():
            return False, []

        bad = per_world_nan.nonzero(as_tuple=False).flatten().cpu().tolist()
        return True, bad

    # ------------------------------------------------------------------
    # Export (cold path -- only on NaN)
    # ------------------------------------------------------------------

    def _export(self, sim_time: float, bad_envs: list[int]) -> None:
        """Dump ring buffer to npz. Only called when NaN detected."""
        n = self._size
        idx = self._write_idx
        ordered = [self._ring[(idx + i) % n] for i in range(n)]
        ordered_diag = [self._diag_ring[(idx + i) % n] for i in range(n)]

        os.makedirs(self._export_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"nan_replay_{ts}"
        npz_path = os.path.join(self._export_path, f"{stem}.npz")

        data = self._dump_full(ordered, n)
        data["exported_env_ids"] = np.array(bad_envs, dtype=np.int32)

        data["buffer_size"] = n
        data["sim_time"] = sim_time
        data["world_count"] = self._world_count

        if self._episode_length_buf is not None:
            ep_len = self._episode_length_buf.cpu().numpy()
            data["episode_length_buf"] = ep_len
            if bad_envs:
                for eid in bad_envs:
                    logger.error("  env %d: episode_step=%d", eid, int(ep_len[eid]))

        # Pre-step state: the solver input that produced NaN
        ps = self._pre_step_state
        if ps.joint_q is not None:
            data["pre_step_joint_q"] = ps.joint_q.numpy()
        if ps.joint_qd is not None:
            data["pre_step_joint_qd"] = ps.joint_qd.numpy()
        if ps.body_q is not None:
            data["pre_step_body_q"] = ps.body_q.numpy()
        if ps.body_qd is not None:
            data["pre_step_body_qd"] = ps.body_qd.numpy()

        if self._last_solver_data is not None:
            opt = self._last_solver_data.get("opt")
            if opt is not None:
                for k, v in opt.items():
                    if v is None:
                        continue
                    if isinstance(v, wp.array):
                        v = v.numpy()
                    data[f"cfg_{k}"] = np.asarray(v)

        self._export_diagnostics(ordered_diag, n, bad_envs, data)

        np.savez_compressed(npz_path, **data)

        # Export USD scene for the NaN'd envs
        if self._scene_exporter is not None:
            usd_path = os.path.join(self._export_path, f"{stem}.usd")
            try:
                self._scene_exporter(usd_path, bad_envs)
                logger.info("Exported scene for envs %s to %s", bad_envs if bad_envs else "all", usd_path)
            except Exception:
                logger.exception("Failed to export scene USD to %s", usd_path)

        # Export pre-NaN mjw_data shadow (the step BEFORE NaN)
        pre_nan_slot = self._mjw_shadow[self._mjw_shadow_idx]  # previous slot
        if pre_nan_slot:
            try:
                mjw_data = {}
                for k, v in pre_nan_slot.items():
                    mjw_data[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                mjw_path = os.path.join(self._export_path, f"{stem}_pre_nan_mjw.npz")
                np.savez_compressed(mjw_path, **mjw_data)
                logger.error("Pre-NaN mjw_data exported to %s", mjw_path)
            except Exception:
                logger.exception("Failed to export pre-NaN mjw_data")

        logger.error(
            "NaN detected (envs %s). Exported %d snapshots to %s",
            bad_envs if bad_envs else "all",
            n,
            npz_path,
        )

        # Track exported envs and check halt condition
        self._exported_envs.update(bad_envs)
        self._export_count += 1
        if self._export_count >= self._max_exports:
            self._nan_halt = True
            logger.error(
                "Reached max NaN exports (%d). Halting debug state buffer.",
                self._max_exports,
            )

    def _export_diagnostics(
        self, ordered_diag: list[dict[str, torch.Tensor]], n: int, bad_envs: list[int], data: dict
    ) -> None:
        """Add diagnostics arrays to the export data dict.

        Converts GPU tensors to CPU numpy only here (cold export path).
        """
        if not ordered_diag or not ordered_diag[0]:
            return

        all_keys: set[str] = set()
        for d in ordered_diag:
            all_keys.update(d.keys())

        for key in sorted(all_keys):
            frames = []
            for i in range(n):
                val = ordered_diag[i].get(key)
                if val is not None:
                    frames.append(val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val))
                else:
                    frames.append(np.zeros_like(frames[-1]) if frames else np.array([0.0]))

            stacked = np.stack(frames)

            data[f"diag_{key}"] = stacked

    @staticmethod
    def _dump_full(ordered: list[State], n: int) -> dict:
        data: dict = {}
        s0 = ordered[0]
        if s0.body_q is not None:
            data["body_q"] = np.stack([ordered[i].body_q.numpy() for i in range(n)])  # type: ignore[union-attr]
        if s0.body_qd is not None:
            data["body_qd"] = np.stack([ordered[i].body_qd.numpy() for i in range(n)])  # type: ignore[union-attr]
        if s0.joint_q is not None:
            data["joint_q"] = np.stack([ordered[i].joint_q.numpy() for i in range(n)])  # type: ignore[union-attr]
        if s0.joint_qd is not None:
            data["joint_qd"] = np.stack([ordered[i].joint_qd.numpy() for i in range(n)])  # type: ignore[union-attr]
        return data

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Release ring buffer."""
        self._ring.clear()
        self._diag_ring.clear()
        self._write_idx = 0
        self._size = 0
        self._last_solver_data = None
