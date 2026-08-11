# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-conditioned full-scene reset replay for cable routing."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import torch

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.mdp.events import SuccessMonitor
from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg

from .reset_robot_targets import CableResetRobotTargetCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

SceneState: TypeAlias = dict[str, dict[str, dict[str, torch.Tensor]]]

__all__ = [
    "CableResetReplay",
    "CableResetReplayCfg",
    "SceneStateBuffer",
    "active_step_progress_from_route_progress",
    "finite_scene_state_rows",
]


def active_step_progress_from_route_progress(
    route_progress: torch.Tensor,
    route_ids: torch.Tensor,
    active_steps: torch.Tensor,
    route_lengths: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Recover within-step progress from normalized whole-route progress."""
    if route_progress.ndim != 1:
        raise ValueError(f"route_progress must be one-dimensional; got {tuple(route_progress.shape)}.")
    if route_ids.shape != route_progress.shape or active_steps.shape != route_progress.shape:
        raise ValueError("route_progress, route_ids, and active_steps must have identical shapes.")
    if not route_progress.is_floating_point():
        raise TypeError("route_progress must use a floating-point dtype.")
    if route_ids.dtype == torch.bool or route_ids.is_floating_point():
        raise TypeError("route_ids must use an integer dtype.")
    if active_steps.dtype == torch.bool or active_steps.is_floating_point():
        raise TypeError("active_steps must use an integer dtype.")
    if route_ids.device != route_progress.device or active_steps.device != route_progress.device:
        raise ValueError("Progress and route metadata must use the same device.")

    lengths = torch.as_tensor(route_lengths, device=route_progress.device, dtype=torch.long)
    if lengths.ndim != 1 or len(lengths) < 1 or bool((lengths < 1).any()):
        raise ValueError("route_lengths must be a non-empty one-dimensional sequence of positive integers.")
    route_ids = route_ids.to(dtype=torch.long)
    active_steps = active_steps.to(dtype=torch.long)
    if bool(((route_ids < 0) | (route_ids >= len(lengths))).any()):
        raise IndexError("route_ids contains an out-of-range route index.")
    selected_lengths = lengths[route_ids]
    if bool(((active_steps < 0) | (active_steps >= selected_lengths)).any()):
        raise IndexError("active_steps contains an out-of-range route step.")
    return route_progress * selected_lengths.to(dtype=route_progress.dtype) - active_steps.to(
        dtype=route_progress.dtype
    )


def finite_scene_state_rows(state: SceneState) -> torch.Tensor:
    """Return rows whose complete nested scene snapshot contains only finite values."""
    row_mask: torch.Tensor | None = None
    num_rows: int | None = None
    device: torch.device | None = None
    for assets in state.values():
        for asset_state in assets.values():
            for value in asset_state.values():
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Every scene-state field must be a torch.Tensor.")
                if value.ndim < 1:
                    raise ValueError("Every scene-state field must have a leading environment dimension.")
                if num_rows is None:
                    num_rows = value.shape[0]
                    device = value.device
                    row_mask = torch.ones(num_rows, device=device, dtype=torch.bool)
                elif value.shape[0] != num_rows or value.device != device:
                    raise ValueError("All scene-state fields must share their leading dimension and device.")
                assert row_mask is not None
                row_mask &= torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
    if row_mask is None:
        raise ValueError("Scene state must contain at least one tensor field.")
    return row_mask


@configclass
class CableResetReplayCfg:
    """Configuration for a fixed, success-conditioned cable reset bank."""

    enabled: bool = True
    """Whether episode resets are restored from the replay bank."""

    buffer_size: int = 4096
    """Number of complete physical snapshots in the fixed bank."""

    success_monitor: SuccessMonitorCfg = SuccessMonitorCfg(
        monitored_history_len=10,
        target_success_rate=0.5,
        kappa=1.0,
        temperature=1.0,
    )
    """Exact Lift success-rate monitor used to weight snapshot slots."""

    robot_targets: CableResetRobotTargetCfg = CableResetRobotTargetCfg()
    """Route-conditioned reach/cage targets used to make robot and cable states coherent."""

    active_progress_range: tuple[float, float] = (0.40, 0.92)
    """Sampled nonterminal progress fraction for the active route step."""

    generation_batch_size: int = 128
    """Maximum scratch worlds projected together during one-time bank construction.

    The fixed-sweep XPBD-style projector scans non-local vertex pairs, so this
    bound is intentionally independent of the much larger training environment
    batch.
    """

    completed_winding: float = 4.0
    """Directed winding [rad] used for route steps before the active step."""

    settle_steps: int = 16
    """Physics steps used to relax each generated cable state before it is banked."""

    max_settle_attempts: int = 64
    """Maximum generate-and-settle attempts used to fill each replay-bank row.

    Only rejected rows are regenerated. A 4096-state bank otherwise amplifies
    rare rejection tails: live OSMO seeds each left one row unfilled at the old
    16-attempt cap despite the other 4095 rows being viable.
    """

    max_settle_linear_speed: float = 0.15
    """Maximum cable-segment linear speed [m/s] accepted after settling."""

    max_settle_angular_speed: float = 15.0
    """Maximum cable-segment angular speed [rad/s] accepted after settling."""

    max_segment_length_relative_error: float = 0.15
    """Maximum relative cable-segment length error accepted after settling."""

    restore_clearance: float = 5.0e-6
    """Fixture and tabletop reserve retained when accepting replay snapshots [m].

    Replay rows are stored relative to one scratch clone and may be restored at any other clone. The 5 micrometer
    reserve is five times the largest float32 relocation loss observed across the clone grid, preventing exact
    surface contact from becoming penetration without rejecting physically clear settled cable states.
    """

    post_settle_progress_tolerance: float = 0.35
    """Maximum absolute normalized route-progress change accepted after settling."""

    maximum_settled_active_progress: float = 0.92
    """Maximum active-step progress accepted after settling.

    This leaves a deliberate completion margin in replay states. It prevents
    ordinary gripper-release actions from turning tiny settling differences
    into immediate route completions while preserving the configured 40%--92%
    frontier curriculum.
    """

    wrap_radius_range: tuple[float, float] = (0.023, 0.029)
    """Cable centerline radius range [m] around round pegs."""

    entry_angle_jitter: float = 0.65
    """Maximum directed-arc entry-angle perturbation [rad]."""

    start_position_jitter: float = 0.004
    """Maximum planar endpoint perturbation [m]."""

    curve_projection_iterations: int = 50
    """Chebyshev-accelerated Warp sweeps before the Taubin and cleanup tail."""

    repulsive_iterations: int | None = None
    """Deprecated alias for :attr:`curve_projection_iterations`."""

    max_curve_attempts: int = 512
    """Maximum guide resamples for each candidate curve.

    Curve rejection is row-local and stops immediately on acceptance, so this
    tail budget does not penalize ordinary candidates. The 12 mm bend gate can
    produce rare long tails for particular fixture/progress combinations; this
    matches the ordinary cable-reset sampler's proven 512-attempt safety bound
    and prevents one difficult row from aborting a 4096-state bank.
    """

    arm_joint_position_jitter: float = 0.05
    """Uniform arm-joint reset perturbation [rad] stored in the bank."""

    seed: int = 2026
    """Deterministic seed for fixed-bank construction."""

    def __post_init__(self) -> None:
        """Validate replay and curve-generation settings."""
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be positive.")
        if self.generation_batch_size < 1:
            raise ValueError("generation_batch_size must be positive.")
        if not 0.0 < self.active_progress_range[0] <= self.active_progress_range[1] < 1.0:
            raise ValueError("active_progress_range must lie strictly inside (0, 1).")
        if self.completed_winding <= 0.0:
            raise ValueError("completed_winding must be positive.")
        if self.settle_steps < 1:
            raise ValueError("settle_steps must be positive.")
        if self.max_settle_attempts < 1:
            raise ValueError("max_settle_attempts must be positive.")
        if not math.isfinite(self.max_settle_linear_speed) or self.max_settle_linear_speed <= 0.0:
            raise ValueError("max_settle_linear_speed must be finite and positive.")
        if not math.isfinite(self.max_settle_angular_speed) or self.max_settle_angular_speed <= 0.0:
            raise ValueError("max_settle_angular_speed must be finite and positive.")
        if (
            not math.isfinite(self.max_segment_length_relative_error)
            or not 0.0 <= self.max_segment_length_relative_error < 1.0
        ):
            raise ValueError("max_segment_length_relative_error must lie in [0, 1).")
        if not math.isfinite(self.restore_clearance) or self.restore_clearance <= 0.0:
            raise ValueError("restore_clearance must be finite and positive.")
        if (
            not math.isfinite(self.post_settle_progress_tolerance)
            or not 0.0 <= self.post_settle_progress_tolerance <= 1.0
        ):
            raise ValueError("post_settle_progress_tolerance must lie in [0, 1].")
        if (
            not math.isfinite(self.maximum_settled_active_progress)
            or not self.active_progress_range[1] <= self.maximum_settled_active_progress < 1.0
        ):
            raise ValueError(
                "maximum_settled_active_progress must include active_progress_range and lie strictly below 1."
            )
        if self.wrap_radius_range[0] <= 0.0 or self.wrap_radius_range[0] > self.wrap_radius_range[1]:
            raise ValueError("wrap_radius_range must be positive and ordered.")
        if min(self.entry_angle_jitter, self.start_position_jitter, self.arm_joint_position_jitter) < 0.0:
            raise ValueError("Reset jitters must be non-negative.")
        if self.repulsive_iterations is not None:
            warnings.warn(
                "repulsive_iterations is deprecated; use curve_projection_iterations.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.curve_projection_iterations = self.repulsive_iterations
        if (
            isinstance(self.curve_projection_iterations, bool)
            or not isinstance(self.curve_projection_iterations, int)
            or self.curve_projection_iterations < 0
        ):
            raise ValueError("curve_projection_iterations must be a non-negative integer.")
        if self.max_curve_attempts < 1:
            raise ValueError("max_curve_attempts must be positive.")


class SceneStateBuffer:
    """Fixed-capacity tensor bank matching :meth:`InteractiveScene.get_state`."""

    def __init__(self, capacity: int, example: SceneState):
        """Allocate one leading bank dimension for every scene-state tensor."""
        if capacity < 1:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._state: SceneState = {
            asset_type: {
                asset_name: {
                    field: torch.empty(
                        (capacity, *value.shape[1:]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    for field, value in asset_state.items()
                }
                for asset_name, asset_state in assets.items()
            }
            for asset_type, assets in example.items()
        }

    @property
    def state(self) -> SceneState:
        """Return the backing state tensors."""
        return self._state

    def store(self, start: int, state: SceneState, env_ids: torch.Tensor) -> None:
        """Copy selected live environments into consecutive bank rows."""
        count = len(env_ids)
        if start < 0 or start + count > self.capacity:
            raise IndexError(f"Rows [{start}, {start + count}) exceed buffer capacity {self.capacity}.")
        destination_rows = torch.arange(start, start + count, device=env_ids.device, dtype=torch.long)
        self.store_rows(destination_rows, state, env_ids)

    def store_rows(self, destination_rows: torch.Tensor, state: SceneState, env_ids: torch.Tensor) -> None:
        """Copy selected live environments into arbitrary replay-bank rows.

        Args:
            destination_rows: Replay-bank rows to overwrite, shape ``(N,)``.
            state: Complete live scene state following :meth:`InteractiveScene.get_state`.
            env_ids: Live environment rows copied into ``destination_rows``, shape ``(N,)``.
        """
        if destination_rows.ndim != 1 or env_ids.ndim != 1:
            raise ValueError("destination_rows and env_ids must both be one-dimensional.")
        if destination_rows.shape != env_ids.shape:
            raise ValueError(
                f"destination_rows and env_ids must have the same shape; got "
                f"{tuple(destination_rows.shape)} and {tuple(env_ids.shape)}."
            )
        if destination_rows.dtype not in (torch.int32, torch.int64) or env_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("destination_rows and env_ids must contain integer indices.")
        if len(destination_rows) == 0:
            return
        if bool(((destination_rows < 0) | (destination_rows >= self.capacity)).any()):
            raise IndexError(f"destination_rows must lie in [0, {self.capacity}); got {destination_rows.tolist()}.")

        sources: list[tuple[torch.Tensor, torch.Tensor]] = []
        source_env_bounds: tuple[int, int] | None = None
        for asset_type, assets in self._state.items():
            for asset_name, fields in assets.items():
                for field, destination in fields.items():
                    source = state[asset_type][asset_name][field]
                    if source.ndim != destination.ndim or source.shape[1:] != destination.shape[1:]:
                        raise ValueError(
                            f"State field {asset_type}/{asset_name}/{field} must have trailing shape "
                            f"{tuple(destination.shape[1:])}; got {tuple(source.shape)}."
                        )
                    if source_env_bounds is None:
                        source_env_min, source_env_max = torch.stack(torch.aminmax(env_ids)).tolist()
                        source_env_bounds = (int(source_env_min), int(source_env_max))
                    if source_env_bounds[0] < 0 or source_env_bounds[1] >= source.shape[0]:
                        raise IndexError(
                            f"env_ids must lie in [0, {source.shape[0]}) for state field "
                            f"{asset_type}/{asset_name}/{field}; got {env_ids.tolist()}."
                        )
                    sources.append((destination, source))

        for destination, source in sources:
            destination[destination_rows] = source[env_ids]

    def gather(self, rows: torch.Tensor) -> SceneState:
        """Gather bank rows in :meth:`InteractiveScene.reset_to` format."""
        return {
            asset_type: {
                asset_name: {field: value[rows] for field, value in fields.items()}
                for asset_name, fields in assets.items()
            }
            for asset_type, assets in self._state.items()
        }


class CableResetReplay:
    """Own the frozen scene bank, Lift monitor, and episode-source attribution."""

    def __init__(self, cfg: CableResetReplayCfg, env: ManagerBasedRLEnv):
        """Allocate source metadata and the exact Lift success monitor."""
        self.cfg = cfg
        self.env = env
        self.monitor: SuccessMonitor = cfg.success_monitor.class_type(
            cfg.success_monitor,
            num_partitions=1,
            partition_size=cfg.buffer_size,
            device=str(env.device),
        )
        self.state_buffer: SceneStateBuffer | None = None
        self.route_id = torch.zeros(cfg.buffer_size, device=env.device, dtype=torch.long)
        self.active_step = torch.zeros(cfg.buffer_size, device=env.device, dtype=torch.long)
        self.interaction_phase = torch.zeros(cfg.buffer_size, device=env.device, dtype=torch.long)
        # Keep the authored within-step target separate from settled normalized
        # whole-route progress; they have different ranges for multi-step goals.
        self.requested_active_progress = torch.zeros(cfg.buffer_size, device=env.device)
        self.start_progress = torch.zeros(cfg.buffer_size, device=env.device)
        self.env_source = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self._route_rows: dict[int, torch.Tensor] = {}
        self.build_candidate_count = 0
        self.build_rejection_count = 0
        self.build_max_attempts = 0
        self.build_duration_s = 0.0
        self.built = False

    def allocate(self, example: SceneState) -> None:
        """Allocate the physical-state bank from one live scene-state example."""
        self.state_buffer = SceneStateBuffer(self.cfg.buffer_size, example)

    def sample_sources(self, route_ids: torch.Tensor) -> torch.Tensor:
        """Draw one target-rate-weighted replay row for each uniformly chosen goal."""
        if route_ids.ndim != 1:
            raise ValueError("route_ids must be one-dimensional.")
        if route_ids.dtype == torch.bool or route_ids.is_floating_point():
            raise TypeError("route_ids must use an integer dtype.")
        if route_ids.device != self.route_id.device:
            raise ValueError("route_ids must be on the replay device.")
        if len(route_ids) == 0:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        weights = self.monitor.target_weights()
        sources = torch.empty_like(route_ids, dtype=torch.long)
        for route_id in torch.unique(route_ids).tolist():
            requested = (route_ids == route_id).nonzero(as_tuple=False).squeeze(-1)
            candidates = self._route_rows.get(route_id)
            if candidates is None:
                candidates = (self.route_id == route_id).nonzero(as_tuple=False).squeeze(-1)
                self._route_rows[route_id] = candidates
            if len(candidates) == 0:
                raise RuntimeError(f"Reset replay contains no rows for requested route {route_id}.")
            sampled = torch.multinomial(weights[candidates], len(requested), replacement=True)
            sources[requested] = candidates[sampled]
        return sources

    def restore(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Restore complete relative scene states into selected environments."""
        if self.state_buffer is None or not self.built:
            raise RuntimeError("Cannot restore an unbuilt reset replay bank.")
        self.env.scene.reset_to(self.state_buffer.gather(rows), env_ids=env_ids, is_relative=True)

    def credit(self, env_ids: torch.Tensor, succeeded: torch.Tensor) -> None:
        """Credit terminal outcomes to the replay snapshots that seeded them."""
        source = self.env_source[env_ids]
        replayed = source >= 0
        self.monitor.success_update(source[replayed], succeeded[replayed])

    def metrics(self, sampled_sources: torch.Tensor | None = None) -> dict[str, float]:
        """Return compact replay diagnostics for the ordinary task logger."""
        measured = self.monitor.success_size > 0
        metrics = {
            "reset_replay/success_rate": self.monitor.get_mean_success_rate(),
            "reset_replay/measured_slot_fraction": float(measured.float().mean()),
            "reset_replay/build_rejection_rate": self.build_rejection_count / max(self.build_candidate_count, 1),
            "reset_replay/build_max_attempts": float(self.build_max_attempts),
            "reset_replay/build_duration_s": self.build_duration_s,
        }
        if sampled_sources is not None and len(sampled_sources) > 0:
            replayed = sampled_sources >= 0
            metrics["reset_replay/buffer_fraction"] = float(replayed.float().mean())
            if bool(replayed.any()):
                rows = sampled_sources[replayed]
                metrics["reset_replay/mean_sampled_slot_rate"] = float(self.monitor.success_rate[rows].mean())
                metrics["reset_replay/mean_start_progress"] = float(self.start_progress[rows].mean())
                metrics["reset_replay/mean_requested_active_progress"] = float(
                    self.requested_active_progress[rows].mean()
                )
                sampled_phase = self.interaction_phase[rows]
                for phase, name in enumerate(("reach", "cage", "bimanual")):
                    metrics[f"reset_replay/{name}_sample_fraction"] = float((sampled_phase == phase).float().mean())
                sampled_route = self.route_id[rows]
                for route_id in torch.unique(self.route_id).tolist():
                    metrics[f"reset_replay/route_{route_id}_sample_fraction"] = float(
                        (sampled_route == route_id).float().mean()
                    )
        for phase, name in enumerate(("reach", "cage", "bimanual")):
            phase_measured = measured & (self.interaction_phase == phase)
            metrics[f"reset_replay/{name}_success_rate"] = (
                float(self.monitor.success_rate[phase_measured].mean()) if bool(phase_measured.any()) else 0.0
            )
        for route_id in torch.unique(self.route_id).tolist():
            route_rows = self.route_id == route_id
            route_measured = measured & route_rows
            metrics[f"reset_replay/route_{route_id}_measured_slot_fraction"] = float(
                route_measured.sum() / route_rows.sum().clamp_min(1)
            )
            metrics[f"reset_replay/route_{route_id}_monitor_success_rate"] = (
                float(self.monitor.success_rate[route_measured].mean()) if bool(route_measured.any()) else 0.0
            )
        return metrics
