# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import CheckpointCatalog, CheckpointEntry, CheckpointLoader
from .config import PolicyDebugCfg
from .slots import SlotAllocator

_TINTS = (
    (1.0, 0.35, 0.35),
    (0.35, 0.75, 1.0),
    (0.45, 1.0, 0.45),
    (1.0, 0.75, 0.3),
    (0.8, 0.45, 1.0),
    (0.2, 1.0, 0.9),
    (1.0, 0.45, 0.8),
)


@dataclass
class ActivePolicy:
    """Loaded policy and the fixed simulation slot it controls."""

    entry: CheckpointEntry
    policy: Any
    slot: int
    tint: tuple[float, float, float]


class PolicyDebugManager:
    """Application-level synchronized multi-checkpoint rollout manager."""

    def __init__(
        self,
        cfg: PolicyDebugCfg,
        env: Any,
        visualizer: Any,
        scenario_adapter: Any,
        policy_factory: Any,
        action_dim: int,
    ):
        self.cfg = cfg
        self.env = env
        self.visualizer = visualizer
        self.scenario_adapter = scenario_adapter
        self.policy_factory = policy_factory
        self.action_dim = action_dim
        self.catalog = CheckpointCatalog(cfg.run_dir, stable_scans=cfg.stable_scans)
        self.loader = CheckpointLoader()
        self.slots = SlotAllocator(cfg.max_policies)
        self.active: dict[Path, ActivePolicy] = {}
        self.overlay = False
        self.ghost_opacity = cfg.ghost_opacity
        self._activation_requests: deque[CheckpointEntry] = deque()
        self._last_scan = float("-inf")
        self._observations = None
        self.catalog.scan()
        self._last_scan = time.monotonic()
        self.visualizer.configure_environment_layers(range(cfg.max_policies))
        comparison_assets = self.scenario_adapter.comparison_visible_assets()
        self._comparison_shape_visibility = (
            self.visualizer.scene_asset_shape_visibility(comparison_assets) if comparison_assets is not None else None
        )
        overlay_assets = self.scenario_adapter.overlay_visible_assets()
        self._overlay_shape_visibility = (
            self.visualizer.scene_asset_shape_visibility(overlay_assets) if overlay_assets is not None else None
        )
        self._sync_visualizer()

    def run(self) -> None:
        """Run until the Newton viewer closes, including while no policy is active."""
        try:
            while self.visualizer.is_running():
                self.step()
        except KeyboardInterrupt:
            pass

    def set_checkpoint_enabled(self, entry: CheckpointEntry, enabled: bool) -> None:
        """Request activation, or immediately remove a currently active policy."""
        if enabled:
            if entry.path not in self.active and entry not in self._activation_requests:
                entry.status = "loading"
                entry.error = None
                self._activation_requests.append(entry)
        else:
            self._activation_requests = deque(item for item in self._activation_requests if item.path != entry.path)
            active = self.active.pop(entry.path, None)
            if active is not None:
                self.slots.release(active.slot)
                entry.status = "ready"
                self._sync_visualizer(frame=not self.overlay)

    def set_overlay(self, enabled: bool) -> None:
        self.overlay = bool(enabled)
        self._sync_visualizer(frame=not self.overlay)

    def set_ghost_opacity(self, opacity: float) -> None:
        self.ghost_opacity = min(1.0, max(0.0, float(opacity)))
        if self.overlay:
            self._sync_visualizer()

    def rescan(self) -> None:
        self.catalog.rescan_from_scratch()
        self._last_scan = time.monotonic()

    def apply_frame_boundary_requests(self) -> None:
        """Load requested policies and synchronize one restart at a safe boundary."""
        activated = False
        while self._activation_requests:
            entry = self._activation_requests.popleft()
            if not entry.ready or entry.path in self.active:
                entry.status = "waiting" if not entry.ready else entry.status
                continue
            slot = None
            try:
                slot = self.slots.allocate(entry.path)
                checkpoint = self.loader.load(entry)
                self.scenario_adapter.validate_checkpoint(checkpoint, self.env)
                policy = self.policy_factory.create(checkpoint)
                tint = _TINTS[slot % len(_TINTS)]
                self.active[entry.path] = ActivePolicy(entry, policy, slot, tint)
                entry.status = "active"
                activated = True
            except Exception as exc:
                if slot is not None:
                    self.slots.release(slot)
                entry.error = str(exc)
                entry.status = "error"
                print(f"[Policy Debug] Could not activate {entry.path.name}: {entry.error}", file=sys.stderr)
        if activated:
            self._sync_visualizer(frame=not self.overlay)
            try:
                self.restart_synchronized()
            except Exception as exc:
                self._deactivate_all_with_error(f"Synchronized scenario reset failed: {exc}")

    def restart_synchronized(self) -> None:
        """Advance the scenario and reset every active policy's recurrent state."""
        slots = self.active_slots
        if not slots:
            self._observations = None
            return
        self.scenario_adapter.reset_synchronized(self.env, slots)
        self._observations = self.env.get_observations()
        self._reset_active_policy_state()

    def _reset_active_policy_state(self) -> None:
        """Clear checkpoint-owned recurrent state at a shared episode boundary."""
        for active in self.active.values():
            active.policy.reset()

    @property
    def active_slots(self) -> list[int]:
        return sorted(active.slot for active in self.active.values())

    @property
    def reference_policy(self) -> ActivePolicy | None:
        """Return the newest active checkpoint used as the opaque overlay reference."""
        return max(self.active.values(), key=lambda active: active.entry.rank, default=None)

    def ghost_tint(self, active: ActivePolicy) -> tuple[float, float, float] | None:
        """Return the displayed tint only when a policy is a translucent ghost."""
        if not self.overlay or active is self.reference_policy:
            return None
        return active.tint

    def step(self) -> None:
        """Apply requests, then advance physics once or only pump the viewer."""
        self._maybe_scan()
        self.apply_frame_boundary_requests()
        if not self.active:
            self.env.unwrapped.sim.render()
            return
        if self._observations is None:
            self.restart_synchronized()
        import torch

        device = self.env.unwrapped.device
        actions = torch.zeros((self.cfg.max_policies, self.action_dim), device=device)
        with torch.inference_mode():
            for active in self.active.values():
                row = slice_observation(self._observations, active.slot)
                action = active.policy(row)
                actions[active.slot] = action.reshape(-1, self.action_dim)[0]
        slots = self.active_slots
        self.scenario_adapter.before_step(self.env, slots)
        try:
            self._observations, _, dones, _ = self.env.step(actions.detach())
        finally:
            self.scenario_adapter.after_step(self.env, slots)
        failure_hook = getattr(self.scenario_adapter, "rollout_failures", None)
        failures = failure_hook(self.env, self.active_slots) if failure_hook is not None else {}
        if failures:
            self._deactivate_slots_with_error(failures)
            try:
                self.restart_synchronized()
            except Exception as exc:
                self._deactivate_all_with_error(f"Synchronized scenario reset failed: {exc}")
            return
        done_tensor = dones if isinstance(dones, torch.Tensor) else torch.as_tensor(dones, device=device)
        if any(bool(done_tensor[slot]) for slot in self.active_slots):
            try:
                if self.scenario_adapter.accept_automatic_reset(self.env, self.active_slots):
                    self._reset_active_policy_state()
                else:
                    self.restart_synchronized()
            except Exception as exc:
                self._deactivate_all_with_error(f"Synchronized scenario reset failed: {exc}")

    def _maybe_scan(self) -> None:
        now = time.monotonic()
        if now - self._last_scan >= self.cfg.scan_interval:
            self.catalog.scan()
            self._last_scan = now

    def _deactivate_all_with_error(self, message: str) -> None:
        """Keep the viewer alive when a task cannot complete a synchronized reset."""
        for active in self.active.values():
            active.entry.error = message
            active.entry.status = "error"
            self.slots.release(active.slot)
            print(f"[Policy Debug] Could not activate {active.entry.path.name}: {message}", file=sys.stderr)
        self.active.clear()
        self._observations = None
        self._sync_visualizer()

    def _deactivate_slots_with_error(self, failures: dict[int, str]) -> None:
        """Contain a numerical failure to the policies that own the affected slots."""
        for path, active in list(self.active.items()):
            reason = failures.get(active.slot)
            if reason is None:
                continue
            message = f"Numerically unstable rollout: {reason}"
            active.entry.error = message
            active.entry.status = "error"
            self.slots.release(active.slot)
            del self.active[path]
            print(f"[Policy Debug] Deactivated {active.entry.path.name}: {message}", file=sys.stderr)
        self._observations = None
        self._sync_visualizer(frame=not self.overlay)

    def _sync_visualizer(self, frame: bool = False) -> None:
        from newton.viewer import LayerRenderStyle

        visible = self.active_slots
        styles = [LayerRenderStyle() for _ in range(self.cfg.max_policies)]
        shape_visibility = [self._comparison_shape_visibility for _ in range(self.cfg.max_policies)]
        for active in self.active.values():
            tint = self.ghost_tint(active)
            if tint is None:
                styles[active.slot] = LayerRenderStyle()
            else:
                styles[active.slot] = LayerRenderStyle(
                    color=tint,
                    opacity=self.ghost_opacity,
                )
                shape_visibility[active.slot] = _intersect_visibility_masks(
                    self._comparison_shape_visibility,
                    self._overlay_shape_visibility,
                )
        self.visualizer.set_visible_environment_ids(visible)
        self.visualizer.set_environment_layout("overlay" if self.overlay else "grid")
        self.visualizer.set_environment_render_styles(styles)
        self.visualizer.set_environment_shape_visibility(shape_visibility)
        if frame:
            self.visualizer.frame_visible_environments()


def _intersect_visibility_masks(
    first: tuple[bool, ...] | None,
    second: tuple[bool, ...] | None,
) -> tuple[bool, ...] | None:
    """Intersect optional model-shape masks while preserving ``None`` as all-visible."""
    if first is None:
        return second
    if second is None:
        return first
    if len(first) != len(second):
        raise ValueError(f"Shape visibility masks differ in length: {len(first)} and {len(second)}")
    return tuple(left and right for left, right in zip(first, second, strict=True))


def slice_observation(observation: Any, row: int) -> Any:
    """Recursively select one environment row while preserving batch rank."""
    if isinstance(observation, dict):
        return {key: slice_observation(value, row) for key, value in observation.items()}
    if isinstance(observation, tuple):
        return tuple(slice_observation(value, row) for value in observation)
    if isinstance(observation, list):
        return [slice_observation(value, row) for value in observation]
    return observation[row : row + 1]
