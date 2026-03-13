# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that enforces governance policy on agent actions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from .governance_policy import GovernancePolicy
from .provenance_logger import ProvenanceLogger

logger = logging.getLogger(__name__)

# Default max elements logged per action vector. Configurable via
# GovernedEnvWrapper(max_log_dims=N).
_DEFAULT_MAX_LOG_DIMS = 64


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test.

    Args:
        x: X coordinate of the test point.
        y: Y coordinate of the test point.
        polygon: List of (x, y) vertices defining the polygon boundary.

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class GovernedEnvWrapper:
    """Wrapper that adds governance checks to any Isaac Lab environment.

    This wrapper intercepts the ``step()`` call and applies governance
    policy checks before forwarding actions to the underlying environment.
    Every action is classified as:

    - **ALLOW**: Action passes all governance checks unchanged.
    - **CLAMP**: Action violates soft constraints and is modified to
      satisfy them (e.g., clipped to bounds, rate-limited).
    - **DENY**: Action violates hard constraints and is replaced with a
      safe default (zeros). Only when ``fail_closed=True``.

    All decisions are logged to a hash-chained provenance ledger.

    This wrapper works with both single-agent (``DirectRLEnv``) and
    multi-agent (``DirectMARLEnv``) environments.

    Example::

        from isaaclab_contrib.governance import GovernedEnvWrapper, GovernancePolicy

        policy = GovernancePolicy(
            action_bounds=(-1.0, 1.0),
            max_rate_of_change=0.5,
        )
        governed_env = GovernedEnvWrapper(env, governance_policy=policy)
        obs, info = governed_env.reset()
        action = policy_network(obs)
        obs, reward, terminated, truncated, info = governed_env.step(action)
        print(info["governance"])  # {"decision": "ALLOW", "violations": []}

    Attributes:
        env: The wrapped Isaac Lab environment.
        policy: The governance policy being enforced.
        provenance: The provenance logger recording all decisions.
        step_count: Number of steps taken since last reset.
    """

    def __init__(
        self,
        env: Any,
        governance_policy: GovernancePolicy,
        log_path: str | None = None,
        position_extractor: Any | None = None,
        max_log_dims: int = _DEFAULT_MAX_LOG_DIMS,
    ) -> None:
        """Initialize the governance wrapper.

        Args:
            env: The Isaac Lab environment to wrap.
            governance_policy: Safety policy to enforce.
            log_path: Optional path for persistent provenance logging.
            position_extractor: Optional callable ``(env, agent_id) -> dict``
                that returns ``{"x": float, "y": float, "z": float}`` for
                the agent's current position. Required for spatial_bounds
                and geofence enforcement. If ``None``, spatial checks that
                need position data are skipped with a warning on first use.
            max_log_dims: Maximum action dimensions to include in provenance
                records. Set to ``0`` to log all dimensions. Default 64.
        """
        self.env = env
        self.policy = governance_policy
        self.provenance = ProvenanceLogger(log_path=log_path)
        self.step_count = 0
        self._prev_actions: dict[str, np.ndarray] = {}
        self._position_extractor = position_extractor
        self._max_log_dims = max_log_dims
        self._warned_no_extractor = False
        # Track latest governed actions per agent for multi-agent separation
        self._latest_actions: dict[str, np.ndarray] = {}

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)

    def reset(self, **kwargs: Any) -> Any:
        """Reset the environment and governance state.

        Args:
            **kwargs: Keyword arguments forwarded to the wrapped env.

        Returns:
            The reset observation from the wrapped environment.
        """
        result = self.env.reset(**kwargs)
        self.step_count = 0
        self._prev_actions.clear()
        self._latest_actions.clear()
        return result

    def step(self, actions: Any) -> Any:
        """Apply governance checks then step the environment.

        Args:
            actions: Proposed actions from the agent policy. Can be a
                tensor (single-agent), numpy array, or dict of tensors
                (multi-agent MARL).

        Returns:
            The step result from the wrapped environment, with governance
            metadata added to the ``info`` dict.
        """
        if isinstance(actions, dict):
            # Multi-agent: govern each agent independently
            governed_actions = {}
            gov_info: dict[str, Any] = {}
            all_agent_ids = list(actions.keys())
            for agent_id, agent_actions in actions.items():
                governed, decision, violations = self._govern_actions(
                    agent_actions, agent_id=str(agent_id),
                    all_agent_ids=all_agent_ids,
                )
                governed_actions[agent_id] = governed
                gov_info[str(agent_id)] = {
                    "decision": decision,
                    "violations": violations,
                }
            result = self.env.step(governed_actions)
            # Inject governance info — preserve existing info dict
            if isinstance(result, tuple) and len(result) >= 4:
                last = result[-1]
                info = last if isinstance(last, dict) else {"_original": last}
                info["governance"] = gov_info
                result = (*result[:-1], info)
        else:
            # Single-agent
            governed, decision, violations = self._govern_actions(
                actions, agent_id="0",
            )
            result = self.env.step(governed)
            if isinstance(result, tuple) and len(result) >= 4:
                last = result[-1]
                info = last if isinstance(last, dict) else {"_original": last}
                info["governance"] = {
                    "decision": decision,
                    "violations": violations,
                }
                result = (*result[:-1], info)

        self.step_count += 1
        return result

    def _get_agent_position(self, agent_id: str) -> dict[str, float] | None:
        """Get agent position via the position extractor.

        Returns:
            Position dict with x/y/z keys, or None if unavailable.
        """
        if self._position_extractor is None:
            if not self._warned_no_extractor:
                has_spatial = (
                    self.policy.spatial_bounds is not None
                    or self.policy.geofence_polygons is not None
                )
                if has_spatial:
                    logger.warning(
                        "Spatial governance checks configured but no "
                        "position_extractor provided — spatial_bounds and "
                        "geofence checks will be skipped. Pass a "
                        "position_extractor to GovernedEnvWrapper to enable."
                    )
                self._warned_no_extractor = True
            return None
        try:
            return self._position_extractor(self.env, agent_id)
        except Exception:
            logger.debug("position_extractor failed for agent %s", agent_id)
            return None

    def _govern_actions(
        self,
        actions: Any,
        agent_id: str = "0",
        all_agent_ids: list[str] | None = None,
    ) -> tuple[Any, str, list[str]]:
        """Apply governance policy to proposed actions.

        Enforces all configured checks in order:
        1. Action bounds (soft — CLAMP)
        2. Rate-of-change limits (soft — CLAMP)
        3. Spatial bounds (hard — DENY if fail_closed)
        4. Geofence containment (hard — DENY if fail_closed)
        5. Minimum agent separation (hard — DENY if fail_closed)

        Args:
            actions: Proposed action tensor or array.
            agent_id: Identifier for provenance logging.
            all_agent_ids: All agent IDs in multi-agent step (for separation check).

        Returns:
            Tuple of (governed_actions, decision, violations).
        """
        violations: list[str] = []
        is_torch = torch is not None and isinstance(actions, torch.Tensor)
        device = actions.device if is_torch else None

        # Convert to numpy for governance checks
        if is_torch:
            actions_np = actions.detach().cpu().numpy().copy()
        else:
            actions_np = np.array(actions, copy=True)

        proposed = actions_np.flatten().tolist()

        # --- Check 1: Action bounds (soft) ---
        if self.policy.action_bounds is not None:
            lo, hi = self.policy.action_bounds
            if np.any(actions_np < lo) or np.any(actions_np > hi):
                violations.append("action_bounds")
                actions_np = np.clip(actions_np, lo, hi)

        # --- Check 2: Rate-of-change (soft) ---
        prev = self._prev_actions.get(agent_id)
        if self.policy.max_rate_of_change is not None and prev is not None:
            delta = actions_np - prev
            max_delta = self.policy.max_rate_of_change
            if np.any(np.abs(delta) > max_delta):
                violations.append("rate_of_change")
                actions_np = prev + np.clip(delta, -max_delta, max_delta)

        # Store for next rate-of-change check (per agent)
        self._prev_actions[agent_id] = actions_np.copy()

        # --- Check 3: Spatial bounds (hard) ---
        if self.policy.spatial_bounds is not None:
            pos = self._get_agent_position(agent_id)
            if pos is not None:
                for axis, (lo, hi) in self.policy.spatial_bounds.items():
                    val = pos.get(axis)
                    if val is not None and (val < lo or val > hi):
                        violations.append("spatial_bounds")
                        break

        # --- Check 4: Geofence containment (hard) ---
        if self.policy.geofence_polygons is not None:
            pos = self._get_agent_position(agent_id)
            if pos is not None:
                px, py = pos.get("x", 0.0), pos.get("y", 0.0)
                inside_any = any(
                    _point_in_polygon(px, py, poly)
                    for poly in self.policy.geofence_polygons
                )
                if not inside_any:
                    violations.append("geofence")

        # --- Check 5: Minimum agent separation (hard) ---
        # Note: Both agents in a too-close pair receive the violation.
        # This is intentional — in safety-critical scenarios, both agents
        # should be denied actuation when separation is violated, not just
        # the one that happened to be checked second.
        if (
            self.policy.min_separation is not None
            and all_agent_ids is not None
            and len(all_agent_ids) > 1
        ):
            pos = self._get_agent_position(agent_id)
            if pos is not None:
                for other_id in all_agent_ids:
                    if other_id == agent_id:
                        continue
                    other_pos = self._get_agent_position(str(other_id))
                    if other_pos is None:
                        continue
                    dist = (
                        (pos.get("x", 0.0) - other_pos.get("x", 0.0)) ** 2
                        + (pos.get("y", 0.0) - other_pos.get("y", 0.0)) ** 2
                        + (pos.get("z", 0.0) - other_pos.get("z", 0.0)) ** 2
                    ) ** 0.5
                    if dist < self.policy.min_separation:
                        violations.append("min_separation")
                        break

        # --- Determine decision ---
        if not violations:
            decision = "ALLOW"
        elif self.policy.fail_closed and self._is_hard_violation(violations):
            decision = "DENY"
            actions_np = np.zeros_like(actions_np)
        else:
            decision = "CLAMP"

        applied = actions_np.flatten().tolist()

        # Truncate action vectors for provenance (configurable)
        log_proposed = proposed[:self._max_log_dims] if self._max_log_dims else proposed
        log_applied = applied[:self._max_log_dims] if self._max_log_dims else applied

        # Log provenance
        self.provenance.log(
            step=self.step_count,
            agent_id=agent_id,
            decision=decision,
            proposed_action=log_proposed,
            applied_action=log_applied,
            violations=violations,
            metadata=self.policy.metadata if self.policy.metadata else None,
        )

        if decision != "ALLOW":
            logger.debug(
                "Governance %s for agent %s at step %d: %s",
                decision, agent_id, self.step_count, violations,
            )

        # Store governed actions for separation checks on next step
        self._latest_actions[agent_id] = actions_np.copy()

        # Convert back to original format
        if is_torch:
            result = torch.from_numpy(actions_np).to(device=device)
        else:
            result = actions_np

        return result, decision, violations

    def _is_hard_violation(self, violations: list[str]) -> bool:
        """Determine if violations constitute a hard failure.

        Spatial boundary violations and separation violations are hard
        failures that result in DENY. Action bounds and rate-of-change
        violations are soft failures that result in CLAMP.

        Args:
            violations: List of violated constraint names.

        Returns:
            True if any violation is a hard failure.
        """
        hard_violations = {"spatial_bounds", "geofence", "min_separation"}
        return bool(hard_violations & set(violations))

    def governance_report(self) -> dict[str, Any]:
        """Generate a governance summary report.

        Returns:
            Dictionary containing decision counts, chain integrity
            status, and policy configuration.
        """
        is_valid, last_epoch = self.provenance.verify_chain()
        return {
            "total_steps": self.step_count,
            "decisions": self.provenance.summary(),
            "chain_intact": is_valid,
            "chain_length": len(self.provenance.records),
            "last_verified_epoch": last_epoch,
            "policy": {
                "action_bounds": self.policy.action_bounds,
                "max_rate_of_change": self.policy.max_rate_of_change,
                "spatial_bounds": self.policy.spatial_bounds,
                "geofence_polygons": len(self.policy.geofence_polygons) if self.policy.geofence_polygons else 0,
                "min_separation": self.policy.min_separation,
                "fail_closed": self.policy.fail_closed,
            },
        }
