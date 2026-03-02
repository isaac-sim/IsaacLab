# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that enforces governance policy on agent actions."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from .governance_policy import GovernancePolicy
from .provenance_logger import ProvenanceLogger

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize the governance wrapper.

        Args:
            env: The Isaac Lab environment to wrap.
            governance_policy: Safety policy to enforce.
            log_path: Optional path for persistent provenance logging.
        """
        self.env = env
        self.policy = governance_policy
        self.provenance = ProvenanceLogger(log_path=log_path)
        self.step_count = 0
        self._prev_actions: torch.Tensor | np.ndarray | None = None

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
        self._prev_actions = None
        return result

    def step(self, actions: torch.Tensor | np.ndarray | dict) -> Any:
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
            for agent_id, agent_actions in actions.items():
                governed, decision, violations = self._govern_actions(
                    agent_actions, agent_id=str(agent_id)
                )
                governed_actions[agent_id] = governed
                gov_info[str(agent_id)] = {
                    "decision": decision,
                    "violations": violations,
                }
            result = self.env.step(governed_actions)
            # Inject governance info
            if isinstance(result, tuple) and len(result) >= 4:
                info = result[-1] if isinstance(result[-1], dict) else {}
                info["governance"] = gov_info
                result = (*result[:-1], info)
        else:
            # Single-agent
            governed, decision, violations = self._govern_actions(
                actions, agent_id="0"
            )
            result = self.env.step(governed)
            if isinstance(result, tuple) and len(result) >= 4:
                info = result[-1] if isinstance(result[-1], dict) else {}
                info["governance"] = {
                    "decision": decision,
                    "violations": violations,
                }
                result = (*result[:-1], info)

        self.step_count += 1
        return result

    def _govern_actions(
        self,
        actions: torch.Tensor | np.ndarray,
        agent_id: str = "0",
    ) -> tuple[torch.Tensor | np.ndarray, str, list[str]]:
        """Apply governance policy to proposed actions.

        Args:
            actions: Proposed action tensor or array.
            agent_id: Identifier for provenance logging.

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
        original = actions_np.copy()

        # 1. Action bounds check
        if self.policy.action_bounds is not None:
            lo, hi = self.policy.action_bounds
            if np.any(actions_np < lo) or np.any(actions_np > hi):
                violations.append("action_bounds")
                actions_np = np.clip(actions_np, lo, hi)

        # 2. Rate-of-change check
        if self.policy.max_rate_of_change is not None and self._prev_actions is not None:
            if torch is not None and isinstance(self._prev_actions, torch.Tensor):
                prev_np = self._prev_actions.detach().cpu().numpy()
            else:
                prev_np = np.asarray(self._prev_actions)
            delta = actions_np - prev_np
            max_delta = self.policy.max_rate_of_change
            if np.any(np.abs(delta) > max_delta):
                violations.append("rate_of_change")
                actions_np = prev_np + np.clip(delta, -max_delta, max_delta)

        # Store for next rate-of-change check
        self._prev_actions = actions_np.copy()

        # Determine decision
        if not violations:
            decision = "ALLOW"
        elif self.policy.fail_closed and self._is_hard_violation(violations):
            decision = "DENY"
            actions_np = np.zeros_like(actions_np)
        else:
            decision = "CLAMP"

        applied = actions_np.flatten().tolist()

        # Log provenance
        self.provenance.log(
            step=self.step_count,
            agent_id=agent_id,
            decision=decision,
            proposed_action=proposed[:8],  # truncate for readability
            applied_action=applied[:8],
            violations=violations,
            metadata=self.policy.metadata if self.policy.metadata else None,
        )

        if decision != "ALLOW":
            logger.debug(
                "Governance %s for agent %s at step %d: %s",
                decision, agent_id, self.step_count, violations,
            )

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
                "fail_closed": self.policy.fail_closed,
            },
        }
