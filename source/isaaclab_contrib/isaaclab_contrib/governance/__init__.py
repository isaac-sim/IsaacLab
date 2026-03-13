# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safety governance wrappers for Isaac Lab environments.

This module provides governance wrappers that add safety verification,
provenance logging, and permission-before-power checks to any Isaac Lab
environment. Designed for sim-to-real pipelines where deployed robot
policies must be auditable and safety-constrained.

Key components:

- :class:`GovernedEnvWrapper`: Wraps any Isaac Lab env to add pre-actuation
  safety checks, action clamping, and hash-chained audit logging.
- :class:`GovernancePolicy`: Defines spatial bounds, rate limits, and
  per-agent permission rules.
- :class:`ProvenanceLogger`: Append-only, hash-chained record of every
  governance decision (ALLOW / CLAMP / DENY).

Example usage::

    from isaaclab_contrib.governance import GovernedEnvWrapper, GovernancePolicy

    policy = GovernancePolicy(
        action_bounds=(-1.0, 1.0),
        max_rate_of_change=0.5,
        spatial_bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (0.0, 3.0)},
    )
    env = GovernedEnvWrapper(base_env, governance_policy=policy)
    obs = env.reset()
    for _ in range(1000):
        action = agent.predict(obs)       # RL policy proposes action
        obs, reward, done, info = env.step(action)  # governance verifies first
        # info["governance"] contains per-step ALLOW/CLAMP/DENY decisions
"""

from .governance_policy import GovernancePolicy
from .governed_env_wrapper import GovernedEnvWrapper
from .provenance_logger import ProvenanceLogger

__all__ = ["GovernancePolicy", "GovernedEnvWrapper", "ProvenanceLogger"]
