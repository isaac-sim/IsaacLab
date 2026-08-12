# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for manager-based multi-agent reinforcement-learning environments."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .manager_based_env_cfg import ManagerBasedEnvCfg


@configclass
class ManagerBasedMARLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for a fixed-agent manager-based reinforcement-learning environment."""

    @configclass
    class AgentCfg:
        """Manager configurations belonging to one agent."""

        actions: object = MISSING
        observations: object = MISSING
        rewards: object = MISSING
        terminations: object = MISSING

    # The base class fields describe singular managers. They are deliberately disabled here:
    # each agent owns its action and observation managers instead.
    actions: object = None
    observations: object = None

    ui_window_class_type: type | str | None = None

    is_finite_horizon: bool = False
    """Whether the task is finite horizon. This is consumed by MARL wrappers."""

    compute_final_obs: bool = False
    """Whether to expose each agent's terminal observation under ``extras[agent]["final_obs"]``."""

    episode_length_s: float = MISSING
    """Duration of an episode [s]."""

    agents: dict[str, AgentCfg] = MISSING
    """Fixed agents in insertion order, keyed by their identifiers."""

    state: object | None = None
    """Optional centralized-state observation manager configuration."""

    curriculum: object | None = None
    """Shared curriculum settings. Defaults to None."""

    commands: object | None = None
    """Shared command settings. Defaults to None."""

    def validate_config(self) -> None:
        """Validate MARL-specific configuration constraints."""
        if self.export_io_descriptors:
            raise ValueError(
                "ManagerBasedMARLEnv does not support export_io_descriptors. "
                "Configure agent-specific descriptors in the task or disable export_io_descriptors."
            )
        if not self.agents:
            raise ValueError("ManagerBasedMARLEnvCfg.agents must contain at least one agent.")
        if "log" in self.agents:
            raise ValueError("Agent identifier 'log' is reserved for shared reset statistics.")
        for agent_id, agent_cfg in self.agents.items():
            if not isinstance(agent_id, str):
                raise ValueError(f"Agent identifier must be a string, got {agent_id!r}.")
            missing_sections = [
                section
                for section in ("actions", "observations", "rewards", "terminations")
                if getattr(agent_cfg, section, MISSING) is MISSING or getattr(agent_cfg, section, None) is None
            ]
            if missing_sections:
                raise ValueError(
                    f"Agent '{agent_id}' must configure actions, observations, rewards, and terminations; "
                    f"missing: {', '.join(missing_sections)}."
                )
        if self.state is not None and not isinstance(self.state, dict) and not hasattr(self.state, "__dict__"):
            raise ValueError("ManagerBasedMARLEnvCfg.state must be an observation manager configuration or None.")
