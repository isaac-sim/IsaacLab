# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner construction shared by RSL-RL entrypoints and benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner

    from ...rsl_rl import RslRlBaseRunnerCfg


def _configure_rsl_rl_env_cfg(env_cfg: Any, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Enable terminal observations requested by the agent before creating the environment."""
    if agent_cfg.compute_final_obs:
        env_cfg.compute_final_obs = True
    if agent_cfg.final_obs_every_step:
        env_cfg.final_obs_every_step = True


def _create_rsl_rl_runner(
    env: Any, agent_cfg: RslRlBaseRunnerCfg, log_dir: str | None = None, *, inference: bool = False
) -> OnPolicyRunner:
    """Construct the configured runner using native RSL-RL callable resolution."""
    from rsl_rl.utils import resolve_callable

    cfg = agent_cfg.to_dict()
    if inference and "use_cuda_graph" in cfg["algorithm"]:
        cfg["algorithm"]["use_cuda_graph"] = False
    return resolve_callable(agent_cfg.class_name)(env, cfg, log_dir=log_dir, device=agent_cfg.device)
