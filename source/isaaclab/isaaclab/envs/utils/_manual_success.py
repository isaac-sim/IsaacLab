# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare manager-based environments for manually evaluated success."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import replace

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import TerminationTermCfg


def _never_terminate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a false termination signal for every environment."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _prepare_success_term(
    env_cfg: ManagerBasedRLEnvCfg, *, disable_terminations: bool = False
) -> TerminationTermCfg | None:
    """Extract success and keep inert termination names for dependent managers.

    Rewards, curricula, and reset events retain their configurations. Replay of
    recorded actions can disable every automatic termination without removing
    the names those managers reference; other callers disable only success.
    """
    if env_cfg.terminations is None:
        return None
    term_cfgs = env_cfg.terminations if isinstance(env_cfg.terminations, dict) else vars(env_cfg.terminations)
    success_term = term_cfgs.get("success")
    for name, term_cfg in term_cfgs.items():
        if term_cfg is not None and (disable_terminations or name == "success"):
            term_cfgs[name] = replace(term_cfg, func=_never_terminate, params={})
    return success_term
