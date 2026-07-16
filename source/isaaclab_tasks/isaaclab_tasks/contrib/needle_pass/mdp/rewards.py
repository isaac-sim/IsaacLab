# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Progress rewards derived from the same measured physical phase state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .terminations import HandoffPhase, HandoffPhaseCfg, update_handoff_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def handoff_phase_progress(env: ManagerBasedRLEnv, phase_cfg: HandoffPhaseCfg) -> torch.Tensor:
    """Return normalised ordered progress; this is not success evidence."""

    phase = update_handoff_phase(env, phase_cfg).phase
    return phase.to(dtype=torch.float32) / float(HandoffPhase.RETAINED_LIFT)


def retained_lift_bonus(env: ManagerBasedRLEnv, phase_cfg: HandoffPhaseCfg) -> torch.Tensor:
    """Return a sparse bonus after the retained-lift dwell has completed."""

    phase = update_handoff_phase(env, phase_cfg).phase
    return (phase == int(HandoffPhase.RETAINED_LIFT)).to(dtype=torch.float32)


__all__ = ["handoff_phase_progress", "retained_lift_bonus"]
