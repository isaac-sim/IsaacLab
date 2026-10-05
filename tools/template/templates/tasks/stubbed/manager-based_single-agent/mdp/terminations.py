# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Implement the task's termination term."""
    raise NotImplementedError("Implement the termination term.")


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Implement the task's episode timeout term."""
    raise NotImplementedError("Implement the timeout term.")
