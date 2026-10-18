# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms specific to the granular-bed locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..g1_mpm_env import G1MPMEnv


def reset_sand_bed(env: G1MPMEnv, env_ids: Sequence[int] | torch.Tensor) -> None:
    """Restore the flat granular bed for the selected environments.

    Args:
        env: Environment instance.
        env_ids: Indices of the environments to reset.
    """
    env.reset_sand_bed(env_ids)
