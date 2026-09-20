# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def clone_obs_buffer(
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Clone a nested observation buffer, using :meth:`torch.Tensor.clone` for every leaf tensor.

    This avoids the overhead of :func:`copy.deepcopy` while still producing an independent
    snapshot of the buffer (new dict objects + cloned tensor storage).

    Args:
        obs_buffer: Observation buffer mapping group names to either a single concatenated
            tensor or a dict of per-term tensors.

    Returns:
        A new dictionary with the same structure whose tensors are clones of the originals.
    """
    result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
    for key, value in obs_buffer.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        else:  # dict[str, torch.Tensor]
            result[key] = {k: v.clone() for k, v in value.items()}
    return result
