# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the shared visual-color target-selection helper.

Backend-agnostic: this is the per-environment selection + shape validation that every visual-color
writer (Newton / OVRTX / Replicator) shares, so it is tested once here rather than per backend.
No Isaac Sim / Kit / GPU required.
"""

import pytest
import torch

from isaaclab.utils.visual_color import select_visual_color_targets

# 3 envs, 2 prims/env -> 6 targets, in writer order (env0-cart, env0-pole, env1-cart, ...)
_ENV_OF_TARGET = [0, 0, 1, 1, 2, 2]
_NUM_TARGETS = 6


def _colors(n: int = _NUM_TARGETS) -> torch.Tensor:
    return torch.zeros((n, 3), dtype=torch.float32)


def test_subset_selects_only_that_envs_targets():
    targets = select_visual_color_targets(torch.tensor([1]), _colors(), _ENV_OF_TARGET, _NUM_TARGETS)
    assert targets == [2, 3]  # env 1's two prims


def test_multiple_envs_select_in_ascending_target_order():
    targets = select_visual_color_targets(torch.tensor([2, 0]), _colors(), _ENV_OF_TARGET, _NUM_TARGETS)
    assert targets == [0, 1, 4, 5]  # env 0 and env 2, ascending by target index


def test_all_envs_select_every_target():
    targets = select_visual_color_targets(torch.arange(3), _colors(), _ENV_OF_TARGET, _NUM_TARGETS)
    assert targets == [0, 1, 2, 3, 4, 5]


def test_empty_env_ids_is_noop():
    # empty subset returns no targets -- and does NOT validate colors (so a no-op never raises)
    empty_env_ids = torch.empty(0, dtype=torch.long)
    assert select_visual_color_targets(empty_env_ids, _colors(99), _ENV_OF_TARGET, _NUM_TARGETS) == []


def test_unresolved_target_env_is_never_selected():
    env_of_target = [0, None, 1, 1, 2, 2]  # target 1's env could not be parsed
    targets = select_visual_color_targets(torch.arange(3), _colors(), env_of_target, _NUM_TARGETS)
    assert 1 not in targets
    assert targets == [0, 2, 3, 4, 5]


@pytest.mark.parametrize("bad", [(_NUM_TARGETS + 1, 3), (_NUM_TARGETS, 4), (_NUM_TARGETS,)])
def test_wrong_colors_shape_raises(bad):
    with pytest.raises(ValueError):
        select_visual_color_targets(torch.tensor([0]), torch.zeros(bad), _ENV_OF_TARGET, _NUM_TARGETS)
