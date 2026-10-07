# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp observation group configuration."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.envs.mdp.observations import last_action
from isaaclab_experimental.managers.observation_manager import ObservationManager

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit


@configclass
class PolicyCfg(ObservationGroupCfg):
    action = ObservationTermCfg(func=last_action)
    scaled_action = ObservationTermCfg(func=last_action, scale=2.0)


@pytest.fixture
def env():
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        action_manager=SimpleNamespace(action=wp.array([[1.0, 2.0]], dtype=wp.float32, device="cpu")),
    )


def test_group_settings_are_not_observation_terms(env):
    manager = ObservationManager({"policy": PolicyCfg()}, env)

    assert manager.active_terms["policy"] == ["action", "scaled_action"]
    torch.testing.assert_close(manager.compute()["policy"], torch.tensor([[1.0, 2.0, 2.0, 4.0]]))


def test_history_remains_unsupported(env):
    cfg = PolicyCfg(history_length=2, history_order="time")

    with pytest.raises(NotImplementedError, match="History reshaping is not implemented"):
        ObservationManager({"policy": cfg}, env)
