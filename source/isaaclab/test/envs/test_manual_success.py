# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for manual success with dependent managers."""

from types import SimpleNamespace

import pytest

from isaaclab.envs.mdp.rewards import is_terminated_term
from isaaclab.envs.utils._manual_success import _prepare_success_term
from isaaclab.managers import TerminationManager

from isaaclab_tasks.utils import parse_env_cfg

pytestmark = pytest.mark.unit


def test_manual_success_preserves_manager_dependencies():
    """Replay disables resets while retaining success rewards and reset dependencies."""
    cfg = parse_env_cfg("Isaac-Reach-UR10", device="cpu", num_envs=1)
    original_success = cfg.terminations.success
    term_names = {name for name, term in vars(cfg.terminations).items() if term is not None}
    rewards, curriculum, events = cfg.rewards, cfg.curriculum, cfg.events
    success = _prepare_success_term(cfg, disable_terminations=True)

    env = SimpleNamespace(num_envs=1, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))
    env.termination_manager = TerminationManager(cfg.terminations, env)
    success_reward = is_terminated_term(cfg.rewards.success, env)
    assert not env.termination_manager.compute().any()
    assert not success_reward(env).any()
    assert set(env.termination_manager.active_terms) == term_names
    assert success is original_success
    assert cfg.rewards is rewards and cfg.curriculum is curriculum and cfg.events is events
