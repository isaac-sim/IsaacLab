# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp observation manager."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import ObservationManager
from isaaclab_experimental.utils import WarpGraphCache
from isaaclab_experimental.utils.noise import ConstantNoiseCfg

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


def constant_obs(env, out: wp.array, out_dim: int):
    """Warp-first observation term; the manager reads its width from ``out_dim``."""
    out.fill_(1.0)


@configclass
class PolicyCfg(ObservationGroupCfg):
    """Group with two terms and every group-level setting at its default."""

    first = ObservationTermCfg(func=constant_obs, params={"out_dim": 2})
    second = ObservationTermCfg(func=constant_obs, params={"out_dim": 3})


def test_group_settings_are_not_parsed_as_terms():
    """Every :class:`ObservationGroupCfg` field, including ``history_order``, is a group setting, not a term."""
    env = SimpleNamespace(num_envs=4, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))

    manager = ObservationManager({"policy": PolicyCfg()}, env)

    assert manager.active_terms["policy"] == ["first", "second"]
    assert manager.group_obs_dim["policy"] == (5,)


def counted_constant_obs(env, out: wp.array, out_dim: int):
    """``constant_obs`` counting the runs of its Python; a recorded stage runs it once per recording."""
    env.recordings += 1
    out.fill_(1.0)


@configclass
class NoisyPolicyCfg(ObservationGroupCfg):
    """Group with one term corrupted by a constant bias."""

    value = ObservationTermCfg(func=counted_constant_obs, params={"out_dim": 2}, noise=ConstantNoiseCfg(bias=0.0))

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class ObservationsCfg:
    policy: NoisyPolicyCfg = NoisyPolicyCfg()


def _override(env, env_ids, data, value):
    return value


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")
def test_modify_term_cfg_applies_noise_to_the_recorded_observation_stage():
    """An observation noise curriculum changes the observations of a recorded stage, as noise ADR expects.

    The noise parameters are kernel arguments read while the stage records, so a replay without recording again
    keeps applying the old noise. Writing the same bias again, as a curriculum does on every reset past its
    threshold, records nothing new.
    """
    graph_cache = WarpGraphCache("cuda:0")
    env = SimpleNamespace(
        num_envs=4,
        device="cuda:0",
        sim=SimpleNamespace(is_playing=lambda: True),
        rng_state_wp=wp.zeros(4, dtype=wp.uint32, device="cuda:0"),
        recordings=0,
        _warp_graph_cache=graph_cache,
    )
    env.observation_manager = ObservationManager(ObservationsCfg(), env)
    params = {
        "address": "observations.policy.value.noise.bias",
        "modify_fn": _override,
        "modify_params": {"value": 0.5},
    }
    curriculum = mdp.modify_term_cfg(CurriculumTermCfg(func=mdp.modify_term_cfg, params=params), env)
    graph_cache.arm()
    graph_cache.call_steps(
        "ObservationManager_compute", env.observation_manager.stage_steps("compute"), return_cloned_output=False
    )
    assert graph_cache.captured_stages == ("ObservationManager_compute",)

    for _ in range(3):
        curriculum(env, None, **params)
        obs = graph_cache.call_steps(
            "ObservationManager_compute", env.observation_manager.stage_steps("compute"), return_cloned_output=False
        )
    wp.synchronize()

    assert torch.equal(obs["policy"], torch.full((4, 2), 1.5, device="cuda:0"))
    assert env.recordings == 2
    graph_cache.close()
