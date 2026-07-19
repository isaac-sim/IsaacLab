# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based environment curriculum terms."""

from types import SimpleNamespace

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg


def test_modify_term_cfg_invalidates_warp_caches_on_change():
    """A changed term parameter should invalidate cached Warp work exactly once."""

    class DummyEnv:
        def __init__(self):
            term_cfg = SimpleNamespace(params={"scale": 1.0})
            self.reward_manager = SimpleNamespace(cfg=SimpleNamespace(term=term_cfg))
            self.invalidations = 0

        def invalidate_wp_graphs(self):
            self.invalidations += 1

    env = DummyEnv()
    cfg = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.term.params.scale",
            "modify_fn": lambda env, env_ids, data: 2.0,
        },
    )
    term = mdp.modify_term_cfg(cfg, env)

    term(env, [], **cfg.params)
    assert env.reward_manager.cfg.term.params["scale"] == 2.0
    assert env.invalidations == 1

    cfg.params["modify_fn"] = lambda env, env_ids, data: mdp.modify_term_cfg.NO_CHANGE
    term(env, [], **cfg.params)
    assert env.invalidations == 1
