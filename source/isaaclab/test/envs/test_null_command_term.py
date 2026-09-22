# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest

from isaaclab.envs.mdp import NullCommandCfg

pytestmark = pytest.mark.unit


def test_null_command_term_has_no_command():
    """The null command term resets and computes without side effects and exposes no command."""
    env = SimpleNamespace(num_envs=20, dt=0.1, device="cpu")
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)

    assert "NullCommand" in str(command_term)
    command_term.reset()
    command_term.compute(dt=env.dt)
    with pytest.raises(RuntimeError):
        command_term.command
