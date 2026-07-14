# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for kitless construction of shared environment configurations."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.test.env_cfgs import (
    make_empty_direct_marl_env_cfg,
    make_empty_manager_based_env_cfg,
    make_empty_manager_based_rl_env_cfg,
)

pytestmark = pytest.mark.unit

_EnvCfg = ManagerBasedEnvCfg | ManagerBasedRLEnvCfg | DirectMARLEnvCfg
_ENV_CFG_FACTORIES: tuple[Callable[..., _EnvCfg], ...] = (
    make_empty_manager_based_env_cfg,
    make_empty_manager_based_rl_env_cfg,
    make_empty_direct_marl_env_cfg,
)


@pytest.mark.parametrize("factory", _ENV_CFG_FACTORIES, ids=lambda factory: factory.__name__)
def test_factory_constructs_and_validates_without_app_launcher(factory: Callable[..., _EnvCfg]):
    """Every shared factory constructs a valid CPU configuration without starting the simulator."""
    cfg = factory(device="cpu")

    validate = getattr(cfg, "validate")
    validate()

    assert cfg.sim.device == "cpu"
