# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the curriculum manager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from isaaclab.managers import CurriculumManager, CurriculumTermCfg
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit


def advance_level(env, env_ids):
    """Advance a persistent level tensor in place and report it with a derived state."""
    env.level += 1.0
    return {"level": env.level, "half_level": env.level / 2.0, "name": "stage"}


def advance_scalar(env, env_ids):
    """Report the persistent level tensor directly."""
    return env.level


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reset_logs_state_snapshots_without_host_reads(device):
    """Reset logs tensor states as device snapshots, so later in-place term updates do not change them."""
    sim = MagicMock()
    sim.is_playing.return_value = False
    env = SimpleNamespace(num_envs=4, device=device, sim=sim, level=torch.zeros((), device=device))
    cfg = {
        "dict_term": CurriculumTermCfg(func=advance_level),
        "scalar_term": CurriculumTermCfg(func=advance_scalar),
    }
    manager = CurriculumManager(cfg, env)
    manager.compute()

    previous = torch.cuda.get_sync_debug_mode() if device.startswith("cuda") else None
    if previous is not None:
        torch.cuda.synchronize(device)
        torch.cuda.set_sync_debug_mode("error")
    try:
        extras = manager.reset()
    finally:
        if previous is not None:
            torch.cuda.set_sync_debug_mode(previous)
    manager.compute()

    assert extras["Curriculum/dict_term/name"] == "stage"
    for key, expected in (
        ("Curriculum/dict_term/level", 1.0),
        ("Curriculum/dict_term/half_level", 0.5),
        ("Curriculum/scalar_term", 1.0),
    ):
        assert extras[key].device == torch.device(device)
        assert extras[key].item() == expected

    assert manager.get_active_iterable_terms(0) == [("dict_term", [2.0, 1.0, "stage"]), ("scalar_term", [2.0])]
    assert isinstance(manager.get_active_iterable_terms(0)[1][1][0], float)
