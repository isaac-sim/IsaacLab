# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Humanoid Mimic tasks keep Isaac Sim PhysX; their environments skip kitless preparation."""

import pytest


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
        "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-Mimic-v0",
        "Isaac-Locomanipulation-G1-Abs-Mimic-v0",
    ],
)
def test_humanoid_mimic_tasks_offer_only_isaac_sim_physx(task):
    from isaaclab_physx.physics import PhysxCfg

    import isaaclab_mimic.envs  # noqa: F401 -- registers the Mimic task IDs

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task, device="cpu", num_envs=1)
    assert type(cfg.sim.physics) is PhysxCfg
    for selector in ("physics=ovphysx", "physics=newton_mjwarp"):
        with pytest.raises(ValueError, match="did not match a selectable preset"):
            parse_env_cfg(task, device="cpu", num_envs=1, overrides=[selector])
