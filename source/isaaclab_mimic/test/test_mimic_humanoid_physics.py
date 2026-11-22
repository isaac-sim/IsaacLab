# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Humanoid Mimic tasks prepare their selected physics backend on construction; the SDG task keeps Isaac Sim PhysX."""

from copy import deepcopy

import pytest


class ConstructionReached(Exception):
    pass


def _construct(monkeypatch, task, cfg, inspect=lambda cfg: None):
    """Construct ``task`` through its registered Mimic environment, stopping before simulation startup."""
    import gymnasium as gym

    from isaaclab.envs import ManagerBasedRLEnv

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        inspect(cfg)
        raise ConstructionReached

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    with pytest.raises(ConstructionReached):
        gym.make(task, cfg=deepcopy(cfg))


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
        "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-Mimic-v0",
        "Isaac-Locomanipulation-G1-Abs-Mimic-v0",
    ],
)
@pytest.mark.parametrize("physics", [None, "isaacsim_physx", "ovphysx", "newton_mjwarp"])
def test_humanoid_mimic_construction_prepares_only_standalone_backends(monkeypatch, task, physics):
    """Mimic tasks offer the teleoperation task's physics selectors and prepare a standalone backend."""
    from isaaclab_isaacsim.physics import PhysxCfg
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_ov.physics import OvPhysxCfg

    import isaaclab_mimic.envs  # noqa: F401 -- registers the Mimic task IDs

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    overrides = [] if physics is None else [f"physics={physics}"]
    cfg = parse_env_cfg(task, device="cpu", num_envs=1, overrides=overrides)
    expected_type = {None: PhysxCfg, "isaacsim_physx": PhysxCfg, "ovphysx": OvPhysxCfg, "newton_mjwarp": NewtonCfg}
    assert type(cfg.sim.physics) is expected_type[physics]
    prepared = physics if physics in ("ovphysx", "newton_mjwarp") else None

    def inspect(cfg):
        assert type(cfg.sim.physics) is expected_type[physics]
        assert cfg._prepared_physics_backend == prepared

    _construct(monkeypatch, task, cfg, inspect)


@pytest.mark.parametrize(
    "task", ["Isaac-NutPour-GR1T2-Pink-IK-Abs-Mimic-v0", "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-Mimic-v0"]
)
def test_gr1t2_mimic_environment_constructs_tasks_without_backend_preparation(monkeypatch, task):
    """Tasks that share the GR1T2 Mimic environment but declare no backend preparation construct unchanged."""
    import isaaclab_mimic.envs  # noqa: F401 -- registers the Mimic task IDs

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task, device="cpu", num_envs=1)
    assert not hasattr(cfg, "backend_preparation")
    _construct(monkeypatch, task, cfg)


def test_humanoid_sdg_task_offers_only_isaac_sim_physx():
    """The SDG environment does not run the standalone backend preparation, so it keeps Isaac Sim PhysX."""
    from isaaclab_isaacsim.physics import PhysxCfg

    import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401 -- registers the SDG task ID

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = "Isaac-G1-SteeringWheel-Locomanipulation"
    cfg = parse_env_cfg(task, device="cpu", num_envs=1)
    assert type(cfg.sim.physics) is PhysxCfg
    for selector in ("physics=ovphysx", "physics=newton_mjwarp"):
        with pytest.raises(ValueError, match="did not match a selectable preset"):
            parse_env_cfg(task, device="cpu", num_envs=1, overrides=[selector])
