# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-level FeatherPGS presets."""

import pytest
from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


@pytest.mark.parametrize(
    (
        "task_name",
        "num_substeps",
        "pgs_iterations",
        "pgs_mode",
        "pgs_beta",
        "pgs_cfm",
        "dt",
        "decimation",
    ),
    [
        pytest.param("Isaac-Reach-Franka", 1, 8, "split", 0.05, 1.0e-6, 1 / 60, 2, id="reach-franka"),
        pytest.param("Isaac-Reach-UR10", 1, 8, "split", 0.05, 1.0e-6, 1 / 60, 2, id="reach-ur10"),
        pytest.param("Isaac-Cartpole", 1, 4, "split", 0.05, 1.0e-6, 1 / 120, 2, id="cartpole-manager"),
        pytest.param("Isaac-Cartpole-Direct", 1, 4, "split", 0.05, 1.0e-6, 1 / 120, 2, id="cartpole-direct"),
        pytest.param("Isaac-Ant", 1, 8, "split", 0.05, 1.0e-6, 1 / 120, 2, id="ant-manager"),
        pytest.param("Isaac-Ant-Direct", 1, 8, "split", 0.05, 1.0e-6, 1 / 120, 2, id="ant-direct"),
        pytest.param(
            "Isaac-Reorient-Cube-Allegro-Direct",
            12,
            16,
            "matrix_free",
            0.02,
            1.0e-5,
            1 / 120,
            2,
            id="allegro-direct",
        ),
        pytest.param(
            "Isaac-Reorient-Cube-Shadow-Direct",
            12,
            24,
            "matrix_free",
            0.005,
            5.0e-5,
            1 / 120,
            2,
            id="shadow-direct",
        ),
        pytest.param(
            "IsaacContrib-Velocity-Flat-AnymalB",
            1,
            16,
            "split",
            0.02,
            1.0e-5,
            0.005,
            4,
            id="anymal-b-flat",
        ),
        pytest.param(
            "IsaacContrib-Velocity-Rough-AnymalB",
            1,
            16,
            "split",
            0.02,
            1.0e-5,
            0.005,
            4,
            id="anymal-b-rough",
        ),
        pytest.param(
            "IsaacContrib-Velocity-Flat-AnymalC",
            2,
            8,
            "split",
            0.05,
            1.0e-6,
            0.005,
            4,
            id="anymal-c-flat",
        ),
        pytest.param(
            "IsaacContrib-Velocity-Rough-AnymalC",
            2,
            4,
            "split",
            0.05,
            1.0e-6,
            0.005,
            4,
            id="anymal-c-rough",
        ),
        pytest.param("Isaac-Velocity-Flat-AnymalD", 1, 4, "split", 0.05, 1.0e-6, 0.005, 4, id="anymal-d-flat"),
        pytest.param(
            "Isaac-Velocity-Rough-AnymalD",
            2,
            2,
            "split",
            0.05,
            1.0e-6,
            0.005,
            4,
            id="anymal-d-rough",
        ),
        pytest.param("IsaacContrib-Velocity-Flat-UnitreeA1", 1, 8, "split", 0.05, 1.0e-6, 0.005, 4, id="a1-flat"),
        pytest.param(
            "IsaacContrib-Velocity-Rough-UnitreeA1",
            1,
            16,
            "split",
            0.02,
            1.0e-5,
            0.005,
            4,
            id="a1-rough",
        ),
        pytest.param("IsaacContrib-Velocity-Flat-UnitreeGo1", 1, 8, "split", 0.05, 1.0e-6, 0.005, 4, id="go1-flat"),
        pytest.param(
            "IsaacContrib-Velocity-Rough-UnitreeGo1",
            1,
            16,
            "split",
            0.02,
            1.0e-5,
            0.005,
            4,
            id="go1-rough",
        ),
        pytest.param("Isaac-Velocity-Flat-UnitreeGo2", 1, 8, "split", 0.05, 1.0e-6, 0.005, 4, id="go2-flat"),
        pytest.param(
            "Isaac-Velocity-Rough-UnitreeGo2",
            1,
            16,
            "split",
            0.01,
            2.0e-5,
            0.005,
            4,
            id="go2-rough",
        ),
        pytest.param("Isaac-Velocity-Flat-Cassie", 1, 16, "split", 0.02, 1.0e-5, 0.005, 4, id="cassie-flat"),
        pytest.param("Isaac-Velocity-Rough-Cassie", 1, 16, "split", 0.02, 1.0e-5, 0.005, 4, id="cassie-rough"),
        pytest.param("Isaac-Velocity-Flat-G1", 1, 4, "matrix_free", 0.05, 1.0e-6, 0.01, 2, id="g1-flat"),
        pytest.param("Isaac-Velocity-Rough-G1", 1, 16, "split", 0.02, 1.0e-5, 0.005, 4, id="g1-rough"),
        pytest.param("Isaac-Velocity-Flat-H1", 2, 8, "split", 0.05, 1.0e-6, 0.005, 4, id="h1-flat"),
        pytest.param("Isaac-Velocity-Rough-H1", 1, 8, "split", 0.05, 1.0e-6, 0.005, 4, id="h1-rough"),
        pytest.param("Isaac-Humanoid", 1, 24, "split", 0.002, 1.0e-4, 1 / 120, 2, id="humanoid-manager"),
        pytest.param("Isaac-Humanoid-Direct", 1, 24, "split", 0.002, 1.0e-4, 1 / 120, 2, id="humanoid-direct"),
        pytest.param("Isaac-Open-Drawer-Franka", 1, 32, "split", 0.02, 1.0e-5, 1 / 60, 1, id="cabinet-franka"),
    ],
)
def test_feather_pgs_task_preset_matches_validated_configuration(
    task_name: str,
    num_substeps: int,
    pgs_iterations: int,
    pgs_mode: str,
    pgs_beta: float,
    pgs_cfm: float,
    dt: float,
    decimation: int,
) -> None:
    raw_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    cfg = resolve_presets(raw_cfg, {"feather_pgs"})

    assert collect_presets(cfg) == {}
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, FeatherPGSSolverCfg)
    assert cfg.sim.physics.num_substeps == num_substeps
    assert cfg.sim.physics.solver_cfg.pgs_iterations == pgs_iterations
    assert cfg.sim.physics.solver_cfg.pgs_mode == pgs_mode
    assert cfg.sim.physics.solver_cfg.pgs_beta == pytest.approx(pgs_beta)
    assert cfg.sim.physics.solver_cfg.pgs_cfm == pytest.approx(pgs_cfm)
    assert cfg.sim.dt == pytest.approx(dt)
    assert cfg.decimation == decimation


@pytest.mark.parametrize("task_name", ["Isaac-Humanoid", "Isaac-Humanoid-Direct"])
def test_feather_pgs_humanoid_preset_applies_stabilization(task_name: str) -> None:
    cfg = resolve_presets(load_cfg_from_registry(task_name, "env_cfg_entry_point"), {"feather_pgs"})

    robot = cfg.robot if hasattr(cfg, "robot") else cfg.scene.robot
    assert robot.actuators["body"].armature == 0.2
    assert cfg.events.robot_body_mass.params["mass_distribution_params"] == (2.0, 2.0)
    assert cfg.events.robot_body_inertia.params["inertia_distribution_params"] == (2.0, 2.0)


def test_feather_pgs_hand_and_cabinet_presets_apply_stabilization() -> None:
    allegro = resolve_presets(
        load_cfg_from_registry("Isaac-Reorient-Cube-Allegro-Direct", "env_cfg_entry_point"),
        {"feather_pgs"},
    )
    shadow = resolve_presets(
        load_cfg_from_registry("Isaac-Reorient-Cube-Shadow-Direct", "env_cfg_entry_point"),
        {"feather_pgs"},
    )
    cabinet = resolve_presets(
        load_cfg_from_registry("Isaac-Open-Drawer-Franka", "env_cfg_entry_point"),
        {"feather_pgs"},
    )

    assert allegro.robot_cfg.actuators["fingers"].armature == 0.1
    assert shadow.robot_cfg.actuators["fingers"].armature == 0.05
    assert shadow.robot_cfg.actuators["fingers"].friction == 0.1
    assert shadow.events.object_physics_inertia.params["inertia_distribution_params"] == (0.1, 0.1)
    assert cabinet.scene.cabinet.actuators["drawers"].damping == 3.0
    assert cabinet.events.robot_body_inertia.params["inertia_distribution_params"] == (0.02, 0.02)
    assert cabinet.events.cabinet_body_inertia.params["inertia_distribution_params"] == (0.05, 0.05)
