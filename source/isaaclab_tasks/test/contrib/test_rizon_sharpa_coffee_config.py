# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration contracts that must hold before constructing the coffee solver."""

import subprocess
import sys
import textwrap

import pytest

from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.contrib.rizon_sharpa_coffee import robot
from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import BOWL_POSITION
from isaaclab_tasks.contrib.rizon_sharpa_coffee.coffee_env_cfg import CoffeeEnvCfg, _particle_cfg, configure_scene


def test_coffee_configs_import_before_usd_runtime():
    """Registration and demo configuration must not initialize USD before app startup."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import importlib.abc
                import sys

                class RejectUsd(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "pxr" or fullname.startswith("pxr."):
                            raise RuntimeError("USD imported before simulator startup")

                sys.meta_path.insert(0, RejectUsd())
                from isaaclab_tasks.contrib.rizon_sharpa_coffee.coffee_env_cfg import CoffeeEnvCfg
                from isaaclab_tasks.contrib.rizon_sharpa_coffee.kit_presentation_cfg import CoffeeKitVisualizerCfg

                assert CoffeeEnvCfg().scene.num_envs == 1
                assert CoffeeKitVisualizerCfg().backdrop == "kitchen"
            """),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_water_override_resolves_profile_without_replacing_explicit_tuning(monkeypatch):
    """Hydra-style late selection and explicit solver overrides must compose."""
    # Only the externally provisioned robot is substituted. Generated props,
    # fill sampling and the final configuration boundary run normally.
    monkeypatch.setattr(robot, "get_robot_cfg", lambda **kwargs: ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot"))
    cfg = CoffeeEnvCfg()
    cfg.from_dict({"water_enabled": True})
    cfg.sim.dt = 1.0 / 400.0
    cfg.decimation = 4
    configure_scene(cfg)
    solver = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    assert cfg.scene.water is not None
    assert (solver.solver, solver.max_iterations) == ("gs", 500)
    assert cfg.sim.dt == 1.0 / 400.0 and cfg.decimation == 4

    tuned = CoffeeEnvCfg(water_enabled=True)
    tuned_solver = tuned.sim.physics.solver_cfg.entries[1].solver_cfg
    tuned_solver.solver = "jacobi"
    tuned_solver.max_iterations = 100
    configure_scene(tuned)
    assert (tuned_solver.solver, tuned_solver.max_iterations) == ("jacobi", 100)

    dry = CoffeeEnvCfg()
    configure_scene(dry)
    dry_solver = dry.sim.physics.solver_cfg.entries[1].solver_cfg
    assert dry.scene.water is None
    assert (dry_solver.solver, dry_solver.max_iterations) == ("auto", 100)


def test_empty_coffee_fill_reports_incompatible_grid_and_height():
    """Individually supported coarse grids and shallow fills can sample no particles."""
    with pytest.raises(ValueError, match="Coffee.*empty"):
        _particle_cfg("Coffee", BOWL_POSITION, 0.060, 0.009, 0.005, False)
