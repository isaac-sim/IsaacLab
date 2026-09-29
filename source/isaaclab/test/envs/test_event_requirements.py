# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction-time errors raised by event terms."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp import (
    randomize_physics_scene_gravity,
    randomize_rigid_body_mass,
    randomize_visual_color,
    randomize_visual_shape,
)
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RenderContext, RendererCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg


def test_shared_events_import_without_backend_packages() -> None:
    """Importing shared terms must not require an unused physics backend."""
    script = """
import importlib.abc
import sys

class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"isaaclab_newton", "isaaclab_ov", "isaaclab_physx"}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockBackends())
from isaaclab.envs.mdp import randomize_physics_scene_gravity
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "visualizers,renderers",
    [(("kit",), ()), (("newton_gl",), ("isaac_rtx",)), ((), ())],
)
def test_shape_event_rejects_consumers_without_shape_storage(visualizers, renderers) -> None:
    env = SimpleNamespace(
        sim=SimpleNamespace(
            resolve_visualizer_types=lambda: list(visualizers),
            render_context=RenderContext([(RendererCfg(renderer_type=name), None) for name in renderers]),
        )
    )
    cfg = EventTermCfg(
        func=randomize_visual_shape,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "channels": {"color": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))}},
    )
    with pytest.raises(NotImplementedError, match="no per-shape visual storage"):
        randomize_visual_shape(cfg, env)


def test_replicator_event_rejects_kitless_runtime() -> None:
    """Report the runtime requirement before loading a Kit extension."""
    cfg = EventTermCfg(
        func=randomize_visual_color,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "colors": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]},
    )
    env = SimpleNamespace(
        cfg=ManagerBasedEnvCfg(scene=InteractiveSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False))
    )
    with pytest.raises(NotImplementedError, match="require Isaac Sim"):
        randomize_visual_color(cfg, env)


@pytest.mark.parametrize(
    "distribution,params,error",
    [
        # gaussian parameters are (mean, std), so a std below the mean is valid
        pytest.param("gaussian", (0.5, 0.1), None, id="gaussian-std-below-mean"),
        pytest.param("gaussian", (1.0, -0.1), "standard deviation must be ≥ 0", id="gaussian-negative-std"),
        pytest.param("gaussian", (0.0, 0.1), "mean must be > 0", id="gaussian-zero-mean"),
        pytest.param("uniform", (0.5, 0.1), "upper bound", id="uniform-reversed-range"),
    ],
)
def test_scale_range_validation_follows_distribution(distribution, params, error) -> None:
    """Scale parameters are checked as (mean, std) for gaussian and as (low, high) for uniform."""
    env = SimpleNamespace(scene={"robot": SimpleNamespace()})
    cfg = EventTermCfg(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": params,
            "operation": "scale",
            "distribution": distribution,
        },
    )
    if error is None:
        randomize_rigid_body_mass(cfg, env)
    else:
        with pytest.raises(ValueError, match=error):
            randomize_rigid_body_mass(cfg, env)


def test_unknown_physics_configuration_fails_at_term_construction():
    """An unsupported physics config must not silently select a different engine."""
    env = SimpleNamespace(sim=SimpleNamespace(cfg=SimulationCfg(physics=PhysicsCfg())))
    cfg = EventTermCfg(func=randomize_physics_scene_gravity, mode="startup")
    with pytest.raises(NotImplementedError, match="unsupported for PhysicsCfg"):
        randomize_physics_scene_gravity(cfg, env)
