# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction-time validation of event requirements and randomization parameters."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp import (
    randomize_fixed_tendon_parameters,
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


def test_unknown_physics_configuration_fails_at_term_construction():
    """An unsupported physics config must not silently select a different engine."""
    env = SimpleNamespace(sim=SimpleNamespace(cfg=SimulationCfg(physics=PhysicsCfg())))
    cfg = EventTermCfg(func=randomize_physics_scene_gravity, mode="startup")
    with pytest.raises(NotImplementedError, match="unsupported for PhysicsCfg"):
        randomize_physics_scene_gravity(cfg, env)


@pytest.mark.parametrize(
    "params,options,error",
    [
        ((0.5, 0.1), {"distribution": "gaussian"}, None),
        ((1.0, 0.0), {"distribution": "gaussian"}, None),
        ((1.0, -0.1), {"distribution": "gaussian", "operation": "add"}, "standard deviation"),
        ((0.0, 0.1), {"distribution": "gaussian"}, "mean must be > 0"),
        ((0.5, 0.1), {}, "upper bound"),
        ((0.5, 1.5), {}, None),
        ((0.0, 1.0), {"distribution": "log_uniform", "operation": "abs"}, "bounds must be > 0"),
        ((0.5, 1.5), {"distribution": "log_uniform"}, None),
        ((float("nan"), 1.0), {}, "must be finite"),
        ((1.0, float("inf")), {"distribution": "gaussian"}, "must be finite"),
        ((0.5, 1.5), {"distribution": "unknown"}, "unsupported distribution"),
        ((0.5, 1.5), {"operation": "unknown"}, "Unsupported randomization operation"),
    ],
)
def test_randomization_parameters_follow_distribution(params, options, error):
    """Validate the configured distribution independently of operation, preserving scale-sign policy."""
    asset = SimpleNamespace()
    env = SimpleNamespace(scene={"robot": asset})
    cfg = EventTermCfg(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": params,
            "operation": "scale",
            **options,
        },
    )
    if error is None:
        assert randomize_rigid_body_mass(cfg, env).asset is asset
    else:
        with pytest.raises(ValueError, match=error):
            randomize_rigid_body_mass(cfg, env)


def test_tendon_defaults_and_signed_parameters():
    """Omitted operation uses abs; optional and signed fields retain their distinct policies."""
    asset = SimpleNamespace()
    env = SimpleNamespace(scene={"robot": asset})
    cfg = EventTermCfg(
        func=randomize_fixed_tendon_parameters,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "damping_distribution_params": (0.0, 0.0)},
    )
    assert randomize_fixed_tendon_parameters(cfg, env).asset is asset
    assert cfg.params["operation"] == "abs"
    assert cfg.params["distribution"] == "uniform"
    cfg.params.update(operation="scale", lower_limit_distribution_params=(-2.0, -1.0))
    assert randomize_fixed_tendon_parameters(cfg, env).asset is asset
    cfg.params.update(distribution="log_uniform", lower_limit_distribution_params=None)
    with pytest.raises(ValueError, match="log-uniform bounds must be > 0"):
        randomize_fixed_tendon_parameters(cfg, env)
