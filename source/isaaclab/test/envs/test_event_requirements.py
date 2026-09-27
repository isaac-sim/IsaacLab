# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction errors for unsupported event backends and runtimes."""

from types import SimpleNamespace

import pytest

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp import randomize_physics_scene_gravity, randomize_visual_color, randomize_visual_shape
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RenderContext, RendererCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg


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
