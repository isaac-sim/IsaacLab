# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visual-material manager terms."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import VisualMaterialCfg
from isaaclab.envs.mdp import randomize_visual_color
from isaaclab.envs.mdp.visual_events import randomize_visual_material, randomize_visual_shape
from isaaclab.managers import EventTermCfg, SceneEntityCfg


class _RenderContext:
    def __init__(self):
        self.calls = []

    def write_visual_materials(self, materials, channels, env_ids):
        self.calls.append((materials, channels, env_ids))


class _Scene:
    num_envs = 4

    def __init__(self, materials):
        self.materials = dict(zip(("body", "legs"), materials, strict=True))

    def __getitem__(self, name):
        return self.materials[name]


@pytest.mark.parametrize("selection,count", [(slice(None), 4), (slice(1, None, 2), 2), (slice(0, 0), 0)])
def test_all_environment_slice_samples_one_gpu_row_per_material_and_environment(selection, count) -> None:
    materials = [
        SimpleNamespace(
            cfg=VisualMaterialCfg(prim_path=f"/World/envs/env_0/Materials/{name}", spawn=sim_utils.PreviewSurfaceCfg()),
            is_per_env=True,
            channels=("color",),
        )
        for name in ("body", "legs")
    ]
    render_context = _RenderContext()
    env = SimpleNamespace(
        scene=_Scene(materials),
        sim=SimpleNamespace(render_context=render_context),
        device="cpu",
        num_envs=4,
    )
    cfg = EventTermCfg(
        func=randomize_visual_material,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("body"), SceneEntityCfg("legs")],
            "channels": {"color": ((0.25, 0.5, 0.75), (0.25, 0.5, 0.75))},
        },
    )

    term = randomize_visual_material(cfg, env)
    term(env, selection, **cfg.params)

    written_materials, channels, env_ids = render_context.calls[0]
    assert written_materials == materials
    assert env_ids is selection
    assert channels["color"].shape == (2, count, 3)
    torch.testing.assert_close(channels["color"], torch.tensor([0.25, 0.5, 0.75]).expand(2, count, 3))


@pytest.mark.parametrize(
    "visualizers,renderers",
    [(("kit",), ()), (("newton_gl",), ("isaac_rtx",)), ((), ())],
)
def test_shape_event_rejects_consumers_without_shape_storage(visualizers, renderers) -> None:
    env = SimpleNamespace(
        sim=SimpleNamespace(
            resolve_visualizer_types=lambda: list(visualizers),
            render_context=SimpleNamespace(renderer_types=renderers),
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
    env = SimpleNamespace(cfg=SimpleNamespace(scene=SimpleNamespace(replicate_physics=False)))
    with pytest.raises(NotImplementedError, match="require Isaac Sim"):
        randomize_visual_color(cfg, env)
