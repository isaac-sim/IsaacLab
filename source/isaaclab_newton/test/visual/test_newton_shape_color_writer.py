# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`NewtonShapeColorWriter`.

Exercises the Newton-specific bits: resolving target shapes from the term's ``mesh_prim_path`` (via
``find_matching_prim_paths``, monkeypatched here so no stage/GPU is needed) and the per-env/per-prim
apply mechanism on ``model.shape_color``. Backend-agnostic selection logic lives in
``test_visual_color_select.py``.
"""

from types import SimpleNamespace

import isaaclab_newton.visual.newton_shape_color_writer as mod
import numpy as np
import torch
import warp as wp
from isaaclab_newton.visual.newton_shape_color_writer import NewtonShapeColorWriter

# Asset-style scene (cartpole-like): each env has cart + pole, each with a visual and a collision shape.
_ASSET_LABELS = [
    label
    for i in range(3)
    for label in (
        f"/World/envs/env_{i}/Robot/cart/visuals/mesh_0",
        f"/World/envs/env_{i}/Robot/cart/collisions/mesh_0",
        f"/World/envs/env_{i}/Robot/pole/visuals/mesh_0",
        f"/World/envs/env_{i}/Robot/pole/collisions/mesh_0",
    )
]
# What the term resolves for an asset with a /visuals layout: the per-body visuals scope prims.
_ASSET_TARGET_PRIMS = [f"/World/envs/env_{i}/Robot/{body}/visuals" for i in range(3) for body in ("cart", "pole")]


class _FakeModel:
    """Minimal stand-in for ``newton.Model`` exposing the two fields the writer reads."""

    def __init__(self, shape_label: list[str], shape_color: wp.array):
        self.shape_label = shape_label
        self.shape_color = shape_color


def _make_writer(monkeypatch, labels, target_prims, num_envs):
    """Construct a writer against a fake model, with ``find_matching_prim_paths`` returning ``target_prims``."""
    model = _FakeModel(labels, wp.zeros(len(labels), dtype=wp.vec3f, device="cpu"))
    monkeypatch.setattr(mod.sim_utils, "find_matching_prim_paths", lambda pattern: list(target_prims))
    # Fake env exposing what the writer reads: the Newton model + the env count.
    env = SimpleNamespace(
        sim=SimpleNamespace(physics_manager=SimpleNamespace(get_model=lambda: model)),
        scene=SimpleNamespace(num_envs=num_envs),
    )
    writer = NewtonShapeColorWriter(env=env, mesh_prim_path="<ignored>")
    return model, writer


def _row(labels: list[str], env: int, body: str) -> int:
    return next(i for i, label in enumerate(labels) if f"/env_{env}/" in label and f"/{body}/visuals/" in label)


def test_resolves_visual_meshes_excluding_collisions(monkeypatch):
    _, writer = _make_writer(monkeypatch, _ASSET_LABELS, _ASSET_TARGET_PRIMS, num_envs=3)
    # cart + pole visuals per env (6); collision shapes excluded
    assert writer.num_targets == 6
    assert all("/visuals/" in label for (_, _, label) in writer.matched_labels)


def test_primitive_bare_geometry_is_targetable(monkeypatch):
    # A spawned primitive cube has no /visuals scope -- its single mesh is /…/geometry/mesh. The
    # consistency fix (resolve from mesh_prim_path, not a /visuals filter) must still target it.
    # find_matching_prim_paths returns only the source env_0 (Newton clones the rest into shape_label
    # without authoring USD) -- the writer must generalize env_0's match to every cloned env.
    labels = [f"/World/envs/env_{i}/cube_a/geometry/mesh" for i in range(2)]
    target_prims = ["/World/envs/env_0/cube_a/geometry"]  # env_0 only, as the real stage returns
    _, writer = _make_writer(monkeypatch, labels, target_prims, num_envs=2)
    assert writer.num_targets == 2  # both envs resolved from the env_0 template


def test_per_env_and_per_prim_distinct(monkeypatch):
    model, writer = _make_writer(monkeypatch, _ASSET_LABELS, _ASSET_TARGET_PRIMS, num_envs=3)
    n = writer.num_targets  # 6
    colors = torch.tensor([[0.1 * (g + 1), 0.02 * g, 0.5 - 0.05 * g] for g in range(n)], dtype=torch.float32)
    writer.write_colors(torch.arange(3), colors)

    buf = model.shape_color.numpy()
    labels = model.shape_label
    # collision shapes are never recolored
    for i, label in enumerate(labels):
        if "/collisions/" in label:
            assert np.allclose(buf[i], 0.0)
    # every visual prim got its own color -> per-prim AND per-env (all distinct)
    visual_rows = [i for i, label in enumerate(labels) if "/visuals/" in label]
    assert len({tuple(np.round(buf[i], 6)) for i in visual_rows}) == len(visual_rows)
    # explicit: per-env (env0 cart != env1 cart) and per-prim (env0 cart != env0 pole)
    assert not np.allclose(buf[_row(labels, 0, "cart")], buf[_row(labels, 1, "cart")])
    assert not np.allclose(buf[_row(labels, 0, "cart")], buf[_row(labels, 0, "pole")])


def test_env_ids_subset_leaves_others_untouched(monkeypatch):
    model, writer = _make_writer(monkeypatch, _ASSET_LABELS, _ASSET_TARGET_PRIMS, num_envs=3)
    n = writer.num_targets
    writer.write_colors(torch.arange(3), torch.full((n, 3), 0.5))
    before = model.shape_color.numpy().copy()

    # recolor only env 1; colors is still the full (num_targets, 3) -- the writer applies env 1's targets only
    new_colors = torch.stack([torch.tensor([1.0, g / 10.0, 0.0]) for g in range(n)])  # all != baseline 0.5
    writer.write_colors(torch.tensor([1]), new_colors)
    after = model.shape_color.numpy()
    labels = model.shape_label

    assert not np.allclose(after[_row(labels, 1, "cart")], 0.5)
    assert not np.allclose(after[_row(labels, 1, "pole")], 0.5)
    for env in (0, 2):
        assert np.allclose(after[_row(labels, env, "cart")], before[_row(labels, env, "cart")])
        assert np.allclose(after[_row(labels, env, "pole")], before[_row(labels, env, "pole")])
