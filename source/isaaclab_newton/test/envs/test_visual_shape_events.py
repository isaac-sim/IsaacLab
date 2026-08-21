# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton's per-link visual event."""

from types import SimpleNamespace

import torch
from isaaclab_newton.envs.mdp import events as events_module
from isaaclab_newton.envs.mdp.events import randomize_visual_shape

from isaaclab.managers import EventTermCfg, SceneEntityCfg


class _Writer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.body_count = 2
        self.model = object()
        self.calls = []
        self.rebound = None

    def rebind(self, model):
        self.model = model
        self.rebound = model

    def __call__(self, colors, env_ids):
        self.calls.append((colors, env_ids))


def test_event_samples_selected_links_per_environment_and_rebinds(monkeypatch) -> None:
    writer = _Writer()
    created = []
    new_model = object()
    asset = SimpleNamespace(num_bodies=3, body_names=["base", "hip", "foot"])
    env = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=4)
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 2])
    cfg = EventTermCfg(
        func=randomize_visual_shape,
        mode="reset",
        params={"asset_cfg": asset_cfg, "channels": {"color": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))}},
    )
    monkeypatch.setattr(
        events_module.NewtonManager,
        "create_visual_shape_color_writer",
        lambda asset, body_names: created.append((asset, body_names)) or writer,
    )
    monkeypatch.setattr(events_module.NewtonManager, "get_model", lambda: new_model)
    monkeypatch.setattr(
        events_module,
        "_compile_distribution",
        lambda spec, device: lambda shape: torch.arange(torch.tensor(shape).prod() * 3).reshape(*shape, 3).float(),
    )

    term = randomize_visual_shape(cfg, env)
    assert created == [(asset, ("base", "foot"))]
    assert not writer.calls
    term(env, torch.tensor([3, 1], dtype=torch.int32), **cfg.params)

    assert len(created) == 1
    assert writer.rebound is new_model
    colors, env_ids = writer.calls[0]
    assert colors.shape == (2, 2, 3)
    torch.testing.assert_close(env_ids, torch.tensor([3, 1], dtype=torch.int32))
    assert not torch.equal(colors[:, 0], colors[:, 1])

    term(env, slice(None), **cfg.params)
    torch.testing.assert_close(writer.calls[1][1], torch.arange(4, dtype=torch.int32))


def test_shape_event_rejects_non_color_channels() -> None:
    env = SimpleNamespace(scene={}, device="cpu", num_envs=1)
    cfg = EventTermCfg(
        func=randomize_visual_shape,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "channels": {"roughness": (0.0, 1.0)}},
    )

    try:
        randomize_visual_shape(cfg, env)
    except NotImplementedError as error:
        assert "only the 'color' channel" in str(error)
    else:
        raise AssertionError("non-color per-shape randomization must fail")
