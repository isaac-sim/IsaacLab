# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PhysX event selection and gravity sampling."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg

from isaaclab import assets as assets_module
from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp import events as events_module
from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.managers import EventManager, EventTermCfg

_NUM_SHAPES_PER_BACKEND_BODY = (1, 2, 3)
_NUM_ENVS = 2
_NUM_SHAPES = sum(_NUM_SHAPES_PER_BACKEND_BODY)
_NONIDENTITY_BODY_ORDERING = SimpleNamespace(
    user_to_backend_indices=(0, 2, 1),
    backend_to_user_indices=(0, 2, 1),
)


class _FakePhysxRootView:
    """Minimal PhysX articulation root view used by material randomization."""

    def __init__(self):
        self.link_paths = [[f"/body_{body_id}" for body_id in range(len(_NUM_SHAPES_PER_BACKEND_BODY))]]
        self.max_shapes = _NUM_SHAPES
        self.materials = torch.zeros((_NUM_ENVS, _NUM_SHAPES, 3))
        self.written_env_ids = None

    def get_material_properties(self):
        return wp.from_torch(self.materials)

    def set_material_properties(self, materials, env_ids):
        self.materials = wp.to_torch(materials).clone()
        self.written_env_ids = wp.to_torch(env_ids).clone()


class _FakePhysicsSimulationView:
    """Return per-link PhysX views in backend body order."""

    def create_rigid_body_view(self, link_path):
        body_id = int(link_path.rsplit("_", maxsplit=1)[1])
        return SimpleNamespace(max_shapes=_NUM_SHAPES_PER_BACKEND_BODY[body_id])


class _FakePhysxArticulation:
    """Minimal articulation surface consumed by the PhysX material term."""

    map_body_ids_to_backend = BaseArticulation.map_body_ids_to_backend

    def __init__(self, body_ordering):
        self.body_ordering = body_ordering
        self.root_view = _FakePhysxRootView()
        self._physics_sim_view = _FakePhysicsSimulationView()


class _FakeScene(dict):
    """Scene exposing the environment count used by material randomization."""

    def __init__(self, **assets):
        super().__init__(assets)
        self.num_envs = _NUM_ENVS


@pytest.mark.parametrize(
    ("body_ordering", "expected_shape_slice", "index_dtype"),
    [
        pytest.param(_NONIDENTITY_BODY_ORDERING, slice(3, 6), torch.int32, id="reordered-int32"),
        pytest.param(None, slice(1, 3), torch.int64, id="default-int64"),
        pytest.param(_NONIDENTITY_BODY_ORDERING, slice(3, 6), slice, id="reordered-slice"),
    ],
)
def test_physx_material_randomization_automatically_converts_public_body_ids_to_backend_shape_range(
    monkeypatch, body_ordering, expected_shape_slice, index_dtype
):
    """PhysX automatically converts public body selections to backend shape ranges."""
    monkeypatch.setattr(assets_module, "BaseArticulation", _FakePhysxArticulation)
    asset = _FakePhysxArticulation(body_ordering)
    asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
    cfg = SimpleNamespace(
        params={
            "static_friction_range": (0.4, 0.4),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.1, 0.1),
            "num_buckets": 1,
        }
    )
    cfg.params["asset_cfg"] = asset_cfg
    env = SimpleNamespace(
        scene=_FakeScene(robot=asset),
        num_envs=_NUM_ENVS,
        device="cpu",
        sim=SimpleNamespace(cfg=SimpleNamespace(physics=PhysxCfg())),
    )
    term = events_module.randomize_rigid_body_material(cfg, env)

    term(
        env,
        slice(0, 1) if index_dtype is slice else torch.tensor([0], dtype=index_dtype),
        static_friction_range=(0.4, 0.4),
        dynamic_friction_range=(0.2, 0.2),
        restitution_range=(0.1, 0.1),
        num_buckets=1,
        asset_cfg=asset_cfg,
    )

    expected = torch.zeros_like(asset.root_view.materials)
    expected[0, expected_shape_slice] = torch.tensor([0.4, 0.2, 0.1])
    torch.testing.assert_close(asset.root_view.materials, expected)
    torch.testing.assert_close(asset.root_view.written_env_ids, torch.tensor([0], dtype=torch.int32))


def test_scene_gravity_uses_configured_distribution() -> None:
    """PhysX uses the distribution configured at initialization."""
    gravity_sink = SimpleNamespace()
    physics_manager = type(
        "GravitySink",
        (),
        {"set_gravity": staticmethod(lambda gravity: setattr(gravity_sink, "value", gravity))},
    )
    physics_cfg = PhysxCfg()
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        sim=SimpleNamespace(
            cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81), physics=physics_cfg),
            physics_manager=physics_manager,
            physics_sim_view=physics_manager,
            is_playing=lambda: True,
        ),
    )
    cfg = EventTermCfg(
        func=randomize_physics_scene_gravity,
        mode="startup",
        params={
            "gravity_distribution_params": ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    manager = EventManager({"gravity": cfg}, env)
    torch.manual_seed(0)
    manager.apply("startup")
    assert gravity_sink.value == pytest.approx((1.0, 2.0, 3.0))
