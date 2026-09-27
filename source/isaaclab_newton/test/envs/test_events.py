# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton implementations of shared MDP event terms."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton import assets as newton_assets
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from newton import ModelFlags
from newton.solvers import SolverKamino

from isaaclab import assets
from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp import events as events_module
from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.managers import EventManager, EventTermCfg

_NUM_SHAPES_PER_BACKEND_BODY = (1, 2, 3)
_NUM_ENVS = 2
_NUM_SHAPES = sum(_NUM_SHAPES_PER_BACKEND_BODY)
_NONIDENTITY_BODY_ORDERING = SimpleNamespace(user_to_backend_indices=(0, 2, 1), backend_to_user_indices=(0, 2, 1))


class _RootView:
    def __init__(self):
        self.attributes = {
            "shape_material_mu": torch.zeros((_NUM_ENVS, 1, _NUM_SHAPES)),
            "shape_material_restitution": torch.zeros((_NUM_ENVS, 1, _NUM_SHAPES)),
            "shape_margin": torch.full((_NUM_ENVS, 1, _NUM_SHAPES), 0.1),
            "shape_gap": torch.full((_NUM_ENVS, 1, _NUM_SHAPES), 0.2),
        }

    def get_attribute(self, name, model):
        return wp.from_torch(self.attributes[name])


class _Articulation:
    map_body_ids_to_backend = BaseArticulation.map_body_ids_to_backend

    def __init__(self, body_ordering):
        self.body_ordering = body_ordering
        self.backend_num_shapes_per_body = list(_NUM_SHAPES_PER_BACKEND_BODY)
        self._root_view = _RootView()


class _Manager:
    _solver = object()
    notifications = []

    @staticmethod
    def get_model():
        return object()

    @classmethod
    def add_model_change(cls, notification):
        cls.notifications.append(notification)


@pytest.fixture
def asset_types(monkeypatch):
    wp.init()
    monkeypatch.setattr(assets, "BaseArticulation", _Articulation)
    monkeypatch.setattr(newton_assets, "Articulation", _Articulation)
    _Manager.notifications.clear()


@pytest.mark.parametrize(
    ("body_ordering", "expected_shape_slice"),
    [
        pytest.param(_NONIDENTITY_BODY_ORDERING, slice(3, 6), id="nonidentity-ordering"),
        pytest.param(None, slice(1, 3), id="default-ordering"),
    ],
)
def test_newton_material_randomization_automatically_converts_public_body_ids_to_backend_shape_range(
    asset_types, body_ordering, expected_shape_slice
):
    """Newton automatically converts public body selections to backend shape ranges."""
    asset = _Articulation(body_ordering)
    asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
    cfg = SimpleNamespace(
        params={
            "static_friction_range": (0.4, 0.4),
            "restitution_range": (0.1, 0.1),
        }
    )
    cfg.params["asset_cfg"] = asset_cfg
    env = SimpleNamespace(
        scene={"robot": asset},
        num_envs=_NUM_ENVS,
        device="cpu",
        sim=SimpleNamespace(
            physics_manager=_Manager, cfg=SimpleNamespace(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
        ),
    )
    term = events_module.randomize_rigid_body_material(cfg, env)

    term(
        env,
        torch.tensor([0], dtype=torch.int32),
        static_friction_range=(0.4, 0.4),
        dynamic_friction_range=(0.2, 0.2),
        num_buckets=1,
        restitution_range=(0.1, 0.1),
        asset_cfg=asset_cfg,
    )

    expected_friction = torch.zeros((_NUM_ENVS, _NUM_SHAPES))
    expected_restitution = torch.zeros_like(expected_friction)
    expected_friction[0, expected_shape_slice] = 0.4
    expected_restitution[0, expected_shape_slice] = 0.1
    torch.testing.assert_close(asset._root_view.attributes["shape_material_mu"][:, 0], expected_friction)
    torch.testing.assert_close(asset._root_view.attributes["shape_material_restitution"][:, 0], expected_restitution)
    assert len(_Manager.notifications) == 1


def test_newton_collider_offsets_translate_only_selected_environments(asset_types):
    """The shared offset term performs Newton margin/gap translation internally."""
    asset = _Articulation(None)
    asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))
    env = SimpleNamespace(
        scene={"robot": asset},
        sim=SimpleNamespace(
            physics_manager=_Manager,
            cfg=SimpleNamespace(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        ),
    )
    term = events_module.randomize_rigid_body_collider_offsets(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)
    term(env, torch.tensor([1]), asset_cfg, (0.3, 0.3), (0.4, 0.4))
    expected_margin = torch.tensor([0.1, 0.3])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    expected_gap = torch.tensor([0.2, 0.1])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    torch.testing.assert_close(asset._root_view.attributes["shape_margin"], expected_margin)
    torch.testing.assert_close(asset._root_view.attributes["shape_gap"], expected_gap)
    term(env, slice(0, 1), asset_cfg, contact_offset_distribution_params=(0.6, 0.6))
    torch.testing.assert_close(asset._root_view.attributes["shape_margin"], expected_margin)
    expected_gap = torch.tensor([0.5, 0.1])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    torch.testing.assert_close(asset._root_view.attributes["shape_gap"], expected_gap)
    term(env, torch.tensor([1]), asset_cfg, contact_offset_distribution_params=(0.2, 0.2))
    expected_gap = torch.tensor([0.5, 0.0])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    torch.testing.assert_close(asset._root_view.attributes["shape_margin"], expected_margin)
    torch.testing.assert_close(asset._root_view.attributes["shape_gap"], expected_gap)


def test_newton_gravity_selectors_preserve_global_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment selectors must exclude Newton's trailing global-world gravity row."""
    wp.init()
    gravity = torch.full((5, 3), -9.81)
    model = SimpleNamespace(gravity=wp.from_torch(gravity, dtype=wp.vec3))
    notifications = []

    class CustomDynamics(NewtonManager):
        pass

    monkeypatch.setattr(CustomDynamics, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(CustomDynamics, "add_model_change", classmethod(lambda cls, flag: notifications.append(flag)))
    env = SimpleNamespace(
        device="cpu",
        num_envs=4,
        sim=SimpleNamespace(
            physics_manager=CustomDynamics,
            is_playing=lambda: True,
            cfg=SimpleNamespace(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        ),
    )
    cfg = EventTermCfg(
        func=randomize_physics_scene_gravity,
        mode="startup",
        params={"gravity_distribution_params": ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), "operation": "abs"},
    )
    manager = EventManager({"gravity": cfg}, env)
    for selector, rows in (
        (None, [0, 1, 2, 3]),
        (slice(None), [0, 1, 2, 3]),
        (slice(1, None, 2), [1, 3]),
        (slice(-2, None), [2, 3]),
        (torch.tensor([1, 3], dtype=torch.int32), [1, 3]),
        (slice(0, 0), []),
    ):
        gravity.fill_(-9.81)
        notifications.clear()
        manager.apply("startup", env_ids=selector)
        expected = torch.full_like(gravity, -9.81)
        expected[rows] = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(gravity, expected)
        assert notifications == ([ModelFlags.MODEL_PROPERTIES] if rows else [])

    term = manager.get_term_cfg("gravity").func
    gravity.fill_(-9.81)
    bounds = ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    for _ in range(2):
        term(env, torch.tensor([1]), bounds, operation="add")
    torch.testing.assert_close(gravity[1], torch.tensor([-7.81, -5.81, -3.81]))
    bounds[0][:] = [2.0, 2.0, 2.0]
    bounds[1][:] = [2.0, 2.0, 2.0]
    for _ in range(2):
        term(env, torch.tensor([1]), bounds, operation="scale")
    torch.testing.assert_close(gravity[1], torch.tensor([-31.24, -23.24, -15.24]))
    torch.testing.assert_close(gravity[[0, 2, 3, 4]], torch.full((4, 3), -9.81))


def test_kamino_material_groups_persist_across_randomization(asset_types, monkeypatch):
    """Shared original materials stay grouped across worlds and successive samples."""
    monkeypatch.setattr(_Manager, "_solver", SolverKamino.__new__(SolverKamino))
    asset = _Articulation(None)
    friction = asset._root_view.attributes["shape_material_mu"][:, 0]
    restitution = asset._root_view.attributes["shape_material_restitution"][:, 0]
    friction[:] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.4, 0.4])
    restitution[:] = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.1, 0.2])
    original_friction = friction.clone()
    original_restitution = restitution.clone()
    asset_cfg = SimpleNamespace(name="robot", body_ids=[2])
    params = {
        "asset_cfg": asset_cfg,
        "static_friction_range": (0.5, 1.0),
        "dynamic_friction_range": (0.0, 0.0),
        "restitution_range": (0.3, 0.9),
        "num_buckets": 1,
    }
    env = SimpleNamespace(
        scene={"robot": asset},
        device="cpu",
        num_envs=_NUM_ENVS,
        sim=SimpleNamespace(
            physics_manager=_Manager, cfg=SimpleNamespace(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
        ),
    )
    term = events_module.randomize_rigid_body_material(SimpleNamespace(params=params), env)
    torch.manual_seed(7)
    for _ in range(2):
        term(env, torch.tensor([1]), **params)
        for values, original in ((friction, original_friction), (restitution, original_restitution)):
            torch.testing.assert_close(values[:, :3], original[:, :3])
            torch.testing.assert_close(values[0, 3:], values[1, 3:])
            torch.testing.assert_close(values[:, 3], values[:, 4])
            assert not torch.equal(values[:, 3], values[:, 5])
        # Make randomized values coincide; the original grouping must survive.
        friction[:, 3:] = 0.7
        restitution[:, 3:] = 0.6
    assert _Manager.notifications == [ModelFlags.SHAPE_PROPERTIES, ModelFlags.SHAPE_PROPERTIES]
