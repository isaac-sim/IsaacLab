# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for articulation ordering in environment event terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp import events as events_module

_NUM_SHAPES_PER_BACKEND_BODY = (1, 2, 3)
_NUM_ENVS = 2
_NUM_SHAPES = sum(_NUM_SHAPES_PER_BACKEND_BODY)
_NONIDENTITY_BODY_ORDERING = SimpleNamespace(
    user_to_backend_indices=(0, 2, 1),
    backend_to_user_indices=(0, 2, 1),
)


def _sample_lower_bound(low, high, shape, device):
    """Return deterministic samples at the lower bound."""
    del high
    low = torch.as_tensor(low, device=device)
    if low.ndim == 0:
        return torch.full(shape, low.item(), device=device)
    return low.expand(shape).clone()


class _FakePhysxRootView:
    """Minimal PhysX articulation root view used by material randomization."""

    def __init__(self):
        self.link_paths = [[f"/body_{body_id}" for body_id in range(len(_NUM_SHAPES_PER_BACKEND_BODY))]]
        self.max_shapes = _NUM_SHAPES
        self.materials = torch.zeros((_NUM_ENVS, _NUM_SHAPES, 3))
        self.written_env_ids = None

    def get_material_properties(self):
        return self.materials

    def set_material_properties(self, materials, env_ids):
        self.materials = materials.clone()
        self.written_env_ids = env_ids.clone()


class _FakePhysicsSimulationView:
    """Return per-link PhysX views in backend body order."""

    def create_rigid_body_view(self, link_path):
        body_id = int(link_path.rsplit("_", maxsplit=1)[1])
        return SimpleNamespace(max_shapes=_NUM_SHAPES_PER_BACKEND_BODY[body_id])


class _FakePhysxArticulation:
    """Minimal articulation surface consumed by the PhysX material term."""

    # exercise the real public translation instead of re-implementing it
    map_body_ids_to_backend = BaseArticulation.map_body_ids_to_backend

    def __init__(self, body_ordering):
        self.body_ordering = body_ordering
        self.root_view = _FakePhysxRootView()
        self._physics_sim_view = _FakePhysicsSimulationView()


class _FakeNewtonRootView:
    """Minimal Newton articulation root view used by material randomization."""

    def __init__(self):
        self.attributes = {
            "shape_material_mu": torch.zeros((_NUM_ENVS, 1, _NUM_SHAPES)),
            "shape_material_restitution": torch.zeros((_NUM_ENVS, 1, _NUM_SHAPES)),
        }

    def get_attribute(self, name, model):
        del model
        return self.attributes[name]


class _FakeNewtonArticulation:
    """Minimal articulation surface consumed by the Newton material term."""

    # exercise the real public translation instead of re-implementing it
    map_body_ids_to_backend = BaseArticulation.map_body_ids_to_backend

    def __init__(self, body_ordering):
        self.body_ordering = body_ordering
        # shape counts are always exposed in backend order by the Newton asset
        self.backend_num_shapes_per_body = list(_NUM_SHAPES_PER_BACKEND_BODY)
        self._root_view = _FakeNewtonRootView()


class _FakeNewtonManager:
    """Capture Newton material-change notifications."""

    _solver = object()
    notifications = []

    @staticmethod
    def get_model():
        return object()

    @classmethod
    def add_model_change(cls, notification):
        cls.notifications.append(notification)


@pytest.fixture
def deterministic_material_sampling(monkeypatch):
    """Keep material samples deterministic and accept torch-backed fake bindings."""
    import isaaclab.assets as assets_module

    _ = assets_module.BaseArticulation
    monkeypatch.setattr(events_module.math_utils, "sample_uniform", _sample_lower_bound)
    monkeypatch.setattr(events_module.wp, "from_torch", lambda tensor, dtype=None: tensor)
    monkeypatch.setattr(events_module.wp, "to_torch", lambda tensor, requires_grad=None: tensor)


@pytest.mark.parametrize(
    ("body_ordering", "expected_shape_slice"),
    [
        pytest.param(_NONIDENTITY_BODY_ORDERING, slice(3, 6), id="nonidentity-ordering"),
        pytest.param(None, slice(1, 3), id="default-ordering"),
    ],
)
def test_physx_material_randomization_automatically_converts_public_body_ids_to_backend_shape_range(
    monkeypatch, deterministic_material_sampling, body_ordering, expected_shape_slice
):
    """PhysX automatically converts public body selections to backend shape ranges."""
    import isaaclab.assets as assets_module

    monkeypatch.setattr(assets_module, "BaseArticulation", _FakePhysxArticulation)
    asset = _FakePhysxArticulation(body_ordering)
    asset_cfg = SimpleNamespace(body_ids=[1])
    cfg = SimpleNamespace(
        params={
            "static_friction_range": (0.4, 0.4),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.1, 0.1),
            "num_buckets": 1,
        }
    )
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=_NUM_ENVS), device="cpu")
    term = events_module._RandomizeRigidBodyMaterialPhysx(cfg, env, asset, asset_cfg)

    term(
        env,
        torch.tensor([0], dtype=torch.int32),
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


@pytest.mark.parametrize(
    ("body_ordering", "expected_shape_slice"),
    [
        pytest.param(_NONIDENTITY_BODY_ORDERING, slice(3, 6), id="nonidentity-ordering"),
        pytest.param(None, slice(1, 3), id="default-ordering"),
    ],
)
def test_newton_material_randomization_automatically_converts_public_body_ids_to_backend_shape_range(
    monkeypatch, deterministic_material_sampling, body_ordering, expected_shape_slice
):
    """Newton automatically converts public body selections to backend shape ranges."""
    newton_assets_module = pytest.importorskip("isaaclab_newton.assets")
    newton_manager_module = pytest.importorskip("isaaclab_newton.physics.newton_manager")

    monkeypatch.setattr(newton_assets_module, "Articulation", _FakeNewtonArticulation)
    monkeypatch.setattr(newton_manager_module, "NewtonManager", _FakeNewtonManager)
    _FakeNewtonManager.notifications.clear()
    asset = _FakeNewtonArticulation(body_ordering)
    asset_cfg = SimpleNamespace(body_ids=[1])
    cfg = SimpleNamespace(
        params={
            "static_friction_range": (0.4, 0.4),
            "restitution_range": (0.1, 0.1),
        }
    )
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=_NUM_ENVS), device="cpu")
    term = events_module._RandomizeRigidBodyMaterialNewton(cfg, env, asset, asset_cfg)

    term(
        env,
        torch.tensor([0], dtype=torch.int32),
        static_friction_range=(0.4, 0.4),
        dynamic_friction_range=(0.2, 0.2),
        restitution_range=(0.1, 0.1),
        num_buckets=1,
        asset_cfg=asset_cfg,
    )

    expected_friction = torch.zeros((_NUM_ENVS, _NUM_SHAPES))
    expected_restitution = torch.zeros_like(expected_friction)
    expected_friction[0, expected_shape_slice] = 0.4
    expected_restitution[0, expected_shape_slice] = 0.1
    torch.testing.assert_close(term._friction_binding, expected_friction)
    torch.testing.assert_close(term._restitution_binding, expected_restitution)
    assert len(_FakeNewtonManager.notifications) == 1
