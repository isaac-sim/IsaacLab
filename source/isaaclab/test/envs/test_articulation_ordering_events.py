# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for articulation ordering in environment event terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.envs.mdp import randomize_rigid_body_material as newton_material
from isaaclab_physx.envs.mdp import randomize_rigid_body_material as physx_material

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
    """Minimal articulation surface consumed by Newton event terms."""

    # exercise the real public translation instead of re-implementing it
    map_body_ids_to_backend = BaseArticulation.map_body_ids_to_backend

    def __init__(self, body_ordering):
        self.body_ordering = body_ordering
        self.device = "cpu"
        # shape counts are always exposed in backend order by the Newton asset
        self.backend_num_shapes_per_body = list(_NUM_SHAPES_PER_BACKEND_BODY)
        self._root_view = _FakeNewtonRootView()
        joint_properties = torch.zeros((_NUM_ENVS, 2))
        self.data = SimpleNamespace(
            joint_friction_coeff=SimpleNamespace(torch=joint_properties.clone()),
            joint_viscous_friction_coeff=SimpleNamespace(torch=joint_properties.clone()),
            joint_armature=SimpleNamespace(torch=joint_properties.clone()),
            joint_pos_limits=SimpleNamespace(torch=torch.zeros((_NUM_ENVS, 2, 2))),
        )
        self.static_friction_writes = []
        self.viscous_friction_writes = []

    def write_joint_friction_coefficient_to_sim_index(
        self, *, joint_dynamic_friction_coeff=None, joint_viscous_friction_coeff=None, **kwargs
    ):
        """Record Newton static-friction writes; like Newton, forward viscous friction and drop dynamic friction."""
        if joint_viscous_friction_coeff is not None:
            self.write_joint_viscous_friction_coefficient_to_sim_index(
                joint_viscous_friction_coeff=joint_viscous_friction_coeff,
                joint_ids=kwargs["joint_ids"],
                env_ids=kwargs["env_ids"],
            )
        self.static_friction_writes.append(kwargs)

    def write_joint_viscous_friction_coefficient_to_sim_index(self, **kwargs):
        """Record Newton passive-damping writes."""
        self.viscous_friction_writes.append(kwargs)


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


class _FakeScene(dict):
    """Dictionary-backed scene with the attributes used by joint randomization."""

    def __init__(self, **assets):
        super().__init__(assets)
        self.num_envs = _NUM_ENVS


@pytest.fixture
def deterministic_material_sampling(monkeypatch):
    """Keep material samples deterministic and accept torch-backed fake bindings."""
    import isaaclab.assets as assets_module

    _ = assets_module.BaseArticulation
    monkeypatch.setattr(events_module.math_utils, "sample_uniform", _sample_lower_bound)
    monkeypatch.setattr(wp, "from_torch", lambda tensor, dtype=None: tensor)
    monkeypatch.setattr(wp, "to_torch", lambda tensor, requires_grad=None: tensor)


@pytest.mark.parametrize(
    ("body_ordering", "expected_shape_slice"),
    [
        pytest.param(_NONIDENTITY_BODY_ORDERING, slice(3, 6), id="nonidentity-ordering"),
        pytest.param(None, slice(1, 3), id="default-ordering"),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, slice])
def test_physx_material_randomization_automatically_converts_public_body_ids_to_backend_shape_range(
    monkeypatch, deterministic_material_sampling, body_ordering, expected_shape_slice, index_dtype
):
    """PhysX automatically converts public body selections to backend shape ranges."""
    import isaaclab.assets as assets_module

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
        sim=SimpleNamespace(physics_manager=_FakeNewtonManager),
    )
    term = physx_material(cfg, env)

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

    monkeypatch.setattr(newton_assets_module, "Articulation", _FakeNewtonArticulation)
    _FakeNewtonManager.notifications.clear()
    asset = _FakeNewtonArticulation(body_ordering)
    asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
    cfg = SimpleNamespace(
        params={
            "friction_range": (0.4, 0.4),
            "restitution_range": (0.1, 0.1),
        }
    )
    cfg.params["asset_cfg"] = asset_cfg
    env = SimpleNamespace(
        scene=_FakeScene(robot=asset),
        num_envs=_NUM_ENVS,
        device="cpu",
        sim=SimpleNamespace(physics_manager=_FakeNewtonManager),
    )
    term = newton_material(cfg, env)

    term(
        env,
        torch.tensor([0], dtype=torch.int32),
        friction_range=(0.4, 0.4),
        restitution_range=(0.1, 0.1),
        asset_cfg=asset_cfg,
    )

    expected_friction = torch.zeros((_NUM_ENVS, _NUM_SHAPES))
    expected_restitution = torch.zeros_like(expected_friction)
    expected_friction[0, expected_shape_slice] = 0.4
    expected_restitution[0, expected_shape_slice] = 0.1
    torch.testing.assert_close(term._friction_binding, expected_friction)
    torch.testing.assert_close(term._restitution_binding, expected_restitution)
    assert len(_FakeNewtonManager.notifications) == 1

    # Native Newton terms accept curriculum range updates at call time.
    term(env, torch.tensor([1]), (0.6, 0.6), (0.3, 0.3), asset_cfg)
    expected_friction[1, expected_shape_slice] = 0.6
    expected_restitution[1, expected_shape_slice] = 0.3
    torch.testing.assert_close(term._friction_binding, expected_friction)
    torch.testing.assert_close(term._restitution_binding, expected_restitution)


@pytest.mark.parametrize("env_ids", [torch.tensor([0], dtype=torch.int32), slice(0, 1)])
def test_newton_joint_parameter_randomization_writes_static_and_viscous_friction(env_ids):
    """Newton randomization writes static friction and passive viscous damping separately."""
    asset = _FakeNewtonArticulation(body_ordering=None)
    asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    cfg = SimpleNamespace(
        params={
            "asset_cfg": asset_cfg,
            "operation": "abs",
            "friction_distribution_params": (0.5, 0.5),
        }
    )
    # the term does not read the physics backend; the asset API decides what it supports
    env = SimpleNamespace(scene=_FakeScene(robot=asset))

    term = events_module.randomize_joint_parameters(cfg, env)
    term(
        env,
        env_ids,
        asset_cfg,
        friction_distribution_params=(0.5, 0.5),
    )

    assert len(asset.static_friction_writes) == 1
    assert len(asset.viscous_friction_writes) == 1
    static_write = asset.static_friction_writes[0]
    viscous_write = asset.viscous_friction_writes[0]
    assert set(static_write) == {"joint_friction_coeff", "joint_ids", "env_ids"}
    assert set(viscous_write) == {"joint_viscous_friction_coeff", "joint_ids", "env_ids"}
    torch.testing.assert_close(static_write["joint_friction_coeff"], torch.full((1, 2), 0.5))
    torch.testing.assert_close(viscous_write["joint_viscous_friction_coeff"], torch.full((1, 2), 0.5))
    assert static_write["env_ids"] is env_ids
    assert viscous_write["env_ids"] is env_ids


def test_fixed_tendon_randomization_writes_limit_stiffness_and_rest_length():
    """Fixed tendon randomization writes every requested property through the asset setters."""
    tendon_values = torch.zeros((_NUM_ENVS, 2))
    writes = {}
    asset = SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(
            fixed_tendon_limit_stiffness=SimpleNamespace(torch=tendon_values.clone()),
            fixed_tendon_rest_length=SimpleNamespace(torch=tendon_values.clone()),
        ),
        set_fixed_tendon_limit_stiffness_index=lambda **kwargs: writes.update(limit_stiffness=kwargs),
        set_fixed_tendon_rest_length_index=lambda **kwargs: writes.update(rest_length=kwargs),
        write_fixed_tendon_properties_to_sim_index=lambda **kwargs: writes.update(sim=kwargs),
    )
    asset_cfg = SimpleNamespace(name="robot", fixed_tendon_ids=slice(None))
    cfg = SimpleNamespace(params={"asset_cfg": asset_cfg, "operation": "abs"})
    env = SimpleNamespace(scene=_FakeScene(robot=asset))

    term = events_module.randomize_fixed_tendon_parameters(cfg, env)
    term(
        env,
        torch.tensor([0]),
        asset_cfg,
        limit_stiffness_distribution_params=(2.0, 2.0),
        rest_length_distribution_params=(0.5, 0.5),
    )

    torch.testing.assert_close(writes["limit_stiffness"]["limit_stiffness"], torch.full((1, 2), 2.0))
    torch.testing.assert_close(writes["rest_length"]["rest_length"], torch.full((1, 2), 0.5))
    assert "sim" in writes


def test_newton_collider_parameters_preserve_unselected_environments(deterministic_material_sampling):
    """Native gap sampling must not subtract margin or write other environments."""
    from isaaclab_newton.envs.mdp import randomize_rigid_body_collider_parameters

    asset = _FakeNewtonArticulation(None)
    asset._root_view.attributes.update(
        shape_margin=torch.full((_NUM_ENVS, 1, _NUM_SHAPES), 0.1),
        shape_gap=torch.full((_NUM_ENVS, 1, _NUM_SHAPES), 0.2),
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))
    env = SimpleNamespace(scene=_FakeScene(robot=asset), sim=SimpleNamespace(physics_manager=_FakeNewtonManager))
    term = randomize_rigid_body_collider_parameters(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)
    term(env, torch.tensor([1]), asset_cfg, (0.3, 0.3), (0.4, 0.4))
    expected_margin = torch.tensor([0.1, 0.3])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    expected_gap = torch.tensor([0.2, 0.4])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    torch.testing.assert_close(asset._root_view.attributes["shape_margin"], expected_margin)
    torch.testing.assert_close(asset._root_view.attributes["shape_gap"], expected_gap)
    term(env, slice(0, 1), asset_cfg, gap_distribution_params=(0.5, 0.5))
    torch.testing.assert_close(asset._root_view.attributes["shape_margin"], expected_margin)
    expected_gap = torch.tensor([0.5, 0.4])[:, None, None].expand(_NUM_ENVS, 1, _NUM_SHAPES)
    torch.testing.assert_close(asset._root_view.attributes["shape_gap"], expected_gap)
