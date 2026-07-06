# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Pure-Python tests for named Newton coupled-solver configuration.

The tests use small model fakes and never start Isaac Sim. They exercise the
selector, ownership, proxy, and ADMM translation performed before Newton
constructs the coupled solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_newton.physics import KaminoSolverCfg, XPBDSolverCfg
from newton import ShapeFlags
from newton.solvers import SolverKamino
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.coupling import coupled_manager
from isaaclab_contrib.coupling.coupled_manager import NewtonCoupledSolverManager
from isaaclab_contrib.coupling.coupled_manager_cfg import (
    CoupledAdmmSolverCfg,
    CoupledProxyCfg,
    CoupledProxySolverCfg,
    CoupledSolverEntryCfg,
)


@dataclass
class _FakeArray:
    """Minimal mutable stand-in for a Warp array."""

    data: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def assign(self, values: np.ndarray) -> None:
        self.data = np.asarray(values).copy()


@dataclass
class _FakeModel:
    """Fields consulted by the manager's pure configuration helpers."""

    body_count: int = 3
    body_label: list[str] = field(
        default_factory=lambda: [
            "/World/envs/env_0/Robot/base",
            "/World/envs/env_0/Robot/hand",
            "/World/envs/env_0/Object/body",
        ]
    )
    joint_count: int = 2
    joint_child: _FakeArray = field(default_factory=lambda: _FakeArray(np.asarray([1, 2], dtype=np.int32)))
    joint_parent: _FakeArray = field(default_factory=lambda: _FakeArray(np.asarray([0, -1], dtype=np.int32)))
    shape_count: int = 4
    shape_body: _FakeArray = field(default_factory=lambda: _FakeArray(np.asarray([0, 1, 2, -1], dtype=np.int32)))
    shape_flags: _FakeArray = field(
        default_factory=lambda: _FakeArray(np.full(4, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32))
    )
    shape_label: list[str] = field(
        default_factory=lambda: [
            "/World/envs/env_0/Robot/base_collision",
            "/World/envs/env_0/Robot/hand_collision",
            "/World/envs/env_0/Object/object_collision",
            "/World/ground",
        ]
    )
    particle_count: int = 3
    shape_material_ke: _FakeArray = field(default_factory=lambda: _float_array(4, 1.0))
    shape_material_kd: _FakeArray = field(default_factory=lambda: _float_array(4, 2.0))
    shape_material_mu: _FakeArray = field(default_factory=lambda: _float_array(4, 3.0))
    shape_margin: _FakeArray = field(default_factory=lambda: _float_array(4, 4.0))
    shape_gap: _FakeArray = field(default_factory=lambda: _float_array(4, 5.0))


def _float_array(size: int, value: float) -> _FakeArray:
    return _FakeArray(np.full(size, value, dtype=np.float32))


@dataclass
class _FakeAsset:
    prim_path: str


@dataclass
class _FakeSceneCfg:
    robot: _FakeAsset = field(default_factory=lambda: _FakeAsset("/World/envs/env_.*/Robot"))
    object: _FakeAsset = field(default_factory=lambda: _FakeAsset("/World/envs/env_.*/Object"))


def _entry(
    name: str,
    *,
    bodies: list[int] | None = None,
    particles: list[int] | None = None,
) -> NewtonCoupledSolverManager._ResolvedEntry:
    """Build an already-resolved entry for validation tests."""
    return NewtonCoupledSolverManager._ResolvedEntry(
        config=CoupledSolverEntryCfg(name=name, solver_cfg=XPBDSolverCfg()),
        bodies=list(bodies or []),
        particles=list(particles or []),
        joints=[],
        shapes=[],
    )


def _valid_proxy_setup(
    *, proxy: CoupledProxyCfg | None = None
) -> tuple[
    CoupledProxySolverCfg,
    list[NewtonCoupledSolverManager._ResolvedEntry],
    list[NewtonCoupledSolverManager._ResolvedProxy],
]:
    """Build a complete two-entry proxy configuration."""
    entries = [
        _entry("rigid", bodies=[0, 1], particles=[0]),
        _entry("soft", bodies=[2], particles=[1, 2]),
    ]
    proxy_cfg = proxy or CoupledProxyCfg(source="rigid", destination="soft", bodies=[1])
    return (
        CoupledProxySolverCfg(
            entries=[entry.config for entry in entries],
            proxies=[proxy_cfg],
        ),
        entries,
        [
            NewtonCoupledSolverManager._ResolvedProxy(
                config=proxy_cfg,
                bodies=[int(body) for body in proxy_cfg.bodies],
                particles=list(proxy_cfg.particles),
            )
        ],
    )


def test_scene_entity_and_string_selectors_resolve_full_body_labels():
    """Scene selectors filter short names while strings match full labels."""
    model = _FakeModel()
    scene_cfg = _FakeSceneCfg()

    assert NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        SceneEntityCfg("robot", body_names=["hand"]),
        scene_cfg,
        "entry 'rigid'",
    ) == [1]
    assert NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        "/World/envs/env_.*/Object/body",
        None,
        "entry 'object'",
    ) == [2]


def test_scene_entity_selector_reports_unmatched_body_pattern():
    with pytest.raises(ValueError, match="could not match body patterns"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            _FakeModel(),
            SceneEntityCfg("robot", body_names=["missing"]),
            _FakeSceneCfg(),
            "entry 'rigid'",
        )


def test_raw_body_label_selector_reports_no_matches():
    with pytest.raises(ValueError, match="matched no Newton bodies"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            _FakeModel(),
            "/World/envs/env_.*/Missing",
            None,
            "entry 'missing'",
        )


def test_three_named_entries_partition_bodies_joints_shapes_and_particles():
    """Derived joint/shape ownership follows each entry's selected bodies."""
    model = _FakeModel()
    scene_cfg = _FakeSceneCfg()
    entries = [
        CoupledSolverEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot")],
        ),
        CoupledSolverEntryCfg(
            name="object",
            solver_cfg=XPBDSolverCfg(),
            bodies=["/World/envs/env_.*/Object"],
            all_particles=True,
        ),
        CoupledSolverEntryCfg(
            name="world",
            solver_cfg=XPBDSolverCfg(),
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCoupledSolverManager._resolve_entry(model, entry, scene_cfg) for entry in entries]

    assert resolved[0].bodies == [0, 1]
    assert resolved[0].joints == [0]
    assert resolved[0].shapes == [0, 1]
    assert resolved[1].bodies == [2]
    assert resolved[1].joints == [1]
    assert resolved[1].shapes == [2]
    assert resolved[1].particles == [0, 1, 2]
    assert resolved[2].bodies == []
    assert resolved[2].joints == []
    assert resolved[2].shapes == [3]
    assert isinstance(entries[0].bodies[0], SceneEntityCfg)
    assert entries[1].bodies == ["/World/envs/env_.*/Object"]

    cfg = CoupledAdmmSolverCfg(entries=entries)
    NewtonCoupledSolverManager._validate_solver_cfg(model, cfg, resolved)


def test_cross_entry_joint_is_left_unowned_for_admm_attachment():
    """A joint spanning two entries must remain visible to the ADMM coupler."""
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    entries = [
        CoupledSolverEntryCfg(
            name="robot",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot")],
        ),
        CoupledSolverEntryCfg(
            name="object",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("object")],
            all_particles=True,
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCoupledSolverManager._resolve_entry(model, entry, _FakeSceneCfg()) for entry in entries]

    assert resolved[0].joints == [0]
    assert resolved[1].joints == []


def test_proxy_validation_rejects_cross_entry_joint():
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    cfg, entries, proxies = _valid_proxy_setup()

    with pytest.raises(ValueError, match="does not support cross-entry joint"):
        NewtonCoupledSolverManager._validate_solver_cfg(model, cfg, entries, proxies)


def test_explicit_particle_ownership_accepts_a_complete_partition():
    model = _FakeModel()
    entries = [
        _entry("a", bodies=[0, 1], particles=[0, 2]),
        _entry("b", bodies=[2], particles=[1]),
    ]
    cfg = CoupledAdmmSolverCfg(entries=[entry.config for entry in entries])
    NewtonCoupledSolverManager._validate_solver_cfg(model, cfg, entries)


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        (
            [_entry("a", bodies=[0, 1], particles=[0]), _entry("b", bodies=[1, 2], particles=[1, 2])],
            "bodies index 1 is owned by both",
        ),
        (
            [_entry("a", bodies=[0], particles=[0]), _entry("b", bodies=[1], particles=[1, 2])],
            "unclaimed bodies",
        ),
        (
            [_entry("a", bodies=[0, 1], particles=[0, 1]), _entry("b", bodies=[2], particles=[1, 2])],
            "particles index 1 is owned by both",
        ),
        (
            [_entry("a", bodies=[0, 1], particles=[0]), _entry("b", bodies=[2], particles=[1])],
            "unclaimed particles",
        ),
    ],
)
def test_validation_rejects_duplicate_or_unclaimed_body_and_particle_ownership(entries, match):
    cfg = CoupledAdmmSolverCfg(entries=[entry.config for entry in entries])
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries)


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ([_entry("a", bodies=[0, 1], particles=[0, 1, 2])], "at least two named entries"),
        (
            [
                _entry("", bodies=[0, 1], particles=[0]),
                _entry("b", bodies=[2], particles=[1, 2]),
            ],
            "name must be non-empty",
        ),
        (
            [
                _entry("same", bodies=[0, 1], particles=[0]),
                _entry("same", bodies=[2], particles=[1, 2]),
            ],
            "names must be unique",
        ),
        (
            [
                _entry("a", bodies=[0, 1, 3], particles=[0]),
                _entry("b", bodies=[2], particles=[1, 2]),
            ],
            "out-of-range bodies index 3",
        ),
        (
            [
                _entry("a", bodies=[0, 1], particles=[0, 3]),
                _entry("b", bodies=[2], particles=[1, 2]),
            ],
            "out-of-range particles index 3",
        ),
    ],
)
def test_validation_rejects_invalid_entry_names_counts_and_indices(entries, match):
    cfg = CoupledAdmmSolverCfg(entries=[entry.config for entry in entries])
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries)


def test_shape_label_patterns_and_static_shape_selection_are_additive():
    entry = NewtonCoupledSolverManager._resolve_entry(
        _FakeModel(),
        CoupledSolverEntryCfg(
            name="special",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot", body_names=["base"])],
            include_body_shapes=False,
            include_static_shapes=True,
            shape_label_patterns=[r".*/Object/object_collision"],
        ),
        _FakeSceneCfg(),
    )
    assert entry.bodies == [0]
    assert entry.shapes == [3, 2]


@pytest.mark.parametrize(
    ("proxy", "match"),
    [
        (CoupledProxyCfg(source="missing", destination="soft", bodies=[0]), "must name coupled entries"),
        (CoupledProxyCfg(source="rigid", destination="rigid", bodies=[0]), "must differ"),
        (CoupledProxyCfg(source="rigid", destination="soft", bodies=[2]), "owned by its source"),
        (CoupledProxyCfg(source="rigid", destination="soft", particles=[2]), "owned by its source"),
    ],
)
def test_proxy_validation_checks_endpoints_and_source_ownership(proxy, match):
    cfg, entries, proxies = _valid_proxy_setup(proxy=proxy)
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries, proxies)


@pytest.mark.parametrize(
    ("proxy", "match"),
    [
        (CoupledProxyCfg(source="rigid", destination="soft"), "map at least one body or particle"),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], mode="invalid"),
            "mode must be 'lagged' or 'staggered'",
        ),
        (CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], mass_scale=0.0), "mass_scale must be > 0"),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], collide_interval=0),
            "collide_interval must be >= 1",
        ),
    ],
)
def test_proxy_validation_rejects_invalid_mapping_options(proxy, match):
    cfg, entries, proxies = _valid_proxy_setup(proxy=proxy)
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries, proxies)


def test_proxy_validation_rejects_more_than_two_entries():
    entries = [
        _entry("a", bodies=[0], particles=[0]),
        _entry("b", bodies=[1], particles=[1]),
        _entry("c", bodies=[2], particles=[2]),
    ]
    proxy_cfg = CoupledProxyCfg(source="a", destination="b", bodies=[0])
    cfg = CoupledProxySolverCfg(
        entries=[entry.config for entry in entries],
        proxies=[proxy_cfg],
    )
    proxies = [NewtonCoupledSolverManager._ResolvedProxy(proxy_cfg, bodies=[0], particles=[])]
    with pytest.raises(ValueError, match="at most two solver entries"):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries, proxies)


def test_proxy_resolution_keeps_only_collidable_selected_bodies():
    model = _FakeModel()
    model.shape_flags = _FakeArray(
        np.asarray([int(ShapeFlags.COLLIDE_SHAPES), 0, int(ShapeFlags.COLLIDE_SHAPES), 0], dtype=np.int32)
    )
    proxy = NewtonCoupledSolverManager._resolve_proxy(
        model,
        CoupledProxyCfg(source="rigid", destination="soft", bodies=[SceneEntityCfg("robot")]),
        _FakeSceneCfg(),
    )
    assert proxy.bodies == [0]


class _RecordingProxy:
    """Capture proxy construction while retaining Newton's config dataclasses."""

    Proxy = SolverCoupledProxy.Proxy
    Config = SolverCoupledProxy.Config

    def __init__(self, *, model, entries, coupling):
        self.model = model
        self.entries = entries
        self.coupling = coupling


def test_proxy_build_uses_custom_and_default_collision_pipelines(monkeypatch):
    def custom_pipeline(model_view):
        return model_view

    model = _FakeModel()
    resolved_entries = [
        _entry("rigid", bodies=[0, 1]),
        _entry("soft", bodies=[2], particles=[0, 1, 2]),
    ]
    cfg = CoupledProxySolverCfg(
        entries=[entry.config for entry in resolved_entries],
        proxies=[
            CoupledProxyCfg(
                source="rigid",
                destination="soft",
                bodies=["/World/envs/env_.*/Robot/base"],
                collision_pipeline_factory=custom_pipeline,
            ),
            CoupledProxyCfg(source="soft", destination="rigid", particles=[0]),
        ],
        iterations=3,
    )
    monkeypatch.setattr(coupled_manager, "SolverCoupledProxy", _RecordingProxy)
    monkeypatch.setattr(
        coupled_manager,
        "CollisionPipeline",
        lambda model_view, *, broad_phase: (model_view, broad_phase),
    )

    solver = NewtonCoupledSolverManager._build_proxy_coupled_solver(model, [], cfg, resolved_entries, _FakeSceneCfg())

    assert solver.coupling.iterations == 3
    assert solver.coupling.proxies[0].collision_pipeline is custom_pipeline
    assert solver.coupling.proxies[1].collision_pipeline("soft-view") == ("soft-view", "explicit")


def test_proxy_shape_overrides_apply_only_to_selected_body_shapes():
    model = _FakeModel()
    proxy = CoupledProxyCfg(
        source="rigid",
        destination="soft",
        bodies=[1],
        shape_material_ke=10.0,
        shape_material_kd=20.0,
        shape_material_mu=0.75,
        shape_margin=0.015,
        shape_gap=0.002,
    )

    resolved_proxy = NewtonCoupledSolverManager._ResolvedProxy(config=proxy, bodies=[1], particles=[])
    NewtonCoupledSolverManager._apply_proxy_shape_overrides(model, [resolved_proxy])

    np.testing.assert_allclose(model.shape_material_ke.data, [1.0, 10.0, 1.0, 1.0])
    np.testing.assert_allclose(model.shape_material_kd.data, [2.0, 20.0, 2.0, 2.0])
    np.testing.assert_allclose(model.shape_material_mu.data, [3.0, 0.75, 3.0, 3.0])
    np.testing.assert_allclose(model.shape_margin.data, [4.0, 0.015, 4.0, 4.0])
    np.testing.assert_allclose(model.shape_gap.data, [5.0, 0.002, 5.0, 5.0])


def test_entry_build_constructs_registered_solver_with_model_view(monkeypatch):
    class _RecordingSolver:
        def __init__(self, model):
            self.model = model

    monkeypatch.setitem(NewtonCoupledSolverManager._SOLVER_CLASS_BY_CFG_TYPE, XPBDSolverCfg, _RecordingSolver)
    entry = NewtonCoupledSolverManager._ResolvedEntry(
        config=CoupledSolverEntryCfg(
            name="entry",
            solver_cfg=XPBDSolverCfg(),
        ),
        bodies=[],
        particles=[],
        joints=[],
        shapes=[],
    )

    solver_entry = NewtonCoupledSolverManager._build_entry(entry)
    solver = solver_entry.solver("entry-view")

    assert solver.model == "entry-view"


def test_kamino_solver_config_remains_supported():
    assert NewtonCoupledSolverManager._resolve_solver_class(KaminoSolverCfg()) is SolverKamino


def test_algorithms_select_expected_outer_collision_pipeline(monkeypatch):
    """Proxy owns destination contacts while ADMM uses the shared outer pipeline."""
    model = _FakeModel()
    proxy_cfg, resolved_entries, _ = _valid_proxy_setup()
    recorded_entries: list[str] = []

    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_resolve_entry",
        classmethod(
            lambda cls, model, entry_cfg, scene_cfg: next(
                entry for entry in resolved_entries if entry.config.name == entry_cfg.name
            )
        ),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_build_entry",
        classmethod(lambda cls, entry: recorded_entries.append(entry.config.name) or entry.config.name),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_build_proxy_coupled_solver",
        classmethod(lambda cls, model, entries, cfg, resolved_entries, scene_cfg: SimpleNamespace(kind="proxy")),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_build_admm_coupled_solver",
        classmethod(lambda cls, model, entries, cfg, resolved_entries: SimpleNamespace(kind="admm")),
    )
    old_solver = coupled_manager.NewtonManager._solver
    old_outer = coupled_manager.NewtonManager._needs_collision_pipeline
    old_contact_support = coupled_manager.NewtonManager._supports_contact_sensors
    old_report_contacts = coupled_manager.NewtonManager._report_contacts
    try:
        NewtonCoupledSolverManager._build_solver(model, proxy_cfg)
        assert coupled_manager.NewtonManager._needs_collision_pipeline is False
        assert coupled_manager.NewtonManager._supports_contact_sensors is False
        assert recorded_entries == ["rigid", "soft"]

        recorded_entries.clear()
        admm_cfg = CoupledAdmmSolverCfg(entries=proxy_cfg.entries)
        NewtonCoupledSolverManager._build_solver(model, admm_cfg)
        assert coupled_manager.NewtonManager._needs_collision_pipeline is True
        assert recorded_entries == ["rigid", "soft"]

        coupled_manager.NewtonManager._report_contacts = True
        with pytest.raises(NotImplementedError, match="contact sensors"):
            NewtonCoupledSolverManager._build_solver(model, proxy_cfg)
    finally:
        coupled_manager.NewtonManager._solver = old_solver
        coupled_manager.NewtonManager._needs_collision_pipeline = old_outer
        coupled_manager.NewtonManager._supports_contact_sensors = old_contact_support
        coupled_manager.NewtonManager._report_contacts = old_report_contacts


class _RecordingAdmm:
    """Capture ADMM construction while retaining Newton's config dataclasses."""

    ContactPair = SolverCoupledADMM.ContactPair
    Config = SolverCoupledADMM.Config

    @classmethod
    def auto_detect_contact_pairs(cls, entries):
        return SolverCoupledADMM.auto_detect_contact_pairs(entries)

    def __init__(self, *, model, entries, coupling):
        self.model = model
        self.entries = entries
        self.coupling = coupling


def test_admm_build_forwards_multiple_pairs_matching_and_proximal_options(monkeypatch):
    model = _FakeModel()
    resolved_entries = [
        _entry("robot", bodies=[0], particles=[0]),
        _entry("object", bodies=[1], particles=[1]),
        _entry("world", bodies=[2], particles=[2]),
    ]
    cfg = CoupledAdmmSolverCfg(
        entries=[entry.config for entry in resolved_entries],
        contact_pairs=[
            ("robot", "object"),
            ("object", "world"),
        ],
        iterations=7,
        joint_proximal_bodies=False,
        joint_proximal_destination_entries=["object", "world"],
        joint_proximal_mass_scale=0.25,
        rigid_contact_matching="latest",
        contact_matching_pos_threshold=0.01,
        contact_matching_normal_dot_threshold=0.8,
        contact_matching_force_scale=0.7,
    )
    monkeypatch.setattr(coupled_manager, "SolverCoupledADMM", _RecordingAdmm)

    solver = NewtonCoupledSolverManager._build_admm_coupled_solver(model, [], cfg, resolved_entries)

    assert [(pair.source, pair.destination) for pair in solver.coupling.contact_pairs] == [
        ("robot", "object"),
        ("object", "world"),
    ]
    assert solver.coupling.iterations == 7
    assert solver.coupling.joint_proximal_bodies is False
    assert solver.coupling.joint_proximal_destination_entries == ["object", "world"]
    assert solver.coupling.joint_proximal_mass_scale == pytest.approx(0.25)
    assert solver.coupling.rigid_contact_matching == "latest"
    assert solver.coupling.contact_matching_pos_threshold == pytest.approx(0.01)
    assert solver.coupling.contact_matching_normal_dot_threshold == pytest.approx(0.8)
    assert solver.coupling.contact_matching_force_scale == pytest.approx(0.7)


def test_admm_build_auto_detects_symmetric_contact_pairs_by_default(monkeypatch):
    model = _FakeModel()
    resolved_entries = [
        _entry("a", bodies=[0], particles=[0]),
        _entry("b", bodies=[1], particles=[1]),
        _entry("c", bodies=[2], particles=[2]),
    ]
    entries = [SimpleNamespace(name=entry.config.name) for entry in resolved_entries]
    cfg = CoupledAdmmSolverCfg(entries=[entry.config for entry in resolved_entries])
    monkeypatch.setattr(coupled_manager, "SolverCoupledADMM", _RecordingAdmm)

    solver = NewtonCoupledSolverManager._build_admm_coupled_solver(model, entries, cfg, resolved_entries)

    assert [(pair.source, pair.destination) for pair in solver.coupling.contact_pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]

    cfg.contact_pairs = []
    solver = NewtonCoupledSolverManager._build_admm_coupled_solver(model, entries, cfg, resolved_entries)
    assert list(solver.coupling.contact_pairs) == []


@pytest.mark.parametrize(
    ("pair", "match"),
    [
        (("missing", "b"), "must name coupled entries"),
        (("a", "a"), "source and destination must differ"),
    ],
)
def test_admm_validation_rejects_invalid_contact_pairs(pair, match):
    entries = [
        _entry("a", bodies=[0, 1], particles=[0]),
        _entry("b", bodies=[2], particles=[1, 2]),
    ]
    cfg = CoupledAdmmSolverCfg(
        entries=[entry.config for entry in entries],
        contact_pairs=[pair],
    )
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries)


def test_admm_validation_rejects_duplicate_symmetric_contact_pairs():
    entries = [
        _entry("a", bodies=[0, 1], particles=[0]),
        _entry("b", bodies=[2], particles=[1, 2]),
    ]
    cfg = CoupledAdmmSolverCfg(
        entries=[entry.config for entry in entries],
        contact_pairs=[("a", "b"), ("b", "a")],
    )

    with pytest.raises(ValueError, match="Duplicate symmetric"):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg, entries)
