# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Pure-Python tests for the named Newton coupler.

The tests use small model fakes and never start Isaac Sim. They exercise the
selector, ownership, proxy, and ADMM translation performed before Newton
constructs the coupled solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    XPBDSolverCfg,
)
from newton import ShapeFlags
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
    coupler,
)
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg


@dataclass
class _FakeArray:
    """Minimal mutable stand-in for a Warp array."""

    data: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def assign(self, values: np.ndarray) -> None:
        self.data = np.asarray(values).copy()


def test_public_coupler_config_resolves_renamed_class():
    """The package-level config points at the renamed coupler implementation."""
    assert CouplerCfg().class_type.__name__ == "NewtonCouplerManager"


@dataclass
class _FakeModel:
    """Fields consulted by the coupler's pure configuration helpers."""

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
) -> NewtonCouplerManager._ResolvedEntry:
    """Build an already-resolved entry for validation tests."""
    return NewtonCouplerManager._ResolvedEntry(
        config=CouplerEntryCfg(name=name, solver_cfg=XPBDSolverCfg()),
        bodies=list(bodies or []),
        particles=list(particles or []),
        joints=[],
        shapes=[],
    )


def _valid_proxy_setup(
    *, proxy: CouplerProxyMappingCfg | None = None
) -> tuple[
    CouplerProxyCfg,
    list[NewtonCouplerManager._ResolvedEntry],
    list[CouplerProxyMappingCfg],
]:
    """Build a complete two-entry proxy configuration."""
    entries = [
        _entry("rigid", bodies=[0, 1], particles=[0]),
        _entry("soft", bodies=[2], particles=[1, 2]),
    ]
    proxy_cfg = proxy or CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=[1])
    return (
        CouplerProxyCfg(
            entries=[entry.config for entry in entries],
            proxies=[proxy_cfg],
        ),
        entries,
        [proxy_cfg],
    )


def test_scene_entity_and_string_selectors_resolve_full_body_labels():
    """Scene selectors filter short names while strings match full labels."""
    model = _FakeModel()
    scene_cfg = _FakeSceneCfg()

    assert NewtonCouplerManager._resolve_entities_to_body_ids(
        model,
        [SceneEntityCfg("robot", body_names=["hand"])],
        scene_cfg,
        "entry 'rigid'",
    ) == [1]
    assert NewtonCouplerManager._resolve_entities_to_body_ids(
        model,
        ["/World/envs/env_.*/Object/body"],
        None,
        "entry 'object'",
    ) == [2]


def test_scene_entity_selector_reports_unmatched_body_pattern():
    with pytest.raises(ValueError, match="could not match body patterns"):
        NewtonCouplerManager._resolve_entities_to_body_ids(
            _FakeModel(),
            [SceneEntityCfg("robot", body_names=["missing"])],
            _FakeSceneCfg(),
            "entry 'rigid'",
        )


def test_raw_body_label_selector_reports_no_matches():
    with pytest.raises(ValueError, match="matched no Newton bodies"):
        NewtonCouplerManager._resolve_entities_to_body_ids(
            _FakeModel(),
            ["/World/envs/env_.*/Missing"],
            None,
            "entry 'missing'",
        )


def test_raw_body_id_selectors_pass_through_and_dedupe():
    """Integer selectors resolve to themselves, preserving order and removing duplicates."""
    assert NewtonCouplerManager._resolve_entities_to_body_ids(_FakeModel(), [2, 0, 2], None, "proxy 'a'->'b'") == [2, 0]


def test_out_of_range_body_id_selector_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        NewtonCouplerManager._resolve_entities_to_body_ids(_FakeModel(), [5], None, "proxy 'a'->'b'")


def test_proxy_resolution_writes_body_ids_into_config_in_place():
    """Resolution replaces the proxy's selectors with body ids and is idempotent."""
    model = _FakeModel()
    proxy = CouplerProxyMappingCfg(
        source="rigid", destination="soft", bodies=[SceneEntityCfg("robot")], particles=[1, 1]
    )

    resolved = NewtonCouplerManager._resolve_proxy(model, proxy, _FakeSceneCfg())

    assert resolved is proxy
    assert proxy.bodies == [0, 1]
    assert proxy.particles == [1]
    # Re-resolving the now-integer selectors yields the same result.
    assert NewtonCouplerManager._resolve_proxy(model, proxy, None).bodies == [0, 1]


def test_three_named_entries_partition_bodies_joints_shapes_and_particles():
    """Derived joint/shape ownership follows each entry's selected bodies."""
    model = _FakeModel()
    scene_cfg = _FakeSceneCfg()
    entries = [
        CouplerEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot")],
        ),
        CouplerEntryCfg(
            name="object",
            solver_cfg=XPBDSolverCfg(),
            bodies=["/World/envs/env_.*/Object"],
            all_particles=True,
        ),
        CouplerEntryCfg(
            name="world",
            solver_cfg=XPBDSolverCfg(),
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCouplerManager._resolve_entry(model, entry, scene_cfg) for entry in entries]

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


def test_cross_entry_joint_is_left_unowned_for_admm_attachment():
    """A joint spanning two entries must remain visible to the ADMM coupler."""
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    entries = [
        CouplerEntryCfg(
            name="robot",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot")],
        ),
        CouplerEntryCfg(
            name="object",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("object")],
            all_particles=True,
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCouplerManager._resolve_entry(model, entry, _FakeSceneCfg()) for entry in entries]

    assert resolved[0].joints == [0]
    assert resolved[1].joints == []


def test_proxy_validation_rejects_cross_entry_joint():
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    cfg, entries, proxies = _valid_proxy_setup()

    with pytest.raises(ValueError, match="does not support cross-entry joint"):
        NewtonCouplerManager._validate_no_cross_entry_proxy_joints(model, {entry.config.name: entry for entry in entries})


def test_shape_label_patterns_and_static_shape_selection_are_additive():
    entry = NewtonCouplerManager._resolve_entry(
        _FakeModel(),
        CouplerEntryCfg(
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


def test_proxy_resolution_keeps_only_collidable_selected_bodies():
    model = _FakeModel()
    model.shape_flags = _FakeArray(
        np.asarray([int(ShapeFlags.COLLIDE_SHAPES), 0, int(ShapeFlags.COLLIDE_SHAPES), 0], dtype=np.int32)
    )
    proxy = NewtonCouplerManager._resolve_proxy(
        model,
        CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=[SceneEntityCfg("robot")]),
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
    cfg = CouplerProxyCfg(
        entries=[entry.config for entry in resolved_entries],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="soft",
                bodies=["/World/envs/env_.*/Robot/base"],
                collision_pipeline=custom_pipeline,
            ),
            CouplerProxyMappingCfg(source="soft", destination="rigid", particles=[0]),
        ],
        iterations=3,
    )
    monkeypatch.setattr(coupler, "SolverCoupledProxy", _RecordingProxy)
    monkeypatch.setattr(
        coupler,
        "CollisionPipeline",
        lambda model_view, *, broad_phase: (model_view, broad_phase),
    )

    proxies = [NewtonCouplerManager._resolve_proxy(model, proxy, _FakeSceneCfg()) for proxy in cfg.proxies]
    solver = NewtonCouplerManager._build_proxy_coupled_solver(model, [], proxies, cfg)

    assert solver.coupling.iterations == 3
    assert solver.coupling.proxies[0].collision_pipeline is custom_pipeline
    assert solver.coupling.proxies[1].collision_pipeline("soft-view") == ("soft-view", "explicit")


def test_proxy_shape_overrides_apply_only_to_selected_body_shapes():
    model = _FakeModel()
    proxy = CouplerProxyMappingCfg(
        source="rigid",
        destination="soft",
        bodies=[1],
        shape_material_ke=10.0,
        shape_material_kd=20.0,
        shape_material_mu=0.75,
        shape_margin=0.015,
        shape_gap=0.002,
    )

    NewtonCouplerManager._apply_proxy_shape_overrides(model, [proxy])

    np.testing.assert_allclose(model.shape_material_ke.data, [1.0, 10.0, 1.0, 1.0])
    np.testing.assert_allclose(model.shape_material_kd.data, [2.0, 20.0, 2.0, 2.0])
    np.testing.assert_allclose(model.shape_material_mu.data, [3.0, 0.75, 3.0, 3.0])
    np.testing.assert_allclose(model.shape_margin.data, [4.0, 0.015, 4.0, 4.0])
    np.testing.assert_allclose(model.shape_gap.data, [5.0, 0.002, 5.0, 5.0])


def test_entry_build_uses_solver_config_class_type():
    class _RecordingManager:
        @classmethod
        def _create_solver(cls, model, solver_cfg):
            return SimpleNamespace(model=model, solver_cfg=solver_cfg)

    solver_cfg = XPBDSolverCfg()
    solver_cfg.class_type = _RecordingManager
    entry = NewtonCouplerManager._ResolvedEntry(
        config=CouplerEntryCfg(
            name="entry",
            solver_cfg=solver_cfg,
        ),
        bodies=[],
        particles=[],
        joints=[],
        shapes=[],
    )

    solver_entry = NewtonCouplerManager._build_entry(entry)
    solver = solver_entry.solver("entry-view")

    assert solver.model == "entry-view"
    assert solver.solver_cfg is entry.config.solver_cfg


@pytest.mark.parametrize(
    "solver_cfg",
    [
        MJWarpSolverCfg(),
        XPBDSolverCfg(),
        FeatherstoneSolverCfg(),
        KaminoSolverCfg(),
        MPMSolverCfg(),
        VBDSolverCfg(),
    ],
)
def test_solver_config_manager_exposes_nested_factory(solver_cfg):
    assert callable(solver_cfg.class_type._create_solver)


def test_mpm_entry_forwards_config_and_execution_policy():
    """MPM construction preserves grid configuration, substeps, and in-place stepping."""

    class _RecordingMpmManager:
        @classmethod
        def _create_solver(cls, model, solver_cfg):
            return SimpleNamespace(model=model, solver_cfg=solver_cfg)

    solver_cfg = MPMSolverCfg(grid_type="fixed", max_active_cell_count=256)
    solver_cfg.class_type = _RecordingMpmManager
    entry = NewtonCouplerManager._ResolvedEntry(
        config=CouplerEntryCfg(
            name="media",
            solver_cfg=solver_cfg,
            substeps=2,
            in_place=True,
        ),
        bodies=[],
        particles=[0],
        joints=[],
        shapes=[],
    )

    solver_entry = NewtonCouplerManager._build_entry(entry)
    solver = solver_entry.solver("media-view")

    assert solver.model == "media-view"
    assert solver.solver_cfg.grid_type == "fixed"
    assert solver.solver_cfg.max_active_cell_count == 256
    assert solver_entry.substeps == 2
    assert solver_entry.in_place is True


def test_mpm_entry_reuses_builder_lifecycle_hooks(monkeypatch):
    """Coupled MPM entries register attributes and normalize kinematic colliders."""
    events: list[tuple[str, object]] = []
    builder = object()
    solver_cfg = CouplerProxyCfg(
        entries=[CouplerEntryCfg(name="media", solver_cfg=MPMSolverCfg())],
    )
    monkeypatch.setattr(coupler.PhysicsManager, "_cfg", SimpleNamespace(solver_cfg=solver_cfg))
    monkeypatch.setattr(
        coupler.NewtonMPMManager,
        "_register_builder_attributes",
        classmethod(lambda cls, value: events.append(("register", value))),
    )
    monkeypatch.setattr(
        coupler.NewtonMPMManager,
        "_prepare_builder_for_finalize",
        classmethod(lambda cls, value: events.append(("finalize", value))),
    )

    NewtonCouplerManager._register_builder_attributes(builder)
    NewtonCouplerManager._prepare_builder_for_finalize(builder)

    assert events == [("register", builder), ("finalize", builder)]


def test_contact_initialization_prepares_coupled_solver_buffers(monkeypatch):
    """Entry-local contact buffers are allocated before graph capture."""
    events: list[tuple[str, object | None]] = []
    contacts = object()
    solver = SimpleNamespace(prepare_contacts=lambda value: events.append(("prepare", value)))
    monkeypatch.setattr(
        coupler.NewtonVBDManager,
        "_initialize_contacts",
        classmethod(lambda cls: events.append(("initialize", None))),
    )
    monkeypatch.setattr(coupler.NewtonManager, "_solver", solver)
    monkeypatch.setattr(coupler.NewtonManager, "_contacts", contacts)

    NewtonCouplerManager._initialize_contacts()

    assert events == [("initialize", None), ("prepare", contacts)]


def test_algorithms_select_expected_outer_collision_pipeline(monkeypatch):
    """Proxy sources get outer contacts when their solver requires them."""
    model = _FakeModel()
    proxy_cfg, resolved_entries, resolved_proxies = _valid_proxy_setup()
    proxy_cfg.scene_cfg = _FakeSceneCfg()
    recorded_entries: list[str] = []
    recorded_scene_cfgs: list[object] = []

    monkeypatch.setattr(
        NewtonCouplerManager,
        "_resolve_entry",
        classmethod(
            lambda cls, model, entry_cfg, scene_cfg: recorded_scene_cfgs.append(scene_cfg)
            or next(entry for entry in resolved_entries if entry.config.name == entry_cfg.name)
        ),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_resolve_proxy",
        classmethod(
            lambda cls, model, proxy_cfg, scene_cfg: next(
                p for p in resolved_proxies if (p.source, p.destination) == (proxy_cfg.source, proxy_cfg.destination)
            )
        ),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_build_entry",
        classmethod(lambda cls, entry: recorded_entries.append(entry.config.name) or entry.config.name),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_build_proxy_coupled_solver",
        classmethod(lambda cls, model, entries, proxies, cfg: SimpleNamespace(kind="proxy")),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_build_admm_coupled_solver",
        classmethod(lambda cls, model, entries, cfg: SimpleNamespace(kind="admm")),
    )
    old_solver = coupler.NewtonManager._solver
    old_outer = coupler.NewtonManager._needs_collision_pipeline
    old_contact_support = coupler.NewtonManager._supports_contact_sensors
    old_report_contacts = coupler.NewtonManager._report_contacts
    old_needs_fk = coupler.NewtonManager._needs_fk_before_step
    try:
        resolved_entries[0].config.solver_cfg = MJWarpSolverCfg()
        NewtonCouplerManager._build_solver(model, proxy_cfg)
        assert coupler.NewtonManager._needs_collision_pipeline is False
        assert coupler.NewtonManager._needs_fk_before_step is False
        assert coupler.NewtonManager._supports_contact_sensors is False
        assert recorded_entries == ["rigid", "soft"]
        assert all(sc is proxy_cfg.scene_cfg for sc in recorded_scene_cfgs)

        resolved_entries[0].config.solver_cfg.use_mujoco_contacts = False
        NewtonCouplerManager._build_solver(model, proxy_cfg)
        assert coupler.NewtonManager._needs_collision_pipeline is True

        resolved_entries[0].config.solver_cfg = MPMSolverCfg()
        NewtonCouplerManager._build_solver(model, proxy_cfg)
        assert coupler.NewtonManager._needs_fk_before_step is True

        recorded_entries.clear()
        admm_cfg = CouplerAdmmCfg(entries=proxy_cfg.entries)
        NewtonCouplerManager._build_solver(model, admm_cfg)
        assert coupler.NewtonManager._needs_collision_pipeline is True
        assert recorded_entries == ["rigid", "soft"]

        coupler.NewtonManager._report_contacts = True
        with pytest.raises(NotImplementedError, match="contact sensors"):
            NewtonCouplerManager._build_solver(model, proxy_cfg)
    finally:
        coupler.NewtonManager._solver = old_solver
        coupler.NewtonManager._needs_collision_pipeline = old_outer
        coupler.NewtonManager._supports_contact_sensors = old_contact_support
        coupler.NewtonManager._report_contacts = old_report_contacts
        coupler.NewtonManager._needs_fk_before_step = old_needs_fk


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
    cfg = CouplerAdmmCfg(
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
    monkeypatch.setattr(coupler, "SolverCoupledADMM", _RecordingAdmm)

    solver = NewtonCouplerManager._build_admm_coupled_solver(model, [], cfg)

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
    cfg = CouplerAdmmCfg(entries=[entry.config for entry in resolved_entries])
    monkeypatch.setattr(coupler, "SolverCoupledADMM", _RecordingAdmm)

    solver = NewtonCouplerManager._build_admm_coupled_solver(model, entries, cfg)

    assert [(pair.source, pair.destination) for pair in solver.coupling.contact_pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]

    cfg.contact_pairs = []
    solver = NewtonCouplerManager._build_admm_coupled_solver(model, entries, cfg)
    assert list(solver.coupling.contact_pairs) == []
