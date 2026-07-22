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
    NewtonCollisionPipelineCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import ShapeFlags
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
    coupler,
)
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg


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


def test_public_coupling_exports_are_importable():
    """The lazy-export stub must not retain deleted public symbols."""
    import isaaclab_contrib.coupling as coupling

    for name in coupling.__all__:
        assert getattr(coupling, name) is not None


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


def _float_array(size: int, value: float) -> _FakeArray:
    return _FakeArray(np.full(size, value, dtype=np.float32))


def _entry(
    name: str,
    *,
    bodies: list[int] | None = None,
    particles: list[int] | None = None,
    shapes: list[int] | None = None,
) -> NewtonCouplerManager._ResolvedEntry:
    """Build an already-resolved entry for validation tests."""
    return NewtonCouplerManager._ResolvedEntry(
        config=CouplerEntryCfg(name=name, solver_cfg=XPBDSolverCfg()),
        bodies=list(bodies or []),
        particles=list(particles or []),
        joints=[],
        shapes=list(shapes or []),
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


def test_config_validation_requires_entries():
    with pytest.raises(ValueError, match="at least one solver entry"):
        NewtonCouplerManager._validate_config(CouplerProxyCfg())


def test_config_validation_requires_nonempty_entry_names():
    cfg = CouplerAdmmCfg(entries=[CouplerEntryCfg(name="", solver_cfg=XPBDSolverCfg())])
    with pytest.raises(ValueError, match="non-empty strings"):
        NewtonCouplerManager._validate_config(cfg)


def test_config_validation_requires_newton_solver_config():
    cfg = CouplerAdmmCfg(
        entries=[CouplerEntryCfg(name="entry", solver_cfg=object())]  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="solver_cfg must be a NewtonSolverCfg"):
        NewtonCouplerManager._validate_config(cfg)


@pytest.mark.parametrize(
    ("solver_cfg", "entry_kwargs", "error_type", "match"),
    [
        (CouplerProxyCfg(), {}, ValueError, "nested CouplerCfg"),
        (VBDSolverCfg(model_cfg=NewtonModelCfg()), {}, ValueError, "model parameters are global"),
        (KaminoSolverCfg(), {}, NotImplementedError, "FK/reset lifecycle"),
        (
            MPMSolverCfg(project_outside_colliders=True),
            {"in_place": True},
            NotImplementedError,
            "post-step projection",
        ),
        (MPMSolverCfg(), {}, ValueError, "must set in_place=True"),
        (MJWarpSolverCfg(use_mujoco_cpu=True), {}, NotImplementedError, "reset-mask lifecycle"),
    ],
)
def test_config_validation_rejects_unsupported_nested_lifecycle(solver_cfg, entry_kwargs, error_type, match):
    cfg = CouplerAdmmCfg(entries=[CouplerEntryCfg(name="entry", solver_cfg=solver_cfg, **entry_kwargs)])
    with pytest.raises(error_type, match=match):
        NewtonCouplerManager._validate_config(cfg)


def test_config_validation_requires_concrete_nested_factory():
    solver_cfg = XPBDSolverCfg()
    solver_cfg.class_type = NewtonManager
    cfg = CouplerAdmmCfg(entries=[CouplerEntryCfg(name="entry", solver_cfg=solver_cfg)])
    with pytest.raises(TypeError, match="does not implement nested solver construction"):
        NewtonCouplerManager._validate_config(cfg)


def test_string_selector_resolves_full_body_labels_and_descendants():
    assert NewtonCouplerManager._resolve_entities_to_body_ids(
        _FakeModel(),
        ["/World/envs/env_.*/Robot"],
        "entry 'rigid'",
    ) == [0, 1]


def test_body_selector_rejects_unknown_types():
    with pytest.raises(TypeError, match="expected a full-label regex string"):
        NewtonCouplerManager._resolve_entities_to_body_ids(
            _FakeModel(),
            [object()],
            "entry 'rigid'",
        )


def test_raw_body_label_selector_reports_no_matches():
    with pytest.raises(ValueError, match="matched no Newton bodies"):
        NewtonCouplerManager._resolve_entities_to_body_ids(
            _FakeModel(),
            ["/World/envs/env_.*/Missing"],
            "entry 'missing'",
        )


def test_raw_body_id_selectors_pass_through_and_dedupe():
    """Integer selectors resolve to themselves, preserving order and removing duplicates."""
    assert NewtonCouplerManager._resolve_entities_to_body_ids(_FakeModel(), [2, 0, 2], "proxy 'a'->'b'") == [2, 0]


def test_out_of_range_body_id_selector_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        NewtonCouplerManager._resolve_entities_to_body_ids(_FakeModel(), [5], "proxy 'a'->'b'")


def test_proxy_resolution_writes_body_ids_into_config_in_place():
    """Resolution replaces the proxy's selectors with body ids and is idempotent."""
    model = _FakeModel()
    proxy = CouplerProxyMappingCfg(
        source="rigid", destination="soft", bodies=[r"/World/envs/env_.*/Robot"], particles=[1, 1]
    )

    resolved = NewtonCouplerManager._resolve_proxy(model, proxy)

    assert resolved is proxy
    assert proxy.bodies == [0, 1]
    assert proxy.particles == [1]
    # Re-resolving the now-integer selectors yields the same result.
    assert NewtonCouplerManager._resolve_proxy(model, proxy).bodies == [0, 1]


def test_three_named_entries_partition_bodies_joints_shapes_and_particles():
    """Derived joint/shape ownership follows each entry's selected bodies."""
    model = _FakeModel()
    entries = [
        CouplerEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(),
            bodies=[r"/World/envs/env_.*/Robot"],
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

    resolved = [NewtonCouplerManager._resolve_entry(model, entry) for entry in entries]

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
    assert entries[0].bodies == [r"/World/envs/env_.*/Robot"]
    assert entries[1].bodies == ["/World/envs/env_.*/Object"]


def test_cross_entry_joint_is_left_unowned_for_admm_attachment():
    """A joint spanning two entries must remain visible to the ADMM coupler."""
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    entries = [
        CouplerEntryCfg(
            name="robot",
            solver_cfg=XPBDSolverCfg(),
            bodies=[r"/World/envs/env_.*/Robot"],
        ),
        CouplerEntryCfg(
            name="object",
            solver_cfg=XPBDSolverCfg(),
            bodies=[r"/World/envs/env_.*/Object"],
            all_particles=True,
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCouplerManager._resolve_entry(model, entry) for entry in entries]

    assert resolved[0].joints == [0]
    assert resolved[1].joints == []


def test_resolved_entry_validation_rejects_inactive_entry():
    with pytest.raises(ValueError, match="neither owns nor receives any model elements"):
        NewtonCouplerManager._validate_resolved_entries(
            _FakeModel(),
            [_entry("empty")],
            CouplerAdmmCfg(),
        )


def test_resolved_entry_validation_accepts_active_proxy_destination():
    NewtonCouplerManager._validate_resolved_entries(
        _FakeModel(),
        [_entry("source", bodies=[0]), _entry("destination")],
        CouplerProxyCfg(),
        {"destination"},
    )


def test_resolved_entry_validation_accepts_static_shape_ownership():
    NewtonCouplerManager._validate_resolved_entries(
        _FakeModel(),
        [_entry("ground", shapes=[3])],
        CouplerAdmmCfg(),
    )


def test_proxy_validation_rejects_cross_entry_joint():
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    cfg, entries, proxies = _valid_proxy_setup()

    with pytest.raises(ValueError, match="does not support cross-entry joint"):
        NewtonCouplerManager._validate_no_cross_entry_proxy_joints(
            model, {entry.config.name: entry for entry in entries}
        )


def test_shape_label_patterns_and_static_shape_selection_are_additive():
    entry = NewtonCouplerManager._resolve_entry(
        _FakeModel(),
        CouplerEntryCfg(
            name="special",
            solver_cfg=XPBDSolverCfg(),
            bodies=[r"/World/envs/env_.*/Robot/base"],
            include_body_shapes=False,
            include_static_shapes=True,
            shape_label_patterns=[r".*/Object/object_collision"],
        ),
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
        CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=[r"/World/envs/env_.*/Robot"]),
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
                mode="staggered",
                mass_scale=0.25,
                collide_interval=4,
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
        lambda model_view, **kwargs: (model_view, kwargs["broad_phase"]),
    )

    proxies = [NewtonCouplerManager._resolve_proxy(model, proxy) for proxy in cfg.proxies]
    solver = NewtonCouplerManager._build_proxy_coupled_solver(model, [], proxies, cfg)

    assert solver.coupling.iterations == 3
    assert solver.coupling.proxies[0].mode == "staggered"
    assert solver.coupling.proxies[0].mass_scale == pytest.approx(0.25)
    assert solver.coupling.proxies[0].collide_interval == 4
    assert solver.coupling.proxies[0].collision_pipeline is custom_pipeline
    assert solver.coupling.proxies[1].collision_pipeline("soft-view") == ("soft-view", "explicit")
    assert isinstance(cfg.proxies[1].collision_pipeline, NewtonCollisionPipelineCfg)


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
        MPMSolverCfg(),
        VBDSolverCfg(),
    ],
)
def test_solver_config_manager_exposes_nested_factory(solver_cfg):
    assert solver_cfg.class_type._create_solver.__func__ is not NewtonManager._create_solver.__func__


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


def test_mpm_entry_does_not_request_external_contacts():
    assert NewtonCouplerManager._requires_external_contacts(MPMSolverCfg()) is False


@pytest.mark.parametrize(("grid_type", "expected"), [("fixed", True), ("sparse", False), ("dense", False)])
def test_mpm_grid_controls_cuda_graph_support(monkeypatch, grid_type, expected):
    cfg = CouplerAdmmCfg(
        entries=[
            CouplerEntryCfg(
                name="media",
                solver_cfg=MPMSolverCfg(grid_type=grid_type),
                in_place=True,
            )
        ]
    )
    monkeypatch.setattr(coupler.PhysicsManager, "_cfg", SimpleNamespace(solver_cfg=cfg))
    assert NewtonCouplerManager._supports_cuda_graph_capture() is expected


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


@pytest.mark.parametrize(
    ("case", "expected_outer"),
    [
        ("mjwarp_internal", False),
        ("mjwarp_external", True),
        ("mpm", False),
        ("destination_fallback", True),
        ("destination_factory_fallback", True),
    ],
)
def test_proxy_selects_expected_outer_collision_pipeline(monkeypatch, case, expected_outer):
    """Proxy entries request shared contacts only when one of their solve paths consumes them."""
    model = _FakeModel()
    if case == "destination_fallback":
        fallback_proxy = CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=[1], collision_pipeline=None)
    elif case == "destination_factory_fallback":
        fallback_proxy = CouplerProxyMappingCfg(
            source="rigid", destination="soft", bodies=[1], collision_pipeline=lambda model_view: None
        )
    else:
        fallback_proxy = None
    proxy_cfg, resolved_entries, resolved_proxies = _valid_proxy_setup(proxy=fallback_proxy)
    recorded_entries: list[str] = []

    if case == "mjwarp_external":
        resolved_entries[0].config.solver_cfg = MJWarpSolverCfg(use_mujoco_contacts=False)
    elif case == "mpm":
        resolved_entries[0].config.solver_cfg = MPMSolverCfg()
        resolved_entries[0].config.in_place = True
    else:
        resolved_entries[0].config.solver_cfg = MJWarpSolverCfg()

    for attribute in (
        "_solver",
        "_use_single_state",
        "_needs_collision_pipeline",
        "_supports_contact_sensors",
        "_report_contacts",
    ):
        monkeypatch.setattr(coupler.NewtonManager, attribute, getattr(coupler.NewtonManager, attribute))

    monkeypatch.setattr(
        NewtonCouplerManager,
        "_resolve_entry",
        classmethod(
            lambda cls, model, entry_cfg: next(
                entry for entry in resolved_entries if entry.config.name == entry_cfg.name
            )
        ),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_resolve_proxy",
        classmethod(
            lambda cls, model, proxy_cfg: next(
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
        classmethod(
            lambda cls, model, entries, proxies, cfg: SimpleNamespace(
                kind="proxy",
                get_proxy_contacts=lambda source, destination: (
                    None if case in {"destination_fallback", "destination_factory_fallback"} else object()
                ),
            )
        ),
    )
    NewtonCouplerManager._build_solver(model, proxy_cfg)

    assert coupler.NewtonManager._needs_collision_pipeline is expected_outer
    assert coupler.NewtonManager._supports_contact_sensors is False
    assert recorded_entries == ["rigid", "soft"]


def test_admm_always_requests_outer_collision_pipeline(monkeypatch):
    model = _FakeModel()
    _, resolved_entries, _ = _valid_proxy_setup()
    cfg = CouplerAdmmCfg(entries=[entry.config for entry in resolved_entries])
    for attribute in (
        "_solver",
        "_use_single_state",
        "_needs_collision_pipeline",
        "_supports_contact_sensors",
        "_report_contacts",
    ):
        monkeypatch.setattr(coupler.NewtonManager, attribute, getattr(coupler.NewtonManager, attribute))
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_resolve_entry",
        classmethod(
            lambda cls, model, entry_cfg: next(
                entry for entry in resolved_entries if entry.config.name == entry_cfg.name
            )
        ),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_build_entry",
        classmethod(lambda cls, entry: entry.config.name),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_build_admm_coupled_solver",
        classmethod(lambda cls, model, entries, cfg: SimpleNamespace(kind="admm")),
    )

    NewtonCouplerManager._build_solver(model, cfg)

    assert coupler.NewtonManager._needs_collision_pipeline is True


def test_contact_sensor_guard_does_not_mutate_manager_state(monkeypatch):
    sentinel_solver = object()
    monkeypatch.setattr(coupler.NewtonManager, "_solver", sentinel_solver)
    monkeypatch.setattr(coupler.NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(coupler.NewtonManager, "_needs_collision_pipeline", True)
    monkeypatch.setattr(coupler.NewtonManager, "_supports_contact_sensors", True)
    monkeypatch.setattr(coupler.NewtonManager, "_report_contacts", True)

    with pytest.raises(NotImplementedError, match="contact sensors"):
        NewtonCouplerManager._build_solver(_FakeModel(), CouplerProxyCfg())

    assert coupler.NewtonManager._solver is sentinel_solver
    assert coupler.NewtonManager._use_single_state is True
    assert coupler.NewtonManager._needs_collision_pipeline is True
    assert coupler.NewtonManager._supports_contact_sensors is True


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
        rho=2.0,
        gamma=0.2,
        baumgarte=0.15,
        joint_stiffness=123.0,
        joint_damping=4.0,
        joint_angular_stiffness=456.0,
        joint_angular_damping=7.0,
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
    assert solver.coupling.rho == pytest.approx(2.0)
    assert solver.coupling.gamma == pytest.approx(0.2)
    assert solver.coupling.baumgarte == pytest.approx(0.15)
    assert solver.coupling.joint_stiffness == pytest.approx(123.0)
    assert solver.coupling.joint_damping == pytest.approx(4.0)
    assert solver.coupling.joint_angular_stiffness == pytest.approx(456.0)
    assert solver.coupling.joint_angular_damping == pytest.approx(7.0)
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
