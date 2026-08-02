# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import sys
import types
from collections import UserList

import numpy as np
import pytest

from pxr import Sdf, Usd

import isaaclab.assets.articulation.ordering_resolvers as ordering_resolvers
from isaaclab.assets.articulation.ordering import (
    ArticulationOrderingConvention,
    apply_articulation_ordering_preset,
    build_articulation_name_map,
    parse_articulation_ordering_convention,
)
from isaaclab.assets.articulation.ordering_resolvers import (
    _resolve_articulation_convention_name_ordering,
    _resolve_articulation_ordering_names,
    get_articulation_name_ordering,
)

sim_stub = types.ModuleType("isaaclab.sim")


class _SimulationContext:
    @staticmethod
    def instance():
        return None


class _SpawnerCfg:
    pass


sim_stub.SimulationContext = _SimulationContext
sim_stub.SpawnerCfg = _SpawnerCfg
simulation_context_stub = types.ModuleType("isaaclab.sim.simulation_context")
simulation_context_stub.SimulationContext = _SimulationContext
_inserted_sim_stub = "isaaclab.sim" not in sys.modules
_inserted_sim_context_stub = "isaaclab.sim.simulation_context" not in sys.modules
sys.modules.setdefault("isaaclab.sim", sim_stub)
sys.modules.setdefault("isaaclab.sim.simulation_context", simulation_context_stub)


asset_base_stub = types.ModuleType("isaaclab.assets.asset_base")


class _AssetBase:
    def __init__(self, cfg):
        self.cfg = cfg


asset_base_stub.AssetBase = _AssetBase
_inserted_asset_base_stub = "isaaclab.assets.asset_base" not in sys.modules
sys.modules.setdefault("isaaclab.assets.asset_base", asset_base_stub)


from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

if _inserted_sim_stub:
    sys.modules.pop("isaaclab.sim", None)
if _inserted_sim_context_stub:
    sys.modules.pop("isaaclab.sim.simulation_context", None)
if _inserted_asset_base_stub:
    sys.modules.pop("isaaclab.assets.asset_base", None)
if _inserted_sim_stub or _inserted_asset_base_stub:
    sys.modules.pop("isaaclab.assets.articulation.base_articulation", None)


def test_parse_articulation_ordering_convention_accepts_none_strings_and_enum() -> None:
    """Parse supported articulation ordering convention inputs."""
    assert parse_articulation_ordering_convention(None) is None
    assert parse_articulation_ordering_convention("physx") is ArticulationOrderingConvention.PHYSX
    assert parse_articulation_ordering_convention("mjwarp") is ArticulationOrderingConvention.MJWARP
    assert parse_articulation_ordering_convention("robot_schema") is ArticulationOrderingConvention.ROBOT_SCHEMA
    assert (
        parse_articulation_ordering_convention(ArticulationOrderingConvention.PHYSX)
        is ArticulationOrderingConvention.PHYSX
    )


def test_parse_articulation_ordering_convention_rejects_unknown_string() -> None:
    """Reject unknown symbolic articulation ordering conventions."""
    with pytest.raises(ValueError, match="Unsupported articulation ordering convention"):
        parse_articulation_ordering_convention("backend")


@pytest.mark.parametrize("entrypoint", ["resolve_names", "resolve_convention", "get_ordering"])
def test_ordering_resolvers_reject_invalid_kind(entrypoint: str) -> None:
    """Reject a misspelled element kind at each distinct resolver validation site.

    ``get_articulation_name_ordering`` validates ``kind`` before parsing the convention, so its
    ``physx``/``mjwarp``/``robot_schema`` aliases all reach one guard; a single public call covers
    them.
    """

    class _Articulation:
        __backend_name__ = "physx"
        backend_joint_names = ("joint",)
        backend_body_names = ("body",)

    articulation = _Articulation()
    with pytest.raises(ValueError, match="kind must be 'joint' or 'body'; got 'dof'"):
        if entrypoint == "resolve_names":
            _resolve_articulation_ordering_names(
                kind="dof",  # type: ignore[arg-type]
                backend_names=("joint",),
                ordering=None,
                active_backend_name="physx",
            )
        elif entrypoint == "resolve_convention":
            _resolve_articulation_convention_name_ordering(
                articulation=articulation,
                convention="physx",
                kind="dof",  # type: ignore[arg-type]
            )
        else:
            get_articulation_name_ordering(articulation, "physx", kind="dof")  # type: ignore[arg-type]


def test_articulation_cfg_accepts_optional_ordering_fields() -> None:
    """Configure explicit or symbolic public articulation ordering."""
    explicit_joint_order = ("shoulder", "elbow", "wrist")
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        actuators={},
        joint_ordering=explicit_joint_order,
        body_ordering="mjwarp",
    )

    assert cfg.joint_ordering == explicit_joint_order
    assert cfg.body_ordering == "mjwarp"


def test_apply_articulation_ordering_preset_sets_joint_and_body_ordering() -> None:
    """Apply one ergonomic ordering preset to both joints and bodies."""
    cfg = ArticulationCfg(prim_path="/World/Robot", actuators={})

    ordered_cfg = apply_articulation_ordering_preset(cfg, "physx")

    assert ordered_cfg is not cfg
    assert ordered_cfg.joint_ordering is ArticulationOrderingConvention.PHYSX
    assert ordered_cfg.body_ordering is ArticulationOrderingConvention.PHYSX
    assert apply_articulation_ordering_preset(cfg, None) is cfg


def test_build_articulation_name_map_rejects_scalar_name_sequences() -> None:
    """Reject a scalar string instead of splitting it into per-character names.

    ``backend_names`` and ``user_names`` share one coercion helper, so one representative field and
    value pin the rule and the offending type.
    """
    with pytest.raises(TypeError, match=r"backend_names must be a list or tuple of strings; got str"):
        build_articulation_name_map(
            kind="joint",
            backend_names="joint_0",
            user_names=("joint_0",),
            device="cpu",
        )


def test_build_articulation_name_map_reports_non_string_name_element() -> None:
    """Identify malformed elements in public map-construction inputs."""
    with pytest.raises(TypeError, match=r"^backend_names element 1 must be str; got 7 \(int\)\.$"):
        build_articulation_name_map(
            kind="joint",
            backend_names=("joint_0", 7),
            user_names=("joint_0", "joint_1"),
            device="cpu",
        )


@pytest.mark.parametrize("sequence_type", [list, tuple])
def test_resolve_articulation_ordering_names_accepts_list_and_tuple(sequence_type) -> None:
    """Resolve explicit orderings from both plain sequence forms, normalized to a tuple."""
    user_names = _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=("joint_0", "joint_1", "joint_2"),
        ordering=sequence_type(["joint_2", "joint_0", "joint_1"]),
        active_backend_name="physx",
    )

    assert user_names == ("joint_2", "joint_0", "joint_1")
    assert type(user_names) is tuple


def test_resolve_articulation_ordering_names_rejects_generic_sequence() -> None:
    """Reject explicit orderings that are not plain lists or tuples."""
    with pytest.raises(
        TypeError, match=r"joint_ordering must be a name list or tuple, convention string/enum, or None"
    ):
        _resolve_articulation_ordering_names(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            ordering=UserList(["joint_2", "joint_0", "joint_1"]),
            active_backend_name="physx",
        )


def test_resolve_articulation_ordering_names_reports_non_string_element() -> None:
    """Report the location and type of invalid explicit ordering elements."""
    with pytest.raises(TypeError) as exc_info:
        _resolve_articulation_ordering_names(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            ordering=("joint_1", 7),
            active_backend_name="physx",
        )

    assert str(exc_info.value) == "joint_ordering element 1 must be str; got 7 (int)."


@pytest.mark.parametrize(
    ("ordering", "type_name"),
    [
        pytest.param(7, "int", id="scalar-int"),
        pytest.param(b"joint_0", "bytes", id="bytes-like"),
    ],
)
def test_resolve_articulation_ordering_names_reports_unsupported_type(ordering, type_name: str) -> None:
    """Report the offending type for unsupported scalar orderings.

    All unsupported scalars hit one ``else`` branch, so a plain scalar and a bytes-like value -- which
    must not be silently iterated into per-character names -- pin the rule and its error message.
    """
    with pytest.raises(TypeError) as exc_info:
        _resolve_articulation_ordering_names(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            ordering=ordering,
            active_backend_name="physx",
        )

    assert str(exc_info.value) == (
        f"joint_ordering must be a name list or tuple, convention string/enum, or None; got {type_name}."
    )


def test_resolve_articulation_ordering_names_keeps_matching_backend_preset_identity() -> None:
    """Resolve same-backend symbolic presets without requiring traversal helpers."""

    class _Articulation:
        __backend_name__ = "physx"
        __backend_native_orderings__ = ("physx",)

    user_names = _resolve_articulation_ordering_names(
        kind="body",
        backend_names=("base", "left_foot", "right_foot"),
        ordering=ArticulationOrderingConvention.PHYSX,
        active_backend_name="physx",
        articulation=_Articulation(),
    )

    assert user_names == ("base", "left_foot", "right_foot")


@pytest.mark.parametrize("backend_name", ["physx", "ovphysx"])
def test_physx_ordering_helper_uses_same_backend_identity_without_discovery(
    monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Return PhysX and OVPhysX backend names without metadata discovery."""

    class _Articulation:
        __backend_name__ = backend_name
        __backend_native_orderings__ = ("physx",)
        backend_joint_names = ("shoulder", "elbow", "wrist")
        backend_body_names = ("base", "arm", "hand")

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_physx_names_from_newton_usd_builder",
        lambda _: pytest.fail("same-backend resolution must not invoke the USD builder"),
    )

    articulation = _Articulation()
    assert get_articulation_name_ordering(articulation, "physx", kind="joint") == articulation.backend_joint_names
    assert get_articulation_name_ordering(articulation, "physx", kind="body") == articulation.backend_body_names


def test_mjwarp_ordering_helper_uses_newton_identity_without_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return Newton backend names without metadata discovery."""

    class _Articulation:
        __backend_name__ = "newton"
        __backend_native_orderings__ = ("mjwarp",)
        backend_joint_names = ("knee", "hip", "ankle")
        backend_body_names = ("base", "thigh", "foot")

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_mjwarp_names_from_newton_usd_builder",
        lambda _: pytest.fail("same-backend resolution must not invoke the USD builder"),
    )

    articulation = _Articulation()
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == articulation.backend_joint_names
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == articulation.backend_body_names


def test_backend_native_orderings_declaration_drives_identity_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend self-declares the conventions its native order satisfies.

    The resolver keys the same-backend identity fast path off
    ``__backend_native_orderings__`` rather than a hardcoded backend-name set,
    so a fourth backend opts into a convention purely by declaring it.
    """

    class _Articulation:
        __backend_name__ = "fourth_backend"
        __backend_native_orderings__ = ("physx",)
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("shoulder", "elbow")
        backend_body_names = ("base", "arm")

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_physx_names_from_newton_usd_builder",
        lambda _: pytest.fail("a declared native convention must not invoke the USD builder"),
    )

    articulation = _Articulation()
    assert get_articulation_name_ordering(articulation, "physx", kind="joint") == articulation.backend_joint_names
    assert get_articulation_name_ordering(articulation, "physx", kind="body") == articulation.backend_body_names


def test_undeclared_convention_bypasses_identity_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A convention absent from ``__backend_native_orderings__`` falls through to discovery."""

    class _Articulation:
        __backend_name__ = "fourth_backend"
        __backend_native_orderings__ = ("physx",)
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("shoulder", "elbow")
        backend_body_names = ("base", "arm")

    monkeypatch.setattr(ordering_resolvers, "_get_mjwarp_names_from_newton_usd_builder", lambda _: None)

    with pytest.raises(NotImplementedError, match="Unable to resolve 'mjwarp'"):
        get_articulation_name_ordering(_Articulation(), "mjwarp", kind="joint")


@pytest.mark.parametrize(
    ("kind", "config_field"),
    [
        ("joint", "joint_ordering"),
        ("body", "body_ordering"),
    ],
)
def test_mjwarp_ordering_helper_reports_actionable_cross_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    config_field: str,
) -> None:
    """Explain how to recover when MJWarp ordering discovery is unavailable."""

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("hip", "knee")
        backend_body_names = ("base", "foot")

    monkeypatch.setattr(ordering_resolvers, "_get_mjwarp_names_from_newton_usd_builder", lambda _: None)

    with pytest.raises(NotImplementedError) as exc_info:
        get_articulation_name_ordering(_Articulation(), "mjwarp", kind=kind)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert f"Unable to resolve 'mjwarp' {kind} ordering" in message
    assert "active backend 'physx'" in message
    assert f"env.scene.robot.{config_field}" in message
    assert f"explicit {kind}-name permutation" in message
    assert "mjwarp_usd_builder: the Newton USD builder returned no articulation names" in message


def test_mjwarp_ordering_helper_reports_incomplete_usd_builder_names_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the missing/extra backend names when the temporary builder view is incomplete."""

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("hip", "knee")
        backend_body_names = ("base", "foot")

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_mjwarp_names_from_newton_usd_builder",
        lambda _: {"joint": ("hip",), "body": ("base", "foot")},
    )

    with pytest.raises(NotImplementedError) as exc_info:
        get_articulation_name_ordering(_Articulation(), "mjwarp", kind="joint")

    message = str(exc_info.value)
    assert "mjwarp_usd_builder: joint names are not a complete permutation" in message
    assert "missing=['knee']" in message
    assert "extra=[]" in message


def test_mjwarp_ordering_helper_surfaces_specific_usd_builder_unavailability_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the specific unavailability reason instead of a generic one when known."""

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("hip", "knee")
        backend_body_names = ("base", "foot")

    monkeypatch.setattr(ordering_resolvers, "_get_mjwarp_names_from_newton_usd_builder", lambda _: None)
    monkeypatch.setattr(
        ordering_resolvers,
        "_describe_newton_usd_builder_unavailability",
        lambda _: "no current USD stage is available",
    )

    with pytest.raises(NotImplementedError) as exc_info:
        get_articulation_name_ordering(_Articulation(), "mjwarp", kind="joint")

    assert "mjwarp_usd_builder: no current USD stage is available" in str(exc_info.value)


def test_describe_newton_usd_builder_unavailability_falls_back_without_cfg() -> None:
    """Return a generic reason instead of raising when the articulation has no cfg."""

    class _Articulation:
        pass

    reason = ordering_resolvers._describe_newton_usd_builder_unavailability(_Articulation())

    assert reason == "the Newton USD builder returned no articulation names"


def test_describe_newton_usd_builder_unavailability_reports_missing_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identify a missing current USD stage as the reason resolution could not proceed."""

    class _UsdPhysics:
        class ArticulationRootAPI:
            pass

    pxr_mod = types.ModuleType("pxr")
    pxr_mod.UsdPhysics = _UsdPhysics
    stage_mod = types.ModuleType("isaaclab.sim.utils.stage")
    stage_mod.get_current_stage = lambda: None
    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = lambda *args, **kwargs: pytest.fail(
        "must not resolve prims when the stage is unavailable"
    )
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "pxr", pxr_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", stage_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)

    class _Articulation:
        cfg = types.SimpleNamespace(prim_path="/World/Robot", articulation_root_prim_path=None)

    reason = ordering_resolvers._describe_newton_usd_builder_unavailability(_Articulation())

    assert reason == "no current USD stage is available"


def test_describe_newton_usd_builder_unavailability_reports_missing_source_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identify a missing source asset prim as the reason resolution could not proceed."""

    class _UsdPhysics:
        class ArticulationRootAPI:
            pass

    class _Stage:
        pass

    pxr_mod = types.ModuleType("pxr")
    pxr_mod.UsdPhysics = _UsdPhysics
    stage_mod = types.ModuleType("isaaclab.sim.utils.stage")
    stage_mod.get_current_stage = lambda: _Stage()
    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = lambda *args, **kwargs: []
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "pxr", pxr_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", stage_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)

    class _Articulation:
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

    reason = ordering_resolvers._describe_newton_usd_builder_unavailability(_Articulation())

    assert reason == "source asset prim matching '/World/envs/env_.*/Robot' was not found"


def _install_source_asset_resolver(monkeypatch: pytest.MonkeyPatch, resolve_matching_prims_from_source) -> None:
    """Route ``resolve_matching_prims_from_source`` through a stubbed ``isaaclab.sim.utils.queries``.

    The resolver imports the source-prim query lazily from ``isaaclab.sim.utils.queries``; the sim
    package is otherwise unavailable in this unit-test env, so we install a minimal module tree that
    exposes only the query the resolver calls.
    """
    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = resolve_matching_prims_from_source
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)


_ROBOT_SCHEMA_PRIM_PATH = "/World/envs/env_0/Robot"
_ROBOT_SCHEMA_SOURCE_EXPR = "/World/envs/env_.*/Robot"


def _author_robot_schema_relationship(prim: Usd.Prim, relationship_name: str, target_paths: list[str]) -> None:
    """Author a robot-schema relationship on a real prim with the given target prim paths."""
    prim.CreateRelationship(relationship_name).SetTargets([Sdf.Path(path) for path in target_paths])


def _author_robot_schema_stage(child_names: tuple[str, ...]) -> tuple[Usd.Stage, Usd.Prim]:
    """Create an in-memory stage with a robot prim and the named leaf child prims."""
    stage = Usd.Stage.CreateInMemory()
    robot_prim = stage.DefinePrim(_ROBOT_SCHEMA_PRIM_PATH, "Xform")
    for child in child_names:
        stage.DefinePrim(f"{_ROBOT_SCHEMA_PRIM_PATH}/{child}", "Xform")
    return stage, robot_prim


def _install_robot_schema_source_asset(monkeypatch: pytest.MonkeyPatch, robot_prim: Usd.Prim) -> None:
    """Route source-asset resolution to a single authored robot prim for robot-schema discovery."""

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        assert path_expr == _ROBOT_SCHEMA_SOURCE_EXPR
        return [(robot_prim, _ROBOT_SCHEMA_SOURCE_EXPR)]

    _install_source_asset_resolver(monkeypatch, _resolve_matching_prims_from_source)


def _make_robot_schema_articulation(
    *,
    backend_joint_names: tuple[str, ...] | None = None,
    backend_body_names: tuple[str, ...] | None = None,
) -> types.SimpleNamespace:
    """Build an articulation surface that resolves robot-schema ordering from an in-memory stage."""
    return types.SimpleNamespace(
        __backend_name__="newton",
        _ordering_convention_name_cache={},
        cfg=types.SimpleNamespace(prim_path=_ROBOT_SCHEMA_SOURCE_EXPR, articulation_root_prim_path=None),
        backend_joint_names=list(backend_joint_names) if backend_joint_names is not None else [],
        backend_body_names=list(backend_body_names) if backend_body_names is not None else [],
    )


def test_robot_schema_ordering_helper_resolves_canonical_name_override_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve joint ordering from authored relationships honoring the canonical ``isaac:NameOverride``."""
    stage, robot_prim = _author_robot_schema_stage(("joint_a_prim", "joint_b", "joint_c"))
    _author_robot_schema_relationship(
        robot_prim,
        "isaac:physics:robotJoints",
        [
            f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_b",
            f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_a_prim",
            f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_c",
        ],
    )
    # The canonical uppercase ``NameOverride`` renames the target prim to its backend joint name.
    stage.GetPrimAtPath(f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_a_prim").CreateAttribute(
        "isaac:NameOverride", Sdf.ValueTypeNames.String
    ).Set("joint_a")
    _install_robot_schema_source_asset(monkeypatch, robot_prim)

    articulation = _make_robot_schema_articulation(backend_joint_names=("joint_a", "joint_b", "joint_c"))

    assert get_articulation_name_ordering(articulation, "robot_schema", kind="joint") == (
        "joint_b",
        "joint_a",
        "joint_c",
    )
    # The public resolver entry point routes through the same authored-relationship discovery.
    assert _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=tuple(articulation.backend_joint_names),
        ordering="robot_schema",
        active_backend_name=articulation.__backend_name__,
        articulation=articulation,
    ) == ("joint_b", "joint_a", "joint_c")


def test_robot_schema_ordering_helper_resolves_fallback_name_override_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve body ordering via the fallback ``isaac:nameOverride`` spelling, dropping extra targets."""
    stage, robot_prim = _author_robot_schema_stage(("base_prim", "tool_site", "thigh", "foot"))
    _author_robot_schema_relationship(
        robot_prim,
        "isaac:physics:robotLinks",
        [
            f"{_ROBOT_SCHEMA_PRIM_PATH}/base_prim",
            f"{_ROBOT_SCHEMA_PRIM_PATH}/tool_site",
            f"{_ROBOT_SCHEMA_PRIM_PATH}/thigh",
            f"{_ROBOT_SCHEMA_PRIM_PATH}/foot",
        ],
    )
    # The fallback lowercase ``nameOverride`` renames the root target; tool_site is absent from the
    # backend names and must be dropped by the complete-permutation filter.
    stage.GetPrimAtPath(f"{_ROBOT_SCHEMA_PRIM_PATH}/base_prim").CreateAttribute(
        "isaac:nameOverride", Sdf.ValueTypeNames.String
    ).Set("base")
    _install_robot_schema_source_asset(monkeypatch, robot_prim)

    articulation = _make_robot_schema_articulation(backend_body_names=("base", "thigh", "foot"))

    assert get_articulation_name_ordering(articulation, "robot_schema", kind="body") == ("base", "thigh", "foot")


def test_robot_schema_ordering_helper_rejects_incomplete_relationships(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject robot schema relationships that are not complete backend-name permutations."""
    stage, robot_prim = _author_robot_schema_stage(("joint_a", "joint_b"))
    _author_robot_schema_relationship(
        robot_prim,
        "isaac:physics:robotJoints",
        [f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_b", f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_a"],
    )
    _install_robot_schema_source_asset(monkeypatch, robot_prim)

    articulation = _make_robot_schema_articulation(backend_joint_names=("joint_a", "joint_b", "joint_c"))

    with pytest.raises(NotImplementedError) as exc_info:
        get_articulation_name_ordering(articulation, "robot_schema", kind="joint")

    message = str(exc_info.value)
    assert "Unable to resolve 'robot_schema' joint ordering" in message
    assert "robot_schema: incomplete permutation (missing=['joint_c'], extra=[])" in message


def test_robot_schema_ordering_helper_reports_unresolvable_relationship_target(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Name the unresolvable target in both the warning log and the terminal failure message."""
    stage, robot_prim = _author_robot_schema_stage(("joint_a",))
    # joint_typo is authored as a relationship target but never defined on the stage, so USD resolves
    # it to an invalid prim.
    unresolved_target_path = f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_typo"
    _author_robot_schema_relationship(
        robot_prim,
        "isaac:physics:robotJoints",
        [f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_a", unresolved_target_path],
    )
    _install_robot_schema_source_asset(monkeypatch, robot_prim)

    # A second backend joint name that the single resolvable target cannot cover, so resolution falls
    # through to the terminal failure instead of an incomplete-but-silent success.
    articulation = _make_robot_schema_articulation(backend_joint_names=("joint_a", "joint_b"))

    caplog.set_level(logging.WARNING, logger="isaaclab.assets.articulation.ordering_resolvers")

    with pytest.raises(NotImplementedError) as exc_info:
        get_articulation_name_ordering(articulation, "robot_schema", kind="joint")

    message = str(exc_info.value)
    assert unresolved_target_path in message
    assert any(unresolved_target_path in record.message for record in caplog.records)


def test_robot_schema_ordering_helper_rejects_duplicate_relationship_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise instead of silently deduplicating a target reached twice through nested robots.

    USD collapses a relationship that lists the same target path twice, so the duplicate a real
    stage can express is the same leaf prim reached through two distinct nested robot-schema
    relationships. That nested case is exactly what the resolver must reject rather than dedupe.
    """
    stage, robot_prim = _author_robot_schema_stage(("sub_a", "sub_b"))
    sub_a_prim = stage.GetPrimAtPath(f"{_ROBOT_SCHEMA_PRIM_PATH}/sub_a")
    sub_b_prim = stage.GetPrimAtPath(f"{_ROBOT_SCHEMA_PRIM_PATH}/sub_b")
    duplicated_target_path = f"{_ROBOT_SCHEMA_PRIM_PATH}/joint_a"
    stage.DefinePrim(duplicated_target_path, "Xform")
    _author_robot_schema_relationship(
        robot_prim,
        "isaac:physics:robotJoints",
        [f"{_ROBOT_SCHEMA_PRIM_PATH}/sub_a", f"{_ROBOT_SCHEMA_PRIM_PATH}/sub_b"],
    )
    _author_robot_schema_relationship(sub_a_prim, "isaac:physics:robotJoints", [duplicated_target_path])
    _author_robot_schema_relationship(sub_b_prim, "isaac:physics:robotJoints", [duplicated_target_path])
    _install_robot_schema_source_asset(monkeypatch, robot_prim)

    articulation = _make_robot_schema_articulation(backend_joint_names=("joint_a", "joint_b"))

    with pytest.raises(ValueError, match=f"Duplicate .* target '{duplicated_target_path}'"):
        get_articulation_name_ordering(articulation, "robot_schema", kind="joint")


def _install_newton_usd_builder_mocks(monkeypatch: pytest.MonkeyPatch, stage: Usd.Stage, calls: dict, view_names: dict):
    """Install mocked Newton modules and a current-stage provider around a real in-memory stage.

    Only the optional Newton dependency (unavailable in this unit-test env) is mocked; ``pxr`` stays
    real so the resolver reads real prims and a real stage. Returns nothing; records builder activity
    into :paramref:`calls` and serves :paramref:`view_names` from the mocked articulation view.
    """

    class _JointType:
        FREE = 0
        FIXED = 1

    class _SolverMuJoCo:
        @staticmethod
        def register_custom_attributes(builder) -> None:
            calls["registered"] += 1

    class _ModelBuilder:
        def __init__(self, up_axis):
            self.up_axis = up_axis

        def add_usd(self, stage, **kwargs):
            calls["add_usd"].append(kwargs)

        def finalize(self, **kwargs):
            calls["finalize"].append(kwargs)
            return object()

    class _ArticulationView:
        def __init__(self, model, pattern, **kwargs):
            calls["views"].append((pattern, kwargs))
            self.joint_dof_names = list(view_names["joint"])
            self.link_names = list(view_names["body"])

    newton_mod = types.ModuleType("newton")
    newton_mod.JointType = _JointType
    newton_mod.ModelBuilder = _ModelBuilder
    newton_mod.solvers = types.SimpleNamespace(SolverMuJoCo=_SolverMuJoCo)
    selection_mod = types.ModuleType("newton.selection")
    selection_mod.ArticulationView = _ArticulationView
    schemas_mod = types.ModuleType("newton._src.usd.schemas")
    schemas_mod.SchemaResolverNewton = lambda: "newton_schema"
    schemas_mod.SchemaResolverPhysx = lambda: "physx_schema"
    stage_mod = types.ModuleType("isaaclab.sim.utils.stage")
    stage_mod.get_current_stage = lambda: stage
    monkeypatch.setitem(sys.modules, "newton", newton_mod)
    monkeypatch.setitem(sys.modules, "newton.selection", selection_mod)
    monkeypatch.setitem(sys.modules, "newton._src.usd.schemas", schemas_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", stage_mod)


def test_mjwarp_ordering_helper_builds_newton_view_from_usd_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve MJWarp ordering from Newton's USD builder path when no Newton root view exists."""
    calls = {"add_usd": [], "finalize": [], "views": [], "registered": 0}

    stage = Usd.Stage.CreateInMemory()
    robot_prim = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    root_prim = stage.DefinePrim("/World/envs/env_0/Robot/base", "Xform")

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        assert path_expr == "/World/envs/env_.*/Robot"
        if predicate is None:
            return [(robot_prim, "/World/envs/env_.*/Robot")]
        return [(root_prim, "/World/envs/env_.*/Robot/base")]

    _install_newton_usd_builder_mocks(
        monkeypatch,
        stage,
        calls,
        {"joint": ["knee", "hip", "ankle"], "body": ["base", "thigh", "foot"]},
    )
    _install_source_asset_resolver(monkeypatch, _resolve_matching_prims_from_source)

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

        @property
        def backend_joint_names(self) -> list[str]:
            return ["hip", "knee", "ankle"]

        @property
        def backend_body_names(self) -> list[str]:
            return ["foot", "base", "thigh"]

    articulation = _Articulation()

    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == ("knee", "hip", "ankle")
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == ("base", "thigh", "foot")
    assert calls["registered"] == 1
    assert len(calls["add_usd"]) == 1
    assert calls["add_usd"][0]["root_path"] == "/World/envs/env_0/Robot"
    assert calls["add_usd"][0]["joint_ordering"] == "dfs"
    assert calls["views"] == [("/World/envs/env_0/Robot/base", {"verbose": False, "exclude_joint_types": [0, 1]})]


def test_physx_ordering_helper_builds_bfs_newton_view_from_usd_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve PhysX ordering from Newton's BFS USD builder path when no PhysX root view exists."""
    calls = {"add_usd": [], "finalize": [], "views": [], "registered": 0}

    stage = Usd.Stage.CreateInMemory()
    robot_prim = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    root_prim = stage.DefinePrim("/World/envs/env_0/Robot/base", "Xform")

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        assert path_expr == "/World/envs/env_.*/Robot"
        if predicate is None:
            return [(robot_prim, "/World/envs/env_.*/Robot")]
        return [(root_prim, "/World/envs/env_.*/Robot/base")]

    _install_newton_usd_builder_mocks(
        monkeypatch,
        stage,
        calls,
        {
            "joint": ["hip", "shoulder", "knee", "elbow"],
            "body": ["base", "upper_arm", "forearm", "hand"],
        },
    )
    _install_source_asset_resolver(monkeypatch, _resolve_matching_prims_from_source)

    class _Articulation:
        __backend_name__ = "newton"
        _ordering_convention_name_cache: dict = {}
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

        @property
        def backend_joint_names(self) -> list[str]:
            return ["hip", "knee", "shoulder", "elbow"]

        @property
        def backend_body_names(self) -> list[str]:
            return ["base", "upper_arm", "hand", "forearm"]

    articulation = _Articulation()

    assert get_articulation_name_ordering(articulation, "physx", kind="joint") == (
        "hip",
        "shoulder",
        "knee",
        "elbow",
    )
    assert get_articulation_name_ordering(articulation, "physx", kind="body") == (
        "base",
        "upper_arm",
        "forearm",
        "hand",
    )
    assert calls["registered"] == 1
    assert len(calls["add_usd"]) == 1
    assert calls["add_usd"][0]["root_path"] == "/World/envs/env_0/Robot"
    assert calls["add_usd"][0]["joint_ordering"] == "bfs"
    assert calls["add_usd"][0]["bodies_follow_joint_ordering"] is True
    assert calls["views"] == [("/World/envs/env_0/Robot/base", {"verbose": False, "exclude_joint_types": [0, 1]})]


def test_symbolic_cross_backend_resolver_uses_newton_builder_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve a cross-backend symbolic preset through the Newton USD builder."""
    calls = []

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("hip", "knee", "ankle")
        backend_body_names = ("foot", "base", "thigh")

    articulation = _Articulation()

    def _build_names(candidate):
        calls.append(candidate)
        return {
            "joint": ("knee", "hip", "ankle"),
            "body": ("base", "thigh", "foot"),
        }

    monkeypatch.setattr(ordering_resolvers, "_get_mjwarp_names_from_newton_usd_builder", _build_names)

    user_names = _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=articulation.backend_joint_names,
        ordering=ArticulationOrderingConvention.MJWARP,
        active_backend_name=articulation.__backend_name__,
        articulation=articulation,
    )

    assert user_names == ("knee", "hip", "ankle")
    assert calls == [articulation]


def test_symbolic_resolver_skips_incomplete_cached_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to the Newton USD builder when cached names are incomplete."""

    class _Articulation:
        __backend_name__ = "physx"
        backend_joint_names = ("joint_0", "joint_1")
        backend_body_names = ("body_0", "body_1")
        _ordering_convention_name_cache = {
            (ArticulationOrderingConvention.MJWARP, "joint"): ("joint_0",),
        }

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_mjwarp_names_from_newton_usd_builder",
        lambda _: {
            "joint": ("joint_1", "joint_0"),
            "body": ("body_1", "body_0"),
        },
    )

    assert get_articulation_name_ordering(_Articulation(), "mjwarp", kind="joint") == ("joint_1", "joint_0")


def test_symbolic_resolver_does_not_cache_incomplete_builder_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache a builder result only when both joint and body orders are complete."""
    calls = []

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("joint_0", "joint_1")
        backend_body_names = ("body_0", "body_1")

    articulation = _Articulation()
    builder_names = {
        "joint": ("joint_0",),
        "body": ("body_1", "body_0"),
    }

    def _build_names(candidate):
        calls.append(candidate)
        return builder_names

    monkeypatch.setattr(ordering_resolvers, "_get_mjwarp_names_from_newton_usd_builder", _build_names)

    with pytest.raises(NotImplementedError, match="Unable to resolve 'mjwarp' joint ordering"):
        get_articulation_name_ordering(articulation, "mjwarp", kind="joint")
    assert articulation._ordering_convention_name_cache == {}

    builder_names["joint"] = ("joint_1", "joint_0")
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == ("joint_1", "joint_0")
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == ("body_1", "body_0")
    assert calls == [articulation, articulation]


def test_symbolic_cross_backend_resolver_normalizes_newton_multi_dof_joint_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize Newton multi-DoF spelling only when each match is unique."""

    class _Articulation:
        __backend_name__ = "physx"
        _ordering_convention_name_cache: dict = {}
        backend_joint_names = ("hinge", "ball_rot_z", "ball_rot_x", "ball_rot_y")
        backend_body_names = ("base",)

    articulation = _Articulation()
    monkeypatch.setattr(
        ordering_resolvers,
        "_get_mjwarp_names_from_newton_usd_builder",
        lambda _: {
            "joint": ("ball:rot_x", "ball:rot_y", "ball:rot_z", "hinge"),
            "body": ("base",),
        },
    )

    user_names = _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=articulation.backend_joint_names,
        ordering=ArticulationOrderingConvention.MJWARP,
        active_backend_name=articulation.__backend_name__,
        articulation=articulation,
    )

    assert user_names == ("ball_rot_x", "ball_rot_y", "ball_rot_z", "hinge")
    name_map = build_articulation_name_map(
        kind="joint",
        backend_names=articulation.backend_joint_names,
        user_names=user_names,
        device="cpu",
    )
    assert name_map.user_to_backend_indices == (2, 3, 1, 0)


def test_multi_dof_name_normalization_keeps_ambiguous_spellings() -> None:
    """Do not rewrite separator variants when canonical backend names collide."""
    convention_names = ("ball:rot_x", "ball_rot:x")
    backend_names = ("ball_rot_x", "ball:rot_x")

    assert ordering_resolvers._match_backend_joint_name_spellings(convention_names, backend_names) == convention_names


def test_base_articulation_data_property_uses_base_data_contract() -> None:
    """Annotate the abstract property with the type implemented by every backend."""
    return_annotation = BaseArticulation.data.fget.__annotations__["return"]
    assert return_annotation == "BaseArticulationData"


def test_base_articulation_keeps_none_ordering_on_default_path() -> None:
    """Keep the default public ordering path free of ordering maps."""
    apply_calls = []
    data = types.SimpleNamespace(
        joint_ordering=None,
        body_ordering=None,
        joint_names=None,
        body_names=None,
        _apply_ordering_maps_after_resolve=lambda: apply_calls.append("called"),
    )
    articulation = types.SimpleNamespace(
        __backend_name__="mock",
        cfg=types.SimpleNamespace(joint_ordering=None, body_ordering=None),
        data=data,
        backend_joint_names=["hip", "knee"],
        backend_body_names=["base", "foot"],
        device="cpu",
    )
    articulation._resolve_axis_ordering = BaseArticulation._resolve_axis_ordering.__get__(articulation)

    BaseArticulation._resolve_and_install_ordering_maps(articulation)

    assert data.joint_ordering is None
    assert data.body_ordering is None
    assert data.joint_names == ["hip", "knee"]
    assert data.body_names == ["base", "foot"]
    # The install site always lets the data container reconcile its staging;
    # backend hooks are no-ops when no ordering is or was active.
    assert apply_calls == ["called"]


class _LegacyArticulation(BaseArticulation):
    """Old-style backend that predates the ordering introspection properties."""

    @property
    def data(self):
        return self._data

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names


class _IncompleteArticulation(BaseArticulation):
    """Backend overriding neither the public nor the backend name properties."""

    @property
    def data(self):
        return self._data


def _make_ordering_resolution_articulation(
    *,
    body_ordering,
    backend_body_names: tuple[str, ...] = ("base", "foot"),
    is_fixed_base: bool,
):
    """Create a minimal articulation-shaped object for base ordering resolution."""
    data = types.SimpleNamespace(
        joint_ordering=None,
        body_ordering=None,
        joint_names=None,
        body_names=None,
        _apply_ordering_maps_after_resolve=lambda: None,
    )
    articulation = types.SimpleNamespace(
        __backend_name__="mock",
        _ordering_convention_name_cache={},
        cfg=types.SimpleNamespace(
            prim_path="/World/Robot",
            joint_ordering=None,
            body_ordering=body_ordering,
        ),
        data=data,
        backend_joint_names=["joint"],
        backend_body_names=list(backend_body_names),
        is_fixed_base=is_fixed_base,
        device="cpu",
    )
    articulation._resolve_axis_ordering = BaseArticulation._resolve_axis_ordering.__get__(articulation)
    return articulation


def test_base_articulation_ordering_contract_preserves_legacy_subclasses() -> None:
    """Keep third-party backends instantiable while ordering properties are deprecated fallbacks."""
    articulation = object.__new__(_LegacyArticulation)
    articulation._joint_names = ["hip", "knee"]
    articulation._body_names = ["base", "foot"]
    articulation._data = types.SimpleNamespace(joint_ordering=None, body_ordering=None)

    with pytest.warns(DeprecationWarning, match="override backend_joint_names"):
        assert articulation.backend_joint_names == ["hip", "knee"]
    with pytest.warns(DeprecationWarning, match="override backend_body_names"):
        assert articulation.backend_body_names == ["base", "foot"]
    assert articulation.joint_ordering is None
    assert articulation.body_ordering is None


def test_backend_name_fallbacks_reject_missing_overrides() -> None:
    """Raise instead of recursing when a backend overrides neither name property of an axis."""
    articulation = object.__new__(_IncompleteArticulation)
    articulation._data = types.SimpleNamespace(
        joint_ordering=None, body_ordering=None, joint_names=None, body_names=None
    )

    with pytest.raises(NotImplementedError, match="joint_names"):
        _ = articulation.backend_joint_names
    with pytest.raises(NotImplementedError, match="body_names"):
        _ = articulation.backend_body_names
    # The public properties funnel into the same fallbacks and must fail identically.
    with pytest.raises(NotImplementedError, match="joint_names"):
        _ = articulation.joint_names
    with pytest.raises(NotImplementedError, match="body_names"):
        _ = articulation.body_names


def _make_map_ids_articulation(*, permuted: bool) -> _LegacyArticulation:
    """Create a name-mapping articulation with a 3-cycle joint/body permutation or identity maps."""
    articulation = object.__new__(_LegacyArticulation)
    articulation._joint_names = ["j_b", "j_c", "j_a"]
    articulation._body_names = ["link", "foot", "base"]
    if permuted:
        # Opposite 3-cycles per axis: joint user_to_backend == (1, 2, 0) differs from its
        # backend_to_user == (2, 0, 1) and from the body maps (u2b == (2, 0, 1)), so neither
        # a direction-swapped nor an axis-swapped implementation can pass.
        joint_ordering = build_articulation_name_map(
            kind="joint", backend_names=("j_a", "j_b", "j_c"), user_names=("j_b", "j_c", "j_a"), device="cpu"
        )
        body_ordering = build_articulation_name_map(
            kind="body", backend_names=("base", "link", "foot"), user_names=("foot", "base", "link"), device="cpu"
        )
    else:
        joint_ordering = None
        body_ordering = None
    articulation._data = types.SimpleNamespace(joint_ordering=joint_ordering, body_ordering=body_ordering)
    return articulation


def test_map_ids_to_backend_applies_three_cycle_permutation() -> None:
    """Translate public indices through the exact user-to-backend direction of a 3-cycle."""
    articulation = _make_map_ids_articulation(permuted=True)

    assert articulation.map_joint_ids_to_backend([0, 1, 2]) == [1, 2, 0]
    assert articulation.map_joint_ids_to_backend([2, 0]) == [0, 1]
    assert articulation.map_body_ids_to_backend([0, 1, 2]) == [2, 0, 1]
    assert articulation.map_body_ids_to_backend([1]) == [0]


def test_map_ids_to_backend_handles_slice_selectors() -> None:
    """Accept slice selectors on both the identity and the permutation paths."""
    permuted = _make_map_ids_articulation(permuted=True)
    assert permuted.map_joint_ids_to_backend(slice(None)) == [1, 2, 0]
    assert permuted.map_joint_ids_to_backend(slice(1, 3)) == [2, 0]
    assert permuted.map_body_ids_to_backend(slice(None)) == [2, 0, 1]
    assert permuted.map_body_ids_to_backend(slice(0, 3, 2)) == [2, 1]

    identity = _make_map_ids_articulation(permuted=False)
    all_items = slice(None)
    assert identity.map_joint_ids_to_backend(all_items) is all_items
    assert identity.map_body_ids_to_backend(all_items) is all_items


def test_ordering_map_helpers_pick_axis_direction_and_identity_fallback() -> None:
    """Return the exact map object per axis and direction, or the backend identity buffer."""
    permuted = _make_map_ids_articulation(permuted=True)
    joint_ordering = permuted._data.joint_ordering
    body_ordering = permuted._data.body_ordering
    assert permuted._joint_user_to_backend_map() is joint_ordering.user_to_backend
    assert permuted._joint_backend_to_user_map() is joint_ordering.backend_to_user
    assert permuted._body_user_to_backend_map() is body_ordering.user_to_backend
    assert permuted._body_backend_to_user_map() is body_ordering.backend_to_user

    identity = _make_map_ids_articulation(permuted=False)
    joint_identity_map = object()
    body_identity_map = object()
    identity._ALL_JOINT_INDICES = joint_identity_map
    identity._ALL_BODY_INDICES = body_identity_map
    assert identity._joint_user_to_backend_map() is joint_identity_map
    assert identity._joint_backend_to_user_map() is joint_identity_map
    assert identity._body_user_to_backend_map() is body_identity_map
    assert identity._body_backend_to_user_map() is body_identity_map


@pytest.mark.parametrize("ordering", [("foot", "base"), ArticulationOrderingConvention.MJWARP])
def test_fixed_base_body_ordering_rejects_root_relocation(ordering, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject explicit and symbolic fixed-base orders that move the root from public index zero."""
    monkeypatch.setattr(
        ordering_resolvers,
        "_get_mjwarp_names_from_newton_usd_builder",
        lambda _: {"joint": ("joint",), "body": ("foot", "base")},
    )
    articulation = _make_ordering_resolution_articulation(body_ordering=ordering, is_fixed_base=True)

    with pytest.raises(ValueError) as exc_info:
        BaseArticulation._resolve_and_install_ordering_maps(articulation)

    assert str(exc_info.value) == (
        "Invalid body_ordering for fixed-base articulation '/World/Robot': root body 'base' must remain at public "
        "index 0, but was requested at index 1. Put 'base' first; all remaining bodies may be reordered freely."
    )
    assert articulation.data.body_ordering is None


def test_floating_base_body_ordering_accepts_root_relocation() -> None:
    """Allow a complete body permutation to move the root for floating-base articulations."""
    articulation = _make_ordering_resolution_articulation(
        body_ordering=("foot", "base"),
        is_fixed_base=False,
    )

    BaseArticulation._resolve_and_install_ordering_maps(articulation)

    assert articulation.data.body_names == ["foot", "base"]


def test_explicit_backend_order_normalizes_to_none() -> None:
    """Normalize orderings that resolve to backend order to ``None`` instead of identity maps."""
    articulation = _make_ordering_resolution_articulation(
        body_ordering=("base", "foot"),
        is_fixed_base=True,
    )
    articulation.cfg.joint_ordering = ("joint",)

    BaseArticulation._resolve_and_install_ordering_maps(articulation)

    assert articulation.data.joint_ordering is None
    assert articulation.data.body_ordering is None
    assert articulation.data.joint_names == ["joint"]
    assert articulation.data.body_names == ["base", "foot"]


def test_base_articulation_data_defines_optional_ordering_maps() -> None:
    """Expose optional ordering maps on articulation data containers."""
    assert hasattr(BaseArticulationData, "joint_ordering")
    assert hasattr(BaseArticulationData, "body_ordering")


def test_build_articulation_name_map_builds_permutation_indices_and_device_maps() -> None:
    """Build CPU and device maps for an explicit user ordering permutation."""
    name_map = build_articulation_name_map(
        kind="body",
        backend_names=("base", "left_foot", "right_foot"),
        user_names=("right_foot", "base", "left_foot"),
        device="cpu",
    )

    assert name_map.user_to_backend_indices == (2, 0, 1)
    assert name_map.backend_to_user_indices == (1, 2, 0)
    assert name_map.user_to_backend is not None
    assert name_map.backend_to_user is not None
    np.testing.assert_array_equal(name_map.user_to_backend.numpy(), np.asarray([2, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(name_map.backend_to_user.numpy(), np.asarray([1, 2, 0], dtype=np.int32))


def test_build_articulation_name_map_rejects_duplicate_backend_names() -> None:
    """Reject backend names that cannot be mapped unambiguously."""
    with pytest.raises(ValueError, match="Duplicate backend joint names"):
        build_articulation_name_map(
            kind="joint",
            backend_names=("hip", "hip"),
            user_names=("hip", "knee"),
            device="cpu",
        )


def test_build_articulation_name_map_rejects_duplicate_requested_names() -> None:
    """Reject requested user names that cannot be mapped unambiguously."""
    with pytest.raises(ValueError, match="Duplicate requested body names"):
        build_articulation_name_map(
            kind="body",
            backend_names=("base", "foot"),
            user_names=("base", "base"),
            device="cpu",
        )


def test_build_articulation_name_map_rejects_incomplete_permutation() -> None:
    """Reject requested names that are not a complete backend-name permutation."""
    with pytest.raises(ValueError, match=r"Missing=\['knee'\], extra=\['wheel'\]"):
        build_articulation_name_map(
            kind="joint",
            backend_names=("hip", "knee"),
            user_names=("hip", "wheel"),
            device="cpu",
        )


def test_build_articulation_name_map_returns_none_for_identity() -> None:
    """Identity orderings have no map object; None is the single identity state."""
    backend_names = ("j_a", "j_b", "j_c")
    assert (
        build_articulation_name_map(kind="joint", backend_names=backend_names, user_names=backend_names, device="cpu")
        is None
    )
    assert build_articulation_name_map(kind="joint", backend_names=backend_names, user_names=None, device="cpu") is None

    permuted = build_articulation_name_map(
        kind="joint", backend_names=backend_names, user_names=("j_c", "j_a", "j_b"), device="cpu"
    )
    assert permuted is not None
    assert permuted.user_to_backend_indices == (2, 0, 1)
    assert permuted.backend_to_user_indices == (1, 2, 0)
    assert not hasattr(permuted, "is_identity")
    assert not hasattr(permuted, "user_names")


def test_ordering_state_lives_only_on_data() -> None:
    """No mirrored ordering flags or maps exist on the asset or data classes."""
    from isaaclab.assets.articulation.base_articulation import BaseArticulation
    from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

    for owner, names in (
        (BaseArticulation, ("_cache_ordering_maps", "_reset_and_cache_ordering_maps")),
        (BaseArticulationData, ("_install_ordering_flags", "_has_joint_ordering", "_has_body_ordering")),
    ):
        for name in names:
            assert not hasattr(owner, name), f"{owner.__name__}.{name} should be deleted"


_NONIDENTITY_ORDERING = types.SimpleNamespace(
    user_to_backend_indices=(0, 2, 1),
    backend_to_user_indices=(0, 2, 1),
    is_identity=False,
)


class _MapBodyIdsSurface:
    """Minimal surface exercising the real body-id translation method."""

    map_body_ids_to_backend = BaseArticulation.map_body_ids_to_backend

    def __init__(self, body_ordering):
        self.body_ordering = body_ordering


class _MapJointIdsSurface:
    """Minimal surface exercising the real joint-id translation method."""

    map_joint_ids_to_backend = BaseArticulation.map_joint_ids_to_backend

    def __init__(self, joint_ordering):
        self.joint_ordering = joint_ordering


def test_map_body_ids_to_backend_returns_input_unchanged_for_default_ordering() -> None:
    """A ``None`` body ordering returns the input object unchanged.

    ``None`` covers identity-configured orderings too: assets normalize
    identity maps away at install time, so this is the only fast-path state.
    """
    asset = _MapBodyIdsSurface(None)
    body_ids = [2, 0, 1]
    assert asset.map_body_ids_to_backend(body_ids) is body_ids


def test_map_body_ids_to_backend_permutes_ids_for_nonidentity_ordering() -> None:
    """A nonidentity body ordering gathers public IDs into backend order."""
    asset = _MapBodyIdsSurface(_NONIDENTITY_ORDERING)
    # user_to_backend_indices == (0, 2, 1): public 1 -> backend 2, public 2 -> backend 1
    assert asset.map_body_ids_to_backend([1, 2]) == [2, 1]


def test_map_joint_ids_to_backend_returns_input_unchanged_for_default_ordering() -> None:
    """A ``None`` (default) joint ordering returns the input object unchanged."""
    asset = _MapJointIdsSurface(None)
    joint_ids = [2, 0, 1]
    assert asset.map_joint_ids_to_backend(joint_ids) is joint_ids


def test_map_joint_ids_to_backend_permutes_ids_for_nonidentity_ordering() -> None:
    """A nonidentity joint ordering gathers public IDs into backend order."""
    asset = _MapJointIdsSurface(_NONIDENTITY_ORDERING)
    # user_to_backend_indices == (0, 2, 1): public 1 -> backend 2, public 2 -> backend 1
    assert asset.map_joint_ids_to_backend([1, 2]) == [2, 1]
