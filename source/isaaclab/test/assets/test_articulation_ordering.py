# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import types
from collections import UserList
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import isaaclab.assets.articulation.ordering_resolvers as ordering_resolvers

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from isaaclab.assets.articulation.ordering import (
    ArticulationNameMap,
    ArticulationOrderingConvention,
    apply_articulation_ordering_preset,
    build_articulation_name_map,
    parse_articulation_ordering_convention,
)
from isaaclab.assets.articulation.ordering_resolvers import (
    _resolve_articulation_convention_name_ordering,
    _resolve_articulation_ordering_names,
    get_mjwarp_articulation_name_ordering,
    get_physx_articulation_name_ordering,
    get_robot_schema_articulation_name_ordering,
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

from isaaclab_newton.actuators.kernels import sync_torque_telemetry

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.assets.articulation.ordering_kernels import (
    reorder_2d_backend_to_user,
    reorder_2d_user_to_backend,
    reorder_3d_backend_to_user,
    write_2d_float_user_to_backend_with_indices,
    write_2d_float_user_to_backend_with_mask,
    write_joint_state_user_to_backend_with_indices,
    write_joint_state_user_to_backend_with_mask,
    write_joint_vel_user_to_backend_with_indices,
    write_joint_vel_user_to_backend_with_mask,
    write_scalar_user_to_backend_with_indices,
    write_scalar_user_to_backend_with_mask,
)

if _inserted_sim_stub:
    sys.modules.pop("isaaclab.sim", None)
if _inserted_sim_context_stub:
    sys.modules.pop("isaaclab.sim.simulation_context", None)
if _inserted_asset_base_stub:
    sys.modules.pop("isaaclab.assets.asset_base", None)


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


@pytest.mark.parametrize(
    "entrypoint",
    ["resolve_names", "resolve_convention", "physx", "mjwarp", "robot_schema"],
)
def test_ordering_resolvers_reject_invalid_kind(entrypoint: str) -> None:
    """Reject misspelled element kinds at every resolver entry point."""

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
        elif entrypoint == "physx":
            get_physx_articulation_name_ordering(articulation, kind="dof")  # type: ignore[arg-type]
        elif entrypoint == "mjwarp":
            get_mjwarp_articulation_name_ordering(articulation, kind="dof")  # type: ignore[arg-type]
        else:
            get_robot_schema_articulation_name_ordering(articulation, kind="dof")  # type: ignore[arg-type]


def test_assets_package_reexports_public_ordering_symbols() -> None:
    """Expose every documented ordering symbol from the public assets package."""
    from isaaclab import assets

    expected_exports = {
        "ArticulationOrderingConvention": ArticulationOrderingConvention,
        "ArticulationNameMap": ArticulationNameMap,
        "apply_articulation_ordering_preset": apply_articulation_ordering_preset,
        "build_articulation_name_map": build_articulation_name_map,
        "parse_articulation_ordering_convention": parse_articulation_ordering_convention,
        "get_mjwarp_articulation_name_ordering": get_mjwarp_articulation_name_ordering,
        "get_physx_articulation_name_ordering": get_physx_articulation_name_ordering,
        "get_robot_schema_articulation_name_ordering": get_robot_schema_articulation_name_ordering,
    }
    for name, expected_export in expected_exports.items():
        assert getattr(assets, name, None) is expected_export, name
    assert not hasattr(assets, "resolve_articulation_ordering_names")
    assert not hasattr(assets, "resolve_articulation_convention_name_ordering")


def test_assets_api_page_defines_ordering_symbols() -> None:
    """Publish exact ordering directives; the preceding test covers re-exports."""
    repo_root = Path(__file__).resolve().parents[4]
    api_page = (repo_root / "docs/source/api/lab/isaaclab.assets.rst").read_text(encoding="utf-8")
    actual_directives = {line.strip() for line in api_page.splitlines() if line.strip().startswith(".. ")}

    expected_directives = {
        ".. currentmodule:: isaaclab.assets",
        ".. autoclass:: ArticulationOrderingConvention",
        ".. autoclass:: ArticulationNameMap",
        ".. autofunction:: apply_articulation_ordering_preset",
        ".. autofunction:: build_articulation_name_map",
        ".. autofunction:: parse_articulation_ordering_convention",
        ".. autofunction:: get_mjwarp_articulation_name_ordering",
        ".. autofunction:: get_physx_articulation_name_ordering",
        ".. autofunction:: get_robot_schema_articulation_name_ordering",
    }
    missing_directives = expected_directives - actual_directives
    assert not missing_directives, f"Missing exact RST directives: {sorted(missing_directives)}"


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


def test_articulation_name_map_direct_constructor_validates_maps() -> None:
    """Reject inconsistent public name maps built without the factory."""
    identity_map = wp.array((0, 1, 2), dtype=wp.int32, device="cpu")
    reversed_map = wp.array((2, 1, 0), dtype=wp.int32, device="cpu")

    with pytest.raises(ValueError, match="inverse"):
        ArticulationNameMap(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            user_names=("joint_2", "joint_1", "joint_0"),
            user_to_backend_indices=(2, 1, 0),
            backend_to_user_indices=(0, 1, 2),
            user_to_backend=reversed_map,
            backend_to_user=identity_map,
            is_identity=False,
        )


def test_articulation_name_map_direct_constructor_validates_name_index_correspondence() -> None:
    """Reject maps whose indices do not map each public name to the same backend name."""
    identity_map = wp.array((0, 1, 2), dtype=wp.int32, device="cpu")

    with pytest.raises(ValueError, match="name and index mappings must agree"):
        ArticulationNameMap(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            user_names=("joint_2", "joint_1", "joint_0"),
            user_to_backend_indices=(0, 1, 2),
            backend_to_user_indices=(0, 1, 2),
            user_to_backend=identity_map,
            backend_to_user=identity_map,
            is_identity=False,
        )


def test_articulation_name_map_direct_constructor_validates_device_maps() -> None:
    """Reject device maps that contradict the validated CPU permutations."""
    identity_map = wp.array((0, 1, 2), dtype=wp.int32, device="cpu")
    reversed_map = wp.array((2, 1, 0), dtype=wp.int32, device="cpu")

    with pytest.raises(ValueError, match="device user_to_backend map must match user_to_backend_indices"):
        ArticulationNameMap(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            user_names=("joint_2", "joint_1", "joint_0"),
            user_to_backend_indices=(2, 1, 0),
            backend_to_user_indices=(2, 1, 0),
            user_to_backend=identity_map,
            backend_to_user=reversed_map,
            is_identity=False,
        )


def _make_identity_articulation_name_map(**overrides) -> ArticulationNameMap:
    """Construct an identity name map with selected direct-constructor overrides."""
    identity_device_map = wp.array((0, 1, 2), dtype=wp.int32, device="cpu")
    fields = {
        "kind": "joint",
        "backend_names": ("joint_0", "joint_1", "joint_2"),
        "user_names": ("joint_0", "joint_1", "joint_2"),
        "user_to_backend_indices": (0, 1, 2),
        "backend_to_user_indices": (0, 1, 2),
        "user_to_backend": identity_device_map,
        "backend_to_user": identity_device_map,
        "is_identity": True,
    }
    fields.update(overrides)
    return ArticulationNameMap(**fields)


@pytest.mark.parametrize("field_name", ["user_to_backend_indices", "backend_to_user_indices"])
@pytest.mark.parametrize("invalid_value", [0.5, "0", True])
def test_articulation_name_map_direct_constructor_rejects_non_integer_indices(field_name: str, invalid_value) -> None:
    """Reject lossy or boolean values in either CPU index permutation."""
    with pytest.raises(TypeError, match=rf"{field_name} element 0"):
        _make_identity_articulation_name_map(**{field_name: (invalid_value, 1, 2)})


@pytest.mark.parametrize("index_type", [int, np.int64])
def test_articulation_name_map_direct_constructor_accepts_integral_indices(index_type) -> None:
    """Accept Python and NumPy integral indices and normalize them to int."""
    indices = tuple(index_type(index) for index in range(3))

    name_map = _make_identity_articulation_name_map(
        user_to_backend_indices=indices,
        backend_to_user_indices=indices,
    )

    assert name_map.user_to_backend_indices == (0, 1, 2)
    assert name_map.backend_to_user_indices == (0, 1, 2)
    assert all(type(index) is int for index in name_map.user_to_backend_indices)
    assert all(type(index) is int for index in name_map.backend_to_user_indices)


@pytest.mark.parametrize("invalid_identity", [1, np.bool_(True)])
def test_articulation_name_map_direct_constructor_rejects_non_bool_identity(invalid_identity) -> None:
    """Require is_identity to be a built-in bool."""
    with pytest.raises(TypeError, match="is_identity must be bool"):
        _make_identity_articulation_name_map(is_identity=invalid_identity)


@pytest.mark.parametrize(
    ("field_name", "backend_names", "user_names", "invalid_type"),
    [
        ("backend_names", "joint_0", ("joint_0",), "str"),
        ("backend_names", b"joint_0", ("joint_0",), "bytes"),
        ("user_names", ("joint_0",), "joint_0", "str"),
        ("user_names", ("joint_0",), bytearray(b"joint_0"), "bytearray"),
    ],
)
def test_build_articulation_name_map_rejects_scalar_name_sequences(
    field_name: str, backend_names, user_names, invalid_type: str
) -> None:
    """Reject scalar strings and byte buffers instead of splitting them into names."""
    with pytest.raises(TypeError, match=rf"^{field_name} must be a sequence of strings; got {invalid_type}\.$"):
        build_articulation_name_map(
            kind="joint",
            backend_names=backend_names,
            user_names=user_names,
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


def test_resolve_articulation_ordering_names_accepts_explicit_sequence() -> None:
    """Resolve explicit public ordering names from config."""
    user_names = _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=("joint_0", "joint_1", "joint_2"),
        ordering=("joint_2", "joint_0", "joint_1"),
        active_backend_name="physx",
    )

    assert user_names == ("joint_2", "joint_0", "joint_1")


def test_resolve_articulation_ordering_names_accepts_generic_sequence() -> None:
    """Resolve explicit public ordering names from a generic sequence."""
    user_names = _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=("joint_0", "joint_1", "joint_2"),
        ordering=UserList(["joint_2", "joint_0", "joint_1"]),
        active_backend_name="physx",
    )

    assert user_names == ("joint_2", "joint_0", "joint_1")


def test_resolve_articulation_ordering_names_reports_non_string_element() -> None:
    """Report the location and type of invalid explicit ordering elements."""
    with pytest.raises(TypeError) as exc_info:
        _resolve_articulation_ordering_names(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            ordering=UserList(["joint_1", 7]),
            active_backend_name="physx",
        )

    assert str(exc_info.value) == "joint_ordering element 1 must be str; got 7 (int)."


@pytest.mark.parametrize(
    ("ordering", "type_name"),
    [
        (7, "int"),
        (b"", "bytes"),
        (b"joint_0", "bytes"),
        (bytearray(), "bytearray"),
        (bytearray(b"joint_0"), "bytearray"),
    ],
)
def test_resolve_articulation_ordering_names_reports_unsupported_type(ordering, type_name: str) -> None:
    """Report accepted resolver inputs for unsupported scalar orderings."""
    with pytest.raises(TypeError) as exc_info:
        _resolve_articulation_ordering_names(
            kind="joint",
            backend_names=("joint_0", "joint_1", "joint_2"),
            ordering=ordering,
            active_backend_name="physx",
        )

    assert str(exc_info.value) == (
        f"joint_ordering must be a name sequence, convention string/enum, or None; got {type_name}."
    )


def test_resolve_articulation_ordering_names_keeps_matching_backend_preset_identity() -> None:
    """Resolve same-backend symbolic presets without requiring traversal helpers."""
    user_names = _resolve_articulation_ordering_names(
        kind="body",
        backend_names=("base", "left_foot", "right_foot"),
        ordering=ArticulationOrderingConvention.PHYSX,
        active_backend_name="physx",
    )

    assert user_names == ("base", "left_foot", "right_foot")


@pytest.mark.parametrize("backend_name", ["physx", "ovphysx"])
def test_physx_ordering_helper_uses_same_backend_identity_without_discovery(
    monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Return PhysX and OVPhysX backend names without metadata discovery."""

    class _Articulation:
        __backend_name__ = backend_name
        backend_joint_names = ("shoulder", "elbow", "wrist")
        backend_body_names = ("base", "arm", "hand")

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_physx_names_from_newton_usd_builder",
        lambda _: pytest.fail("same-backend resolution must not invoke the USD builder"),
    )

    articulation = _Articulation()
    assert get_physx_articulation_name_ordering(articulation, kind="joint") == articulation.backend_joint_names
    assert get_physx_articulation_name_ordering(articulation, kind="body") == articulation.backend_body_names


def test_mjwarp_ordering_helper_uses_newton_identity_without_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return Newton backend names without metadata discovery."""

    class _Articulation:
        __backend_name__ = "newton"
        backend_joint_names = ("knee", "hip", "ankle")
        backend_body_names = ("base", "thigh", "foot")

    monkeypatch.setattr(
        ordering_resolvers,
        "_get_mjwarp_names_from_newton_usd_builder",
        lambda _: pytest.fail("same-backend resolution must not invoke the USD builder"),
    )

    articulation = _Articulation()
    assert get_mjwarp_articulation_name_ordering(articulation, kind="joint") == articulation.backend_joint_names
    assert get_mjwarp_articulation_name_ordering(articulation, kind="body") == articulation.backend_body_names


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
        backend_joint_names = ("hip", "knee")
        backend_body_names = ("base", "foot")

    monkeypatch.setattr(ordering_resolvers, "_get_mjwarp_names_from_newton_usd_builder", lambda _: None)

    with pytest.raises(NotImplementedError) as exc_info:
        get_mjwarp_articulation_name_ordering(_Articulation(), kind=kind)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert f"Unable to resolve 'mjwarp' {kind} ordering" in message
    assert "active backend 'physx'" in message
    assert f"env.scene.robot.{config_field}" in message
    assert f"explicit {kind}-name permutation" in message


def test_robot_schema_ordering_helper_reads_authored_relationships(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve explicit robot schema ordering from authored USD relationship targets."""

    class _Path:
        def __init__(self, value: str):
            self.pathString = value

        def __str__(self) -> str:
            return self.pathString

    class _Attribute:
        def __init__(self, value: str):
            self._value = value

        def Get(self):
            return self._value

    class _Relationship:
        def __init__(self, targets: list[str]):
            self._targets = [_Path(target) for target in targets]

        def GetTargets(self):
            return self._targets

    class _Prim:
        def __init__(
            self,
            path: str,
            *,
            attrs: dict[str, str] | None = None,
            relationships: dict[str, _Relationship] | None = None,
        ):
            self._path = _Path(path)
            self._attrs = attrs or {}
            self._relationships = relationships or {}
            self._stage = None

        def GetPath(self):
            return self._path

        def GetName(self) -> str:
            return str(self._path).rsplit("/", maxsplit=1)[-1]

        def GetAttribute(self, name: str):
            if name not in self._attrs:
                return None
            return _Attribute(self._attrs[name])

        def GetRelationship(self, name: str):
            return self._relationships.get(name)

        def GetStage(self):
            return self._stage

        def IsValid(self) -> bool:
            return True

    class _Stage:
        def __init__(self, prims: list[_Prim]):
            self._prims = {str(prim.GetPath()): prim for prim in prims}
            for prim in prims:
                prim._stage = self

        def GetPrimAtPath(self, path):
            return self._prims[str(path)]

    robot_prim = _Prim(
        "/World/envs/env_0/Robot",
        relationships={
            "isaac:physics:robotJoints": _Relationship(
                [
                    "/World/envs/env_0/Robot/joint_b",
                    "/World/envs/env_0/Robot/joint_a_prim",
                    "/World/envs/env_0/Robot/joint_c",
                ]
            ),
            "isaac:physics:robotLinks": _Relationship(
                [
                    "/World/envs/env_0/Robot/base_prim",
                    "/World/envs/env_0/Robot/tool_site",
                    "/World/envs/env_0/Robot/thigh",
                    "/World/envs/env_0/Robot/foot",
                ]
            ),
        },
    )
    _Stage(
        [
            robot_prim,
            _Prim("/World/envs/env_0/Robot/joint_a_prim", attrs={"isaac:NameOverride": "joint_a"}),
            _Prim("/World/envs/env_0/Robot/joint_b"),
            _Prim("/World/envs/env_0/Robot/joint_c"),
            _Prim("/World/envs/env_0/Robot/base_prim", attrs={"isaac:nameOverride": "base"}),
            _Prim("/World/envs/env_0/Robot/tool_site"),
            _Prim("/World/envs/env_0/Robot/thigh"),
            _Prim("/World/envs/env_0/Robot/foot"),
        ]
    )

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        assert path_expr == "/World/envs/env_.*/Robot"
        return [(robot_prim, "/World/envs/env_.*/Robot")]

    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = _resolve_matching_prims_from_source
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)

    class _Articulation:
        __backend_name__ = "newton"
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

        @property
        def backend_joint_names(self) -> list[str]:
            return ["joint_a", "joint_b", "joint_c"]

        @property
        def backend_body_names(self) -> list[str]:
            return ["base", "thigh", "foot"]

    articulation = _Articulation()

    assert get_robot_schema_articulation_name_ordering(articulation, kind="joint") == (
        "joint_b",
        "joint_a",
        "joint_c",
    )
    assert get_robot_schema_articulation_name_ordering(articulation, kind="body") == ("base", "thigh", "foot")
    assert _resolve_articulation_ordering_names(
        kind="joint",
        backend_names=articulation.backend_joint_names,
        ordering="robot_schema",
        active_backend_name=articulation.__backend_name__,
        articulation=articulation,
    ) == ("joint_b", "joint_a", "joint_c")


def test_robot_schema_ordering_helper_rejects_incomplete_relationships(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject robot schema relationships that are not complete backend-name permutations."""

    class _Path:
        def __init__(self, value: str):
            self.pathString = value

        def __str__(self) -> str:
            return self.pathString

    class _Relationship:
        def __init__(self, targets: list[str]):
            self._targets = [_Path(target) for target in targets]

        def GetTargets(self):
            return self._targets

    class _Prim:
        def __init__(self, path: str, relationships: dict[str, _Relationship] | None = None):
            self._path = _Path(path)
            self._relationships = relationships or {}
            self._stage = None

        def GetPath(self):
            return self._path

        def GetName(self) -> str:
            return str(self._path).rsplit("/", maxsplit=1)[-1]

        def GetAttribute(self, name: str):
            return None

        def GetRelationship(self, name: str):
            return self._relationships.get(name)

        def GetStage(self):
            return self._stage

        def IsValid(self) -> bool:
            return True

    class _Stage:
        def __init__(self, prims: list[_Prim]):
            self._prims = {str(prim.GetPath()): prim for prim in prims}
            for prim in prims:
                prim._stage = self

        def GetPrimAtPath(self, path):
            return self._prims[str(path)]

    robot_prim = _Prim(
        "/World/envs/env_0/Robot",
        relationships={
            "isaac:physics:robotJoints": _Relationship(
                [
                    "/World/envs/env_0/Robot/joint_b",
                    "/World/envs/env_0/Robot/joint_a",
                ]
            )
        },
    )
    _Stage(
        [
            robot_prim,
            _Prim("/World/envs/env_0/Robot/joint_a"),
            _Prim("/World/envs/env_0/Robot/joint_b"),
        ]
    )

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        return [(robot_prim, "/World/envs/env_.*/Robot")]

    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = _resolve_matching_prims_from_source
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)

    class _Articulation:
        __backend_name__ = "newton"
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

        @property
        def backend_joint_names(self) -> list[str]:
            return ["joint_a", "joint_b", "joint_c"]

    with pytest.raises(NotImplementedError, match="Unable to resolve 'robot_schema' joint ordering"):
        get_robot_schema_articulation_name_ordering(_Articulation(), kind="joint")


def test_mjwarp_ordering_helper_builds_newton_view_from_usd_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve MJWarp ordering from Newton's USD builder path when no Newton root view exists."""
    calls = {"add_usd": [], "finalize": [], "views": [], "registered": 0}

    class _Path:
        def __init__(self, value: str):
            self.pathString = value

    class _Prim:
        def __init__(self, path: str):
            self._path = _Path(path)

        def GetPath(self):
            return self._path

        def HasAPI(self, api) -> bool:
            return api is _UsdPhysics.ArticulationRootAPI

    class _Stage:
        pass

    class _UsdGeom:
        @staticmethod
        def GetStageUpAxis(stage):
            return "Z"

    class _UsdPhysics:
        class ArticulationRootAPI:
            pass

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
            self.joint_dof_names = ["knee", "hip", "ankle"]
            self.link_names = ["base", "thigh", "foot"]

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        assert path_expr == "/World/envs/env_.*/Robot"
        if predicate is None:
            return [(_Prim("/World/envs/env_0/Robot"), "/World/envs/env_.*/Robot")]
        return [(_Prim("/World/envs/env_0/Robot/base"), "/World/envs/env_.*/Robot/base")]

    newton_mod = types.ModuleType("newton")
    newton_mod.JointType = _JointType
    newton_mod.ModelBuilder = _ModelBuilder
    newton_mod.solvers = types.SimpleNamespace(SolverMuJoCo=_SolverMuJoCo)
    selection_mod = types.ModuleType("newton.selection")
    selection_mod.ArticulationView = _ArticulationView
    schemas_mod = types.ModuleType("newton._src.usd.schemas")
    schemas_mod.SchemaResolverNewton = lambda: "newton_schema"
    schemas_mod.SchemaResolverPhysx = lambda: "physx_schema"
    pxr_mod = types.ModuleType("pxr")
    pxr_mod.UsdGeom = _UsdGeom
    pxr_mod.UsdPhysics = _UsdPhysics
    stage_mod = types.ModuleType("isaaclab.sim.utils.stage")
    stage_mod.get_current_stage = lambda: _Stage()
    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = _resolve_matching_prims_from_source
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "newton", newton_mod)
    monkeypatch.setitem(sys.modules, "newton.selection", selection_mod)
    monkeypatch.setitem(sys.modules, "newton._src.usd.schemas", schemas_mod)
    monkeypatch.setitem(sys.modules, "pxr", pxr_mod)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", _UsdGeom)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", _UsdPhysics)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", stage_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)

    class _Articulation:
        __backend_name__ = "physx"
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

        @property
        def backend_joint_names(self) -> list[str]:
            return ["hip", "knee", "ankle"]

        @property
        def backend_body_names(self) -> list[str]:
            return ["foot", "base", "thigh"]

    articulation = _Articulation()

    assert get_mjwarp_articulation_name_ordering(articulation, kind="joint") == ("knee", "hip", "ankle")
    assert get_mjwarp_articulation_name_ordering(articulation, kind="body") == ("base", "thigh", "foot")
    assert calls["registered"] == 1
    assert len(calls["add_usd"]) == 1
    assert calls["add_usd"][0]["root_path"] == "/World/envs/env_0/Robot"
    assert calls["add_usd"][0]["joint_ordering"] == "dfs"
    assert calls["views"] == [("/World/envs/env_0/Robot/base", {"verbose": False, "exclude_joint_types": [0, 1]})]


def test_physx_ordering_helper_builds_bfs_newton_view_from_usd_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve PhysX ordering from Newton's BFS USD builder path when no PhysX root view exists."""
    calls = {"add_usd": [], "finalize": [], "views": [], "registered": 0}

    class _Path:
        def __init__(self, value: str):
            self.pathString = value

    class _Prim:
        def __init__(self, path: str):
            self._path = _Path(path)

        def GetPath(self):
            return self._path

        def HasAPI(self, api) -> bool:
            return api is _UsdPhysics.ArticulationRootAPI

    class _Stage:
        pass

    class _UsdGeom:
        @staticmethod
        def GetStageUpAxis(stage):
            return "Z"

    class _UsdPhysics:
        class ArticulationRootAPI:
            pass

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
            self.joint_dof_names = ["hip", "shoulder", "knee", "elbow"]
            self.link_names = ["base", "upper_arm", "forearm", "hand"]

    def _resolve_matching_prims_from_source(path_expr, predicate=None, expected_num_matches=None):
        assert path_expr == "/World/envs/env_.*/Robot"
        if predicate is None:
            return [(_Prim("/World/envs/env_0/Robot"), "/World/envs/env_.*/Robot")]
        return [(_Prim("/World/envs/env_0/Robot/base"), "/World/envs/env_.*/Robot/base")]

    newton_mod = types.ModuleType("newton")
    newton_mod.JointType = _JointType
    newton_mod.ModelBuilder = _ModelBuilder
    newton_mod.solvers = types.SimpleNamespace(SolverMuJoCo=_SolverMuJoCo)
    selection_mod = types.ModuleType("newton.selection")
    selection_mod.ArticulationView = _ArticulationView
    schemas_mod = types.ModuleType("newton._src.usd.schemas")
    schemas_mod.SchemaResolverNewton = lambda: "newton_schema"
    schemas_mod.SchemaResolverPhysx = lambda: "physx_schema"
    pxr_mod = types.ModuleType("pxr")
    pxr_mod.UsdGeom = _UsdGeom
    pxr_mod.UsdPhysics = _UsdPhysics
    stage_mod = types.ModuleType("isaaclab.sim.utils.stage")
    stage_mod.get_current_stage = lambda: _Stage()
    queries_mod = types.ModuleType("isaaclab.sim.utils.queries")
    queries_mod.resolve_matching_prims_from_source = _resolve_matching_prims_from_source
    sim_utils_mod = types.ModuleType("isaaclab.sim.utils")
    monkeypatch.setattr(sim_stub, "__path__", [], raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_stub)
    monkeypatch.setitem(sys.modules, "newton", newton_mod)
    monkeypatch.setitem(sys.modules, "newton.selection", selection_mod)
    monkeypatch.setitem(sys.modules, "newton._src.usd.schemas", schemas_mod)
    monkeypatch.setitem(sys.modules, "pxr", pxr_mod)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", _UsdGeom)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", _UsdPhysics)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", sim_utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", stage_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries", queries_mod)

    class _Articulation:
        __backend_name__ = "newton"
        cfg = types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot", articulation_root_prim_path=None)

        @property
        def backend_joint_names(self) -> list[str]:
            return ["hip", "knee", "shoulder", "elbow"]

        @property
        def backend_body_names(self) -> list[str]:
            return ["base", "upper_arm", "hand", "forearm"]

    articulation = _Articulation()

    assert get_physx_articulation_name_ordering(articulation, kind="joint") == (
        "hip",
        "shoulder",
        "knee",
        "elbow",
    )
    assert get_physx_articulation_name_ordering(articulation, kind="body") == (
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

    assert get_mjwarp_articulation_name_ordering(_Articulation(), kind="joint") == ("joint_1", "joint_0")


def test_symbolic_resolver_does_not_cache_incomplete_builder_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache a builder result only when both joint and body orders are complete."""
    calls = []

    class _Articulation:
        __backend_name__ = "physx"
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
        get_mjwarp_articulation_name_ordering(articulation, kind="joint")
    assert not hasattr(articulation, "_ordering_convention_name_cache")

    builder_names["joint"] = ("joint_1", "joint_0")
    assert get_mjwarp_articulation_name_ordering(articulation, kind="joint") == ("joint_1", "joint_0")
    assert get_mjwarp_articulation_name_ordering(articulation, kind="body") == ("body_1", "body_0")
    assert calls == [articulation, articulation]


def test_symbolic_cross_backend_resolver_normalizes_newton_multi_dof_joint_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize Newton multi-DoF spelling only when each match is unique."""

    class _Articulation:
        __backend_name__ = "physx"
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

    BaseArticulation._resolve_and_install_ordering_maps(articulation)

    assert data.joint_ordering is None
    assert data.body_ordering is None
    assert data.joint_names == ["hip", "knee"]
    assert data.body_names == ["base", "foot"]
    assert apply_calls == []


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
    return types.SimpleNamespace(
        __backend_name__="mock",
        cfg=types.SimpleNamespace(
            prim_path="/World/Robot",
            joint_ordering=None,
            body_ordering=body_ordering,
        ),
        data=data,
        backend_joint_names=["joint"],
        backend_body_names=list(backend_body_names),
        _mjwarp_body_names=("foot", "base"),
        is_fixed_base=is_fixed_base,
        device="cpu",
    )


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


@pytest.mark.parametrize("ordering", [("foot", "base"), ArticulationOrderingConvention.MJWARP])
def test_fixed_base_body_ordering_rejects_root_relocation(ordering) -> None:
    """Reject explicit and symbolic fixed-base orders that move the root from public index zero."""
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


def test_base_articulation_data_defines_optional_ordering_maps() -> None:
    """Expose optional ordering maps on articulation data containers."""
    assert hasattr(BaseArticulationData, "joint_ordering")
    assert hasattr(BaseArticulationData, "body_ordering")


def test_build_articulation_name_map_uses_identity_device_maps() -> None:
    """Build an identity articulation name map with identity device maps."""
    name_map = build_articulation_name_map(
        kind="joint",
        backend_names=("hip", "knee", "ankle"),
        user_names=None,
        device="cpu",
    )

    assert name_map.kind == "joint"
    assert name_map.backend_names == ("hip", "knee", "ankle")
    assert name_map.user_names == ("hip", "knee", "ankle")
    assert name_map.user_to_backend_indices == (0, 1, 2)
    assert name_map.backend_to_user_indices == (0, 1, 2)
    assert name_map.user_to_backend is not None
    assert name_map.backend_to_user is not None
    np.testing.assert_array_equal(name_map.user_to_backend.numpy(), np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(name_map.backend_to_user.numpy(), np.asarray([0, 1, 2], dtype=np.int32))
    assert name_map.is_identity


def test_build_articulation_name_map_builds_permutation_indices_and_device_maps() -> None:
    """Build CPU and device maps for an explicit user ordering permutation."""
    name_map = build_articulation_name_map(
        kind="body",
        backend_names=("base", "left_foot", "right_foot"),
        user_names=("right_foot", "base", "left_foot"),
        device="cpu",
    )

    assert name_map.kind == "body"
    assert name_map.backend_names == ("base", "left_foot", "right_foot")
    assert name_map.user_names == ("right_foot", "base", "left_foot")
    assert name_map.user_to_backend_indices == (2, 0, 1)
    assert name_map.backend_to_user_indices == (1, 2, 0)
    assert name_map.user_to_backend is not None
    assert name_map.backend_to_user is not None
    np.testing.assert_array_equal(name_map.user_to_backend.numpy(), np.asarray([2, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(name_map.backend_to_user.numpy(), np.asarray([1, 2, 0], dtype=np.int32))
    assert not name_map.is_identity


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


def test_reorder_2d_backend_to_user_gathers_user_axis() -> None:
    """Reorder a 2-D backend buffer into public user order."""
    backend_data = wp.array(
        np.asarray(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(backend_data)

    wp.launch(
        reorder_2d_backend_to_user,
        dim=user_data.shape,
        inputs=[backend_data, user_to_backend],
        outputs=[user_data],
        device="cpu",
    )

    expected = np.asarray([[12.0, 10.0, 11.0], [22.0, 20.0, 21.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected)


def test_sync_torque_telemetry_reads_backend_effort_buffers_in_user_order() -> None:
    """Report torque telemetry in public joint order from backend-order effort buffers."""
    joint_pos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    joint_vel = wp.zeros_like(joint_pos)
    joint_pos_target = wp.zeros_like(joint_pos)
    joint_vel_target = wp.zeros_like(joint_pos)
    joint_stiffness = wp.zeros_like(joint_pos)
    joint_damping = wp.zeros_like(joint_pos)
    effort_limit = wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu")
    joint_modes = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    sim_bind_joint_effort = wp.array(
        np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    actuator_computed_effort = wp.array(
        np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    computed = wp.zeros_like(joint_pos)
    applied = wp.zeros_like(joint_pos)

    wp.launch(
        sync_torque_telemetry,
        dim=joint_pos.shape,
        inputs=[
            joint_pos,
            joint_vel,
            joint_pos_target,
            joint_vel_target,
            joint_stiffness,
            joint_damping,
            effort_limit,
            joint_modes,
            sim_bind_joint_effort,
            actuator_computed_effort,
            user_to_backend,
            True,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    np.testing.assert_allclose(computed.numpy(), np.asarray([[30.0, 100.0, 20.0]], dtype=np.float32))
    np.testing.assert_allclose(applied.numpy(), np.asarray([[300.0, 100.0, 200.0]], dtype=np.float32))


def test_sync_torque_telemetry_keeps_user_order_effort_buffers_unmapped() -> None:
    """Report torque telemetry directly from user-order actuator buffers."""
    joint_pos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    joint_modes = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_effort = wp.array(np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    user_computed_effort = wp.array(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    computed = wp.zeros_like(joint_pos)
    applied = wp.zeros_like(joint_pos)

    wp.launch(
        sync_torque_telemetry,
        dim=joint_pos.shape,
        inputs=[
            joint_pos,
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu"),
            joint_modes,
            user_effort,
            user_computed_effort,
            user_to_backend,
            False,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    np.testing.assert_allclose(computed.numpy(), np.asarray([[10.0, 200.0, 30.0]], dtype=np.float32))
    np.testing.assert_allclose(applied.numpy(), np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32))


def test_reorder_2d_user_to_backend_scatters_backend_axis() -> None:
    """Reorder a 2-D public user buffer into backend order."""
    user_data = wp.array(
        np.asarray(
            [
                [12.0, 10.0, 11.0],
                [22.0, 20.0, 21.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    backend_to_user = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    backend_data = wp.zeros_like(user_data)

    wp.launch(
        reorder_2d_user_to_backend,
        dim=backend_data.shape,
        inputs=[user_data, backend_to_user],
        outputs=[backend_data],
        device="cpu",
    )

    expected = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32)
    np.testing.assert_allclose(backend_data.numpy(), expected)


def test_reorder_3d_backend_to_user_gathers_user_axis() -> None:
    """Reorder a 3-D backend buffer whose second axis is the public user axis."""
    backend_data_np = np.asarray(
        [
            [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
            [[200.0, 201.0], [210.0, 211.0], [220.0, 221.0]],
        ],
        dtype=np.float32,
    )
    backend_data = wp.array(backend_data_np, dtype=wp.float32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(backend_data)

    wp.launch(
        reorder_3d_backend_to_user,
        dim=user_data.shape,
        inputs=[backend_data, user_to_backend],
        outputs=[user_data],
        device="cpu",
    )

    np.testing.assert_allclose(user_data.numpy(), backend_data_np[:, [1, 2, 0], :])


@pytest.mark.parametrize("selection", ["indices", "mask"])
@pytest.mark.parametrize("writer", ["scalar", "float", "velocity", "state"])
def test_fused_identity_writer_supports_aliased_outputs(selection: str, writer: str) -> None:
    """Preserve identity-order writes when public and backend outputs alias."""
    user_to_backend = wp.array(np.asarray([0, 1, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    env_ids = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")
    env_mask = wp.array(np.asarray([True, False]), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, False]), dtype=wp.bool, device="cpu")
    initial = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_data = wp.array(
        np.asarray([[101.0, 102.0, 103.0], [104.0, 105.0, 106.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )

    def expected_with_selected_value(source: np.ndarray, value: float) -> np.ndarray:
        expected = source.copy()
        expected[0, 1] = value
        return expected

    if writer == "scalar":
        data = wp.array(initial, dtype=wp.float32, device="cpu")
        if selection == "indices":
            wp.launch(
                write_scalar_user_to_backend_with_indices,
                dim=(env_ids.shape[0], user_ids.shape[0]),
                inputs=[42.0, env_ids, user_ids, user_to_backend, False],
                outputs=[data, data],
                device="cpu",
            )
        else:
            wp.launch(
                write_scalar_user_to_backend_with_mask,
                dim=data.shape,
                inputs=[42.0, env_mask, user_mask, user_to_backend, False],
                outputs=[data, data],
                device="cpu",
            )
        np.testing.assert_allclose(data.numpy(), expected_with_selected_value(initial, 42.0))
        return

    if writer == "float":
        data = wp.array(initial, dtype=wp.float32, device="cpu")
        if selection == "indices":
            wp.launch(
                write_2d_float_user_to_backend_with_indices,
                dim=(env_ids.shape[0], user_ids.shape[0]),
                inputs=[input_data, env_ids, user_ids, user_to_backend, False, True],
                outputs=[data, data],
                device="cpu",
            )
        else:
            wp.launch(
                write_2d_float_user_to_backend_with_mask,
                dim=data.shape,
                inputs=[input_data, env_mask, user_mask, user_to_backend, False],
                outputs=[data, data],
                device="cpu",
            )
        np.testing.assert_allclose(data.numpy(), expected_with_selected_value(initial, 102.0))
        return

    initial_velocity = initial + 10.0
    initial_previous_velocity = initial + 20.0
    initial_acceleration = initial + 30.0
    velocity = wp.array(initial_velocity, dtype=wp.float32, device="cpu")
    previous_velocity = wp.array(initial_previous_velocity, dtype=wp.float32, device="cpu")
    acceleration = wp.array(initial_acceleration, dtype=wp.float32, device="cpu")

    if writer == "velocity":
        if selection == "indices":
            wp.launch(
                write_joint_vel_user_to_backend_with_indices,
                dim=(env_ids.shape[0], user_ids.shape[0]),
                inputs=[input_data, env_ids, user_ids, user_to_backend, False, True],
                outputs=[velocity, previous_velocity, acceleration, velocity],
                device="cpu",
            )
        else:
            wp.launch(
                write_joint_vel_user_to_backend_with_mask,
                dim=velocity.shape,
                inputs=[input_data, env_mask, user_mask, user_to_backend, False],
                outputs=[velocity, previous_velocity, acceleration, velocity],
                device="cpu",
            )
        np.testing.assert_allclose(velocity.numpy(), expected_with_selected_value(initial_velocity, 102.0))
        np.testing.assert_allclose(
            previous_velocity.numpy(), expected_with_selected_value(initial_previous_velocity, 102.0)
        )
        np.testing.assert_allclose(acceleration.numpy(), expected_with_selected_value(initial_acceleration, 0.0))
        return

    position_data = wp.array(
        np.asarray([[201.0, 202.0, 203.0], [204.0, 205.0, 206.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    velocity_data = wp.array(
        np.asarray([[301.0, 302.0, 303.0], [304.0, 305.0, 306.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    initial_position = initial + 40.0
    position = wp.array(initial_position, dtype=wp.float32, device="cpu")
    if selection == "indices":
        wp.launch(
            write_joint_state_user_to_backend_with_indices,
            dim=(env_ids.shape[0], user_ids.shape[0]),
            inputs=[position_data, velocity_data, env_ids, user_ids, user_to_backend, False, True],
            outputs=[position, velocity, previous_velocity, acceleration, position, velocity],
            device="cpu",
        )
    else:
        wp.launch(
            write_joint_state_user_to_backend_with_mask,
            dim=position.shape,
            inputs=[position_data, velocity_data, env_mask, user_mask, user_to_backend, False],
            outputs=[position, velocity, previous_velocity, acceleration, position, velocity],
            device="cpu",
        )
    np.testing.assert_allclose(position.numpy(), expected_with_selected_value(initial_position, 202.0))
    np.testing.assert_allclose(velocity.numpy(), expected_with_selected_value(initial_velocity, 302.0))
    np.testing.assert_allclose(
        previous_velocity.numpy(), expected_with_selected_value(initial_previous_velocity, 302.0)
    )
    np.testing.assert_allclose(acceleration.numpy(), expected_with_selected_value(initial_acceleration, 0.0))


def test_write_scalar_user_to_backend_with_indices_updates_user_and_backend_buffers() -> None:
    """Fuse indexed scalar writes into user and backend-order buffers."""
    env_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    backend_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_scalar_user_to_backend_with_indices,
        dim=(env_ids.shape[0], user_ids.shape[0]),
        inputs=[4.5, env_ids, user_ids, user_to_backend, True],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[4.5, 0.0, 4.5], [0.0, 0.0, 0.0], [4.5, 0.0, 4.5]], dtype=np.float32)
    expected_backend = np.asarray([[4.5, 4.5, 0.0], [0.0, 0.0, 0.0], [4.5, 4.5, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_scalar_user_to_backend_with_mask_updates_selected_entries() -> None:
    """Fuse masked scalar writes into user and backend-order buffers."""
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    backend_data = wp.zeros((2, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_scalar_user_to_backend_with_mask,
        dim=user_data.shape,
        inputs=[7.25, env_mask, user_mask, user_to_backend, True],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 7.25, 7.25], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[7.25, 0.0, 7.25], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_2d_float_user_to_backend_with_indices_updates_user_and_backend_buffers() -> None:
    """Fuse partial user-order writes into user and backend-order buffers."""
    input_data = wp.array(
        np.asarray(
            [
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    env_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    backend_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_2d_float_user_to_backend_with_indices,
        dim=input_data.shape,
        inputs=[input_data, env_ids, user_ids, user_to_backend, True, False],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[11.0, 0.0, 10.0], [0.0, 0.0, 0.0], [21.0, 0.0, 20.0]], dtype=np.float32)
    expected_backend = np.asarray([[10.0, 11.0, 0.0], [0.0, 0.0, 0.0], [20.0, 21.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_2d_float_user_to_backend_with_mask_updates_selected_entries() -> None:
    """Fuse masked user-order writes into user and backend-order buffers."""
    input_data = wp.array(
        np.asarray(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(input_data)
    backend_data = wp.zeros_like(input_data)

    wp.launch(
        write_2d_float_user_to_backend_with_mask,
        dim=input_data.shape,
        inputs=[input_data, env_mask, user_mask, user_to_backend, True],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[12.0, 0.0, 11.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_joint_state_user_to_backend_with_indices_updates_history_and_backend_buffers() -> None:
    """Fuse indexed joint state writes into user cache, history, and backend buffers."""
    position = wp.array(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    velocity = wp.array(np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    env_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_pos = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    user_vel = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    user_prev_vel = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    user_acc = wp.ones((3, 3), dtype=wp.float32, device="cpu")
    backend_pos = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    backend_vel = wp.zeros((3, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_joint_state_user_to_backend_with_indices,
        dim=position.shape,
        inputs=[position, velocity, env_ids, user_ids, user_to_backend, True, False],
        outputs=[user_pos, user_vel, user_prev_vel, user_acc, backend_pos, backend_vel],
        device="cpu",
    )

    expected_user_pos = np.asarray([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0], [4.0, 0.0, 3.0]], dtype=np.float32)
    expected_user_vel = np.asarray([[6.0, 0.0, 5.0], [0.0, 0.0, 0.0], [8.0, 0.0, 7.0]], dtype=np.float32)
    expected_backend_pos = np.asarray([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    expected_backend_vel = np.asarray([[5.0, 6.0, 0.0], [0.0, 0.0, 0.0], [7.0, 8.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_pos.numpy(), expected_user_pos)
    np.testing.assert_allclose(user_vel.numpy(), expected_user_vel)
    np.testing.assert_allclose(user_prev_vel.numpy(), expected_user_vel)
    np.testing.assert_allclose(backend_pos.numpy(), expected_backend_pos)
    np.testing.assert_allclose(backend_vel.numpy(), expected_backend_vel)
    np.testing.assert_allclose(user_acc.numpy()[[0, 2]][:, [0, 2]], np.zeros((2, 2), dtype=np.float32))


def test_write_joint_vel_user_to_backend_with_mask_updates_selected_entries() -> None:
    """Fuse masked joint velocity writes into user cache, history, and backend buffers."""
    velocity = wp.array(
        np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_vel = wp.zeros_like(velocity)
    user_prev_vel = wp.zeros_like(velocity)
    user_acc = wp.ones_like(velocity)
    backend_vel = wp.zeros_like(velocity)

    wp.launch(
        write_joint_vel_user_to_backend_with_mask,
        dim=velocity.shape,
        inputs=[velocity, env_mask, user_mask, user_to_backend, True],
        outputs=[user_vel, user_prev_vel, user_acc, backend_vel],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[12.0, 0.0, 11.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_vel.numpy(), expected_user)
    np.testing.assert_allclose(user_prev_vel.numpy(), expected_user)
    np.testing.assert_allclose(backend_vel.numpy(), expected_backend)
    np.testing.assert_allclose(user_acc.numpy()[0, 1:], np.zeros((2,), dtype=np.float32))


def test_newton_actuator_defaults_follow_requested_public_joint_order() -> None:
    """Convert Newton actuator gain snapshots and managed IDs into public joint order."""
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    controller = types.SimpleNamespace(
        kp=wp.array((10.0, 30.0, 11.0, 31.0), dtype=wp.float32, device="cpu"),
        kd=wp.array((1.0, 3.0, 1.1, 3.1), dtype=wp.float32, device="cpu"),
    )
    actuator = types.SimpleNamespace(
        controller=controller,
        indices=wp.array((0, 2, 3, 5), dtype=wp.uint32, device="cpu"),
    )

    stiffness, damping, managed = build_newton_actuator_defaults(
        actuators=[actuator],
        num_envs=2,
        num_joints=3,
        dof_offset=0,
        device="cpu",
        joint_user_to_backend_indices=(2, 0, 1),
    )

    torch.testing.assert_close(stiffness, torch.tensor([[30.0, 10.0, 0.0], [31.0, 11.0, 0.0]]))
    torch.testing.assert_close(damping, torch.tensor([[3.0, 1.0, 0.0], [3.1, 1.1, 0.0]]))
    torch.testing.assert_close(managed, torch.tensor([0, 1], dtype=torch.int32))


def test_newton_actuator_defaults_reject_incomplete_joint_permutation() -> None:
    """Reject malformed actuator-default ordering maps with an actionable error."""
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    with pytest.raises(
        ValueError,
        match=(
            r"joint_user_to_backend_indices must contain each backend joint index exactly once; "
            r"expected a permutation of 0\.\.2, got \(0, 0, 2\)\."
        ),
    ):
        build_newton_actuator_defaults(
            actuators=[],
            num_envs=1,
            num_joints=3,
            dof_offset=0,
            device="cpu",
            joint_user_to_backend_indices=(0, 0, 2),
        )
