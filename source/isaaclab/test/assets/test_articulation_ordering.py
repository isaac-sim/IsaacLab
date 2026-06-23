# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import types

import numpy as np
import pytest
import warp as wp

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from isaaclab.assets.articulation.ordering import (
    ArticulationOrderingConvention,
    apply_articulation_ordering_preset,
    build_articulation_name_map,
    get_mjwarp_articulation_name_ordering,
    get_physx_articulation_name_ordering,
    parse_articulation_ordering_convention,
    resolve_articulation_convention_name_ordering,
    resolve_articulation_ordering_names,
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
sys.modules.setdefault("isaaclab.sim", sim_stub)
sys.modules.setdefault("isaaclab.sim.simulation_context", simulation_context_stub)


asset_base_stub = types.ModuleType("isaaclab.assets.asset_base")


class _AssetBase:
    def __init__(self, cfg):
        self.cfg = cfg


asset_base_stub.AssetBase = _AssetBase
sys.modules.setdefault("isaaclab.assets.asset_base", asset_base_stub)

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
    write_joint_vel_user_to_backend_with_mask,
)


def test_parse_articulation_ordering_convention_accepts_none_strings_and_enum() -> None:
    """Parse supported articulation ordering convention inputs."""
    assert parse_articulation_ordering_convention(None) is None
    assert parse_articulation_ordering_convention("physx") is ArticulationOrderingConvention.PHYSX
    assert parse_articulation_ordering_convention("mjwarp") is ArticulationOrderingConvention.MJWARP
    assert (
        parse_articulation_ordering_convention(ArticulationOrderingConvention.PHYSX)
        is ArticulationOrderingConvention.PHYSX
    )


def test_parse_articulation_ordering_convention_rejects_unknown_string() -> None:
    """Reject unknown symbolic articulation ordering conventions."""
    with pytest.raises(ValueError, match="Unsupported articulation ordering convention"):
        parse_articulation_ordering_convention("backend")


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


def test_resolve_articulation_ordering_names_accepts_explicit_sequence() -> None:
    """Resolve explicit public ordering names from config."""
    user_names = resolve_articulation_ordering_names(
        kind="joint",
        backend_names=("joint_0", "joint_1", "joint_2"),
        ordering=("joint_2", "joint_0", "joint_1"),
        active_backend_name="physx",
    )

    assert user_names == ("joint_2", "joint_0", "joint_1")


def test_resolve_articulation_ordering_names_keeps_matching_backend_preset_identity() -> None:
    """Resolve same-backend symbolic presets without requiring traversal helpers."""
    user_names = resolve_articulation_ordering_names(
        kind="body",
        backend_names=("base", "left_foot", "right_foot"),
        ordering=ArticulationOrderingConvention.PHYSX,
        active_backend_name="physx",
    )

    assert user_names == ("base", "left_foot", "right_foot")


def test_physx_ordering_helper_reads_physx_root_view_names() -> None:
    """Resolve the PhysX convention from PhysX root-view name metadata."""

    class _SharedMetatype:
        dof_names = ["shoulder", "elbow", "wrist"]
        link_names = ["base", "arm", "hand"]

    class _RootView:
        shared_metatype = _SharedMetatype()

    class _Articulation:
        __backend_name__ = "mock"
        root_view = _RootView()

    articulation = _Articulation()

    assert get_physx_articulation_name_ordering(articulation, kind="joint") == ("shoulder", "elbow", "wrist")
    assert get_physx_articulation_name_ordering(articulation, kind="body") == ("base", "arm", "hand")


def test_mjwarp_ordering_helper_reads_newton_root_view_names() -> None:
    """Resolve the MJWarp convention from Newton root-view name metadata."""

    class _RootView:
        joint_dof_names = ["knee", "hip", "ankle"]
        link_names = ["base", "thigh", "foot"]

    class _Articulation:
        __backend_name__ = "newton"
        root_view = _RootView()

        @property
        def backend_joint_names(self) -> list[str]:
            return self.root_view.joint_dof_names

        @property
        def backend_body_names(self) -> list[str]:
            return self.root_view.link_names

    articulation = _Articulation()

    assert get_mjwarp_articulation_name_ordering(articulation, kind="joint") == ("knee", "hip", "ankle")
    assert get_mjwarp_articulation_name_ordering(articulation, kind="body") == ("base", "thigh", "foot")


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


def test_symbolic_cross_backend_resolver_uses_articulation_convention_helper() -> None:
    """Resolve a symbolic preset through articulation convention metadata."""

    class _RootView:
        joint_dof_names = ["knee", "hip", "ankle"]
        link_names = ["base", "thigh", "foot"]

    class _Articulation:
        __backend_name__ = "mock"
        root_view = _RootView()

        @property
        def backend_joint_names(self) -> list[str]:
            return ["hip", "knee", "ankle"]

        @property
        def backend_body_names(self) -> list[str]:
            return ["foot", "base", "thigh"]

    articulation = _Articulation()
    user_names = resolve_articulation_ordering_names(
        kind="joint",
        backend_names=articulation.backend_joint_names,
        ordering=ArticulationOrderingConvention.MJWARP,
        active_backend_name=articulation.__backend_name__,
        convention_name_resolver=lambda convention, kind: resolve_articulation_convention_name_ordering(
            articulation=articulation,
            convention=convention,
            kind=kind,
        ),
    )

    assert user_names == ("knee", "hip", "ankle")


def test_base_articulation_exposes_ordering_introspection_contract() -> None:
    """Require concrete articulations to expose backend and public ordering state."""
    for property_name in ("backend_joint_names", "backend_body_names", "joint_ordering", "body_ordering"):
        property_value = getattr(BaseArticulation, property_name)
        assert getattr(property_value.fget, "__isabstractmethod__", False)


def test_base_articulation_data_defines_optional_ordering_maps() -> None:
    """Expose optional ordering maps on articulation data containers."""
    assert hasattr(BaseArticulationData, "joint_ordering")
    assert hasattr(BaseArticulationData, "body_ordering")


def test_build_articulation_name_map_uses_identity_without_device_maps() -> None:
    """Build an identity articulation name map without allocating device maps."""
    name_map = build_articulation_name_map(
        kind="joint",
        backend_names=("hip", "knee", "ankle"),
        user_names=None,
        device="cpu",
    )

    assert name_map.kind == "joint"
    assert name_map.backend_names == ("hip", "knee", "ankle")
    assert name_map.user_names == ("hip", "knee", "ankle")
    assert name_map.user_to_backend_indices is None
    assert name_map.backend_to_user_indices is None
    assert name_map.user_to_backend is None
    assert name_map.backend_to_user is None
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
        inputs=[input_data, env_ids, user_ids, user_to_backend, False],
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
        inputs=[input_data, env_mask, user_mask, user_to_backend],
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
        inputs=[position, velocity, env_ids, user_ids, user_to_backend, False],
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
        inputs=[velocity, env_mask, user_mask, user_to_backend],
        outputs=[user_vel, user_prev_vel, user_acc, backend_vel],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[12.0, 0.0, 11.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_vel.numpy(), expected_user)
    np.testing.assert_allclose(user_prev_vel.numpy(), expected_user)
    np.testing.assert_allclose(backend_vel.numpy(), expected_backend)
    np.testing.assert_allclose(user_acc.numpy()[0, 1:], np.zeros((2,), dtype=np.float32))
