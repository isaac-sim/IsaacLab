# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the kitless OVPhysX deformable object asset."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_ovphysx.assets.deformable_object.deformable_object import (  # noqa: E402
    _REQUIRED_DEFORMABLE_TENSOR_TYPES,
    DeformableObject,
    _detect_deformable_type,
    _find_deformable_material,
    _get_api_schemas,
    _require_deformable_tensor_types,
    _to_ovphysx_pattern,
)
from isaaclab_ovphysx.assets.deformable_object.deformable_object_data import (  # noqa: E402
    DeformableObjectData,
)
from isaaclab_ovphysx.assets.deformable_object.kernels import vec6f  # noqa: E402

from pxr import Sdf, Usd  # noqa: E402

from isaaclab.assets.deformable_object import (  # noqa: E402
    BaseDeformableObject,
    BaseDeformableObjectData,
)
from isaaclab.assets.deformable_object import (  # noqa: E402
    DeformableObject as DeformableObjectFactory,
)

wp.init()
wp.set_device("cpu")
_TEST_STAGES: list[Usd.Stage] = []


def _make_prim_with_api_metadata(schemas: list[str]) -> Usd.Prim:
    stage = Usd.Stage.CreateInMemory()
    _TEST_STAGES.append(stage)
    prim = stage.DefinePrim("/Soft", "Xform")
    _set_api_schemas(prim, schemas)
    return prim


def _set_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    """Author API schema metadata on a test prim."""
    schema_metadata = Sdf.TokenListOp()
    schema_metadata.prependedItems = schemas
    prim.SetMetadata("apiSchemas", schema_metadata)


def _make_deformable_prims(
    material_schemas: list[str], child_types: list[str], root_schemas: list[str] | None = None
) -> tuple[Usd.Prim, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    _TEST_STAGES.append(stage)
    root = stage.DefinePrim("/Soft", "Xform")
    _set_api_schemas(root, root_schemas or [])
    for index, child_type in enumerate(child_types):
        stage.DefinePrim(f"/Soft/Geometry_{index}", child_type)
    material = stage.DefinePrim("/Material", "Material")
    _set_api_schemas(material, material_schemas)
    return root, material


class _FakeBodyView:
    """Minimal PhysX-style deformable body view used by the asset shell."""

    def __init__(self, num_instances: int, num_vertices: int) -> None:
        self.count = num_instances
        self.max_simulation_nodes_per_body = num_vertices
        self.max_simulation_elements_per_body = 2
        self.max_collision_elements_per_body = 3
        self.max_collision_nodes_per_body = 0
        self.positions = wp.zeros((num_instances, num_vertices, 3), dtype=wp.float32, device="cpu")
        self.velocities = wp.zeros((num_instances, num_vertices, 3), dtype=wp.float32, device="cpu")
        self.targets = wp.zeros((num_instances, num_vertices, 4), dtype=wp.float32, device="cpu")
        self.position_reads = 0
        self.velocity_reads = 0
        self.position_write_count = 0
        self.velocity_write_count = 0
        self.target_write_count = 0
        self.last_indices: torch.Tensor | None = None
        self.last_values: wp.array | None = None
        self.destroyed = False

    def read_simulation_nodal_positions_into(self, values: wp.array(dtype=wp.float32)) -> None:
        self.position_reads += 1
        wp.copy(values, self.positions)

    def read_simulation_nodal_velocities_into(self, values: wp.array(dtype=wp.float32)) -> None:
        self.velocity_reads += 1
        wp.copy(values, self.velocities)

    def get_simulation_nodal_positions(self) -> wp.array(dtype=wp.float32):
        return self.positions

    def get_simulation_nodal_velocities(self) -> wp.array(dtype=wp.float32):
        return self.velocities

    def get_simulation_nodal_kinematic_targets(self) -> wp.array(dtype=wp.float32):
        return self.targets

    def set_simulation_nodal_positions(
        self, values: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32) | None = None
    ) -> None:
        self.position_write_count += 1
        self.last_values = values
        self.last_indices = wp.to_torch(indices) if indices is not None else None

    def set_simulation_nodal_velocities(
        self, values: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32) | None = None
    ) -> None:
        self.velocity_write_count += 1
        self.last_values = values
        self.last_indices = wp.to_torch(indices) if indices is not None else None

    def set_simulation_nodal_kinematic_targets(
        self, values: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32) | None = None
    ) -> None:
        self.target_write_count += 1
        self.last_values = values
        self.last_indices = wp.to_torch(indices) if indices is not None else None

    def destroy(self) -> None:
        self.destroyed = True


class _FakeMaterialView:
    """Minimal optional material view used to verify cleanup."""

    count = 1

    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeVisualizer:
    """Capture marker positions from the debug visualization callback."""

    def __init__(self) -> None:
        self.positions: torch.Tensor | None = None

    def visualize(self, positions: torch.Tensor) -> None:
        self.positions = positions


def _make_asset_shell(
    *,
    deformable_type: str,
    num_instances: int = 2,
    num_vertices: int = 4,
    material_view: _FakeMaterialView | None = None,
) -> DeformableObject:
    asset = object.__new__(DeformableObject)
    asset._device = "cpu"
    asset._check_shapes = True
    asset._DTYPE_TO_TORCH_TRAILING_DIMS = {**asset._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}
    asset._deformable_type = deformable_type
    asset._root_physx_view = _FakeBodyView(num_instances, num_vertices)
    asset._material_physx_view = material_view
    asset._data = DeformableObjectData(asset._root_physx_view, asset._device)
    asset._ALL_INDICES = wp.array(range(num_instances), dtype=wp.int32, device=asset.device)
    asset._nodal_pos_w_f32 = None
    asset._nodal_vel_w_f32 = None
    asset._is_initialized = True
    asset._debug_vis_handle = None
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    return asset


def test_get_api_schemas_includes_unregistered_authored_metadata():
    prim = _make_prim_with_api_metadata(["OmniPhysicsDeformableBodyAPI"])
    assert "OmniPhysicsDeformableBodyAPI" in _get_api_schemas(prim)


@pytest.mark.parametrize(
    "material_schemas,child_type,expected",
    [
        (["PhysxDeformableMaterialAPI"], "TetMesh", "volume"),
        (["PhysxSurfaceDeformableMaterialAPI"], "Mesh", "surface"),
        ([], "TetMesh", "volume"),
        ([], "Mesh", "surface"),
    ],
)
def test_detect_deformable_type_matches_physx_rules(
    material_schemas: list[str], child_type: str, expected: str
) -> None:
    root, material = _make_deformable_prims(material_schemas, [child_type])
    assert _detect_deformable_type(root, material) == expected


@pytest.mark.parametrize(
    "body_schema,expected",
    [
        ("OmniPhysicsVolumeDeformableSimAPI", "volume"),
        ("OmniPhysicsSurfaceDeformableSimAPI", "surface"),
    ],
)
def test_detect_deformable_type_uses_authored_body_schema(body_schema: str, expected: str) -> None:
    root, material = _make_deformable_prims([], [], root_schemas=[body_schema])
    assert _detect_deformable_type(root, material) == expected


def test_detect_deformable_type_rejects_conflicting_explicit_body_schemas() -> None:
    root, material = _make_deformable_prims(
        [], [], root_schemas=["OmniPhysicsVolumeDeformableSimAPI", "OmniPhysicsSurfaceDeformableSimAPI"]
    )

    with pytest.raises(RuntimeError, match="Detected deformable types: \\['surface', 'volume'\\]"):
        _detect_deformable_type(root, material)


def test_detect_deformable_type_rejects_missing_evidence() -> None:
    root, material = _make_deformable_prims([], [])

    with pytest.raises(RuntimeError, match="Detected deformable types: \\[]"):
        _detect_deformable_type(root, material)


def test_detect_deformable_type_normalizes_inherited_generic_surface_material_schema() -> None:
    root, material = _make_deformable_prims(
        ["PhysxDeformableMaterialAPI", "PhysxSurfaceDeformableMaterialAPI"], ["Mesh"]
    )
    assert _detect_deformable_type(root, material) == "surface"


def test_detect_deformable_type_ignores_generic_material_schema_for_surface_mesh() -> None:
    root, material = _make_deformable_prims(["PhysxDeformableMaterialAPI"], ["Mesh"])

    assert _detect_deformable_type(root, material) == "surface"


def test_detect_deformable_type_rejects_surface_material_volume_topology_conflict() -> None:
    material_schemas = ["PhysxSurfaceDeformableMaterialAPI"]
    child_type = "TetMesh"
    root, material = _make_deformable_prims(material_schemas, [child_type])

    with pytest.raises(RuntimeError) as exc_info:
        _detect_deformable_type(root, material)

    message = str(exc_info.value)
    assert "Root schemas:" in message
    assert f"Material schemas: {sorted(material_schemas)}" in message
    assert f"Hierarchy types: {sorted({'Xform', child_type})}" in message
    assert "Detected deformable types: ['surface', 'volume']" in message


def test_detect_deformable_type_uses_descendant_volume_schema_with_visual_mesh() -> None:
    stage = Usd.Stage.CreateInMemory()
    _TEST_STAGES.append(stage)
    root = stage.DefinePrim("/Soft", "Xform")
    stage.DefinePrim("/Soft/visual", "Mesh")
    simulation_tet_mesh = stage.DefinePrim("/Soft/simulation", "TetMesh")
    _set_api_schemas(simulation_tet_mesh, ["OmniPhysicsVolumeDeformableSimAPI"])
    material = stage.DefinePrim("/Material", "Material")

    assert _detect_deformable_type(root, material) == "volume"


def test_detect_deformable_type_mixed_topology_prefers_tet_mesh_without_schemas() -> None:
    root, material = _make_deformable_prims([], ["TetMesh", "Mesh"])

    assert _detect_deformable_type(root, material) == "volume"


@pytest.mark.parametrize(
    "path_expr,expected",
    [
        ("/World/{ENV_REGEX_NS}/Soft", "/World/*/Soft"),
        ("/World/envs/env_.*/Soft", "/World/envs/env_*/Soft"),
        ("/World/Soft", "/World/Soft"),
    ],
)
def test_to_ovphysx_pattern_converts_isaaclab_expressions(path_expr: str, expected: str) -> None:
    assert _to_ovphysx_pattern(path_expr) == expected


def test_required_tensor_check_includes_surface_members():
    missing_name = "SURFACE_DEFORMABLE_SIM_POSITION"
    available = SimpleNamespace(
        **{name: object() for name in _REQUIRED_DEFORMABLE_TENSOR_TYPES if name != missing_name}
    )

    with pytest.raises(RuntimeError, match=missing_name):
        _require_deformable_tensor_types(available)


def test_unbound_material_is_optional():
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Soft", "Xform")
    asset = _make_asset_shell(deformable_type="volume")

    assert _find_deformable_material(root) is None
    assert asset.material_physx_view is None


def test_surface_kinematic_target_write_matches_physx_error():
    asset = _make_asset_shell(deformable_type="surface")
    with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
        asset.write_nodal_kinematic_target_to_sim_index(torch.zeros((2, 4, 4), device=asset.device))


def test_indexed_position_write_updates_full_internal_buffer():
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=4)
    selected = torch.ones((1, 4, 3), device=asset.device)
    asset.write_nodal_pos_to_sim_index(selected, env_ids=torch.tensor([2], device=asset.device))
    assert torch.count_nonzero(asset.data.nodal_pos_w.torch[0:2]) == 0
    torch.testing.assert_close(asset.data.nodal_pos_w.torch[2], selected[0])
    assert asset.root_view.last_indices.tolist() == [2]


def test_indexed_state_write_updates_position_and_velocity_buffers():
    asset = _make_asset_shell(deformable_type="volume", num_instances=2, num_vertices=4)
    selected = torch.cat(
        (torch.full((1, 4, 3), 2.0), torch.full((1, 4, 3), -3.0)),
        dim=-1,
    )

    asset.write_nodal_state_to_sim_index(selected, env_ids=[1])

    torch.testing.assert_close(asset.data.nodal_pos_w.torch[1], selected[0, :, :3])
    torch.testing.assert_close(asset.data.nodal_vel_w.torch[1], selected[0, :, 3:])
    assert asset.data._nodal_state_w.timestamp == -1.0


def test_indexed_state_write_refreshes_unselected_stale_cache_rows():
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)
    initial_positions = torch.full((3, 2, 3), -1.0, device=asset.device)
    initial_velocities = torch.full((3, 2, 3), -2.0, device=asset.device)
    asset.root_view.positions = wp.from_torch(initial_positions.contiguous(), dtype=wp.float32)
    asset.root_view.velocities = wp.from_torch(initial_velocities.contiguous(), dtype=wp.float32)

    asset.data.nodal_pos_w
    asset.data.nodal_vel_w

    latest_positions = torch.tensor(
        [
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]],
            [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]],
        ],
        device=asset.device,
    )
    latest_velocities = torch.tensor(
        [
            [[-10.0, -11.0, -12.0], [-13.0, -14.0, -15.0]],
            [[-20.0, -21.0, -22.0], [-23.0, -24.0, -25.0]],
            [[-30.0, -31.0, -32.0], [-33.0, -34.0, -35.0]],
        ],
        device=asset.device,
    )
    asset.root_view.positions = wp.from_torch(latest_positions.contiguous(), dtype=wp.float32)
    asset.root_view.velocities = wp.from_torch(latest_velocities.contiguous(), dtype=wp.float32)
    asset.update(0.1)

    selected_state = torch.cat(
        (
            torch.full((1, 2, 3), 100.0, device=asset.device),
            torch.full((1, 2, 3), -100.0, device=asset.device),
        ),
        dim=-1,
    )
    asset.write_nodal_state_to_sim_index(selected_state, env_ids=[0])

    expected_positions = latest_positions.clone()
    expected_positions[0] = selected_state[0, :, :3]
    expected_velocities = latest_velocities.clone()
    expected_velocities[0] = selected_state[0, :, 3:]

    assert asset.data._nodal_pos_w.timestamp == asset.data._sim_timestamp
    assert asset.data._nodal_vel_w.timestamp == asset.data._sim_timestamp
    torch.testing.assert_close(asset.data.nodal_pos_w.torch, expected_positions)
    torch.testing.assert_close(asset.data.nodal_vel_w.torch, expected_velocities)
    torch.testing.assert_close(
        asset.data.nodal_state_w.torch, torch.cat((expected_positions, expected_velocities), dim=-1)
    )
    torch.testing.assert_close(asset.data.root_pos_w.torch, expected_positions.mean(dim=1))
    torch.testing.assert_close(asset.data.root_vel_w.torch, expected_velocities.mean(dim=1))


@pytest.mark.parametrize(
    ("property_name", "write_method_name", "simulator_attribute", "command_value"),
    [
        ("nodal_pos_w", "write_nodal_pos_to_sim_index", "positions", 100.0),
        ("nodal_vel_w", "write_nodal_velocity_to_sim_index", "velocities", -100.0),
    ],
)
def test_indexed_full_data_write_preserves_retained_aliased_edits(
    property_name: str, write_method_name: str, simulator_attribute: str, command_value: float
) -> None:
    """A retained public buffer preserves selected edits while stale rows hydrate."""
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)
    initial = torch.full((3, 2, 3), -1.0, device=asset.device)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(initial.contiguous(), dtype=wp.float32))
    retained = getattr(asset.data, property_name)
    retained_torch = retained.torch

    latest = torch.arange(18, dtype=torch.float32, device=asset.device).reshape(3, 2, 3)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(latest.contiguous(), dtype=wp.float32))
    asset.update(0.1)
    retained_torch[1].fill_(command_value)

    getattr(asset, write_method_name)(retained, env_ids=[1], full_data=True)

    expected = latest.clone()
    expected[1].fill_(command_value)
    assert getattr(asset.data, property_name) is retained
    torch.testing.assert_close(retained.torch, expected)


def test_full_overwrite_stale_cache_does_not_read_simulator() -> None:
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)

    asset.data.nodal_pos_w
    asset.data.nodal_vel_w
    position_reads = asset.root_view.position_reads
    velocity_reads = asset.root_view.velocity_reads
    asset.update(0.1)

    full_state = torch.arange(36, dtype=torch.float32, device=asset.device).reshape(3, 2, 6)
    asset.write_nodal_state_to_sim_index(full_state, full_data=True)

    assert asset.root_view.position_reads == position_reads
    assert asset.root_view.velocity_reads == velocity_reads
    torch.testing.assert_close(asset.data.nodal_state_w.torch, full_state)


@pytest.mark.parametrize("trailing_dimension", [4, 5, 7])
def test_malformed_state_write_fails_before_mutating_or_writing(trailing_dimension: int):
    asset = _make_asset_shell(deformable_type="volume", num_instances=2, num_vertices=4)
    original_positions = asset.data.nodal_pos_w.torch.clone()
    original_velocities = asset.data.nodal_vel_w.torch.clone()
    malformed_state = torch.ones((1, 4, trailing_dimension), device=asset.device)

    with pytest.raises(AssertionError, match="nodal_state.*Shape mismatch"):
        asset.write_nodal_state_to_sim_index(malformed_state, env_ids=[1])

    torch.testing.assert_close(asset.data.nodal_pos_w.torch, original_positions)
    torch.testing.assert_close(asset.data.nodal_vel_w.torch, original_velocities)
    assert asset.root_view.position_write_count == 0
    assert asset.root_view.velocity_write_count == 0


def test_material_path_expr_does_not_rewrite_sibling_prefix():
    deformable_object_module = importlib.import_module("isaaclab_ovphysx.assets.deformable_object.deformable_object")

    path_expr = deformable_object_module._resolve_material_path_expr(
        Sdf.Path("/World/SoftMaterial"),
        Sdf.Path("/World/Soft"),
        "/World/envs/env_.*/Soft",
    )

    assert path_expr == "/World/SoftMaterial"


def test_volume_target_initialization_sets_free_flags_and_writes_full_buffer():
    asset = _make_asset_shell(deformable_type="volume")

    asset._create_buffers()

    assert asset.data.nodal_kinematic_target is not None
    torch.testing.assert_close(
        asset.data.nodal_kinematic_target.torch[..., 3],
        torch.ones((asset.num_instances, asset.max_sim_vertices_per_body)),
    )
    assert asset.root_view.target_write_count == 1
    assert asset.root_view.last_indices is None


def test_surface_buffer_initialization_has_no_kinematic_target():
    asset = _make_asset_shell(deformable_type="surface")

    asset._create_buffers()

    assert asset.data.nodal_kinematic_target is None
    assert asset.root_view.target_write_count == 0


def test_lazy_root_means_refresh_in_place_after_update():
    root_view = _FakeBodyView(num_instances=2, num_vertices=2)
    root_view.positions = wp.array(
        [[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]]],
        dtype=wp.float32,
    )
    data = DeformableObjectData(root_view, "cpu")

    root_pos = data.root_pos_w
    torch.testing.assert_close(root_pos.torch, torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    assert root_view.position_reads == 1

    root_view.positions = wp.array(
        [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], [[6.0, 6.0, 6.0], [8.0, 8.0, 8.0]]],
        dtype=wp.float32,
    )
    data.update(0.1)

    assert data.root_pos_w is root_pos
    torch.testing.assert_close(root_pos.torch, torch.tensor([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]))
    assert root_view.position_reads == 2


def test_surface_debug_visualization_uses_below_ground_sentinel():
    asset = _make_asset_shell(deformable_type="surface")
    asset.target_visualizer = _FakeVisualizer()

    asset._debug_vis_callback(None)

    torch.testing.assert_close(asset.target_visualizer.positions, torch.tensor([[0.0, 0.0, -10.0]]))


def test_invalidation_destroys_body_and_material_views(monkeypatch: pytest.MonkeyPatch):
    material_view = _FakeMaterialView()
    asset = _make_asset_shell(deformable_type="volume", material_view=material_view)
    root_view = asset.root_view
    monkeypatch.setattr(BaseDeformableObject, "_invalidate_initialize_callback", lambda self, event: None)

    asset._invalidate_initialize_callback(None)

    assert root_view.destroyed
    assert material_view.destroyed
    assert asset._root_physx_view is None
    assert asset._material_physx_view is None


def test_data_implements_backend_neutral_interface():
    asset = _make_asset_shell(deformable_type="volume")
    assert isinstance(asset.data, BaseDeformableObjectData)


def test_factory_export_is_present():
    backend_module = importlib.import_module("isaaclab_ovphysx.assets.deformable_object")
    assets_module = importlib.import_module("isaaclab_ovphysx.assets")

    assert backend_module.DeformableObject is DeformableObject
    assert assets_module.DeformableObject is DeformableObject
    assert DeformableObjectFactory._get_module_name("ovphysx") == "isaaclab_ovphysx.assets.deformable_object"
    assert DeformableObject.__backend_name__ == "ovphysx"
