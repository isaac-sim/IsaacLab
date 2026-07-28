# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX deformable object."""

from __future__ import annotations

import multiprocessing
import queue
import sys
import traceback
from typing import Any

import ovphysx.types  # noqa: F401
import pytest
import torch
import warp as wp
from flaky import flaky
from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from ..deformable_utils import (  # noqa: E402
    pre_tetrahedralized_deformable_spawn_cfg,
    pretriangulated_surface_deformable_spawn_cfg,
)

wp.init()


@configclass
class DeformableSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration for cloned volume deformables."""

    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=pre_tetrahedralized_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class MixedDeformableRigidSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration for cloned deformable and rigid assets."""

    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=pre_tetrahedralized_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 1.0)),
    )


@configclass
class HeterogeneousMixedDeformableRigidSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration with two rigid variants and a deformable."""

    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=pre_tetrahedralized_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    shape: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shape",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
                sim_utils.SphereCfg(radius=0.05),
            ],
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 1.0)),
    )


def _ovphysx_sim_context(device: str, *, gravity_enabled: bool = True):
    """Build a kitless OVPhysX simulation context."""
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=0.01, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=True)


def _generate_deformable_scene(
    spawn: sim_utils.SpawnerCfg,
    num_objects: int = 2,
    height: float = 1.0,
    initial_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> DeformableObject:
    """Create independently authored deformables beneath matching parent prims."""
    for index in range(num_objects):
        sim_utils.create_prim(f"/World/Table_{index}", "Xform", translation=(index * 1.0, 0.0, height))
    cfg = DeformableObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height), rot=initial_rot),
    )
    return DeformableObject(cfg=cfg)


def _assert_finite_deformable_state(deformable: DeformableObject) -> None:
    """Assert finite nodal and derived root state."""
    assert torch.isfinite(deformable.data.nodal_state_w.torch).all()
    assert torch.isfinite(deformable.data.root_pos_w.torch).all()
    assert torch.isfinite(deformable.data.root_vel_w.torch).all()


def _canonical_connectivity(connectivity: torch.Tensor) -> list[set[tuple[int, ...]]]:
    """Return unordered elements with each element's vertex indices sorted."""
    return [{tuple(sorted(element)) for element in body} for body in connectivity.cpu().tolist()]


def _assert_rest_positions_match_authored(
    rest_positions: torch.Tensor, authored_points: torch.Tensor, prim_paths: list[str]
) -> None:
    """Assert each body contains the authored local rest points transformed into world space."""
    assert rest_positions.shape == (rest_positions.shape[0], authored_points.shape[0], 3)
    assert torch.is_floating_point(rest_positions)
    assert torch.isfinite(rest_positions).all()
    assert len(prim_paths) == rest_positions.shape[0]
    stage = sim_utils.get_current_stage()
    xform_cache = UsdGeom.XformCache()
    for body_points, prim_path in zip(rest_positions, prim_paths):
        local_to_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))
        expected = torch.tensor(
            [tuple(local_to_world.Transform(Gf.Vec3d(*point.tolist()))) for point in authored_points.cpu()],
            dtype=rest_positions.dtype,
            device=rest_positions.device,
        )
        distances = torch.cdist(body_points, expected)
        torch.testing.assert_close(
            distances.min(dim=0).values, torch.zeros(expected.shape[0], device=expected.device), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            distances.min(dim=1).values, torch.zeros(expected.shape[0], device=expected.device), atol=1e-6, rtol=0.0
        )


def _run_cpu_deformable_initialization(result_queue: Any) -> None:
    """Run the CPU initialization contract in a fresh spawned process."""
    try:
        with _ovphysx_sim_context(device="cpu") as sim:
            deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=5)
            assert sys.getrefcount(deformable) < 10
            try:
                sim.reset()
            except RuntimeError as error:
                result = ("runtime_error", str(error), deformable.is_initialized)
            else:
                result = ("no_error", "", deformable.is_initialized)
    except BaseException:
        result = ("child_error", traceback.format_exc(), None)
    result_queue.put(result)


@pytest.mark.parametrize(
    "num_objects, material_path",
    [
        (1, "material"),
        (2, None),
        (2, "/World/SoftMaterial"),
        (2, "material"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_initialization(num_objects: int, material_path: str | None):
    """Test volume deformable initialization and public buffer shapes."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(
            pre_tetrahedralized_deformable_spawn_cfg(material_path=material_path), num_objects=num_objects
        )

        assert sys.getrefcount(deformable) < 10
        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == num_objects
        assert deformable.num_bodies == 1
        assert deformable.root_view.count == num_objects
        if material_path is None:
            assert deformable.material_physx_view is None
        elif material_path.startswith("/"):
            assert deformable.material_physx_view is not None
            assert deformable.material_physx_view.count == 1
        else:
            assert deformable.material_physx_view is not None
            assert deformable.material_physx_view.count == num_objects
        assert deformable.data.nodal_state_w.torch.shape == (
            num_objects,
            deformable.max_sim_vertices_per_body,
            6,
        )
        assert deformable.data.nodal_kinematic_target is not None
        assert deformable.data.nodal_kinematic_target.torch.shape == (
            num_objects,
            deformable.max_sim_vertices_per_body,
            4,
        )
        assert deformable.data.root_pos_w.torch.shape == (num_objects, 3)
        assert deformable.data.root_vel_w.torch.shape == (num_objects, 3)

        deformable._invalidate_initialize_callback(None)
        assert deformable._root_physx_view is None
        assert deformable._material_physx_view is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_absolute_material_sibling_prefix_is_not_expanded():
    """Keep a shared absolute material exact when its path only textually prefixes the asset path."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        material_path = "/World/Table_0/ObjectSiblingMaterial"
        deformable = _generate_deformable_scene(
            pre_tetrahedralized_deformable_spawn_cfg(material_path=material_path), num_objects=2
        )
        distractor_cfg = PhysxDeformableBodyMaterialCfg()
        distractor_cfg.func("/World/Table_1/ObjectSiblingMaterial", distractor_cfg)

        sim.reset()

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_initialization_surface_deformable():
    """Test surface deformable initialization and unsupported target writes."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        num_objects = 2
        deformable = _generate_deformable_scene(pretriangulated_surface_deformable_spawn_cfg(), num_objects=num_objects)

        sim.reset()

        assert deformable.is_initialized
        assert deformable._deformable_type == "surface"
        assert deformable.num_instances == num_objects
        assert deformable.root_view.count == num_objects
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == num_objects
        assert deformable.data.nodal_state_w.torch.shape == (
            num_objects,
            deformable.max_sim_vertices_per_body,
            6,
        )
        assert deformable.data.root_pos_w.torch.shape == (num_objects, 3)
        assert deformable.data.root_vel_w.torch.shape == (num_objects, 3)
        assert deformable.data.nodal_kinematic_target is None

        dummy_targets = torch.zeros(num_objects, deformable.max_sim_vertices_per_body, 4, device=sim.device)
        with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
            deformable.write_nodal_kinematic_target_to_sim_index(dummy_targets)


@pytest.mark.isaacsim_ci
def test_initialization_on_device_cpu():
    """Test that OVPhysX deformable initialization rejects a CPU simulation."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_run_cpu_deformable_initialization, args=(result_queue,))
    process.start()
    process.join(timeout=30.0)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("CPU deformable initialization child process timed out.")
    assert process.exitcode == 0

    try:
        result_kind, message, is_initialized = result_queue.get(timeout=5.0)
    except queue.Empty:
        pytest.fail("CPU deformable initialization child process returned no result.")
    finally:
        result_queue.close()
        result_queue.join_thread()

    assert result_kind == "runtime_error", message
    assert message == "OVPhysX deformable tensors require a CUDA simulation device; received 'cpu'."
    assert is_initialized is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_set_nodal_state():
    """Test combined nodal state writes while independently randomizing position and velocity."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        num_objects = 2
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=num_objects)
        sim.reset()

        for state_type_to_randomize in ["nodal_pos_w", "nodal_vel_w"]:
            state_dict = {
                "nodal_pos_w": torch.zeros_like(deformable.data.nodal_pos_w.torch),
                "nodal_vel_w": torch.zeros_like(deformable.data.nodal_vel_w.torch),
            }

            for _ in range(5):
                deformable.reset()
                state_dict[state_type_to_randomize] = torch.randn(
                    num_objects, deformable.max_sim_vertices_per_body, 3, device=sim.device
                )

                for _ in range(5):
                    nodal_state = torch.cat([state_dict["nodal_pos_w"], state_dict["nodal_vel_w"]], dim=-1)
                    deformable.write_nodal_state_to_sim_index(nodal_state)
                    torch.testing.assert_close(deformable.data.nodal_state_w.torch, nodal_state, rtol=1e-5, atol=1e-5)

                    sim.step()
                    deformable.update(sim.cfg.dt)


@pytest.mark.parametrize(
    ("property_name", "write_method_name", "tensor_type", "command_value"),
    [
        ("nodal_pos_w", "write_nodal_pos_to_sim_index", "deformable_sim_nodal_position", 100.0),
        ("nodal_vel_w", "write_nodal_velocity_to_sim_index", "deformable_sim_nodal_velocity", -100.0),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_indexed_partial_write_preserves_retained_aliased_slice_in_simulator(
    property_name: str, write_method_name: str, tensor_type: str, command_value: float
) -> None:
    """Preserve a retained selected command while hydrating stale rows from OVPhysX."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=2)
        sim.reset()

        retained = getattr(deformable.data, property_name).torch
        sim.step()
        deformable.update(sim.cfg.dt)
        latest = wp.to_torch(deformable.root_view.get_attribute(tensor_type)).clone()
        selected = retained[1:2]
        selected.fill_(command_value)

        getattr(deformable, write_method_name)(selected, env_ids=torch.tensor([1], device=sim.device))
        readback = wp.to_torch(deformable.root_view.get_attribute(tensor_type))

        torch.testing.assert_close(readback[0], latest[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback[1], selected[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "num_objects, randomize_pos, randomize_rot",
    [
        (1, False, False),
        (1, True, False),
        (1, False, True),
        (2, True, True),
    ],
)
@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_set_nodal_state_with_applied_transform(num_objects: int, randomize_pos: bool, randomize_rot: bool):
    """Test combined nodal state writes after applying rigid transforms."""
    with _ovphysx_sim_context(device="cuda:0", gravity_enabled=False) as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=num_objects)
        sim.reset()

        for _ in range(5):
            nodal_state = deformable.data.default_nodal_state_w.torch.clone()
            mean_nodal_pos_default = nodal_state[..., :3].mean(dim=1)

            if randomize_pos:
                pos_w = 0.5 * torch.rand(deformable.num_instances, 3, device=sim.device)
                pos_w[:, 2] += 0.5
            else:
                pos_w = None
            if randomize_rot:
                quat_w = math_utils.random_orientation(deformable.num_instances, device=sim.device)
            else:
                quat_w = None

            nodal_state[..., :3] = deformable.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            mean_nodal_pos_init = nodal_state[..., :3].mean(dim=1)

            if pos_w is None:
                torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default, rtol=1e-5, atol=1e-5)
            else:
                torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default + pos_w, rtol=1e-5, atol=1e-5)

            deformable.write_nodal_state_to_sim_index(nodal_state)
            deformable.reset()

            for _ in range(50):
                sim.step()
                deformable.update(sim.cfg.dt)

            torch.testing.assert_close(deformable.data.root_pos_w.torch, mean_nodal_pos_init, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_set_kinematic_targets():
    """Test pinning one volume deformable while another falls under gravity."""
    with _ovphysx_sim_context(device="cuda:0", gravity_enabled=True) as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=2, height=1.0)
        sim.reset()

        nodal_kinematic_targets = wp.to_torch(
            deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET)
        ).clone()

        for _ in range(5):
            deformable.write_nodal_state_to_sim_index(deformable.data.default_nodal_state_w.torch)
            default_root_pos = deformable.data.default_nodal_state_w.torch[..., :3].mean(dim=1)
            deformable.reset()

            nodal_kinematic_targets[1:, :, 3] = 1.0
            nodal_kinematic_targets[0, :, 3] = 0.0
            nodal_kinematic_targets[0, :, :3] = deformable.data.default_nodal_state_w.torch[0, :, :3]
            deformable.write_nodal_kinematic_target_to_sim_index(
                nodal_kinematic_targets[0:1], env_ids=torch.tensor([0], device=sim.device)
            )

            for _ in range(20):
                sim.step()
                deformable.update(sim.cfg.dt)

                torch.testing.assert_close(
                    deformable.data.nodal_pos_w.torch[0],
                    nodal_kinematic_targets[0, :, :3],
                    rtol=1e-5,
                    atol=1e-5,
                )
                assert torch.all(deformable.data.root_pos_w.torch[1:, 2] < default_root_pos[1:, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_volume_deformable_reads_writes_targets_materials_and_steps():
    """Exercise authored volume state, topology, targets, materials, and stepping."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg())

        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == 2
        assert deformable.num_bodies == 1
        assert deformable.root_view.count == 2
        assert deformable.max_sim_vertices_per_body == 5
        assert deformable.max_sim_elements_per_body == 2
        assert deformable.max_collision_elements_per_body == 2
        assert deformable.max_collision_vertices_per_body == 5

        nodal_state = deformable.data.nodal_state_w.torch
        nodal_pos = deformable.data.nodal_pos_w.torch
        nodal_vel = deformable.data.nodal_vel_w.torch
        assert nodal_state.shape == (2, 5, 6)
        assert deformable.data.default_nodal_state_w.torch.shape == (2, 5, 6)
        assert deformable.data.root_pos_w.torch.shape == (2, 3)
        assert deformable.data.root_vel_w.torch.shape == (2, 3)
        rest_positions = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_REST_NODAL_POSITION))
        _assert_rest_positions_match_authored(
            rest_positions,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.2],
                    [0.2, 0.2, 0.2],
                ],
                device=rest_positions.device,
            ),
            deformable.root_view.prim_paths,
        )
        torch.testing.assert_close(deformable.data.root_pos_w.torch, nodal_pos.mean(dim=1))
        torch.testing.assert_close(deformable.data.root_vel_w.torch, nodal_vel.mean(dim=1))

        element_indices = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_ELEMENT_INDICES))
        collision_indices = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_COLLISION_ELEMENT_INDICES))
        assert element_indices.shape == (2, 2, 4)
        assert collision_indices.shape == (2, 2, 4)
        assert element_indices.dtype == torch.int32
        assert collision_indices.dtype == torch.int32
        assert torch.all((element_indices >= 0) & (element_indices < 5))
        assert torch.all((collision_indices >= 0) & (collision_indices < 5))
        expected_tetrahedra = {(0, 1, 2, 3), (1, 2, 3, 4)}
        assert all(elements == expected_tetrahedra for elements in _canonical_connectivity(element_indices))
        assert all(elements == expected_tetrahedra for elements in _canonical_connectivity(collision_indices))

        updated_pos = nodal_pos[1:2].clone()
        updated_pos[..., 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device=sim.device))
        readback_pos = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_NODAL_POSITION))
        torch.testing.assert_close(readback_pos[0], nodal_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        updated_vel = nodal_vel[0:1].clone()
        updated_vel[..., 1] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel, env_ids=torch.tensor([0]))
        readback_vel = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_NODAL_VELOCITY))
        torch.testing.assert_close(readback_vel[0], updated_vel[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_vel[1], nodal_vel[1], rtol=1e-5, atol=1e-5)

        targets = deformable.data.nodal_kinematic_target
        assert targets is not None
        assert targets.torch.shape == (2, 5, 4)
        torch.testing.assert_close(targets.torch[..., 3], torch.ones_like(targets.torch[..., 3]))
        updated_targets = targets.torch[1:2].clone()
        updated_targets[..., :3] = readback_pos[1:2] + torch.tensor([0.0, 0.0, 0.03], device=sim.device)
        updated_targets[..., 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(
            updated_targets, env_ids=torch.tensor([1], device=sim.device)
        )
        readback_targets = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET))
        torch.testing.assert_close(readback_targets[0, :, 3], torch.ones_like(readback_targets[0, :, 3]))
        torch.testing.assert_close(readback_targets[1], updated_targets[0], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 2
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION)), torch.full((2,), 0.5)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), torch.full((2,), 1000.0)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_POISSONS_RATIO)), torch.full((2,), 0.3)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING)), torch.full((2,), 0.005)
        )

        updated_youngs = torch.tensor([1000.0, 1500.0])
        material_view.set_attribute(
            TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS,
            wp.from_torch(updated_youngs),
            indices=wp.array([1], dtype=wp.int32),
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), updated_youngs.cpu()
        )

        for _ in range(5):
            sim.step()
            deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_surface_deformable_reads_writes_materials_and_steps():
    """Exercise authored surface state, topology, materials, and stepping."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(pretriangulated_surface_deformable_spawn_cfg())

        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == 2
        assert deformable.root_view.count == 2
        assert deformable.max_sim_vertices_per_body == 4
        assert deformable.max_sim_elements_per_body == 2
        assert deformable.max_collision_elements_per_body == 0
        assert deformable.data.nodal_state_w.torch.shape == (2, 4, 6)
        assert deformable.data.root_pos_w.torch.shape == (2, 3)
        assert deformable.data.root_vel_w.torch.shape == (2, 3)
        rest_positions = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_REST_POSITION))
        _assert_rest_positions_match_authored(
            rest_positions,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.2, 0.2, 0.0],
                    [0.0, 0.2, 0.0],
                ],
                device=rest_positions.device,
            ),
            deformable.root_view.prim_paths,
        )

        element_indices = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES))
        assert element_indices.shape == (2, 2, 3)
        assert element_indices.dtype == torch.int32
        assert torch.all((element_indices >= 0) & (element_indices < 4))
        expected_triangles = {(0, 1, 2), (0, 2, 3)}
        assert all(elements == expected_triangles for elements in _canonical_connectivity(element_indices))

        assert deformable.data.nodal_kinematic_target is None
        dummy_targets = torch.zeros((2, 4, 4), device=sim.device)
        with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
            deformable.write_nodal_kinematic_target_to_sim_index(dummy_targets)

        nodal_pos = deformable.data.nodal_pos_w.torch
        updated_pos = nodal_pos[1:2].clone()
        updated_pos[..., 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device=sim.device))
        readback_pos = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_POSITION))
        torch.testing.assert_close(readback_pos[0], nodal_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        nodal_vel = deformable.data.nodal_vel_w.torch
        updated_vel = nodal_vel[0:1].clone()
        updated_vel[..., 1] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel, env_ids=torch.tensor([0]))
        readback_vel = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_VELOCITY))
        torch.testing.assert_close(readback_vel[0], updated_vel[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_vel[1], nodal_vel[1], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 2
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION)), torch.full((2,), 0.4)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), torch.full((2,), 2000.0)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_POISSONS_RATIO)), torch.full((2,), 0.25)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING)), torch.full((2,), 0.03)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_STIFFNESS)), torch.full((2,), 0.6)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_THICKNESS)), torch.full((2,), 0.02)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_DAMPING)), torch.full((2,), 0.04)
        )

        updated_bending_damping = torch.tensor([0.08, 0.04])
        material_view.set_attribute(
            TT.DEFORMABLE_MATERIAL_BENDING_DAMPING,
            wp.from_torch(updated_bending_damping),
            indices=wp.array([0], dtype=wp.int32),
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_DAMPING)),
            updated_bending_damping.cpu(),
        )

        for _ in range(5):
            sim.step()
            deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_deformable_interactive_scene_uses_full_authored_stage():
    """Initialize cloned deformable bodies and materials from the full authored stage."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        scene = InteractiveScene(DeformableSceneCfg(num_envs=3, env_spacing=0.75, lazy_sensor_update=False))

        sim.reset()

        deformable = scene["deformable"]
        assert deformable.num_instances == 3
        assert deformable.root_view.count == 3
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == 3

        sim.step()
        scene.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_forced_rewarm_rebuilds_deformable_bindings():
    """Replace deformable bindings when a forced re-warm replaces the attached stage."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=2)
        sim.reset()

        original_view = deformable.root_view
        original_binding = original_view.binding_for(TT.DEFORMABLE_SIM_NODAL_POSITION)

        OvPhysxManager._warmup_done = False
        sim.reset()

        assert deformable.is_initialized
        assert deformable.root_view is not original_view
        assert deformable.root_view.binding_for(TT.DEFORMABLE_SIM_NODAL_POSITION) is not original_binding
        _assert_finite_deformable_state(deformable)
        sim.step()
        deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_mixed_deformable_rigid_scene_does_not_duplicate_runtime_clones():
    """Keep deformable, material, and rigid clone counts aligned in a mixed scene."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        scene = InteractiveScene(MixedDeformableRigidSceneCfg(num_envs=3, env_spacing=0.75, lazy_sensor_update=False))

        sim.reset()

        deformable = scene["deformable"]
        cube = scene["cube"]
        assert deformable.num_instances == 3
        assert deformable.root_view.count == 3
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == 3
        assert cube.num_instances == 3
        assert cube.root_view.count == 3

        sim.step()
        scene.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)
        assert torch.isfinite(cube.data.root_pos_w.torch).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_heterogeneous_mixed_deformable_rigid_scene_materializes_missing_targets(
    monkeypatch: pytest.MonkeyPatch,
):
    """Materialize missing rigid targets beside full-stage deformable clones without duplicates."""
    from isaaclab import cloner

    clone_cfg_type = cloner.CloneCfg
    monkeypatch.setattr(
        cloner,
        "CloneCfg",
        lambda device: clone_cfg_type(device=device, clone_strategy=cloner.sequential),
    )

    with _ovphysx_sim_context(device="cuda:0") as sim:
        num_envs = 4
        scene = InteractiveScene(
            HeterogeneousMixedDeformableRigidSceneCfg(
                num_envs=num_envs,
                env_spacing=1.0,
                lazy_sensor_update=False,
            )
        )
        plan = scene.clone_plan
        assert plan is not None
        shape_rows = plan.cfg_rows[id(scene.cfg.shape)]
        shape_mask = plan.clone_mask[list(shape_rows)]
        assert shape_mask.sum(dim=1).tolist() == [2, 2]
        assert shape_mask.sum(dim=0).tolist() == [1, 1, 1, 1]

        expected_paths = {f"/World/envs/env_{index}/Shape" for index in range(num_envs)}
        source_paths = {plan.sources[row] for row in shape_rows}
        assert source_paths == {"/World/envs/env_0/Shape", "/World/envs/env_1/Shape"}
        stage = sim_utils.get_current_stage()
        ancestor_path = "/World/envs/env_2/Shape"
        camera_path = f"{ancestor_path}/Camera"
        UsdGeom.Xform.Define(stage, camera_path)
        authored_paths = {path for path in expected_paths if stage.GetPrimAtPath(path).IsValid()}
        assert authored_paths == source_paths | {ancestor_path}
        authored_deformable_paths = {f"/World/envs/env_{index}/Object/simulation" for index in range(num_envs)}
        deformable_rows = plan.cfg_rows[id(scene.cfg.deformable)]
        deformable_source_paths = {f"{plan.sources[row]}/simulation" for row in deformable_rows}
        assert {
            path for path in authored_deformable_paths if stage.GetPrimAtPath(path).IsValid()
        } == deformable_source_paths

        sim.reset()

        deformable = scene["deformable"]
        shape = scene["shape"]
        assert deformable.root_view.count == num_envs, deformable.root_view.prim_paths
        assert shape.root_view.count == num_envs, shape.root_view.prim_paths
        runtime_paths = shape.root_view.prim_paths
        assert set(runtime_paths) == expected_paths
        assert len(runtime_paths) == len(set(runtime_paths)) == num_envs
        assert OvPhysxManager._stage_usda is not None
        layer = Sdf.Layer.CreateAnonymous("materialized.usda")
        assert layer.ImportFromString(OvPhysxManager._stage_usda)
        materialized_stage = Usd.Stage.Open(layer)
        assert materialized_stage.GetPrimAtPath(camera_path).IsValid()
        assert all(materialized_stage.GetPrimAtPath(path).IsValid() for path in authored_deformable_paths)

        sim.step()
        scene.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)
        assert torch.isfinite(shape.data.root_pos_w.torch).all()
