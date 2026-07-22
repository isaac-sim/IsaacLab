# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for the OVPhysX deformable object."""

from __future__ import annotations

import ovphysx.types  # noqa: F401
import pytest
import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
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


def _ovphysx_sim_context(device: str, *, gravity_enabled: bool = False):
    """Build a kitless OVPhysX simulation context."""
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 60.0, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=True)


def _generate_deformable_scene(spawn: sim_utils.SpawnerCfg, num_objects: int = 2) -> DeformableObject:
    """Create independently authored deformables beneath matching parent prims."""
    for index in range(num_objects):
        sim_utils.create_prim(f"/World/Table_{index}", "Xform", translation=(index * 0.5, 0.0, 1.0))
    cfg = DeformableObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn,
        init_state=DeformableObjectCfg.InitialStateCfg(),
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

        nodal_state = deformable.data.nodal_state_w.torch
        nodal_pos = deformable.data.nodal_pos_w.torch
        nodal_vel = deformable.data.nodal_vel_w.torch
        assert nodal_state.shape == (2, 5, 6)
        assert deformable.data.default_nodal_state_w.torch.shape == (2, 5, 6)
        assert deformable.data.root_pos_w.torch.shape == (2, 3)
        assert deformable.data.root_vel_w.torch.shape == (2, 3)
        torch.testing.assert_close(deformable.data.root_pos_w.torch, nodal_pos.mean(dim=1))
        torch.testing.assert_close(deformable.data.root_vel_w.torch, nodal_vel.mean(dim=1))

        element_indices = wp.to_torch(deformable.root_view.get_simulation_element_indices())
        collision_indices = wp.to_torch(deformable.root_view.get_collision_element_indices())
        assert element_indices.shape == (2, 2, 4)
        assert collision_indices.shape == (2, 2, 4)
        assert element_indices.dtype == torch.int32
        assert collision_indices.dtype == torch.int32
        assert torch.all((element_indices >= 0) & (element_indices < 5))
        assert torch.all((collision_indices >= 0) & (collision_indices < 5))
        expected_tetrahedra = {(0, 1, 2, 3), (1, 2, 3, 4)}
        assert all(elements == expected_tetrahedra for elements in _canonical_connectivity(element_indices))
        assert all(elements == expected_tetrahedra for elements in _canonical_connectivity(collision_indices))
        torch.testing.assert_close(collision_indices, element_indices)

        updated_pos = nodal_pos[1:2].clone()
        updated_pos[..., 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device=sim.device))
        readback_pos = wp.to_torch(deformable.root_view.get_simulation_nodal_positions())
        torch.testing.assert_close(readback_pos[0], nodal_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        updated_vel = nodal_vel[0:1].clone()
        updated_vel[..., 1] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel, env_ids=torch.tensor([0]))
        readback_vel = wp.to_torch(deformable.root_view.get_simulation_nodal_velocities())
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
        readback_targets = wp.to_torch(deformable.root_view.get_simulation_nodal_kinematic_targets())
        torch.testing.assert_close(readback_targets[0, :, 3], torch.ones_like(readback_targets[0, :, 3]))
        torch.testing.assert_close(readback_targets[1], updated_targets[0], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 2
        torch.testing.assert_close(wp.to_torch(material_view.get_dynamic_frictions()), torch.full((2,), 0.5))
        torch.testing.assert_close(wp.to_torch(material_view.get_youngs_moduli()), torch.full((2,), 1000.0))
        torch.testing.assert_close(wp.to_torch(material_view.get_poissons_ratios()), torch.full((2,), 0.3))
        torch.testing.assert_close(wp.to_torch(material_view.get_elasticity_dampings()), torch.full((2,), 0.005))

        updated_youngs = torch.tensor([1000.0, 1500.0], device=sim.device)
        material_view.set_youngs_moduli(updated_youngs, indices=torch.tensor([1], device=sim.device))
        torch.testing.assert_close(wp.to_torch(material_view.get_youngs_moduli()), updated_youngs.cpu())

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

        element_indices = wp.to_torch(deformable.root_view.get_simulation_element_indices())
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
        readback_pos = wp.to_torch(deformable.root_view.get_simulation_nodal_positions())
        torch.testing.assert_close(readback_pos[0], nodal_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        nodal_vel = deformable.data.nodal_vel_w.torch
        updated_vel = nodal_vel[0:1].clone()
        updated_vel[..., 1] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel, env_ids=torch.tensor([0]))
        readback_vel = wp.to_torch(deformable.root_view.get_simulation_nodal_velocities())
        torch.testing.assert_close(readback_vel[0], updated_vel[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_vel[1], nodal_vel[1], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 2
        torch.testing.assert_close(wp.to_torch(material_view.get_dynamic_frictions()), torch.full((2,), 0.4))
        torch.testing.assert_close(wp.to_torch(material_view.get_youngs_moduli()), torch.full((2,), 2000.0))
        torch.testing.assert_close(wp.to_torch(material_view.get_poissons_ratios()), torch.full((2,), 0.25))
        torch.testing.assert_close(wp.to_torch(material_view.get_elasticity_dampings()), torch.full((2,), 0.03))
        torch.testing.assert_close(wp.to_torch(material_view.get_bending_stiffnesses()), torch.full((2,), 0.6))
        torch.testing.assert_close(wp.to_torch(material_view.get_thicknesses()), torch.full((2,), 0.02))
        torch.testing.assert_close(wp.to_torch(material_view.get_bending_dampings()), torch.full((2,), 0.04))

        updated_bending_damping = torch.tensor([0.08, 0.04], device=sim.device)
        material_view.set_bending_dampings(updated_bending_damping, indices=torch.tensor([0], device=sim.device))
        torch.testing.assert_close(wp.to_torch(material_view.get_bending_dampings()), updated_bending_damping.cpu())

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
