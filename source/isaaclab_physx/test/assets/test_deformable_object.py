# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX deformable-object coverage on locally authored surface and volume meshes.

The meshes are authored directly with their simulation topology, so no mesher or collision cooking runs. Each
deformable holds two environments so that partial writes can target environment 1 and prove that environment 0
is preserved in the real PhysX state. The CPU failure test owns its simulation context and is defined first:
pytest runs it before the composite scene is created, and a new simulation context would replace that stage.
"""

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation(physics="isaacsim_physx")

import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.assets import DeformableObject
from isaaclab_physx.sim import PhysxDeformableBodyMaterialCfg, PhysxSurfaceDeformableBodyMaterialCfg

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.math import quat_apply

_NUM_ENVS = 2
_VOLUME_POINTS = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2), (0.2, 0.2, 0.2)]
_VOLUME_TETS = [(0, 1, 2, 3), (1, 2, 3, 4)]
_VOLUME_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 4), (2, 3, 4), (3, 1, 4)]
_SURFACE_POINTS = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 0.2, 0.0), (0.0, 0.2, 0.0)]
_SURFACE_TRIANGLES = [(0, 1, 2), (0, 2, 3)]


def _author_deformable_body(
    body_prim: Usd.Prim,
    points: list[Gf.Vec3f],
    sim_api: str,
    material_cfg: PhysxDeformableBodyMaterialCfg | PhysxSurfaceDeformableBodyMaterialCfg,
) -> None:
    """Author the deformable body schemas, rest shape, and a bound physics material on a mesh prim.

    Args:
        body_prim: Mesh or tetrahedral mesh prim that becomes the deformable body.
        points: Rest positions of the simulation nodes in the prim frame [m].
        sim_api: Surface or volume deformable simulation API schema to apply.
        material_cfg: Physics material to create below the prim and bind to it.
    """
    applied_schemas = Sdf.TokenListOp()
    applied_schemas.explicitItems = [
        "OmniPhysicsDeformableBodyAPI",
        sim_api,
        "OmniPhysicsDeformablePoseAPI:default",
        "PhysicsCollisionAPI",
    ]
    body_prim.SetMetadata("apiSchemas", applied_schemas)
    body_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        ["bindPose"]
    )
    body_prim.CreateAttribute("omniphysics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("velocities", Sdf.ValueTypeNames.Vector3fArray).Set([Gf.Vec3f()] * len(points))
    material_prim = material_cfg.func(f"{body_prim.GetPath()}/material", material_cfg)
    UsdShade.MaterialBindingAPI.Apply(body_prim)
    UsdShade.MaterialBindingAPI(body_prim).Bind(
        UsdShade.Material(material_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _spawn_volume_deformables(root: str, y_offset: float) -> DeformableObject:
    """Author one five-node, two-tetrahedron volume deformable per environment."""
    stage = sim_utils.get_current_stage()
    points = [Gf.Vec3f(*point) for point in _VOLUME_POINTS]
    for env_index in range(_NUM_ENVS):
        object_path = f"{root}/Env_{env_index}/Object"
        sim_utils.create_prim(f"{root}/Env_{env_index}", "Xform", translation=(env_index * 1.0, y_offset, 1.0))
        UsdGeom.Xform.Define(stage, object_path)
        tet_mesh = UsdGeom.TetMesh.Define(stage, f"{object_path}/simulation")
        tet_mesh.CreatePointsAttr(points)
        tet_mesh.CreateTetVertexIndicesAttr([Gf.Vec4i(*tet) for tet in _VOLUME_TETS])
        tet_mesh.CreateSurfaceFaceVertexIndicesAttr([Gf.Vec3i(*face) for face in _VOLUME_FACES])
        body_prim = tet_mesh.GetPrim()
        _author_deformable_body(
            body_prim, points, "OmniPhysicsVolumeDeformableSimAPI", PhysxDeformableBodyMaterialCfg()
        )
        body_prim.CreateAttribute("omniphysics:restTetVtxIndices", Sdf.ValueTypeNames.Int4Array).Set(
            [Gf.Vec4i(*tet) for tet in _VOLUME_TETS]
        )
        visual = UsdGeom.Mesh.Define(stage, f"{object_path}/visual")
        visual.CreatePointsAttr(points)
        visual.CreateFaceVertexCountsAttr([3] * len(_VOLUME_FACES))
        visual.CreateFaceVertexIndicesAttr([index for face in _VOLUME_FACES for index in face])
    return DeformableObject(DeformableObjectCfg(prim_path=f"{root}/Env_[^/]*/Object"))


def _spawn_surface_deformables(root: str, y_offset: float) -> DeformableObject:
    """Author one four-node, two-triangle surface deformable per environment."""
    stage = sim_utils.get_current_stage()
    points = [Gf.Vec3f(*point) for point in _SURFACE_POINTS]
    for env_index in range(_NUM_ENVS):
        sim_utils.create_prim(f"{root}/Env_{env_index}", "Xform", translation=(env_index * 1.0, y_offset, 1.0))
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/Env_{env_index}/Object")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * len(_SURFACE_TRIANGLES))
        mesh.CreateFaceVertexIndicesAttr([index for triangle in _SURFACE_TRIANGLES for index in triangle])
        body_prim = mesh.GetPrim()
        _author_deformable_body(
            body_prim,
            points,
            "OmniPhysicsSurfaceDeformableSimAPI",
            PhysxSurfaceDeformableBodyMaterialCfg(density=900.0, youngs_modulus=2000.0, surface_thickness=0.02),
        )
        body_prim.CreateAttribute("omniphysics:restTriVtxIndices", Sdf.ValueTypeNames.Int3Array).Set(
            [Gf.Vec3i(*triangle) for triangle in _SURFACE_TRIANGLES]
        )
    return DeformableObject(DeformableObjectCfg(prim_path=f"{root}/Env_[^/]*/Object"))


@pytest.mark.isaacsim_ci
def test_initialization_on_device_cpu() -> None:
    """Test that initialization fails with deformable body API on the CPU."""
    with build_simulation_context(device="cpu", sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, gravity=(0.0, 0.0, 0.0))) as sim:
        sim._app_control_on_stop_handle = None
        deformable = _spawn_volume_deformables("/World/Volume", 0.0)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(deformable) < 10

        with pytest.raises(RuntimeError, match="Failed to create deformable body at"):
            sim.reset()


@dataclass
class _DeformableScene:
    """Volume and surface deformables that share one real PhysX lifecycle."""

    sim: sim_utils.SimulationContext
    device: str
    volume: DeformableObject
    surface: DeformableObject
    refcounts: dict[str, int]

    def step(self, num_steps: int = 1) -> None:
        """Write, step, and update both deformables."""
        for _ in range(num_steps):
            for deformable in (self.volume, self.surface):
                deformable.write_data_to_sim()
            self.sim.step()
            for deformable in (self.volume, self.surface):
                deformable.update(self.sim.cfg.dt)


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def deformable_scene(request) -> Iterator[_DeformableScene]:
    """Initialize the deformables once for this module; PhysX deformables require CUDA."""
    device = request.param
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, gravity=(0.0, 0.0, 0.0))) as sim:
        sim._app_control_on_stop_handle = None
        volume = _spawn_volume_deformables("/World/Volume", 0.0)
        surface = _spawn_surface_deformables("/World/Surface", 2.0)
        refcounts = {"volume": sys.getrefcount(volume), "surface": sys.getrefcount(surface)}
        sim.reset()
        yield _DeformableScene(sim=sim, device=device, volume=volume, surface=surface, refcounts=refcounts)


@pytest.mark.isaacsim_ci
def test_deformable_initialization(deformable_scene: _DeformableScene) -> None:
    """Resolve the volume and surface views, their materials, and their nodal buffers."""
    scene = deformable_scene
    for name, deformable, num_vertices in (
        ("volume", scene.volume, len(_VOLUME_POINTS)),
        ("surface", scene.surface, len(_SURFACE_POINTS)),
    ):
        # Check that the framework doesn't hold excessive strong references.
        assert scene.refcounts[name] < 10
        assert deformable.is_initialized
        assert deformable.num_instances == _NUM_ENVS
        assert deformable.root_view.count == _NUM_ENVS
        # Each environment binds its own material.
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == _NUM_ENVS
        assert deformable.max_sim_vertices_per_body == num_vertices
        assert deformable.data.nodal_state_w.torch.shape == (_NUM_ENVS, num_vertices, 6)
        assert deformable.data.root_pos_w.torch.shape == (_NUM_ENVS, 3)
        assert deformable.data.root_vel_w.torch.shape == (_NUM_ENVS, 3)
    # Only volume deformables carry kinematic targets.
    assert scene.volume.data.nodal_kinematic_target.torch.shape == (_NUM_ENVS, len(_VOLUME_POINTS), 4)
    assert scene.surface.data.nodal_kinematic_target is None
    dummy_targets = torch.zeros(_NUM_ENVS, len(_SURFACE_POINTS), 4, device=scene.device)
    with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
        scene.surface.write_nodal_kinematic_target_to_sim_index(dummy_targets)


@pytest.mark.isaacsim_ci
def test_deformable_nodal_state_writes(deformable_scene: _DeformableScene) -> None:
    """Partial nodal state writes reach only the selected environment of each real view."""
    scene = deformable_scene
    device = scene.device
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    for deformable in (scene.volume, scene.surface):
        initial_state = deformable.data.nodal_state_w.torch.clone()
        state = initial_state[1:].clone()
        state[:, 0, :3] += torch.tensor([0.01, 0.02, 0.03], device=device)
        state[:, 0, 3] = 0.25
        deformable.write_nodal_state_to_sim_index(state, env_ids=env_ids)
        expected = torch.cat((initial_state[:1], state))
        torch.testing.assert_close(deformable.data.nodal_state_w.torch, expected)
        for getter, component in (
            (deformable.root_view.get_simulation_nodal_positions, slice(0, 3)),
            (deformable.root_view.get_simulation_nodal_velocities, slice(3, 6)),
        ):
            raw = wp.to_torch(getter()).to(device).reshape(expected[..., component].shape)
            torch.testing.assert_close(raw, expected[..., component])

    # A transformed default state keeps its shape and moves its mean to the requested position.
    volume = scene.volume
    nodal_state = volume.data.default_nodal_state_w.torch.clone()
    pos_w = torch.tensor([[0.1, 0.2, 1.3], [0.4, -0.3, 1.1]], device=device)
    quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071068, 0.7071068]], device=device)
    nodal_pos = volume.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
    default_offsets = nodal_state[..., :3] - nodal_state[..., :3].mean(dim=1, keepdim=True)
    torch.testing.assert_close(nodal_pos.mean(dim=1), nodal_state[..., :3].mean(dim=1) + pos_w)
    torch.testing.assert_close(
        nodal_pos - nodal_pos.mean(dim=1, keepdim=True),
        quat_apply(quat_w[:, None].expand_as(nodal_state[..., :4]), default_offsets),
    )
    nodal_state[..., :3] = nodal_pos
    nodal_state[..., 3:] = 0.0
    volume.write_nodal_state_to_sim_index(nodal_state)
    raw_nodal_pos = wp.to_torch(volume.root_view.get_simulation_nodal_positions()).to(device)
    torch.testing.assert_close(raw_nodal_pos.reshape(nodal_pos.shape), nodal_pos)


@pytest.mark.isaacsim_ci
def test_volume_kinematic_targets_hold_selected_nodes(deformable_scene: _DeformableScene) -> None:
    """Kinematic targets of environment 0 hold its nodes while environment 1 keeps moving freely."""
    scene = deformable_scene
    device = scene.device
    volume = scene.volume
    targets = volume.data.nodal_kinematic_target.torch.clone()
    raw_targets_before = wp.to_torch(volume.root_view.get_simulation_nodal_kinematic_targets()).clone()
    targets[0, :, :3] = volume.data.nodal_pos_w.torch[0] + torch.tensor([0.01, 0.02, 0.03], device=device)
    targets[0, :, 3] = 0.0
    volume.write_nodal_kinematic_target_to_sim_index(targets[:1], env_ids=torch.tensor([0], device=device))
    torch.testing.assert_close(volume.data.nodal_kinematic_target.torch, targets)
    raw_targets = wp.to_torch(volume.root_view.get_simulation_nodal_kinematic_targets()).reshape(targets.shape)
    torch.testing.assert_close(raw_targets[0].to(device), targets[0])
    torch.testing.assert_close(raw_targets[1], raw_targets_before.reshape(targets.shape)[1])

    # Environment 1 moves with a uniform nodal velocity; environment 0 stays on its targets.
    state = volume.data.nodal_state_w.torch.clone()
    state[1, :, 3:] = torch.tensor([0.0, 0.0, 0.5], device=device)
    volume.write_nodal_state_to_sim_index(state[1:], env_ids=torch.tensor([1], device=device))
    free_root_pos = volume.data.root_pos_w.torch[1].clone()
    scene.step(5)
    torch.testing.assert_close(volume.data.nodal_pos_w.torch[0], targets[0, :, :3], rtol=1e-5, atol=1e-5)
    assert volume.data.root_pos_w.torch[1, 2] > free_root_pos[2] + 1e-2
