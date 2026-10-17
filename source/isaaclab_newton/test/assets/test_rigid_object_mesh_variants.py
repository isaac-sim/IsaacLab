# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for Newton rigid-object mesh variants."""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager
from newton import ShapeFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.configclass import configclass

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _spawn_named_cone(prim_path, cfg, **kwargs):
    """Give the second variant a different body path, as independently authored USDs do."""
    return sim_utils.spawn_mesh_cone(prim_path + "/large_part", cfg, **kwargs)


def _mesh_variant_object(prim_name: str, height: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.MeshConeCfg(
                    radius=0.02,
                    height=0.08,
                    mass_props=sim_utils.MassCfg(mass=1.0),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                ),
                sim_utils.MeshConeCfg(
                    func=_spawn_named_cone,
                    radius=0.05,
                    height=0.05,
                    mass_props=sim_utils.MassCfg(mass=8.0),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                ),
            ]
        ),
        mesh_variants_enabled=True,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )


@configclass
class _MeshVariantSceneCfg(InteractiveSceneCfg):
    object: RigidObjectCfg = _mesh_variant_object("Object", 0.2)


def _snapshot(scene: InteractiveScene) -> dict[str, np.ndarray]:
    """Observe collision geometry, inertia, and public rigid-object state after a write or step."""
    model = NewtonManager.get_model()
    arrays = {}
    for name in (
        "shape_scale",
        "shape_transform",
        "shape_collision_radius",
        "shape_source_ptr",
        "shape_edge_range",
        "shape_collision_aabb_lower",
        "shape_collision_aabb_upper",
        "body_mass",
        "body_inv_mass",
        "body_com",
        "body_inertia",
        "body_inv_inertia",
    ):
        arrays[f"model.{name}"] = getattr(model, name)
    for name, obj in scene.rigid_objects.items():
        arrays[f"{name}.variant_ids"] = obj.mesh_variant_ids.warp
        arrays[f"{name}.pose"] = obj.data.root_link_pose_w.warp
        arrays[f"{name}.velocity"] = obj.data.root_com_vel_w.warp
        arrays[f"{name}.acceleration"] = obj.data.body_com_acc_w.warp
    wp.synchronize_device()
    return {name: array.numpy().copy() for name, array in arrays.items()}


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_write_mesh_variant_updates_selected_environment(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch geometry and inertia without replacing views or changing the unselected world."""
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_MeshVariantSceneCfg(num_envs=2, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()

        asset = scene["object"]
        ids = asset.mesh_variant_ids.torch
        torch.testing.assert_close(ids.cpu(), torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([1.0, 8.0]))
        initial = _snapshot(scene)
        model = NewtonManager.get_model()
        bodies = [np.flatnonzero(model.body_world.numpy() == world).item() for world in range(2)]
        colliding = model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES) != 0
        shapes = [np.flatnonzero((model.shape_body.numpy() == body) & colliding) for body in bodies]

        asset.write_mesh_variant_to_sim(
            torch.tensor([1], dtype=torch.int32, device=sim.device),
            torch.tensor([0], dtype=torch.int32, device=sim.device),
        )
        selected = _snapshot(scene)
        assert asset.mesh_variant_ids.torch.data_ptr() == ids.data_ptr()
        torch.testing.assert_close(ids.cpu(), torch.tensor([1, 1], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 8.0]))
        for name, before in initial.items():
            if name.startswith("model.shape_"):
                np.testing.assert_array_equal(selected[name][shapes[1]], before[shapes[1]], err_msg=name)
                np.testing.assert_array_equal(selected[name][shapes[0]], before[shapes[1]], err_msg=name)
            elif name.startswith("model.body_"):
                np.testing.assert_array_equal(selected[name][bodies[1]], before[bodies[1]], err_msg=name)
                np.testing.assert_array_equal(selected[name][bodies[0]], before[bodies[1]], err_msg=name)
            else:
                np.testing.assert_array_equal(selected[name][1], before[1], err_msg=name)

        sim.step()
        scene.update(sim.cfg.dt)
        assert torch.isfinite(asset.data.root_link_pose_w.torch).all()
        assert torch.isfinite(asset.data.root_com_vel_w.torch).all()
        torch.testing.assert_close(
            asset.data.root_link_pose_w.torch[1].cpu(), torch.from_numpy(initial["object.pose"][1])
        )

        asset.write_mesh_variant_to_sim(torch.tensor([0], dtype=torch.int32, device=sim.device), (1,))
        sim.step()
        scene.update(sim.cfg.dt)

        torch.testing.assert_close(ids.cpu(), torch.tensor([1, 0], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 1.0]))

        # A hard reset rebuilds backend allocations; public selection views must follow the new owner.
        sim.reset()
        rebuilt_ids = asset.mesh_variant_ids.torch
        assert rebuilt_ids.data_ptr() != ids.data_ptr()
        asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=sim.device), (0,))
        torch.testing.assert_close(rebuilt_ids.cpu(), torch.tensor([1, 1], dtype=torch.int32))

        captured = {}

        def capture_indices(name, *, variant_ids, env_ids):
            captured[name] = variant_ids, env_ids

        monkeypatch.setattr(NewtonManager, "_set_mesh_variant_index", capture_indices)
        variant_ids = torch.tensor([0, 1], dtype=torch.int32, device=sim.device)
        env_ids = torch.tensor([1, 0], dtype=torch.int32, device=sim.device)
        asset.write_mesh_variant_to_sim(variant_ids, env_ids)
        captured_variants, captured_envs = captured[asset.cfg.prim_path]
        assert wp.to_torch(captured_variants).data_ptr() != variant_ids.data_ptr()
        assert wp.to_torch(captured_envs).data_ptr() != env_ids.data_ptr()
        torch.cuda.synchronize(sim.device)
        variant_ids.fill_(1)
        env_ids.fill_(0)

        torch.testing.assert_close(wp.to_torch(captured_variants).cpu(), torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(wp.to_torch(captured_envs).cpu(), torch.tensor([1, 0], dtype=torch.int32))

        asset.write_mesh_variant_to_sim(variant_ids, (0, 1))
        _, captured_envs = captured[asset.cfg.prim_path]
        torch.testing.assert_close(wp.to_torch(captured_envs).cpu(), torch.tensor([0, 1], dtype=torch.int32))
