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
from isaaclab_newton.physics import MJWarpSolverCfg, XPBDSolverCfg
from newton import Model, ShapeFlags
from newton.viewer import ViewerBase, ViewerNull

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner import path as cloner_path
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )


@configclass
class _MeshVariantSceneCfg(InteractiveSceneCfg):
    object: RigidObjectCfg = _mesh_variant_object("Object", 0.2)


def _snapshot(scene: InteractiveScene, model: Model) -> dict[str, np.ndarray]:
    """Observe collision geometry, inertia, and public rigid-object state after a write or step."""
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
@pytest.mark.parametrize("use_mujoco_contacts", [False, True])
def test_initially_unused_variants_remain_resources(device: str, use_mujoco_contacts: bool) -> None:
    """Unused alternatives stay out of live worlds and are authored only for supported reset-time selection."""
    scene_cfg = _MeshVariantSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.object.spawn.assets_cfg[1].func = sim_utils.spawn_mesh_cone
    scene_cfg.imu = ImuCfg(prim_path=scene_cfg.object.prim_path)
    third = scene_cfg.object.spawn.assets_cfg[0].copy()
    third.radius, third.mass_props.mass = 0.06, 27.0
    scene_cfg.object.spawn.assets_cfg.append(third)
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=use_mujoco_contacts),
            use_cuda_graph=False,
            load_visual_shapes=True,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        sim.reset()
        asset = scene["object"]
        model = sim.backend_manager.get_model()
        assert model.body_count == 1
        paths = cloner_path.get_asset_prototype_paths(sim.get_clone_plan())
        if use_mujoco_contacts:
            assert sum(path is not None for path in paths) == 1
            with pytest.raises(NotImplementedError, match="Reset-time mesh variant selection is unavailable"):
                asset.write_mesh_variant_to_sim(wp.array([2], dtype=wp.int32, device=device))
            return
        assert sum(path is not None for path in paths) == len(scene_cfg.object.spawn.assets_cfg)
        assert all(sim.stage.GetPrimAtPath(path) for path in paths if path is not None)
        bank = next(iter(sim.backend_manager.backend.mesh_variants.values()))
        viewer = ViewerNull()
        viewer.set_model(
            model, shape_choices=[ViewerBase.ShapeChoices(bank.render_model, bank.body_indices, bank.variant_ids.warp)]
        )
        batches = [batch for batch in viewer._shape_instances.values() if batch.selected is bank.variant_ids.warp]
        assert batches
        shape_count = model.shape_count
        geometry_count = np.count_nonzero(model.shape_flags.numpy() & int(ShapeFlags.SITE) == 0)
        for variant, mass in enumerate((1.0, 8.0, 27.0)):
            asset.write_mesh_variant_to_sim(wp.array([variant], dtype=wp.int32, device=device))
            assert asset.mesh_variant_ids.torch.item() == variant
            assert asset.data.body_mass.torch.item() == pytest.approx(mass)
            assert (model.body_count, model.shape_count) == (1, shape_count)
            viewer.log_state(sim.backend_manager.get_state_0())
            for batch in batches:
                active = np.any(batch.scales.numpy() != 0.0, axis=1)
                np.testing.assert_array_equal(active, batch.choice_indices.numpy() == variant)
        viewer.close()
        sim.reset()
        model = sim.backend_manager.get_model()
        assert model.body_count == 1
        assert np.count_nonzero(model.shape_flags.numpy() & int(ShapeFlags.SITE) == 0) == geometry_count
        asset.write_mesh_variant_to_sim(wp.array([2], dtype=wp.int32, device=device))
        assert asset.data.body_mass.torch.item() == pytest.approx(27.0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.parametrize(
    ("solver_cfg", "supports_selection"),
    [
        pytest.param(MJWarpSolverCfg(use_mujoco_contacts=False), True, id="mjwarp-external"),
        pytest.param(MJWarpSolverCfg(use_mujoco_contacts=True), False, id="mjwarp-internal"),
        pytest.param(XPBDSolverCfg(), False, id="xpbd"),
    ],
)
def test_write_mesh_variant_updates_selected_environment(device: str, solver_cfg, supports_selection: bool) -> None:
    """Keep static catalogs usable; supported solvers switch only the selected world's geometry and inertia."""
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_MeshVariantSceneCfg(num_envs=2, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()

        asset = scene["object"]
        assert asset.num_mesh_variants == 2
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([1.0, 8.0]))
        if not supports_selection:
            with pytest.raises(NotImplementedError, match="Reset-time mesh variant selection is unavailable"):
                _ = asset.mesh_variant_ids
            with pytest.raises(NotImplementedError, match="Reset-time mesh variant selection is unavailable"):
                asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=sim.device), (0,))
            poses = asset.data.root_link_pose_w.torch.clone()
            sim.step()
            scene.update(sim.cfg.dt)
            torch.testing.assert_close(asset.data.root_link_pose_w.torch, poses)
            return
        ids = asset.mesh_variant_ids.torch
        torch.testing.assert_close(ids.cpu(), torch.tensor([0, 1], dtype=torch.int32))
        model = sim.backend_manager.get_model()
        initial = _snapshot(scene, model)
        bodies = [np.flatnonzero(model.body_world.numpy() == world).item() for world in range(2)]
        colliding = model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES) != 0
        shapes = [np.flatnonzero((model.shape_body.numpy() == body) & colliding) for body in bodies]

        asset.write_mesh_variant_to_sim(
            torch.tensor([1], dtype=torch.int32, device=sim.device),
            torch.tensor([0], dtype=torch.int32, device=sim.device),
        )
        selected = _snapshot(scene, model)
        assert asset.mesh_variant_ids.torch.data_ptr() == ids.data_ptr()
        torch.testing.assert_close(ids.cpu(), torch.tensor([1, 1], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 8.0]))
        for name, before in initial.items():
            if name.startswith("model.shape_"):
                np.testing.assert_array_equal(selected[name][shapes[1]], before[shapes[1]], err_msg=name)
                np.testing.assert_array_equal(selected[name][shapes[0]], before[shapes[1]], err_msg=name)
            elif name.startswith("model.body_"):
                np.testing.assert_array_equal(selected[name][bodies[1]], before[bodies[1]], err_msg=name)
                if name == "model.body_inv_inertia":
                    np.testing.assert_allclose(
                        selected[name][bodies[0]], before[bodies[1]], rtol=2 * np.finfo(np.float32).eps, atol=0
                    )
                else:
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

        asset.write_mesh_variant_to_sim(
            wp.array([0], dtype=wp.int32, device=sim.device), wp.array([1], dtype=wp.int64, device=sim.device)
        )
        sim.step()
        scene.update(sim.cfg.dt)

        torch.testing.assert_close(ids.cpu(), torch.tensor([1, 0], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 1.0]))

        # A hard reset rebuilds backend allocations; public selection views must follow the new owner.
        shape_count = model.shape_count
        sim.reset()
        assert sim.backend_manager.get_model().shape_count == shape_count
        rebuilt_initial = _snapshot(scene, sim.backend_manager.get_model())
        rebuilt_ids = asset.mesh_variant_ids.torch
        assert rebuilt_ids.data_ptr() != ids.data_ptr()
        asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=sim.device), (0,))
        torch.testing.assert_close(rebuilt_ids.cpu(), torch.tensor([1, 1], dtype=torch.int32))

        # Warm the int64 launch before queuing delayed production on a non-default caller stream.
        asset.write_mesh_variant_to_sim(
            torch.ones(2, dtype=torch.int64, device=sim.device),
            torch.arange(2, dtype=torch.int64, device=sim.device),
        )
        desired_indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=sim.device)
        torch.cuda.synchronize(sim.device)
        caller_stream = torch.cuda.Stream(device=sim.device)
        with torch.cuda.stream(caller_stream):
            variant_ids = torch.ones(2, dtype=torch.int64, device=sim.device)
            env_ids = torch.zeros(2, dtype=torch.int64, device=sim.device)
            torch.cuda._sleep(100_000_000)
            variant_ids.copy_(desired_indices[0])
            env_ids.copy_(desired_indices[1])
            asset.write_mesh_variant_to_sim(variant_ids, env_ids)
            # Immediate caller reuse must not change the queued geometry selection.
            variant_ids.fill_(1)
            env_ids.fill_(0)
            del variant_ids, env_ids
            torch.empty(4, dtype=torch.int64, device=sim.device).zero_()
        model = sim.backend_manager.get_model()
        streamed = _snapshot(scene, model)
        np.testing.assert_array_equal(streamed["object.variant_ids"], (1, 0))
        np.testing.assert_allclose(streamed["model.body_mass"][bodies], (8.0, 1.0))
        for world, source_world in ((0, 1), (1, 0)):
            np.testing.assert_array_equal(
                streamed["model.shape_source_ptr"][shapes[world]],
                rebuilt_initial["model.shape_source_ptr"][shapes[source_world]],
            )

        # Warp-only writes remain capture-safe and consume the replay's current selector values.
        variant_ids = wp.array([0], dtype=wp.int64, device=sim.device)
        env_ids = wp.array([0], dtype=wp.int32, device=sim.device)
        asset.write_mesh_variant_to_sim(variant_ids, env_ids)
        with wp.ScopedCapture(device=sim.device) as capture:
            asset.write_mesh_variant_to_sim(variant_ids, env_ids)
        variant_ids.fill_(1)
        wp.capture_launch(capture.graph)
        captured = _snapshot(scene, model)
        np.testing.assert_array_equal(captured["object.variant_ids"], (1, 0))
        np.testing.assert_allclose(captured["model.body_mass"][bodies], (8.0, 1.0))
