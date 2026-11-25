# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration coverage for multi-asset spawners backed by Newton body variants."""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, XPBDSolverCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner import InclusionSet
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.configclass import configclass

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _spawn_named_cone(prim_path, cfg, **kwargs):
    """Give the second variant a different body path."""
    return sim_utils.spawn_mesh_cone(prim_path + "/large_part", cfg, **kwargs)


def _spawn_compound_cone(prim_path, cfg, **kwargs):
    prim = sim_utils.spawn_mesh_cone(prim_path, cfg, **kwargs)
    sim_utils.spawn_sphere(
        prim_path + "/ExtraCollider",
        sim_utils.SphereCfg(radius=0.01, collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        translation=(0.0, 0.0, 0.04),
    )
    return prim


def _variant_object() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            static=False,
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
            ],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )


@configclass
class _VariantSceneCfg(InteractiveSceneCfg):
    object: RigidObjectCfg = _variant_object()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_native_selection_preserves_initial_worlds_and_unused_variant(device: str) -> None:
    scene_cfg = _VariantSceneCfg(num_envs=2, env_spacing=2.0)
    scene_cfg.object.spawn.assets_cfg[1].func = sim_utils.spawn_mesh_cone
    third = scene_cfg.object.spawn.assets_cfg[0].copy()
    third.radius, third.mass_props.mass = 0.06, 27.0
    third.func = _spawn_compound_cone
    scene_cfg.object.spawn.assets_cfg.append(third)
    scene_cfg.object.mesh_variant_inertia_diagonal_offset = 1.0e-5
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            use_cuda_graph=False,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        sim.reset()
        asset, model = scene["object"], sim.backend_manager.backend.model
        assert model.body_count == 2
        shape_count = model.shape_count
        assert asset.num_mesh_variants == 3
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([0, 2], dtype=torch.int32))
        asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=device), (0,))
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([1, 2], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 27.0]))
        asset.write_mesh_variant_to_sim(
            wp.array([0], dtype=wp.int32, device=device), wp.array([1], dtype=wp.int64, device=device)
        )
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 1.0]))
        assert model.shape_count == shape_count
        forces = torch.zeros((2, 1, 3), device=device)
        forces[:, 0, 0] = 8.0
        asset.permanent_wrench_composer.set_forces_and_torques_index(forces=forces)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
        torch.testing.assert_close(
            asset.data.root_com_vel_w.torch[:, 0],
            torch.tensor([sim.cfg.dt, 8.0 * sim.cfg.dt], device=device),
            rtol=1.0e-4,
            atol=1.0e-6,
        )
        assert torch.isfinite(asset.data.root_link_pose_w.torch).all()
        assert torch.isfinite(asset.data.root_com_vel_w.torch).all()

        sim.reset()
        assert sim.backend_manager.backend.model.shape_count == shape_count
        asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int64, device=device), (0,))
        assert asset.data.body_mass.torch[0, 0].item() == pytest.approx(8.0)

        # The caller coordinates Torch and Warp on the same stream before reusing its input buffers.
        caller_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(caller_stream), wp.ScopedStream(wp.stream_from_torch(caller_stream), sync_exit=True):
            variants = torch.zeros(2, dtype=torch.int64, device=device)
            worlds = torch.arange(2, dtype=torch.int64, device=device)
            variants.copy_(torch.tensor([2, 0], dtype=torch.int64, device=device))
            asset.write_mesh_variant_to_sim(variants, worlds)
            variants.fill_(0)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([2, 0], dtype=torch.int32))

        variants = wp.array([0], dtype=wp.int32, device=device)
        worlds = wp.array([1], dtype=wp.int32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            asset.write_mesh_variant_to_sim(variants, worlds)
        variants.fill_(1)
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([2, 1], dtype=torch.int32))


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_semantic_worlds_choose_initial_variants(device: str) -> None:
    """Semantic rules select the right native variants, including inherited wrapper tags."""

    scene_cfg = _VariantSceneCfg(num_envs=4, env_spacing=2.0)
    third = scene_cfg.object.spawn.assets_cfg[0].copy()
    third.radius, third.mass_props.mass = 0.06, 27.0
    scene_cfg.object.spawn.assets_cfg.append(third)
    scene_cfg.fixture = _variant_object()
    scene_cfg.fixture.prim_path = "{ENV_REGEX_NS}/Fixture"
    scene_cfg.fixture.spawn.static = True
    scene_cfg.fixture.init_state.pos = (0.5, 0.0, 0.2)
    for obj, tags in zip(scene_cfg.object.spawn.assets_cfg, ([("assembly", "spare")], [], [("assembly", "drive")])):
        obj.semantic_tags = tags
    scene_cfg.fixture.spawn.assets_cfg[0].semantic_tags = [("assembly", "spare")]
    scene_cfg.fixture.spawn.assets_cfg[1].semantic_tags = [("assembly", "spare"), ("assembly", "drive")]
    scene_cfg.object.spawn.semantic_tags = [("domain", "mechanical")]
    scene_cfg.fixture.spawn.semantic_tags = [("domain", "mechanical")]
    scene_cfg.clone_cfg.clone_combinations = [
        InclusionSet(assets=["object", "fixture"], match_tags=["assembly", "domain"]),
    ]
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        sim.reset()
        asset, fixture = scene["object"], scene["fixture"]
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([0, 0, 2, 2], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([1.0, 1.0, 27.0, 27.0]))
        torch.testing.assert_close(fixture.data.body_mass.torch[:, 0].cpu(), torch.tensor([1.0, 8.0, 8.0, 8.0]))


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.parametrize(
    "solver_cfg, static",
    [
        pytest.param(MJWarpSolverCfg(use_mujoco_contacts=False), True, id="static-multi-asset"),
        pytest.param(MJWarpSolverCfg(use_mujoco_contacts=True), False, id="mjwarp-internal"),
        pytest.param(XPBDSolverCfg(), False, id="xpbd"),
    ],
)
def test_multi_asset_variants_across_solvers(device: str, solver_cfg, static: bool) -> None:
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = _VariantSceneCfg(num_envs=2, env_spacing=2.0)
        scene_cfg.object.spawn.static = static
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        if isinstance(solver_cfg, MJWarpSolverCfg) and solver_cfg.use_mujoco_contacts:
            with pytest.raises(NotImplementedError, match="supports body variants only with Newton contacts"):
                sim.reset()
            return
        sim.reset()
        asset = scene["object"]
        assert asset.num_mesh_variants == (1 if static else 2)
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([1.0, 8.0]))
        if static:
            with pytest.raises(NotImplementedError, match="Reset-time mesh variant selection is unavailable"):
                _ = asset.mesh_variant_ids
            with pytest.raises(NotImplementedError, match="Reset-time mesh variant selection is unavailable"):
                asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=device), (0,))
        else:
            torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([0, 1], dtype=torch.int32))
            asset.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=device), (0,))
            torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 8.0]))
            sim.step()
