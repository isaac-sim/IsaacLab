# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX renders the alternative that Newton selects for a reset-time mesh variant."""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_newton", "isaaclab_ov", "ovrtx", "ovstage")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.ov,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.sim import SimulationCfg
    from isaaclab.test.utils import launch_test_simulation

    launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

    import torch
    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import build_simulation_context
    from isaaclab.utils import configclass


def _silhouette_pixels(sim, scene, camera):
    """Settle the renderer after a change and return the object's silhouette size per environment."""
    for _ in range(3):
        sim.step()
        scene.update(sim.cfg.dt)
    depth = camera.data.output["depth"].torch[..., 0]
    return (torch.isfinite(depth) & (depth > 0.05) & (depth < 10.0)).sum(dim=(1, 2)).cpu()


@pytest.mark.parametrize("use_ovstage", ["0", "1"], ids=["native", "ovstage"])
def test_switched_mesh_variants_are_rendered(monkeypatch: pytest.MonkeyPatch, use_ovstage: str) -> None:
    """Each world shows the alternative Newton selects, before and after reset-time switches."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", use_ovstage)
    physics = dict(
        mass_props=sim_utils.MassCfg(mass=1.0),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        part = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Part",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                static=False,
                assets_cfg=[
                    sim_utils.MeshConeCfg(radius=0.02, height=0.08, **physics),
                    sim_utils.MeshCuboidCfg(size=(0.3, 0.3, 0.3), **physics),
                ],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            height=48,
            width=64,
            data_types=["depth"],
            renderer_cfg=OVRTXRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
            ),
        )

    sim_cfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(SceneCfg(num_envs=2, env_spacing=2.0, lazy_sensor_update=False))
        sim.register_interactive_scene(scene)
        sim.reset()
        scene.reset()
        part, camera = scene["part"], scene["camera"]
        assert camera._renderer._use_ovstage is (use_ovstage == "1")
        eyes = scene.env_origins + torch.tensor([1.0, 0.0, 0.2], device=sim.device)
        camera.set_world_poses_from_view(eyes, scene.env_origins + torch.tensor([0.0, 0.0, 0.2], device=sim.device))
        torch.testing.assert_close(part.mesh_variant_ids.torch.cpu(), torch.tensor([0, 1], dtype=torch.int32))

        cone, cube = _silhouette_pixels(sim, scene, camera).tolist()
        assert 0 < cone < cube / 4, (cone, cube)

        part.write_mesh_variant_to_sim(torch.tensor([1], dtype=torch.int32, device=sim.device), (0,))
        switched = _silhouette_pixels(sim, scene, camera).tolist()
        assert switched[1] == cube and abs(switched[0] - cube) <= 2, (switched, cone, cube)

        # A copy of an alternative that another world parked earlier must still appear in its own world.
        part.write_mesh_variant_to_sim(torch.tensor([0], dtype=torch.int32, device=sim.device), (1,))
        swapped = _silhouette_pixels(sim, scene, camera).tolist()
        assert abs(swapped[0] - cube) <= 2 and abs(swapped[1] - cone) <= 2, (swapped, cone, cube)

        part.write_mesh_variant_to_sim(torch.tensor([0, 0], dtype=torch.int32, device=sim.device))
        reverted = _silhouette_pixels(sim, scene, camera).tolist()
        assert all(abs(pixels - cone) <= 2 for pixels in reverted), (reverted, cone, cube)
