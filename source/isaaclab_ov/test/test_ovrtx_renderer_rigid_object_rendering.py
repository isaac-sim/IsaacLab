# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX adapter for the shared rigid-object rendering contract."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from isaaclab.sim import SimulationCfg, build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_scale_and_pose_rendering_contract,
)

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "ovphysx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_OVSTAGE_AVAILABLE = importlib.util.find_spec("ovstage") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
else:
    OVRTXRendererCfg = None
    OvPhysxCfg = None


def test_kinematic_rigid_object_scale_and_pose_are_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """One opted-in OVStage clone supplies physics and rendered poses, scale and calibration."""
    import ovphysx
    import ovrtx
    import ovstage

    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    monkeypatch.delenv("ISAAC_LAB_OVPHYSX_USE_OVSTAGE", raising=False)
    monkeypatch.setenv("ISAAC_LAB_SHARE_OVSTAGE", "1")
    copies, attached = [], []
    clone = ovstage.Stage.clone
    attach_physics = ovphysx.PhysX.attach_ovstage
    attach_renderer = ovrtx.Renderer.attach_ovstage

    def record_clone(stage, source, targets, **kwargs):
        copies.extend((source, target) for target in targets)
        return clone(stage, source, targets, **kwargs)

    def attach_physx(physics, stage, **kwargs):
        attached.append(stage)
        return attach_physics(physics, stage, **kwargs)

    def attach_rtx(renderer, stage):
        assert attached == [stage]
        return attach_renderer(renderer, stage)

    def reject_native_clone(*args, **kwargs):
        pytest.fail("OVPhysX + OVRTX must clone their shared OVStage only")

    monkeypatch.setattr(ovstage.Stage, "clone", record_clone)
    monkeypatch.setattr(ovphysx.PhysX, "clone", reject_native_clone)
    monkeypatch.setattr(ovrtx.Renderer, "clone_usd", reject_native_clone)
    monkeypatch.setattr(ovphysx.PhysX, "attach_ovstage", attach_physx)
    monkeypatch.setattr(ovrtx.Renderer, "attach_ovstage", attach_rtx)
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0), physics=OvPhysxCfg())

    @contextmanager
    def simulation():
        with build_simulation_context(sim_cfg=sim_cfg) as sim:
            yield sim
            from isaaclab_ov.cloner import OvPhysxReplicateContext, OvrtxReplicateContext, OvstageReplicateContext

            assert OvstageReplicateContext in sim.clone_contexts
            assert not {OvPhysxReplicateContext, OvrtxReplicateContext}.intersection(sim.clone_contexts)

    run_rigid_object_scale_and_pose_rendering_contract(
        RigidObjectRenderingBackend(
            name="OVPhysX + OVRTX shared stage",
            simulation_context_factory=simulation,
            renderer_cfg=OVRTXRendererCfg(),
            with_articulation=True,
        )
    )
    assert copies and len(copies) == len(set(copies))


def test_shared_stage_isolates_contacts_and_interleaves_rendering_with_gravity(monkeypatch):
    """Overlapping worlds retain distinct contacts and images across shared-stage control writes."""
    import ovrtx
    import torch

    from pxr import UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass, replace

    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.setenv("ISAAC_LAB_OVPHYSX_USE_OVSTAGE", "1")
    monkeypatch.delenv("ISAAC_LAB_SHARE_OVSTAGE", raising=False)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        body = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Body",
            spawn=sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.5)),
        )
        floor = replace(
            body,
            prim_path="{ENV_REGEX_NS}/Floor",
            spawn=replace(body.spawn, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            height=32,
            width=32,
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 6.0), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
            renderer_cfg=OVRTXRendererCfg(),
        )
        other_camera = replace(
            camera,
            prim_path="{ENV_REGEX_NS}/OtherCamera",
            renderer_cfg=OVRTXRendererCfg(enable_shadows=True),
        )

    submissions = []
    native_step = ovrtx.Renderer.step

    def record_products(renderer, **kwargs):
        submissions.append(set(kwargs["render_products"]))
        return native_step(renderer, **kwargs)

    monkeypatch.setattr(ovrtx.Renderer, "step", record_products)
    cfg = SimulationCfg(device="cuda:0", dt=1 / 120, gravity=(0.0, 0.0, -10.0), physics=OvPhysxCfg())
    with build_simulation_context(sim_cfg=cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(SceneCfg(num_envs=2, env_spacing=0.0, filter_collisions=True))
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()
            camera, other = scene["camera"], scene["other_camera"]
            assert camera._renderer.backend is other._renderer.backend
            assert sim.physics_manager.backend.stage is camera._renderer.scene.stage
            floor_pose = scene["floor"].data.root_link_pose_w.torch.clone()
            floor_pose[1, 2] += 1.0
            scene["floor"].write_root_pose_to_sim_index(root_pose=floor_pose)
            for step in range(180):
                if step in (30, 60):
                    sim.physics_manager.set_gravity((0.0, 0.0, -5.0 if step == 30 else -10.0))
                sim.step(render=False)
                scene.update(cfg.dt)
                if step % 30 == 0:
                    sim.render()
                    assert torch.isfinite(camera.data.output["depth"].torch[:, 16, 16]).all()
            expected = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 2.5]], device="cuda:0")
            torch.testing.assert_close(scene["body"].data.root_link_pos_w.torch, expected, atol=0.02, rtol=0.0)
            products = {sensor._render_data.render_product_path for sensor in (camera, other)}
            assert submissions and all(submitted == products for submitted in submissions)
            for sensor in (camera, other):
                depth = sensor.data.output["depth"].torch[:, 16, 16, 0]
                torch.testing.assert_close(depth, 6.0 - expected[:, 2] - 0.5, atol=0.02, rtol=0.0)
            for soft in (True, False):
                if not soft:
                    from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

                    UsdPhysics.MassAPI(sim.stage.GetPrimAtPath("/World/envs/env_0/Body")).GetMassAttr().Set(2.0)
                    marker = UsdGeom.Cube.Define(sim.stage, "/World/ReloadMarker")
                    marker.GetSizeAttr().Set(1.0)
                    marker.AddTranslateOp().Set((0.0, 0.0, 5.0))
                    OvPhysxManager._warmup_done = False
                sim.reset(soft=soft)
                scene.reset()
                sim.physics_manager.set_gravity((0.0, 0.0, 0.0))
                sim.render()
                scene.update(cfg.dt)
                torch.testing.assert_close(camera.data.output["depth"].torch, other.data.output["depth"].torch)
                if not soft:
                    torch.testing.assert_close(
                        scene["body"].data.body_mass.torch, torch.full((2, 1), 2.0, device="cuda:0")
                    )
                    for sensor in (camera, other):
                        torch.testing.assert_close(
                            sensor.data.output["depth"].torch[:, 16, 16, 0],
                            torch.full((2,), 0.5, device="cuda:0"),
                            atol=0.02,
                            rtol=0.0,
                        )
        finally:
            sim.register_interactive_scene(None)
            scene["camera"]._invalidate_initialize_callback(None)
            scene["other_camera"]._invalidate_initialize_callback(None)
