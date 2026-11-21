# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cameras showing images that another component writes into the simulation's OVStage."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_ov.renderers import OvstageImageRenderer, OvstageImageRendererCfg
from isaaclab_ov.stage import OvstageBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

pytestmark = [pytest.mark.kitless, pytest.mark.ov]

N, H, W = 2, 6, 8
CAMERAS = [f"/World/envs/env_{env}/Camera" for env in range(N)]


@configclass
class SceneCfg(InteractiveSceneCfg):
    # Newton clones the environments' worlds from their bodies, and the camera views need those worlds.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=H,
        width=W,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(),
        renderer_cfg=OvstageImageRendererCfg(),
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_the_camera_shows_the_latest_image_of_each_environment(device):
    from isaaclab_newton.physics import NewtonCfg

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 30, device=device, gravity=(0.0, 0.0, 0.0), physics=NewtonCfg())
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        # The producer creates the simulation stage before the cameras initialize.
        stage = sim.get_or_create_backend(OvstageBackendCfg())
        scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=1.0))
        sim.reset()
        attribute = stage.paths.intern_token("sensor:rgb")

        def produce(envs, values, size=H * W * 3):
            """Write one uniform image per listed environment and seal it, then step and read the camera."""
            with stage.query([CAMERAS[env] for env in envs]) as query:
                images = [torch.full((size,), value, dtype=torch.uint8, device=device) for value in values]
                stage.stage.write_attribute(query, attribute, stage.ordinal, images, is_array=True).wait()
            stage.commit()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            output = scene["camera"].data.output
            return output["rgb"].torch, output["rgba"].torch

        # An environment without an image keeps its previous one; the stage returns only the image it has.
        rgb, _ = produce([1], [11])
        assert rgb[:, 0, 0, 0].tolist() == [0, 11]

        rgb, rgba = produce([0, 1], [10, 12])
        assert rgb.tolist() == [[[[10] * 3] * W] * H, [[[12] * 3] * W] * H]
        assert (rgba[..., 3] == 255).all()

        with pytest.raises(ValueError, match="expects images of"):
            produce([0], [0], size=H * W)


def test_a_camera_needs_the_simulation_stage():
    renderer = OvstageImageRenderer(OvstageImageRendererCfg(), sim=SimpleNamespace(get_backends=lambda cfg: ()))
    with pytest.raises(RuntimeError, match="nothing has created it"):
        renderer.create_render_data(SimpleNamespace(cfg=SceneCfg().camera, num_instances=N, camera_prim_paths=()))
