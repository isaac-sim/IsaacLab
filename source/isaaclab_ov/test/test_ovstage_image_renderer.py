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
SIZE = H * W * 3


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
    # A second camera reads its own prims from the same stage.
    side_camera = camera.replace(prim_path="{ENV_REGEX_NS}/SideCamera")


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

        def produce(images):
            """Write uniform images, ``{(camera, env): (value, size)}``, and seal them; then step and read the
            cameras' RGB. The side camera's images come from the host, so on CUDA its read copies them to the device."""
            for (camera, env), (value, size) in images.items():
                with stage.query([f"/World/envs/env_{env}/{camera}"]) as query:
                    source = device if camera == "Camera" else "cpu"
                    image = torch.full((size,), value, dtype=torch.uint8, device=source)
                    stage.stage.write_attribute(query, attribute, stage.ordinal, [image], is_array=True).wait()
            stage.commit()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            return scene["camera"].data.output["rgb"].torch, scene["side_camera"].data.output["rgb"].torch

        # An environment without an image keeps its previous one; the stage returns only the image it has.
        rgb, side = produce({("Camera", 1): (11, SIZE)})
        assert rgb[:, 0, 0, 0].tolist() == [0, 11]
        assert (side == 0).all()

        rgb, side = produce({("Camera", 0): (10, SIZE), ("Camera", 1): (12, SIZE), ("SideCamera", 0): (20, SIZE)})
        assert rgb.tolist() == [[[[10] * 3] * W] * H, [[[12] * 3] * W] * H]
        assert side[:, 0, 0, 0].tolist() == [20, 0]
        assert (scene["camera"].data.output["rgba"].torch[..., 3] == 255).all()

        # A reset of one environment keeps its image until the producer writes the new episode's first one.
        scene.reset(env_ids=[0])
        rgb, _ = produce({})
        assert rgb[:, 0, 0, 0].tolist() == [10, 12]
        rgb, _ = produce({("Camera", 0): (30, SIZE)})
        assert rgb[:, 0, 0, 0].tolist() == [30, 12]

        # A wrong-sized image fails the read before any image is copied.
        with pytest.raises(ValueError, match="expects images of"):
            produce({("Camera", 0): (40, SIZE), ("Camera", 1): (41, H * W)})
        assert rgb[:, 0, 0, 0].tolist() == [30, 12]


def test_a_camera_needs_the_simulation_stage():
    renderer = OvstageImageRenderer(OvstageImageRendererCfg(), sim=SimpleNamespace(get_backends=lambda cfg: ()))
    with pytest.raises(RuntimeError, match="nothing has created it"):
        renderer.create_render_data(SimpleNamespace(cfg=SceneCfg().camera, num_instances=N, camera_prim_paths=()))
