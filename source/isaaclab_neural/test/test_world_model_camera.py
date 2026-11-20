# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A camera fed by the world-model backend."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.test.utils import test_devices
from isaaclab.utils import configclass

from isaaclab_neural.renderers import WorldModelRenderer, WorldModelRendererCfg

from .conftest import DT, H, J, N, T, W, world_model_sim


@configclass
class SceneCfg(InteractiveSceneCfg):
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=H,
        width=W,
        data_types=["rgb"],
        frame_history_length=T,
        spawn=PinholeCameraCfg(focal_length=13.6, clipping_range=(0.001, 20.0)),
        renderer_cfg=WorldModelRendererCfg(),
    )


@pytest.fixture
def world(tmp_path):
    with world_model_sim(tmp_path, envs_per_reset_record=1, split="validation") as sim:
        scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=1.0))
        sim.backend_manager.bind(N, J)
        sim.reset()
        sim.backend_manager.reset_envs(torch.ones(N, dtype=torch.bool, device=sim.device))
        yield sim, scene


def advance(sim, scene, steps):
    """Step like an environment does: the observation reads the camera after every step."""
    for _ in range(steps):
        sim.backend_manager.joint_target.zero_()
        sim.step(render=False)
        scene.update(DT)
        scene["camera"].data


def test_the_start_frame_reaches_the_camera(world):
    sim, scene = world
    scene.update(DT)
    data = scene["camera"].data
    assert data.output["rgb"].torch[:, 0, 0, 0].tolist() == [200, 201]
    assert (data.frame_history.torch[1] == 201).all()


def test_a_chunk_fills_the_history_and_rgb_is_its_last_frame(world):
    sim, scene = world
    advance(sim, scene, T)
    data = scene["camera"].data
    assert data.frame_history.torch[:, :, 0, 0, 0].tolist() == [[10, 11, 12, 13]] * N
    assert torch.equal(data.output["rgb"].torch, data.frame_history.torch[:, -1])
    assert data.frame_history_mask.torch.all()


def test_the_mask_is_clear_between_chunks(world):
    sim, scene = world
    advance(sim, scene, T + 1)
    data = scene["camera"].data
    assert not data.frame_history_mask.torch.any()
    assert data.frame_history.torch[0, -1, 0, 0, 0] == 13


@pytest.mark.parametrize("device", test_devices())
def test_three_cameras_read_their_views_and_partial_resets(tmp_path, device):
    camera = SceneCfg().camera

    @configclass
    class ThreeViewSceneCfg(InteractiveSceneCfg):
        front = camera.replace(prim_path="{ENV_REGEX_NS}/Front", renderer_cfg=WorldModelRendererCfg(view="front"))
        left = camera.replace(prim_path="{ENV_REGEX_NS}/Left", renderer_cfg=WorldModelRendererCfg(view="left"))
        right = camera.replace(prim_path="{ENV_REGEX_NS}/Right", renderer_cfg=WorldModelRendererCfg(view="right"))

    views = ("front", "left", "right")
    with world_model_sim(tmp_path, envs_per_reset_record=1, split="validation", device=device, views=views) as sim:
        scene = InteractiveScene(ThreeViewSceneCfg(num_envs=N, env_spacing=1.0))
        manager = sim.backend_manager
        manager.bind(N, J)
        sim.reset()
        manager.reset_envs(torch.ones(N, dtype=torch.bool, device=sim.device))
        scene.update(DT)
        for v, view in enumerate(views):
            assert scene[view].data.output["rgb"].torch[:, 0, 0, 0].tolist() == [200 + 20 * v, 201 + 20 * v]
        for _ in range(T):
            sim.step(render=False)
            scene.update(DT)
        # Reading in a different order does not change which view or chunk each camera receives.
        for view, value in (("right", 213), ("front", 13), ("left", 113)):
            data = scene[view].data
            assert (data.output["rgb"].torch == value).all()
            assert data.frame_history.torch[0, :, 0, 0, 0].tolist() == list(range(value - T + 1, value + 1))
            assert data.frame_history_mask.torch.all()
        manager.reset_envs(torch.tensor([True, False], device=sim.device))
        scene.reset(env_ids=torch.tensor([0], device=device))
        scene.update(DT)
        for v, view in enumerate(views):
            data = scene[view].data
            assert data.output["rgb"].torch[:, 0, 0, 0].tolist() == [200 + 20 * v, 13 + 100 * v]
            assert data.frame_history_mask.torch.tolist() == [True, False]


def test_camera_binding_rejects_the_wrong_backend_and_history_size(tmp_path):
    renderer = WorldModelRenderer(WorldModelRendererCfg(), sim=SimpleNamespace(backend_manager=object(), device="cpu"))
    with pytest.raises(ValueError, match="world-model physics backend"):
        renderer.create_render_data(SimpleNamespace(cfg=SceneCfg().camera, num_instances=N))
    with world_model_sim(tmp_path) as sim:
        scene = InteractiveScene(
            SceneCfg(num_envs=N, env_spacing=1.0).replace(camera=SceneCfg().camera.replace(frame_history_length=0))
        )
        sim.backend_manager.bind(N, J)
        with pytest.raises(ValueError, match="frame_history_length"):
            sim.reset()
        assert not scene["camera"].is_initialized
