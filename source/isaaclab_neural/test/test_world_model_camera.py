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
from isaaclab.utils import configclass

from isaaclab_neural.renderers import NeuralRenderer, NeuralRendererCfg

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
        renderer_cfg=NeuralRendererCfg(),
    )


@pytest.fixture
def world(tmp_path):
    with world_model_sim(tmp_path, envs_per_reset_record=1, split="validation") as sim:
        scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=1.0))
        sim.backend_manager.bind(N, J)
        sim.reset()
        sim.backend_manager.reset_envs(torch.ones(N, dtype=torch.bool))
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
    assert (data.output["rgba"].torch[..., 3] == 255).all()
    assert data.frame_history_mask.torch.all()


def test_a_camera_is_bound_to_the_world_model_backend_and_must_match_the_model(tmp_path):
    # Another physics backend has no world-model frames to publish.
    renderer = NeuralRenderer(NeuralRendererCfg(), sim=SimpleNamespace(backend_manager=object(), device="cpu"))
    with pytest.raises(ValueError, match="neural physics backend"):
        renderer.create_render_data(SimpleNamespace(cfg=SceneCfg().camera, num_instances=N))
    # The camera is checked against the model when it binds, before any step, and must name an existing view.
    with world_model_sim(tmp_path) as sim:
        scene = InteractiveScene(  # kept alive until the camera initializes
            SceneCfg(num_envs=N, env_spacing=1.0).replace(camera=SceneCfg().camera.replace(frame_history_length=0))
        )
        sim.backend_manager.bind(N, J)
        with pytest.raises(ValueError, match="frame_history_length=4"):
            sim.reset()
        assert not scene["camera"].is_initialized


def test_each_camera_shows_the_view_it_selects(tmp_path):
    camera = SceneCfg().camera

    @configclass
    class TwoViewSceneCfg(InteractiveSceneCfg):
        left = camera.replace(prim_path="{ENV_REGEX_NS}/Left", renderer_cfg=NeuralRendererCfg(view="left"))
        right = camera.replace(prim_path="{ENV_REGEX_NS}/Right", renderer_cfg=NeuralRendererCfg(view="right"))

    with world_model_sim(tmp_path, envs_per_reset_record=1, split="validation", views=("left", "right")) as sim:
        scene = InteractiveScene(TwoViewSceneCfg(num_envs=N, env_spacing=1.0))
        sim.backend_manager.bind(N, J)
        sim.reset()
        sim.backend_manager.reset_envs(torch.ones(N, dtype=torch.bool))
        scene.update(DT)
        # Each view starts from its own start image.
        assert scene["left"].data.output["rgb"].torch[:, 0, 0, 0].tolist() == [200, 201]
        assert scene["right"].data.output["rgb"].torch[:, 0, 0, 0].tolist() == [220, 221]
        for _ in range(T):
            sim.step(render=False)
            scene.update(DT)
        assert (scene["left"].data.output["rgb"].torch == 13).all()
        assert (scene["right"].data.output["rgb"].torch == 113).all()


def test_the_mask_is_clear_between_chunks(world):
    sim, scene = world
    advance(sim, scene, T + 1)
    data = scene["camera"].data
    assert not data.frame_history_mask.torch.any()
    assert data.frame_history.torch[0, -1, 0, 0, 0] == 13
