# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A camera fed by the world-model backend."""

import numpy as np
import pytest
import torch

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import configclass

from isaaclab_neural.renderers import WorldModelRendererCfg

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
        sim.backend_manager.reset_envs(np.arange(N))
        yield sim, scene


def advance(sim, scene, steps):
    """Step like an environment does: the observation reads the camera after every step."""
    for _ in range(steps):
        sim.backend_manager.set_joint_target(np.zeros((N, J), np.float32))
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
