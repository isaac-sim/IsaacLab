# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test of a frame-stacked :class:`image_rgb` term through :class:`ObservationManager`.

Launches Kit + sim so the obs manager's construction-time shape probe and per-step compute
exercise the real lifecycle. The camera is a stand-in exposing ``data.output`` as a
:class:`ProxyArray`, so no scene or rendering is required.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.observations import image_rgb
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]

NUM_ENVS = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 3
DEVICE = "cuda:0"


@pytest.fixture
def env_with_sim():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=DEVICE)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_buf = torch.randint(0, 255, (NUM_ENVS, HEIGHT, WIDTH, CHANNELS), dtype=torch.uint8, device=DEVICE)
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": ProxyArray(wp.from_torch(camera_buf))}))
    scene = SimpleNamespace(sensors={"tiled_camera": camera})
    env = namedtuple("Env", ["num_envs", "device", "sim", "scene"])(NUM_ENVS, DEVICE, sim, scene)
    env.sim._app_control_on_stop_handle = None
    env.sim.reset()
    yield env
    sim.clear_instance()


def _make_cfg(frame_stack: int):
    @configclass
    class ObsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            img: ObservationTermCfg = ObservationTermCfg(
                func=image_rgb,
                params={"frame_stack": frame_stack},
            )

        policy: ObservationGroupCfg = PolicyCfg()

    return ObsCfg()


def test_obs_manager_compute_returns_stacked_output(env_with_sim):
    """``compute()`` after construction returns the channel-stacked obs tensor."""
    manager = ObservationManager(_make_cfg(frame_stack=3), env_with_sim)
    obs = manager.compute()
    # the manager probes the term at construction and infers the channel-stacked shape
    assert manager.group_obs_dim["policy"] == (HEIGHT, WIDTH, CHANNELS * 3)
    assert obs["policy"].shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 3)
