# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test of :class:`stacked_image` through :class:`ObservationManager`.

Launches Kit + sim so the obs manager's construction-time shape probe and per-step compute
exercise the real lifecycle. Mocks the camera-pull function ``image`` so no scene/sensors
are required.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from collections import namedtuple
from unittest import mock

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.observations import stacked_image
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.isaacsim_ci

NUM_ENVS = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 3
DEVICE = "cuda:0"


def _fake_image(
    env, sensor_cfg=None, data_type="rgb", convert_perspective_to_orthogonal=False, normalize=True, clone=True
):
    """Stand-in for ``isaaclab.envs.mdp.observations.image`` — returns a constant frame.

    The value is keyed to the call count so consecutive calls produce distinct frames
    (used by the channel-shift assertion).
    """
    value = float(_fake_image.call_count)
    _fake_image.call_count += 1
    return torch.full((env.num_envs, HEIGHT, WIDTH, CHANNELS), value, dtype=torch.float32, device=env.device)


_fake_image.call_count = 0


@pytest.fixture
def env_with_sim():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=DEVICE)
    sim = sim_utils.SimulationContext(sim_cfg)
    env = namedtuple("Env", ["num_envs", "device", "sim"])(NUM_ENVS, DEVICE, sim)
    env.sim._app_control_on_stop_handle = None
    env.sim.reset()
    _fake_image.call_count = 0
    yield env
    sim.clear_instance()


def _make_cfg(frame_stack: int):
    @configclass
    class ObsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            img: ObservationTermCfg = ObservationTermCfg(
                func=stacked_image,
                params={"frame_stack": frame_stack},
            )

        policy: ObservationGroupCfg = PolicyCfg()

    return ObsCfg()


def test_obs_manager_infers_channel_stacked_shape(env_with_sim):
    """ObservationManager probes ``stacked_image`` at construction and infers the stacked shape."""
    with mock.patch("isaaclab.envs.mdp.observations.image", side_effect=_fake_image):
        manager = ObservationManager(_make_cfg(frame_stack=2), env_with_sim)
    assert manager.group_obs_dim["policy"] == (HEIGHT, WIDTH, CHANNELS * 2)


def test_obs_manager_compute_returns_stacked_output(env_with_sim):
    """``compute()`` after construction returns the channel-stacked obs tensor."""
    with mock.patch("isaaclab.envs.mdp.observations.image", side_effect=_fake_image):
        manager = ObservationManager(_make_cfg(frame_stack=3), env_with_sim)
        obs = manager.compute()
    assert obs["policy"].shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 3)


def test_obs_manager_reset_clears_term_state(env_with_sim):
    """``manager.reset()`` forwards to ``stacked_image.reset()``; next compute fills slots with the new frame."""
    with mock.patch("isaaclab.envs.mdp.observations.image", side_effect=_fake_image):
        manager = ObservationManager(_make_cfg(frame_stack=2), env_with_sim)
        manager.compute()
        manager.compute()  # ring fills with two distinct frames
        manager.reset()
        # Next compute should treat the frame as fresh init — both channel-slots identical.
        obs = manager.compute()
    oldest = obs["policy"][..., :CHANNELS]
    newest = obs["policy"][..., CHANNELS:]
    assert torch.equal(oldest, newest), "After reset, init path should fill all slots with the same frame."
