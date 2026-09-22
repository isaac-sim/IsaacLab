# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test of :func:`stacked_image` through :class:`ObservationManager`.

The manager's construction-time shape probe, per-step compute and reset drive the real term. The camera-pull
function ``image`` is replaced by a counter so no scene or sensors are required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import observations
from isaaclab.envs.mdp.observations import stacked_image
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit

NUM_ENVS = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 3


@pytest.fixture
def env(monkeypatch):
    """Environment double whose ``image`` term returns a distinct uint8 frame on every call."""
    frame_count = {"value": 0}

    def fake_image(
        env, sensor_cfg=None, data_type="rgb", convert_perspective_to_orthogonal=False, normalize=True, clone=True
    ):
        frame_count["value"] += 1
        # raw uint8 camera output with a spatial gradient so that normalization (which subtracts the spatial
        # mean) keeps consecutive frames distinguishable
        pattern = torch.arange(HEIGHT * WIDTH, dtype=torch.uint8, device=env.device).reshape(1, HEIGHT, WIDTH, 1) % 16
        return (pattern * frame_count["value"]).repeat(env.num_envs, 1, 1, CHANNELS)

    monkeypatch.setattr(observations, "image", fake_image)
    return SimpleNamespace(num_envs=NUM_ENVS, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))


def _make_cfg(frame_stack: int):
    @configclass
    class ObsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            img: ObservationTermCfg = ObservationTermCfg(func=stacked_image, params={"frame_stack": frame_stack})

        policy: ObservationGroupCfg = PolicyCfg()

    return ObsCfg()


@pytest.mark.parametrize("frame_stack", [2, 3])
def test_obs_manager_stacks_channels_and_resets_term_state(env, frame_stack):
    """The manager infers the channel-stacked shape, computes stacked frames and resets the term's ring buffer."""
    manager = ObservationManager(_make_cfg(frame_stack), env)
    assert manager.group_obs_dim["policy"] == (HEIGHT, WIDTH, CHANNELS * frame_stack)

    # the ring fills with distinct frames over consecutive computes
    for _ in range(frame_stack):
        obs = manager.compute()["policy"]
    assert obs.shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * frame_stack)
    assert not torch.equal(obs[..., :CHANNELS], obs[..., -CHANNELS:])

    # after a reset the next frame fills every slot
    manager.reset()
    obs = manager.compute()["policy"]
    assert torch.equal(obs[..., :CHANNELS], obs[..., -CHANNELS:])
