# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Action assembly at the generation loop's environment-step boundary."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_mimic.datagen import generation

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("device", test_devices(DeviceScope.ALL))
def test_action_batch_device_and_environment_order(device, monkeypatch):
    """Deliver out-of-order queued actions to their owning environments on the simulation device."""
    monkeypatch.setattr(generation, "num_success", 0)
    monkeypatch.setattr(generation, "num_failures", 0)
    monkeypatch.setattr(generation, "num_attempts", 0)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
    action_queue = asyncio.Queue()
    for env_id in (2, 0, 1):
        action_queue.put_nowait((env_id, expected[env_id]))
    received = []

    def step(actions):
        received.append(actions)

    env = SimpleNamespace(
        num_envs=3,
        device=device,
        action_space=SimpleNamespace(shape=(3, 2)),
        step=step,
        sim=SimpleNamespace(is_stopped=lambda: True),
    )
    previous_loop = asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        generation.env_loop(env, asyncio.Queue(), action_queue, None, loop)
        assert len(received) == 1
        assert received[0].device == expected.device
        torch.testing.assert_close(received[0], expected, rtol=0, atol=0)
        assert action_queue.empty()
        loop.run_until_complete(asyncio.wait_for(action_queue.join(), timeout=1.0))
    finally:
        loop.close()
        asyncio.set_event_loop(previous_loop)
