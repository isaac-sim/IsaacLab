# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import instantiate
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.unit

NUM_ENVS, NUM_JOINTS = 4, 2


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_masked_reset_matches_index_reset(device):
    """A boolean-mask reset delays the same environments as the equivalent index reset, without synchronizing CUDA."""
    joint_names = [f"joint_{j}" for j in range(NUM_JOINTS)]
    cfg = DelayedPDActuatorCfg(joint_names_expr=joint_names, stiffness=10.0, damping=1.0, min_delay=2, max_delay=2)
    by_ids, by_mask = (
        instantiate(cfg, joint_names=joint_names, joint_ids=slice(None), num_envs=NUM_ENVS, device=device)
        for _ in range(2)
    )
    env_ids = torch.tensor([1, 2], device=device)
    env_mask = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
    env_mask[env_ids] = True
    zeros = torch.zeros(NUM_ENVS, NUM_JOINTS, device=device)

    def step(actuator, target: float) -> torch.Tensor:
        action = ArticulationActions(joint_positions=zeros + target, joint_velocities=zeros, joint_efforts=zeros)
        actuator.compute(action, joint_pos=zeros, joint_vel=zeros)
        return actuator.computed_effort.clone()

    # A lag of one in every environment differs from the configured lag the reset samples.
    for actuator in (by_ids, by_mask):
        for delay_buffer in (
            actuator.positions_delay_buffer,
            actuator.velocities_delay_buffer,
            actuator.efforts_delay_buffer,
        ):
            delay_buffer.set_time_lag(1)
        for target in range(3):
            step(actuator, float(target))

    by_ids.reset(env_ids)
    torch.cuda.synchronize(device)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        by_mask.reset(env_mask=env_mask)
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    for target in range(3, 7):
        torch.testing.assert_close(step(by_mask, float(target)), step(by_ids, float(target)))
