# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent numerical coverage of episode boundaries in the local PPO runner."""

import numpy as np
import torch

from scripts.reinforcement_learning.gr00t.model_client import _advantages


def test_timeout_bootstraps_terminal_state_without_crossing_reset():
    """A timeout bootstraps its final state; failure and both reset boundaries stop GAE."""
    critic = torch.nn.Linear(8, 1, bias=False).to("cuda")
    with torch.no_grad():
        critic.weight.zero_()
        critic.weight[0, 0] = 1
    samples = []
    for value, reward, terminated, truncated in [(2, 1, False, False), (3, 4, False, True), (10, 7, True, False)]:
        state = torch.zeros(1, 8)
        state[0, 0] = value
        samples.append({"state": state, "reward": reward, "terminated": terminated, "truncated": truncated})
    final_state = np.zeros((1, 8), dtype=np.float32)
    final_state[0, 0] = 5
    samples[1]["final_state"] = final_state
    advantages, returns = _advantages(samples, critic, torch.full((1, 8), 100.0))
    # Independent two-step return: 1 + .99*3 + (.99*.95)*(4 + .99*5 - 3).
    torch.testing.assert_close(returns.cpu(), torch.tensor([9.565975, 8.95, 7.0]))
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(advantages.mean(), torch.zeros((), device="cuda"), atol=1e-6, rtol=0)
