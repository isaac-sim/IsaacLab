# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid AMP observations keep the extras that the step set before them."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv, compute_obs

pytestmark = pytest.mark.unit

NUM_ENVS, NUM_JOINTS, NUM_KEY_BODIES, NUM_AMP_OBSERVATIONS = 2, 3, 2, 2


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    """Robot data fields expose their tensor through ``.torch``."""
    return SimpleNamespace(torch=tensor)


def test_amp_observations_keep_terminal_observations_and_other_extras() -> None:
    """Computing observations adds ``amp_obs`` without dropping ``final_obs`` or the time-outs."""
    env = object.__new__(HumanoidAmpEnv)
    num_bodies = 1 + NUM_KEY_BODIES
    quat = torch.zeros(NUM_ENVS, num_bodies, 4)
    quat[..., 0] = 1.0
    env.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=_proxy(torch.rand(NUM_ENVS, NUM_JOINTS)),
            joint_vel=_proxy(torch.rand(NUM_ENVS, NUM_JOINTS)),
            body_pos_w=_proxy(torch.rand(NUM_ENVS, num_bodies, 3)),
            body_quat_w=_proxy(quat),
            body_lin_vel_w=_proxy(torch.rand(NUM_ENVS, num_bodies, 3)),
            body_ang_vel_w=_proxy(torch.rand(NUM_ENVS, num_bodies, 3)),
        )
    )
    env.ref_body_index = 0
    env.key_body_indexes = list(range(1, num_bodies))
    # size the AMP history from one observation of this robot
    obs_size = compute_obs(
        env.robot.data.joint_pos.torch,
        env.robot.data.joint_vel.torch,
        env.robot.data.body_pos_w.torch[:, 0],
        env.robot.data.body_quat_w.torch[:, 0],
        env.robot.data.body_lin_vel_w.torch[:, 0],
        env.robot.data.body_ang_vel_w.torch[:, 0],
        env.robot.data.body_pos_w.torch[:, 1:],
    ).shape[1]
    env.cfg = SimpleNamespace(num_amp_observations=NUM_AMP_OBSERVATIONS)
    env.amp_observation_size = NUM_AMP_OBSERVATIONS * obs_size
    env.amp_observation_buffer = torch.zeros(NUM_ENVS, NUM_AMP_OBSERVATIONS, obs_size)
    final_obs = {"policy": torch.full((NUM_ENVS, obs_size), 7.0)}
    time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
    env.extras = {"final_obs": final_obs, "time_outs": time_outs}

    obs = env._get_observations()

    assert env.extras["final_obs"] is final_obs
    assert env.extras["time_outs"] is time_outs
    assert env.extras["amp_obs"].shape == (NUM_ENVS, env.amp_observation_size)
    torch.testing.assert_close(env.extras["amp_obs"][:, :obs_size], obs["policy"])
