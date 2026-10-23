# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the quaternion convention at the Humanoid AMP motion-to-env boundaries."""

import os
import types

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.humanoid_amp import humanoid_amp_env
from isaaclab_tasks.contrib.humanoid_amp.humanoid_amp_env_cfg import MOTIONS_DIR
from isaaclab_tasks.contrib.humanoid_amp.motions.motion_loader import MotionLoader

NUM_ENVS = 4
NUM_AMP_OBSERVATIONS = 2
AMP_OBSERVATION_SPACE = 81
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]


def _env_without_sim(motion_file: str) -> humanoid_amp_env.HumanoidAmpEnv:
    """Build the env around real motion data, skipping the simulator-only construction."""
    env = object.__new__(humanoid_amp_env.HumanoidAmpEnv)
    env.cfg = types.SimpleNamespace(
        num_amp_observations=NUM_AMP_OBSERVATIONS, amp_observation_space=AMP_OBSERVATION_SPACE
    )
    env.amp_observation_size = NUM_AMP_OBSERVATIONS * AMP_OBSERVATION_SPACE
    env.amp_observation_buffer = torch.zeros((NUM_ENVS, NUM_AMP_OBSERVATIONS, AMP_OBSERVATION_SPACE))
    loader = MotionLoader(motion_file=motion_file, device=torch.device("cpu"))
    env._motion_loader = loader
    # the motion skeleton's own names stand in for the robot's, so the indexes map one to one
    env.motion_dof_indexes = loader.get_dof_index(loader.dof_names)
    env.motion_ref_body_index = loader.get_body_index(["torso"])[0]
    env.motion_key_body_indexes = loader.get_body_index(KEY_BODY_NAMES)
    default_root_pose = torch.zeros((NUM_ENVS, 7))
    default_root_pose[:, 6] = 1.0
    env.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            default_root_pose=types.SimpleNamespace(torch=default_root_pose),
            default_root_vel=types.SimpleNamespace(torch=torch.zeros((NUM_ENVS, 6))),
        )
    )
    env.scene = types.SimpleNamespace(env_origins=torch.zeros((NUM_ENVS, 3)))
    return env


def test_motion_rotations_reach_xyzw_consumers_upright():
    """The loader's wxyz rotations are converted before the env's xyzw consumers read them.

    Both consumers of the sampled torso rotation assume xyzw: the root pose written back to
    the simulation on reset and ``quat_apply`` inside the reference AMP observation. Applied
    as xyzw, the walk clip's torso reads upside down (normal z about -1); converted first,
    it reads upright (normal z about +1).
    """
    np.random.seed(0)
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    env = _env_without_sim(motion_file)
    env_ids = torch.arange(NUM_ENVS)

    root_state, _, _ = env._reset_strategy_random(env_ids, start=False)
    up = torch.zeros((NUM_ENVS, 3))
    up[:, 2] = 1.0
    torso_up_z = quat_apply(root_state[:, 3:7], up)[:, 2]
    assert torch.all(torso_up_z > 0.9), f"reset root pose is not upright: torso up-vector z = {torso_up_z}"

    reference = env.collect_reference_motions(NUM_ENVS).view(NUM_ENVS, NUM_AMP_OBSERVATIONS, -1)
    # one observation: DOF pos and vel, root height, tangent, normal, linear and angular
    # velocity, key body relative positions
    normal_offset = env._motion_loader.num_dofs * 2 + 1 + 3
    torso_normal_z = reference[:, :, normal_offset + 2]
    assert torch.all(torso_normal_z > 0.9), (
        f"reference AMP observation is not upright: torso normal z = {torso_normal_z}"
    )
