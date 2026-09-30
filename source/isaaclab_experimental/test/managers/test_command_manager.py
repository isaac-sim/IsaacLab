# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp command manager."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental.envs.mdp.commands import UniformVelocityCommand
from isaaclab_experimental.managers import CommandManager
from isaaclab_experimental.utils import WarpGraphCache
from isaaclab_experimental.utils.warp import resolve_1d_mask

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"
NUM_ENVS = 4
DT = 0.02


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=UniformVelocityCommand,
        asset_name="robot",
        # shorter than a step, so every step resamples
        resampling_time_range=(DT / 2, DT / 2),
        heading_command=False,
        rel_standing_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )


def _override(env, env_ids, data, value):
    return value


def _env(graph_cache: WarpGraphCache) -> SimpleNamespace:
    """The environment surface the command manager and the velocity command read."""
    pose = np.zeros((NUM_ENVS, 7), dtype=np.float32)
    pose[:, 6] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pose_w=SimpleNamespace(warp=wp.array(pose, dtype=wp.transformf, device=DEVICE)),
            root_com_vel_w=SimpleNamespace(warp=wp.zeros(NUM_ENVS, dtype=wp.spatial_vectorf, device=DEVICE)),
        )
    )
    all_mask = wp.ones(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    scratch_mask = wp.zeros(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    return SimpleNamespace(
        num_envs=NUM_ENVS,
        device=DEVICE,
        scene={"robot": robot},
        rng_state_wp=wp.array(np.arange(NUM_ENVS, dtype=np.uint32), device=DEVICE),
        resolve_env_mask=lambda env_ids=None, env_mask=None: resolve_1d_mask(
            ids=env_ids, mask=env_mask, all_mask=all_mask, scratch_mask=scratch_mask, device=DEVICE
        ),
        sim=SimpleNamespace(
            is_playing=lambda: True,
            vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None),
        ),
        _warp_graph_cache=graph_cache,
    )


def test_modify_term_cfg_applies_to_the_recorded_command_stage():
    """A curriculum that changes a command range after the command stage recorded changes the next commands.

    The range is a kernel argument read while the stage records, so a replay without recording again keeps
    sampling from the old range.
    """
    graph_cache = WarpGraphCache(DEVICE)
    env = _env(graph_cache)
    env.command_manager = CommandManager(CommandsCfg(), env)
    params = {
        "address": "commands.base_velocity.ranges.lin_vel_x",
        "modify_fn": _override,
        "modify_params": {"value": (1.5, 1.5)},
    }
    curriculum = mdp.modify_term_cfg(CurriculumTermCfg(func=mdp.modify_term_cfg, params=params), env)
    graph_cache.arm()
    graph_cache.call_steps("CommandManager_compute", env.command_manager.stage_steps("compute"), dt=DT)

    curriculum(env, None, **params)
    graph_cache.call_steps("CommandManager_compute", env.command_manager.stage_steps("compute"), dt=DT)
    wp.synchronize()

    command = env.command_manager.get_command("base_velocity")
    assert torch.equal(command[:, 0], torch.full((NUM_ENVS,), 1.5, device=DEVICE))
    graph_cache.close()
