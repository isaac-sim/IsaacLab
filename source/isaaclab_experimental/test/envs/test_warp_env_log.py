# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logged values of the Warp environments keep the value of the step that returned them.

Warp managers and tasks log persistent buffers that later steps overwrite. RL libraries keep the
``extras["log"]`` of every step and average them at the end of an iteration, so the environments must return
copies, like the fresh tensors of the stable environments.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch
from isaaclab_experimental.envs import ManagerBasedEnvWarp
from isaaclab_experimental.envs.frontend import WarpFrontend

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 16
_STEPS = 120


def _make_env(task_id: str):
    env_cfg, _ = resolve_task_config(task_id, "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    sim_utils.create_new_stage()
    return WarpFrontend.build_env(env_cfg, task_id).unwrapped


def test_reset_logs_keep_their_values():
    env = _make_env("Isaac-Cartpole")
    logged = []
    try:
        env.reset()
        # different pushes drive the carts out of bounds at different steps, so episode sums differ
        push = torch.linspace(-0.3, 0.3, _NUM_ENVS, device=env.device)[:, None]
        for _ in range(_STEPS):
            _, _, terminated, truncated, extras = env.step(push)
            if (terminated | truncated).any():
                log = extras["log"]
                logged.append((log, {key: value.clone() for key, value in log.items() if torch.is_tensor(value)}))
    finally:
        env.close()

    first_values = logged[0][1]
    assert any(not torch.equal(values[key], first_values[key]) for _, values in logged[1:] for key in first_values)
    for log, values in logged:
        for key, value in values.items():
            assert torch.equal(log[key], value), key


def test_base_env_reset_logs_keep_their_values(monkeypatch: pytest.MonkeyPatch):
    env_cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    WarpFrontend.adapt_cfg(env_cfg)
    sim_utils.create_new_stage()
    env = ManagerBasedEnvWarp(cfg=env_cfg)
    logged = []
    try:
        # a manager logs a persistent buffer that it refreshes on every reset
        metric = torch.zeros((), device=env.device)
        monkeypatch.setattr(env.recorder_manager, "reset", lambda env_ids=None: {"Metrics/value": metric})
        for value in range(3):
            metric.fill_(value)
            _, extras = env.reset()
            logged.append(extras["log"])
    finally:
        env.close()

    assert [log["Metrics/value"].item() for log in logged] == [0.0, 1.0, 2.0]


def test_direct_logs_keep_their_values():
    env = _make_env("Isaac-Cartpole-Direct")
    logged = []
    try:
        env.reset()
        # a task logs a persistent buffer that its kernels refresh
        metric = torch.zeros((), device=env.device)
        env.extras["log"] = {"Metrics/value": metric}
        for step in range(5):
            metric.fill_(step)
            _, _, _, _, extras = env.step(torch.zeros((_NUM_ENVS, 1), device=env.device))
            logged.append(extras["log"])
    finally:
        env.close()

    assert [log["Metrics/value"].item() for log in logged] == [0.0, 1.0, 2.0, 3.0, 4.0]
