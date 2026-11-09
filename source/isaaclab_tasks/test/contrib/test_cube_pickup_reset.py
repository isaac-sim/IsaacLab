# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for pickup rewards, completed episodes, and reset-bank persistence."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.reset import ResetStateBuffer, ResetStateBufferCfg
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ObservationManager,
    RecorderManager,
    RewardManager,
    TerminationManager,
    TerminationTermCfg,
)
from isaaclab.utils import update_from_dict

from isaaclab_tasks.contrib.cube_pickup.cube_pickup_env import CubePickupEnv
from isaaclab_tasks.contrib.cube_pickup.cube_pickup_env_cfg import CubePickupEnvCfg
from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def _success(env) -> torch.Tensor:
    return env.success_flags


def test_terminal_reward_stays_binary_after_timing_overrides() -> None:
    """Post-construction CLI overrides must not rescale the terminal +1 reward."""
    cfg = CubePickupEnvCfg()
    update_from_dict(cfg, {"decimation": 2, "sim": {"dt": 1.0 / 240.0}})
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=cfg.sim.dt * cfg.decimation,
        sim=SimpleNamespace(is_playing=lambda: True),
        success_flags=torch.tensor([True, False]),
    )
    env.termination_manager = TerminationManager({"success": TerminationTermCfg(func=_success)}, env)
    env.termination_manager.compute()
    rewards = RewardManager(cfg.rewards, env).compute(dt=env.step_dt)
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0]))


def test_viewer_reset_does_not_record_an_unfinished_episode() -> None:
    """A viewer termination alias must not credit a manual reset as a completed failure."""
    env = CubePickupEnv.__new__(CubePickupEnv)
    env._is_closed = True
    env.cfg = CubePickupEnvCfg()
    env.sim = SimpleNamespace(device="cpu", is_playing=lambda: True)
    env.scene = SimpleNamespace(num_envs=2, _ALL_INDICES=torch.arange(2), reset=lambda env_ids: None)
    # Only physics is omitted; the real task and manager reset bookkeeping still run.
    env._restore_reset_state = lambda state, env_ids: None
    env.extras = {}
    env.episode_length_buf = torch.tensor([5, 5])
    env.success_flags = torch.tensor([True, False])
    env.termination_manager = TerminationManager({"success": TerminationTermCfg(func=_success)}, env)
    for name, manager in (
        ("action_manager", ActionManager),
        ("command_manager", CommandManager),
        ("curriculum_manager", CurriculumManager),
        ("event_manager", EventManager),
        ("observation_manager", ObservationManager),
        ("recorder_manager", RecorderManager),
        ("reward_manager", RewardManager),
    ):
        setattr(env, name, manager(SimpleNamespace(), env))
    env.reset_buffer = ResetStateBuffer(ResetStateBufferCfg(size=2))
    env.reset_buffer.collect(torch.arange(2), lambda ids: None, lambda ids: {"row": ids[:, None]})
    env.reset_sampler = ResetStateSampler(ResetStateSamplerCfg(), 2, 2, "cpu")
    env.reset_sampler.row_ids[:] = torch.tensor([0, 1])
    env.reset_sampler.success_monitor.success_update(torch.tensor([1]), torch.tensor([True]))

    # step() retains the natural compute result before the viewer marks its terminated alias.
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_terminated[1] = True
    env._reset_idx(torch.tensor([0, 1]))

    torch.testing.assert_close(env.reset_sampler.success_monitor.get_success_rate(), torch.tensor([1.0, 1.0]))
    assert env.extras["log"]["Reset/default_success"].item() == 1.0


def test_saving_reset_bank_rejects_its_json_report_path(tmp_path: Path) -> None:
    """Reject an output collision before replacing an existing catalog."""
    env = CubePickupEnv.__new__(CubePickupEnv)
    env._is_closed = True
    env.reset_buffer = ResetStateBuffer(ResetStateBufferCfg(size=1))
    env.generation_report = {}
    path = tmp_path / "bank.json"
    path.write_bytes(b"existing catalog")
    with pytest.raises(ValueError, match="different paths"):
        env.save_reset_buffer(path)
    assert path.read_bytes() == b"existing catalog"
