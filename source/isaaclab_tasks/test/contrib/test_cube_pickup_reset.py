# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for pickup rewards, completed episodes, and reset-bank persistence."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.benchmark.metrics import SuccessRateTracker
from isaaclab.envs.reset import ResetStateBuffer, ResetStateBufferCfg
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ObservationManager,
    RecorderManager,
    RewardManager,
    SceneEntityCfg,
    TerminationManager,
    TerminationTermCfg,
)
from isaaclab.utils import update_from_dict

from isaaclab_tasks.contrib.cube_pickup import mdp
from isaaclab_tasks.contrib.cube_pickup.cube_pickup_env import CubePickupEnv
from isaaclab_tasks.contrib.cube_pickup.cube_pickup_env_cfg import CubePickupEnvCfg
from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def _success(env) -> torch.Tensor:
    return env.success_flags


def test_goal_requires_a_grasp_at_the_transformed_target() -> None:
    """Root-frame goals and rotated TCP offsets use the selected hand and both selected fingers."""
    quarter_turn = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    body_positions = torch.tensor(
        [
            [[9.0, 9.0, 9.0], [1.0, 2.4, 0.10]],
            [[9.0, 9.0, 9.0], [1.0, 2.4, 0.10]],
            [[9.0, 9.0, 9.0], [1.0, 2.2, 0.10]],
            [[9.0, 9.0, 9.0], [1.0, 2.6, 0.10]],
        ]
    )
    body_orientations = torch.zeros(4, 2, 4)
    body_orientations[:, 0, 3] = 1.0
    body_orientations[:, 1] = quarter_turn
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([1.0, 2.0, 0.0]).expand(4, -1)),
            root_quat_w=SimpleNamespace(torch=quarter_turn.expand(4, -1)),
            body_link_pos_w=SimpleNamespace(torch=body_positions),
            body_link_quat_w=SimpleNamespace(torch=body_orientations),
            joint_pos=SimpleNamespace(
                torch=torch.tensor(
                    [[0.0, 0.02, 0.0, 0.02], [0.0, 0.02, 0.0, 0.04], [0.0, 0.02, 0.0, 0.02], [0.0, 0.02, 0.0, 0.02]]
                )
            ),
        )
    )
    cube = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(
                torch=torch.tensor([[1.0, 2.5, 0.15], [1.0, 2.5, 0.15], [1.0, 2.5, 0.15], [1.0, 2.7, 0.15]])
            )
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot, "object": cube},
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.tensor([0.5, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0]).expand(4, -1)
        ),
        _tcp_offset_pos=torch.tensor([0.10, 0.0, 0.05]).expand(4, -1),
    )
    # A 90-degree root turn places the goal at (1, 2.5, .15). The hand's same turn
    # maps its TCP offset to (0, .10, .05), independently of the decoy body at index 0.
    reached = mdp.goal_reached(
        env,
        command_name="object_pose",
        threshold=0.04,
        hand_cfg=SceneEntityCfg("robot", body_ids=[1]),
        gripper_cfg=SceneEntityCfg("robot", joint_ids=[1, 3]),
    )
    # Valid grasp, one open finger, distant hand, and a grasp outside the goal.
    torch.testing.assert_close(reached, torch.tensor([True, False, False, False]))


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
    """Only natural outcomes reach row history and the benchmark success tracker."""
    env = CubePickupEnv.__new__(CubePickupEnv)
    env._is_closed = True
    env.cfg = CubePickupEnvCfg()
    env.sim = SimpleNamespace(
        device="cpu",
        is_playing=lambda: True,
        is_rendering=False,
        step=lambda render: None,
        forward=lambda: None,
        manual_reset=False,
    )
    env.sim.consume_reset_request = lambda: env.sim.manual_reset
    env.scene = SimpleNamespace(
        num_envs=2,
        _ALL_INDICES=torch.arange(2),
        reset=lambda env_ids: None,
        write_data_to_sim=lambda: None,
        update=lambda dt: None,
    )
    env._restore_reset_state = lambda state, env_ids: None
    env.extras = {}
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.success_flags = torch.zeros(2, dtype=torch.bool)
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env._physics_handles_decimation = False
    env._physics_handles_actions = False
    env.render_enabled = False
    env.has_rtx_sensors = False
    env.video_recorders = []
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
    env._reset_group_ids = env.reset_buffer.group_ids
    env.reset_sampler = ResetStateSampler(ResetStateSamplerCfg(), 2, 2, "cpu")
    tracker = SuccessRateTracker(threshold=0.5, window=1, num_steps_per_env=1)
    _, extras = env.reset()
    tracker.record_step(extras)
    observations = [tracker.end_iteration()]
    env.reset_sampler.row_ids[:] = torch.tensor([0, 1])
    env.reset_sampler.success_monitor.success_update(torch.tensor([1]), torch.tensor([True]))

    env.episode_length_buf[:] = 5
    env.success_flags[0] = True
    env.sim.manual_reset = True
    tracker.record_step(env.step(torch.empty(2, 0))[4])
    observations.append(tracker.end_iteration())

    torch.testing.assert_close(env.reset_sampler.success_monitor.get_success_rate(), torch.tensor([1.0, 1.0]))
    env.success_flags.zero_()
    for manual_reset in (False, True):
        env.sim.manual_reset = manual_reset
        tracker.record_step(env.step(torch.empty(2, 0))[4])
        observations.append(tracker.end_iteration())
    assert observations == [None, 1.0, None, None]


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
