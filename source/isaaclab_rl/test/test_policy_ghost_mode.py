# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy Ghosts switch on and off inside a running playback."""

from types import SimpleNamespace

import pytest

from isaaclab_rl.entrypoints.policy_ghosts import PolicyGhostMode, PolicyGhostRestart, relaunch_argv

pytestmark = pytest.mark.unit


class _Visualizer:
    def __init__(self):
        self.events = []

    def attach_policy_ghosts(self, session, restart, disable):
        self.events.append(("attach", session, disable))

    def detach_policy_ghosts(self):
        self.events.append(("detach",))


class _Session:
    def __init__(self):
        self.steps, self.closed = 0, False

    def step(self):
        self.steps += 1

    def close(self):
        self.closed = True


def _mode(num_envs=64, fail=False):
    log = []
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=num_envs), reset=lambda: log.append("reset"))
    visualizer, sessions = _Visualizer(), []

    def create(size, slots):
        if fail:
            raise RuntimeError("no checkpoint")
        sessions.append(_Session())
        log.append(("create", size, slots))
        return sessions[-1]

    mode = PolicyGhostMode(
        env,
        slice_size=4,
        num_slots=3,
        create_session=create,
        step=lambda: log.append("plain"),
        resume=lambda: log.append("resume"),
        restart=PolicyGhostRestart(),
        visualizers=[visualizer],
    )
    return mode, visualizer, sessions, log


def test_a_pool_with_enough_environments_switches_in_place() -> None:
    assert _mode(12)[0].in_place and not _mode(11)[0].in_place


def test_enabling_runs_the_session_and_disabling_resets_to_the_plain_playback() -> None:
    mode, visualizer, sessions, log = _mode()
    mode.step()
    mode.request_enable()
    mode.step()
    mode.step()

    assert log == ["plain", ("create", 4, 3)] and sessions[0].steps == 2
    assert mode.active and visualizer.events[0][0] == "attach"

    mode.request_disable()
    mode.step()

    assert visualizer.events[-1] == ("detach",) and sessions[0].closed and not mode.active
    assert log[2:] == ["reset", "resume", "plain"]


def test_a_request_that_matches_the_current_state_does_nothing() -> None:
    mode, _, sessions, _ = _mode()
    mode.request_disable()
    mode.step()
    mode.request_enable()
    mode.step()
    mode.request_enable()
    mode.step()

    assert len(sessions) == 1


def test_a_failed_switch_keeps_the_plain_playback_and_reports_why() -> None:
    mode, visualizer, _, log = _mode(fail=True)
    mode.request_enable()
    mode.step()

    assert not mode.active and "no checkpoint" in mode.message and log == ["plain"]
    assert visualizer.events == [("detach",)]  # Nothing stays half attached.


def test_the_layout_can_be_chosen_within_the_slot_limits() -> None:
    mode, *_ = _mode(20)
    mode.set_layout(0, 99)
    assert (mode.slice_size, mode.num_slots) == (1, 8)
    mode.set_layout(5, 1)
    assert (mode.slice_size, mode.num_slots) == (5, 2) and mode.in_place and mode.pool_size == 20
    mode.set_layout(11, 2)
    assert not mode.in_place  # 22 environments do not fit in 20.


def test_a_restart_carries_the_chosen_layout_into_the_command_line() -> None:
    mode, *_ = _mode(4)
    mode.set_layout(2, 3)
    mode.request_restart()
    restart = mode.restart
    assert (restart.requested, restart.enable, restart.envs, restart.slots) == (True, True, 2, 3)
    argv = ["python", "play", "--num_envs", "4", "--policy_ghost_slots=8", "--task", "T"]
    assert relaunch_argv(argv, True, 2, 3) == [
        "python", "play", "--num_envs", "2", "--policy_ghost_slots=3", "--task", "T", "--policy_ghosts"
    ]  # fmt: skip


def test_a_failed_switch_detaches_the_visualizer_and_closes_the_new_session() -> None:
    mode, visualizer, sessions, _ = _mode()

    def failing_attach(session, restart, disable):
        raise RuntimeError("viewer closed")

    visualizer.attach_policy_ghosts = failing_attach
    mode.request_enable()
    mode.step()

    assert not mode.active and sessions[0].closed and ("detach",) in visualizer.events
