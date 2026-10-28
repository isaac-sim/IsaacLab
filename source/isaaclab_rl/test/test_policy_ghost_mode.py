# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy Ghosts switch on and off inside a running playback."""

from types import SimpleNamespace

import pytest

from isaaclab_rl.entrypoints.policy_ghosts import PolicyGhostMode, PolicyGhostRestart

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
    assert not visualizer.events
