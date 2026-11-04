# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A ghost that was just added or changed starts from the spawn state even when no step follows, such as when paused."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint, PolicyGhostLayout

from isaaclab_rl.entrypoints.policy_ghosts import PolicyGhostSession


def _session(pending: bool) -> tuple[PolicyGhostSession, Mock, Mock]:
    session = object.__new__(PolicyGhostSession)  # the real constructor needs a running environment
    env = Mock()
    env.reset.return_value = ("observations", {})
    env.unwrapped = SimpleNamespace(device="cpu")
    policy, metrics = Mock(), Mock()
    session._env, session._metrics = env, metrics
    session._policies = [policy, None, None]
    session.layout = PolicyGhostLayout(2, 3)
    session._restart_pending = pending
    return session, env, policy


def test_a_pending_restart_resets_the_environments_and_the_policies_once():
    session, env, policy = _session(pending=True)

    session.apply_pending_restart()
    session.apply_pending_restart()

    env.reset.assert_called_once_with()
    assert session._obs == "observations"
    assert session._restart_pending is False
    assert torch.equal(policy.reset.call_args.args[0], torch.ones(2, dtype=torch.bool))


def test_without_a_pending_restart_nothing_is_reset():
    session, env, policy = _session(pending=False)

    session.apply_pending_restart()

    env.reset.assert_not_called()
    policy.reset.assert_not_called()


def _checkpoint(name: str) -> PolicyGhostCheckpoint:
    return PolicyGhostCheckpoint(f"/runs/{name}.pt", "run", f"{name}.pt", 0.0)


def _session_with_ghost(factory: Mock) -> PolicyGhostSession:
    session, _, _ = _session(pending=False)
    session._policy_factory = factory
    session._load(1, _checkpoint("old"))
    return session


def test_replacing_a_ghost_loads_the_checkpoint_into_that_slot_and_restarts_together():
    factory = Mock(side_effect=["old policy", "new policy"])
    session = _session_with_ghost(factory)

    error = session.replace_ghost(1, _checkpoint("new"))

    assert error is None
    assert session._policies[1] == "new policy"
    assert session.layout.slots[1].checkpoint.name == "new.pt"
    assert session._restart_pending is True


def test_a_checkpoint_that_fails_to_load_leaves_the_ghost_unchanged():
    factory = Mock(side_effect=["old policy", RuntimeError("size mismatch")])
    session = _session_with_ghost(factory)

    error = session.replace_ghost(1, _checkpoint("bad"))

    assert "size mismatch" in error
    assert session._policies[1] == "old policy"
    assert session.layout.slots[1].checkpoint.name == "old.pt"
    assert session._restart_pending is False


def test_the_reference_slot_cannot_be_replaced():
    session = _session_with_ghost(Mock(return_value="policy"))

    assert session.replace_ghost(0, _checkpoint("new")) is not None
