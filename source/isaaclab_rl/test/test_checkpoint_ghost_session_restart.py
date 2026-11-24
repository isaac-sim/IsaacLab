# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A ghost that was just added or changed starts from the spawn state even when no step follows, such as when paused."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.visualizers.checkpoint_ghost import CheckpointGhostCheckpoint, CheckpointGhostLayout

from isaaclab_rl.entrypoints.checkpoint_ghosts import CheckpointGhostSession

CHECKPOINT = CheckpointGhostCheckpoint("/runs/a/model_1.pt", "a", "model_1.pt", 0.0)


def _session(pending: bool) -> tuple[CheckpointGhostSession, Mock, Mock]:
    session = object.__new__(CheckpointGhostSession)  # the real constructor needs a running environment
    env = Mock()
    env.reset.return_value = ("observations", {})
    env.unwrapped = SimpleNamespace(device="cpu", num_envs=6)
    policy = Mock()
    session._env, session._sync = env, None
    session._policies = [policy, None, None]
    session.layout = CheckpointGhostLayout(6, 3)
    session.layout.slots[0].checkpoint = CHECKPOINT
    session._restart_pending = pending
    return session, env, policy


def test_a_pending_restart_resets_the_environments_and_the_policies_once():
    session, env, policy = _session(pending=True)

    session.apply_pending_restart()
    session.apply_pending_restart()

    env.reset.assert_called_once_with()
    assert session._obs == "observations"
    assert session._restart_pending is False
    assert torch.equal(policy.reset.call_args.args[0], torch.ones(6, dtype=torch.bool))


def test_without_a_pending_restart_nothing_is_reset():
    session, env, policy = _session(pending=False)

    session.apply_pending_restart()

    env.reset.assert_not_called()
    policy.reset.assert_not_called()


def _checkpoint(name: str) -> CheckpointGhostCheckpoint:
    return CheckpointGhostCheckpoint(f"/runs/{name}.pt", "run", f"{name}.pt", 0.0)


def _session_with_ghost(factory: Mock) -> CheckpointGhostSession:
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

    assert "bad.pt" in error and "does not fit this task" in error
    assert session._policies[1] == "old policy"
    assert session.layout.slots[1].checkpoint.name == "old.pt"
    assert session._restart_pending is False


def test_the_reference_slot_cannot_be_replaced():
    session = _session_with_ghost(Mock(return_value="policy"))

    assert session.replace_ghost(0, _checkpoint("new")) is not None


@pytest.mark.parametrize(
    ("dones", "expected"),
    [
        ([False, False, False, True, False, False], [False, False, False]),  # a ghost fell alone: keeps its memory
        ([True, False, False, False, False, False], [True, False, False]),  # the reference reset, so the ghost did too
    ],
    ids=["ghost-falls-alone", "reference-resets"],
)
def test_a_ghost_policy_resets_with_its_reference(dones, expected):
    session, env, reference = _session(pending=False)
    ghost = Mock()
    session._policies = [reference, ghost, None]
    session.layout.slots[1].checkpoint = CHECKPOINT  # two loaded slots split the six envs into slices of three
    session._obs = torch.zeros(6, 1)
    env.num_envs, env.num_actions = 6, 1
    reference.return_value = torch.zeros(3, 1)
    ghost.return_value = torch.zeros(3, 1)
    env.step.return_value = (torch.zeros(6, 1), torch.zeros(6), torch.tensor(dones), {})

    session.step()

    assert ghost.reset.call_args.args[0].tolist() == expected


def _start(monkeypatch, observations):
    from isaaclab_rl.entrypoints import checkpoint_ghosts

    sync = Mock()
    monkeypatch.setattr(checkpoint_ghosts, "CheckpointGhostSync", sync)
    env = Mock(unwrapped=SimpleNamespace(device="cpu", step_dt=0.02, num_envs=6))
    env.get_observations.side_effect = observations
    session = CheckpointGhostSession(
        env,
        num_slots=3,
        task="Isaac-Reach-Franka",
        checkpoint_root="/runs",
        checkpoint_pattern=r"model_.*\.pt",
        policy_factory=Mock(),
        reference=CHECKPOINT,
    )
    return session, sync


def test_a_failing_start_never_leaves_the_environment_patched(monkeypatch):
    with pytest.raises(RuntimeError, match="observations"):
        _start(monkeypatch, RuntimeError("observations"))


def test_adding_and_removing_a_ghost_splits_the_environments_again(monkeypatch):
    session, sync = _start(monkeypatch, [None])
    sync.assert_not_called()  # the reference alone owns every environment
    assert session.layout.slice_size == 6

    assert session.add_ghost(CHECKPOINT) is None
    sync.assert_called_once_with(session._env.unwrapped, 3, 2)
    assert session.layout.slice_size == 3 and session._restart_pending

    first = sync.return_value
    session.remove_ghost(1)

    first.close.assert_called_once()  # the old split's terrain is restored before the next one
    assert sync.call_count == 1 and session._sync is None
    assert session.layout.slice_size == 6


def test_a_slot_is_labelled_with_the_file_name_then_its_timestamp():
    import time

    session, _, _ = _session(pending=False)
    session._policy_factory = Mock()
    modified = time.mktime((2026, 8, 12, 21, 14, 3, 0, 0, -1))
    checkpoint = CheckpointGhostCheckpoint("/runs/a/model_1200.pt", "a", "model_1200.pt", modified)

    session._load(1, checkpoint)

    assert session.layout.slots[1].label == "model_1200.pt / 2026-08-12 21:14"


def test_browsing_lists_folders_newest_first_and_only_loadable_checkpoint_files(tmp_path):
    import os

    session, _, _ = _session(pending=False)
    session._pattern = r"model_.*\.pt"
    for name, age in (("old_run", 100), ("new_run", 0), (".hidden", 0)):
        folder = tmp_path / name
        folder.mkdir()
        os.utime(folder, (1_000_000 - age, 1_000_000 - age))
    for name in ("model_10.pt", "model_9.pt", "notes.txt", "skrl_agent.pt"):
        (tmp_path / name).touch()

    folders, files = session.browse(str(tmp_path))

    assert folders == ["new_run", "old_run"]
    assert [f.name for f in files] == ["model_9.pt", "model_10.pt"]
    assert all(os.path.isabs(f.path) for f in files)


def test_browsing_an_unreadable_directory_raises_os_error(tmp_path):
    session, _, _ = _session(pending=False)

    with pytest.raises(OSError):
        session.browse(str(tmp_path / "missing"))


def test_a_run_of_another_task_is_named_and_a_matching_or_unknown_directory_is_not(tmp_path):
    import json

    session, _, _ = _session(pending=False)
    session._task = "Isaac-Reach-Franka"
    for name, task in (("same", "isaac:Isaac-Reach-Franka"), ("osc", "Isaac-Reach-Franka-OSC")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.json").write_text(json.dumps({"task": task}))
    (tmp_path / "plain").mkdir()
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "run.json").write_text("not json")

    assert session.other_task(str(tmp_path / "osc")) == "Isaac-Reach-Franka-OSC"
    assert session.other_task(str(tmp_path / "same")) is None  # a namespace prefix does not make it another task
    assert session.other_task(str(tmp_path / "plain")) is None
    assert session.other_task(str(tmp_path / "bad")) is None


def test_a_load_failure_is_reported_in_one_line():
    from isaaclab_rl.entrypoints.checkpoint_ghosts import _describe_load_error

    mismatch = RuntimeError("Error(s) in loading state_dict:\n size mismatch for mlp.0.weight: copying a param")

    assert "does not fit this task" in _describe_load_error(mismatch)
    assert _describe_load_error(RuntimeError("file is corrupt\nmore detail")) == "file is corrupt"
