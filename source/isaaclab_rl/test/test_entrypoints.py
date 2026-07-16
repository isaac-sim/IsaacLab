# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reusable unified reinforcement learning entrypoints."""

from __future__ import annotations

from isaaclab_rl.entrypoints import PlaybackRequest, TrainingRequest, api, dispatch


def test_train_request_adapts_typed_parameters_to_cli(monkeypatch) -> None:
    """Training requests use typed parameters rather than parser namespaces."""
    received: list[str] = []

    def fake_run_train_cli(argv: list[str]) -> int:
        received.extend(argv)
        return 7

    monkeypatch.setattr(api, "run_train_cli", fake_run_train_cli)

    result = api.train(
        TrainingRequest(
            backend="rsl_rl",
            task="Isaac-Cartpole",
            num_envs=32,
            max_iterations=10,
            distributed=True,
            hydra_args=("physics=newton_mjwarp",),
        )
    )

    assert result == 7
    assert received == [
        "--rl_library",
        "rsl_rl",
        "--task",
        "Isaac-Cartpole",
        "--num_envs",
        "32",
        "--max_iterations",
        "10",
        "--distributed",
        "physics=newton_mjwarp",
    ]


def test_play_request_uses_rlinf_argument_names(monkeypatch) -> None:
    """RLinf requests map shared fields to its focused backend arguments."""
    received: list[str] = []

    def fake_run_play_cli(argv: list[str]) -> int:
        received.extend(argv)
        return 0

    monkeypatch.setattr(api, "run_play_cli", fake_run_play_cli)

    api.play(PlaybackRequest(backend="rlinf", task="Isaac-Task", checkpoint="model", video=True))

    assert received == [
        "--rl_library",
        "rlinf",
        "--task",
        "Isaac-Task",
        "--model_path",
        "model",
        "--video",
    ]


def test_train_dispatches_selected_backend(monkeypatch) -> None:
    """The unified training dispatcher forwards only backend arguments."""
    received: dict[str, object] = {}

    def _fake_run_backend(module_name: str, argv: list[str], *, run_as_script: bool) -> None:
        received.update(module_name=module_name, argv=argv, run_as_script=run_as_script)

    monkeypatch.setattr(dispatch, "_run_backend", _fake_run_backend)

    assert dispatch.run_train_cli(["--rl_library", "rsl_rl", "--task", "Isaac-Cartpole"]) == 0
    assert received == {
        "module_name": "isaaclab_rl.entrypoints.backends.train_rsl_rl",
        "argv": ["--task", "Isaac-Cartpole"],
        "run_as_script": False,
    }


def test_dispatch_fuses_option_like_kit_args(monkeypatch) -> None:
    """Space-separated option-like Kit arguments are fused before backend parsing."""
    received: dict[str, object] = {}
    monkeypatch.setattr(
        dispatch, "_run_backend", lambda module_name, argv, *, run_as_script: received.update(argv=argv)
    )

    dispatch.run_train_cli(["--rl_library", "rsl_rl", "--kit_args", "--foo=/bar"])

    assert received["argv"] == ["--kit_args=--foo=/bar"]


def test_dispatch_requires_a_backend() -> None:
    """Missing backend selection returns the conventional CLI error status."""
    assert dispatch.run_train_cli(["--task", "Isaac-Cartpole"]) == 2
