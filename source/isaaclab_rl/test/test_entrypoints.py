# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reusable unified reinforcement learning entrypoints."""

from __future__ import annotations

import importlib
import runpy
import sys
import types

import pytest

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


def test_train_request_maps_checkpoint_to_backend_argument(monkeypatch) -> None:
    """Training requests forward a checkpoint so training can resume from it."""
    received: list[str] = []

    def fake_run_train_cli(argv: list[str]) -> int:
        received.extend(argv)
        return 0

    monkeypatch.setattr(api, "run_train_cli", fake_run_train_cli)

    api.train(TrainingRequest(backend="rsl_rl", task="Isaac-Cartpole", checkpoint="latest"))
    assert received == ["--rl_library", "rsl_rl", "--task", "Isaac-Cartpole", "--checkpoint", "latest"]

    received.clear()
    api.train(TrainingRequest(backend="rlinf", task="Isaac-Task", checkpoint="model"))
    assert received == ["--rl_library", "rlinf", "--task", "Isaac-Task", "--model_path", "model"]


def test_run_backend_restores_sys_argv_after_training(monkeypatch) -> None:
    """A training backend that mutates ``sys.argv`` must not leak the change to the caller."""
    module = types.ModuleType("fake_train_backend")

    def run(argv: list[str]) -> None:
        # backends call set_hydra_args, which overwrites sys.argv while parsing
        sys.argv = [sys.argv[0]] + argv

    module.run = run
    monkeypatch.setattr(importlib, "import_module", lambda name: module)

    original_argv = list(sys.argv)
    dispatch._run_backend("fake_train_backend", ["physics=newton_mjwarp"], run_as_script=False)

    assert sys.argv == original_argv


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


def _torch_backend_state() -> tuple[bool, bool, bool, bool]:
    import torch

    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def test_scoped_backend_state_restores_values_after_exception() -> None:
    """Backend-global settings are restored when a scoped operation fails."""
    from isaaclab_rl.entrypoints.common import preserve_attribute, scoped_torch_backend_flags

    original = _torch_backend_state()
    holder = types.SimpleNamespace(value="original")

    with pytest.raises(RuntimeError, match="failed"):
        with (
            scoped_torch_backend_flags(
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=True,
                cudnn_deterministic=True,
                cudnn_benchmark=False,
            ),
            preserve_attribute(holder, "value"),
        ):
            holder.value = "temporary"
            assert _torch_backend_state() == (False, True, True, False)
            raise RuntimeError("failed")

    assert _torch_backend_state() == original
    assert holder.value == "original"


def test_rejected_rsl_training_preserves_torch_backend_state(monkeypatch) -> None:
    """A rejected in-process RSL-RL request does not mutate its caller."""
    import torch

    caller_state = (False, False, True, True)
    settings = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
    )
    for (target, name), value in zip(settings, caller_state):
        monkeypatch.setattr(target, name, value)

    with pytest.raises(SystemExit):
        dispatch._run_backend("isaaclab_rl.entrypoints.backends.train_rsl_rl", ["--help"], run_as_script=False)

    assert _torch_backend_state() == caller_state


def test_failed_rsl_training_restores_torch_backend_state(monkeypatch) -> None:
    """RSL-RL training restores its caller's Torch settings after a failure."""
    import torch

    from isaaclab_rl.entrypoints.backends import train_rsl_rl

    caller_state = (False, False, True, True)
    settings = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
    )
    for (target, name), value in zip(settings, caller_state):
        monkeypatch.setattr(target, name, value)

    monkeypatch.setattr(train_rsl_rl, "_parse_args", lambda argv: types.SimpleNamespace())

    def fail_after_mutation(_args_cli) -> None:
        assert _torch_backend_state() == (True, True, False, False)
        raise RuntimeError("failed")

    monkeypatch.setattr(train_rsl_rl, "_run", fail_after_mutation)
    with pytest.raises(RuntimeError, match="failed"):
        train_rsl_rl.run([])

    assert _torch_backend_state() == caller_state


def test_skrl_training_restores_jax_backend(monkeypatch) -> None:
    """SKRL training removes the JAX backend setting it created after an exception."""
    skrl = pytest.importorskip("skrl")

    from isaaclab_rl.entrypoints.backends import train_skrl

    monkeypatch.delattr(skrl.config.jax, "backend", raising=False)
    monkeypatch.setattr(train_skrl, "_parse_args", lambda argv: types.SimpleNamespace(ml_framework="jax"))

    def fail_after_mutation(_args_cli) -> None:
        assert skrl.config.jax.backend == "jax"
        skrl.config.jax.backend = "mutated"
        raise RuntimeError("failed")

    monkeypatch.setattr(train_skrl, "_run", fail_after_mutation)
    with pytest.raises(RuntimeError, match="failed"):
        train_skrl.run([])

    assert not hasattr(skrl.config.jax, "backend")


def test_skrl_play_main_restores_jax_backend(monkeypatch) -> None:
    """Direct SKRL play calls remove the JAX backend setting they created."""
    pytest.importorskip("skrl")
    monkeypatch.setattr(sys, "argv", ["play_skrl.py"])
    namespace = runpy.run_module("isaaclab_rl.entrypoints.backends.play_skrl", run_name="test_play_skrl")
    skrl = namespace["skrl"]
    namespace["args_cli"].ml_framework = "jax"
    monkeypatch.delattr(skrl.config.jax, "backend", raising=False)

    def fail_after_mutation() -> None:
        skrl.config.jax.backend = "mutated"
        raise RuntimeError("failed")

    namespace["main"].__globals__["_main"] = fail_after_mutation
    with pytest.raises(RuntimeError, match="failed"):
        namespace["main"]()

    assert not hasattr(skrl.config.jax, "backend")
