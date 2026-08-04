# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-GPU training benchmark launcher."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from isaaclab.benchmark.entrypoints import training_multigpu as launcher

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _build_command(argv: list[str]) -> list[str]:
    args_cli, forwarded_args = launcher._parse_args(argv)
    return launcher._build_distributed_command(args_cli, forwarded_args)


def test_launcher_targets_benchmark_training_and_adds_private_mode(monkeypatch: pytest.MonkeyPatch):
    """The benchmark launcher should select torchrun and the benchmark training child."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    command = _build_command(["--rl_library", "rsl_rl", "--num_gpus", "2", "--task", "X"])

    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"]
    assert str(_REPO_ROOT / "scripts" / "benchmarks" / "training.py") in command
    assert command[-2:] == ["--distributed", "--benchmark_multigpu"]


def test_launcher_forwards_multi_node_rendezvous(monkeypatch: pytest.MonkeyPatch):
    """Multi-node launcher options should reach torchrun unchanged."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    command = _build_command(
        [
            "--num_gpus",
            "2",
            "--nnodes",
            "3",
            "--node_rank",
            "1",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "host0:29400",
            "--rdzv_id",
            "bench",
            "--task",
            "X",
        ]
    )

    assert command[command.index("--nnodes") + 1] == "3"
    assert command[command.index("--node_rank") + 1] == "1"
    assert command[command.index("--rdzv_endpoint") + 1] == "host0:29400"
    assert command[command.index("--rdzv_id") + 1] == "bench"


@pytest.mark.parametrize(
    "argv",
    [
        ["--rl_library", "sb3", "--task", "X"],
        ["--rl_library", "skrl", "--ml_framework", "jax", "--task", "X"],
        ["--video", "--task", "X"],
        ["--capture_env_sensors", "1", "--task", "X"],
        ["--check_success", "--task", "X"],
    ],
)
def test_launcher_rejects_unsupported_modes(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    """Unsupported distributed benchmark modes should fail before workers launch."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SystemExit):
        _build_command(argv)


def test_launcher_allows_zero_sensor_capture_and_opaque_kit_args(monkeypatch: pytest.MonkeyPatch):
    """Disabled capture and option-like Kit values should remain valid forwarded input."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    command = _build_command(
        ["--capture_env_sensors=0", "--kit_args", "--video", "presets=isaacsim_physx", "--task", "X"]
    )

    assert "--capture_env_sensors=0" in command
    assert command[command.index("--kit_args") + 1] == "--video"
    assert "presets=isaacsim_physx" in command


def test_dry_run_is_shell_parsable(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Dry-run output should preserve one token per forwarded argument."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert launcher.main(["--dry_run", "--num_gpus", "2", "--task", "X"]) == 0

    tokens = shlex.split(capsys.readouterr().out)
    assert tokens[-2:] == ["--distributed", "--benchmark_multigpu"]
