# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``<workflow>-multigpu`` benchmark launcher command building."""

from __future__ import annotations

import shlex
import sys

import pytest

from isaaclab.benchmark.entrypoints import multigpu
from isaaclab.cli.multigpu import build_launch_command, parse_launcher_args


@pytest.fixture(autouse=True)
def two_visible_gpus(monkeypatch: pytest.MonkeyPatch):
    """Let ``--num_gpus 2`` pass the visible-device check without a GPU present."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")


def _command(workflow: str, argv: list[str]) -> list[str]:
    """Build the launcher command for a workflow the way the CLI does."""
    cfg = multigpu.launcher_cfg(workflow)
    args_cli, worker_args = parse_launcher_args(argv, cfg)
    return build_launch_command(args_cli, worker_args, cfg)


def _worker_argv(command: list[str]) -> list[str]:
    """Return the arguments each rank receives."""
    return command[command.index(multigpu.WORKER_SCRIPT) + 1 :]


def test_worker_runs_the_requested_workflow_in_distributed_mode():
    """Each rank runs this module, selects the workflow, and receives ``--distributed``."""
    command = _command("runtime", ["--num_gpus", "2", "--task", "Isaac-Cartpole-Direct"])

    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"]
    assert _worker_argv(command) == ["--workflow", "runtime", "--task", "Isaac-Cartpole-Direct", "--distributed"]


def test_training_workflow_selects_a_distributed_capable_library():
    """Training forwards ``--rl_library``; the other workflows have no library to choose."""
    command = _command("training", ["--rl_library", "skrl", "--num_gpus", "2", "--task", "X"])

    assert _worker_argv(command) == ["--workflow", "training", "--rl_library", "skrl", "--task", "X", "--distributed"]
    assert "--rl_library" not in _command("startup", ["--num_gpus", "2", "--task", "X"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--rl_library", "sb3", "--task", "X"],
        ["--rl_library", "skrl", "--ml_framework", "jax", "--task", "X"],
        ["--num_gpus", "4", "--task", "X"],
    ],
)
def test_unsupported_launches_fail_before_any_worker_starts(argv: list[str]):
    """SB3, skrl JAX, and over-subscribed GPUs are rejected by the launcher itself."""
    with pytest.raises(SystemExit):
        _command("training", argv)


def test_multi_node_rendezvous_options_reach_torchrun():
    """Multi-node launches forward their rendezvous configuration unchanged."""
    command = _command(
        "training",
        ["--num_gpus", "2", "--nnodes", "3", "--node_rank", "1", "--rdzv_endpoint", "host0:29400", "--task", "X"],
    )

    assert command[command.index("--nnodes") + 1] == "3"
    assert command[command.index("--node_rank") + 1] == "1"
    assert command[command.index("--rdzv_endpoint") + 1] == "host0:29400"


def test_virtual_local_rank_is_opt_in_and_forwarded_without_a_value():
    """``--virtual_local_rank`` reaches torchrun as a bare flag and never reaches the workers."""
    enabled = _command("training", ["--num_gpus", "2", "--virtual_local_rank", "--task", "X"])

    assert "--virtual_local_rank" in enabled[: enabled.index(multigpu.WORKER_SCRIPT)]
    assert "True" not in enabled  # torchrun rejects the flag if it is given a value
    assert "--virtual_local_rank" not in _worker_argv(enabled)
    assert "--virtual_local_rank" not in _command("training", ["--num_gpus", "2", "--task", "X"])


def test_dry_run_prints_a_shell_parsable_command(capsys: pytest.CaptureFixture[str]):
    """``--dry_run`` reports the exact command instead of launching workers."""
    status = multigpu.run_multigpu_benchmark_cli("startup", ["--dry_run", "--num_gpus", "2", "--task", "X"])

    assert status == 0
    tokens = shlex.split(capsys.readouterr().out)
    assert tokens == _command("startup", ["--num_gpus", "2", "--task", "X"])
