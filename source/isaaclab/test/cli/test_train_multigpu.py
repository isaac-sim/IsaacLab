# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-GPU training launcher."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock

from isaaclab.cli.utils import ISAACLAB_ROOT


def _load_train_multigpu_module():
    """Load the train_multigpu script as a test module."""
    module_path = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "train_multigpu.py"
    spec = importlib.util.spec_from_file_location("isaaclab_test_train_multigpu", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN_MULTIGPU = _load_train_multigpu_module()


def test_builds_single_node_rsl_rl_torchrun_command():
    """Multi-GPU launcher should preserve training args and inject distributed mode."""
    args_cli, train_args = TRAIN_MULTIGPU._parse_args(
        [
            "--num_gpus",
            "4",
            "--master_port",
            "29504",
            "--task=Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
            "--headless",
            "--num_envs=4096",
            "--max_iterations=100",
            "--run_name=gpu4_vis",
            "presets=newton",
        ]
    )

    command = TRAIN_MULTIGPU._build_torchrun_command(args_cli, train_args)
    train_script_index = command.index(str(TRAIN_MULTIGPU.TRAIN_SCRIPT))

    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4"]
    assert command[5:7] == ["--master_port", "29504"]
    assert command[train_script_index + 1 : train_script_index + 3] == ["--rl_library", "rsl_rl"]
    assert command[-6:] == [
        "--headless",
        "--num_envs=4096",
        "--max_iterations=100",
        "--run_name=gpu4_vis",
        "presets=newton",
        "--distributed",
    ]


def test_builds_multi_node_skrl_torchrun_command():
    """Multi-node torchrun settings should be forwarded before the training script."""
    args_cli, train_args = TRAIN_MULTIGPU._parse_args(
        [
            "--rl_library",
            "skrl",
            "--nproc_per_node",
            "2",
            "--nnodes",
            "2",
            "--node_rank",
            "1",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "host.example.com:5555",
            "--rdzv_id",
            "job-1",
            "--task",
            "Isaac-Cartpole-v0",
            "--distributed",
        ]
    )

    command = TRAIN_MULTIGPU._build_torchrun_command(args_cli, train_args)
    train_script_index = command.index(str(TRAIN_MULTIGPU.TRAIN_SCRIPT))

    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"]
    assert command[5:train_script_index] == [
        "--nnodes",
        "2",
        "--node_rank",
        "1",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "host.example.com:5555",
        "--rdzv_id",
        "job-1",
    ]
    assert command[train_script_index + 1 : train_script_index + 3] == ["--rl_library", "skrl"]
    assert command.count("--distributed") == 1


def test_dry_run_prints_command_without_launching(capsys):
    """Dry-run mode should not start torchrun."""
    with mock.patch.object(subprocess, "Popen") as mock_popen:
        result = TRAIN_MULTIGPU.main(["--dry_run", "--num_gpus", "2", "--task", "Isaac-Cartpole-v0"])

    assert result == 0
    mock_popen.assert_not_called()
    output = capsys.readouterr().out
    assert "torch.distributed.run" in output
    assert "--nproc_per_node 2" in output
    assert "--task Isaac-Cartpole-v0 --distributed" in output


def test_main_forwards_signals_to_subprocess():
    """Main should forward termination signals to the torchrun subprocess."""
    proc = mock.Mock()
    proc.wait.return_value = 7

    with (
        mock.patch.object(subprocess, "Popen", return_value=proc) as mock_popen,
        mock.patch.object(
            TRAIN_MULTIGPU.signal, "signal", side_effect=["old_sigterm", "old_sigint", None, None]
        ) as mock_signal,
    ):
        result = TRAIN_MULTIGPU.main(["--num_gpus", "2", "--task", "Isaac-Cartpole-v0"])

    assert result == 7
    mock_popen.assert_called_once()
    proc.wait.assert_called_once_with()

    sigterm_handler = mock_signal.call_args_list[0].args[1]
    sigint_handler = mock_signal.call_args_list[1].args[1]
    sigterm_handler(TRAIN_MULTIGPU.signal.SIGTERM, None)
    sigint_handler(TRAIN_MULTIGPU.signal.SIGINT, None)
    assert proc.terminate.call_count == 2

    assert mock_signal.call_args_list[0].args[0] == TRAIN_MULTIGPU.signal.SIGTERM
    assert mock_signal.call_args_list[1].args[0] == TRAIN_MULTIGPU.signal.SIGINT
    assert mock_signal.call_args_list[2] == mock.call(TRAIN_MULTIGPU.signal.SIGTERM, "old_sigterm")
    assert mock_signal.call_args_list[3] == mock.call(TRAIN_MULTIGPU.signal.SIGINT, "old_sigint")


def test_cli_helper_runs_multigpu_script():
    """The isaaclab CLI helper should dispatch to the multi-GPU training script."""
    from isaaclab import cli

    with mock.patch("isaaclab.cli.run_python_command") as mock_run:
        cli.train_multigpu(["--dry_run"])

    mock_run.assert_called_once_with(
        Path(ISAACLAB_ROOT) / "scripts" / "reinforcement_learning" / "train_multigpu.py",
        ["--dry_run"],
        check=True,
    )
