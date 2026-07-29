# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 LEAPP export integration test."""

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "sb3" / "export.py"
_EXPORT_MODULE_NAME = "_isaaclab_sb3_leapp_export"
_THIS_SCRIPT = Path(__file__).resolve()
_TASKS = ["Isaac-Cartpole"]
_EXPORT_BATCH_TIMEOUT = 600
_OUTPUT_TAIL_CHARS = 5000
_PROCESS_FAILURE_PATTERNS = (
    "Traceback (most recent call last):",
    "FileNotFoundError:",
    "[ERROR]",
    "Segmentation fault",
)


def _export_dir(task_name: str) -> str:
    """Return the directory where export.py writes artifacts for *task_name*."""
    train_task = task_name.replace("-Play", "")
    return os.path.join(_REPO_ROOT, ".pretrained_checkpoints", "sb3", train_task, task_name)


def _ensure_text(output: str | bytes | None) -> str:
    """Return subprocess output as text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _leapp_log_tail(export_dir: str) -> str:
    """Return the tail of the LEAPP log when it exists."""
    log_txt_path = os.path.join(export_dir, "log.txt")
    if not os.path.isfile(log_txt_path):
        return ""
    with open(log_txt_path) as file:
        last_lines = file.readlines()[-50:]
    return f"\n--- leapp log.txt (last 50 lines) ---\n{''.join(last_lines)}"


def _fail_on_process_error(result: subprocess.CompletedProcess[str]) -> None:
    """Fail when Isaac Sim reports an error but exits with a successful status."""
    output = f"{result.stdout}\n{result.stderr}"
    for pattern in _PROCESS_FAILURE_PATTERNS:
        if pattern in output:
            pytest.fail(
                f"export batch reported {pattern!r} for {_TASKS}.\n"
                f"--- stdout tail ---\n{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n"
                f"--- stderr tail ---\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
            )


def _load_export_module():
    """Load the SB3 export script as an importable module."""
    module = sys.modules.get(_EXPORT_MODULE_NAME)
    if module is not None and hasattr(module, "export_sb3_agent"):
        return module

    sys.modules.pop(_EXPORT_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, _EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {_EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _clean_hydra_argv():
    """Temporarily hide pytest arguments from Hydra config resolution."""
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        yield
    finally:
        sys.argv = original_argv


def _export_args(task_name: str):
    """Build the export argument namespace for *task_name*."""
    export_module = _load_export_module()
    args_cli, _ = export_module.parse_export_args(
        [
            "--task",
            task_name,
            "--use_pretrained_checkpoint",
            "--disable_graph_visualization",
            "--headless",
        ]
    )
    return args_cli


def _run_export_task(task_name: str, simulation_app, sim_utils, get_settings_manager, resolve_task_config) -> None:
    """Export one task inside an already running Isaac Sim process."""
    export_dir = _export_dir(task_name)
    export_module = _load_export_module()

    try:
        sim_utils.create_new_stage()
        get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)

        args_cli = _export_args(task_name)
        with _clean_hydra_argv():
            env_cfg, agent_cfg = resolve_task_config(task_name, args_cli.agent)
        exported = export_module.export_sb3_agent(args_cli, env_cfg, agent_cfg, simulation_app)

        assert exported, "Expected export to produce LEAPP artifacts"
        assert os.path.isfile(os.path.join(export_dir, f"{task_name}.onnx")), "Missing .onnx export"
        assert os.path.isfile(os.path.join(export_dir, f"{task_name}.yaml")), "Missing .yaml export"
        assert os.path.isfile(os.path.join(export_dir, "log.txt")), "Missing log.txt"
    except Exception as exc:
        raise RuntimeError(f"export.py failed for {task_name}: {exc!r}{_leapp_log_tail(export_dir)}") from exc
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


def _run_export_batch(task_names: list[str]) -> None:
    """Run a batch of exports inside a single Isaac Sim process."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.app.settings_manager import get_settings_manager

    from isaaclab_tasks.utils.hydra import resolve_task_config

    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    try:
        for task_name in task_names:
            _run_export_task(task_name, simulation_app, sim_utils, get_settings_manager, resolve_task_config)
    finally:
        simulation_app.close()


def _export_batch_command(task_names: list[str]) -> list[str]:
    """Build the subprocess command for an export batch."""
    return [sys.executable, str(_THIS_SCRIPT), "--export-flow-batch", *task_names]


def _run_export_batch_entrypoint() -> None:
    """Run the helper subprocess entrypoint."""
    tasks = sys.argv[2:]
    if not tasks:
        raise ValueError("Expected at least one task for --export-flow-batch")
    _run_export_batch(tasks)


def test_export_flow_fails_on_sim_traceback():
    """Catch simulator failures even when the process reports success."""
    result = subprocess.CompletedProcess(
        args=["export-flow"],
        returncode=0,
        stdout="Traceback (most recent call last):\nFileNotFoundError: missing asset",
        stderr="",
    )

    with pytest.raises(pytest.fail.Exception):
        _fail_on_process_error(result)


def test_sb3_export_flow():
    """Run SB3 export.py and assert the expected artifacts are created."""
    try:
        result = subprocess.run(
            _export_batch_command(_TASKS),
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_EXPORT_BATCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)
        pytest.fail(
            f"export batch timed out after {_EXPORT_BATCH_TIMEOUT}s for {_TASKS}.\n"
            f"--- stdout tail ---\n{stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{stderr[-_OUTPUT_TAIL_CHARS:]}"
        )

    if "Unfortunately a pre-trained checkpoint is currently unavailable" in result.stdout:
        pytest.skip("No pretrained SB3 checkpoint available for test task")
    if "KeyError: 'EXP_PATH'" in result.stderr:
        pytest.skip("Isaac Sim EXP_PATH is not configured for export-flow test")
    if result.returncode != 0:
        pytest.fail(
            f"export batch exited with code {result.returncode} for {_TASKS}.\n"
            f"--- stdout tail ---\n{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
        )

    _fail_on_process_error(result)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--export-flow-batch":
    _run_export_batch_entrypoint()
