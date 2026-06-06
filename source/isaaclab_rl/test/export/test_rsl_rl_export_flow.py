# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export pipeline integration tests.

Each test launches Isaac Sim once for a batch of tasks. This avoids the
per-task Kit startup churn while keeping each Kit process short enough to avoid
accumulating PhysX GPU allocations across the full export matrix.
"""

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Root of the repository (three levels up from this file).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "rsl_rl" / "export.py"
_EXPORT_MODULE_NAME = "_isaaclab_rsl_rl_leapp_export"
_THIS_SCRIPT = Path(__file__).resolve()
_EXPORT_BATCH_SIZE = 8
_EXPORT_BATCH_TIMEOUT = 600
_OUTPUT_TAIL_CHARS = 5000


# Tasks with confirmed pretrained checkpoints (Direct and no-checkpoint tasks excluded).
TASKS = [
    # Classic
    "Isaac-Ant",
    "Isaac-Cartpole",
    # Navigation
    "Isaac-Navigation-Flat-Anymal-C-v0",
    "Isaac-Navigation-Flat-Anymal-C-Play-v0",
    # Locomotion Velocity
    "Isaac-Velocity-Flat-Anymal-B-v0",
    "Isaac-Velocity-Flat-Anymal-B-Play-v0",
    "Isaac-Velocity-Rough-Anymal-B-v0",
    "Isaac-Velocity-Rough-Anymal-B-Play-v0",
    "Isaac-Velocity-Flat-Anymal-C-v0",
    "Isaac-Velocity-Flat-Anymal-C-Play-v0",
    "Isaac-Velocity-Rough-Anymal-C-v0",
    "Isaac-Velocity-Rough-Anymal-C-Play-v0",
    "Isaac-Velocity-Flat-Anymal-D-v0",
    "Isaac-Velocity-Flat-Anymal-D-Play-v0",
    "Isaac-Velocity-Rough-Anymal-D-v0",
    "Isaac-Velocity-Rough-Anymal-D-Play-v0",
    "Isaac-Velocity-Flat-Cassie-v0",
    "Isaac-Velocity-Flat-Cassie-Play-v0",
    "Isaac-Velocity-Rough-Cassie-v0",
    "Isaac-Velocity-Rough-Cassie-Play-v0",
    "Isaac-Velocity-Flat-G1-v0",
    "Isaac-Velocity-Flat-G1-Play-v0",
    "Isaac-Velocity-Rough-G1-v0",
    "Isaac-Velocity-Rough-G1-Play-v0",
    "Isaac-Velocity-Flat-H1-v0",
    "Isaac-Velocity-Flat-H1-Play-v0",
    "Isaac-Velocity-Rough-H1-v0",
    "Isaac-Velocity-Rough-H1-Play-v0",
    "Isaac-Velocity-Flat-Spot-v0",
    "Isaac-Velocity-Flat-Spot-Play-v0",
    "Isaac-Velocity-Flat-Unitree-A1-v0",
    "Isaac-Velocity-Flat-Unitree-A1-Play-v0",
    "Isaac-Velocity-Rough-Unitree-A1-v0",
    "Isaac-Velocity-Rough-Unitree-A1-Play-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-Play-v0",
    "Isaac-Velocity-Rough-Unitree-Go1-v0",
    "Isaac-Velocity-Rough-Unitree-Go1-Play-v0",
    "Isaac-Velocity-Flat-Unitree-Go2-v0",
    "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    "Isaac-Velocity-Rough-Unitree-Go2-v0",
    "Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    # Manipulation Reach
    "Isaac-Reach-Franka",
    "Isaac-Reach-Franka-Play",
    "Isaac-Reach-UR10",
    "Isaac-Reach-UR10-Play",
    # Manipulation Lift
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Lift-Cube-Franka-Play-v0",
    # Manipulation Cabinet
    "Isaac-Open-Drawer-Franka-v0",
    "Isaac-Open-Drawer-Franka-Play-v0",
    # Dexsuite
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0",
]


def _export_dir(task_name: str) -> str:
    """Return the directory where export.py writes artifacts for *task_name*."""
    train_task = task_name.replace("-Play", "")
    return os.path.join(_REPO_ROOT, ".pretrained_checkpoints", "rsl_rl", train_task, task_name)


def _task_batches(tasks: list[str]) -> list[list[str]]:
    """Split export tasks into batches that share one Kit process."""
    return [tasks[index : index + _EXPORT_BATCH_SIZE] for index in range(0, len(tasks), _EXPORT_BATCH_SIZE)]


def _batch_id(task_names: list[str]) -> str:
    """Return a compact pytest id for a task batch."""
    first = task_names[0].replace("Isaac-", "")
    last = task_names[-1].replace("Isaac-", "")
    return f"{first}__to__{last}"


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
    with open(log_txt_path) as f:
        last_lines = f.readlines()[-50:]
    return f"\n--- leapp log.txt (last 50 lines) ---\n{''.join(last_lines)}"


def _load_export_module():
    """Load the LEAPP RSL-RL export script as an importable module."""
    module = sys.modules.get(_EXPORT_MODULE_NAME)
    if module is not None:
        return module

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
        try:
            with _clean_hydra_argv():
                env_cfg, agent_cfg = resolve_task_config(task_name, args_cli.agent)
            exported = export_module.export_rsl_rl_agent(args_cli, env_cfg, agent_cfg, simulation_app)
        except Exception as exc:
            if "actor_state_dict" in str(exc):
                return
            raise RuntimeError(f"export.py failed for {task_name}: {exc!r}{_leapp_log_tail(export_dir)}") from exc

        # Gracefully skip tasks whose checkpoint isn't published yet
        if not exported:
            return

        assert os.path.isfile(os.path.join(export_dir, f"{task_name}.onnx")), "Missing .onnx export"
        assert os.path.isfile(os.path.join(export_dir, f"{task_name}.yaml")), "Missing .yaml export"
        assert os.path.isfile(os.path.join(export_dir, "log.txt")), "Missing log.txt"

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

    # This flag matches the environment wrapper tests and avoids random stalls
    # when many environments are constructed sequentially in one Kit process.
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


@pytest.mark.parametrize("task_names", _task_batches(TASKS), ids=_batch_id)
def test_export_flow(task_names: list[str]):
    """Run export.py for a task batch and assert the expected artifacts are created."""
    try:
        result = subprocess.run(
            _export_batch_command(task_names),
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_EXPORT_BATCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)
        pytest.fail(
            f"export batch timed out after {_EXPORT_BATCH_TIMEOUT}s for {task_names}.\n"
            f"--- stdout tail ---\n{stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{stderr[-_OUTPUT_TAIL_CHARS:]}"
        )

    if result.returncode != 0:
        pytest.fail(
            f"export batch exited with code {result.returncode} for {task_names}.\n"
            f"--- stdout tail ---\n{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
        )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--export-flow-batch":
    _run_export_batch_entrypoint()
