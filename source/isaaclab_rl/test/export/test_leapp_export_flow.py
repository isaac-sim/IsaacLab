# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-backend LEAPP export integration tests.

For each backend/task pair:
1. Create an initialized checkpoint in a subprocess.
2. Run the backend export.py CLI in a subprocess.
3. Assert the expected LEAPP artifacts exist.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from leapp_initialized_checkpoints import discover_backend_tasks, resolved_path_file

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEAPP_ROOT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp"
_CHECKPOINT_SCRIPT = Path(__file__).resolve().parent / "leapp_initialized_checkpoints.py"
_SUBPROCESS_TIMEOUT = 600
_OUTPUT_TAIL_CHARS = 5000


@dataclass(frozen=True)
class ExportFlowBackend:
    """Configuration for one RL-library LEAPP export flow."""

    rl_library: str
    export_script: Path
    tasks: tuple[str, ...]

    @property
    def agent_cfg_entry_points(self) -> tuple[str, ...]:
        """Gym registry keys that identify tasks supported by this backend."""
        return (f"{self.rl_library}_cfg_entry_point",)


# These representative tasks all support Newton MJWarp so the export flow uses
# one physics backend consistently without requiring optional runtime wheels.
_EXPORT_BACKENDS = (
    ExportFlowBackend(
        rl_library="rsl_rl",
        export_script=_LEAPP_ROOT / "rsl_rl" / "export.py",
        tasks=(
            "Isaac-Cartpole",
            "Isaac-Reach-Franka",
            "Isaac-Reach-UR10",
            "Isaac-Lift-KukaAllegro",
            "Isaac-Open-Drawer-Franka",
            "Isaac-Reorient-Franka",
            "Isaac-Velocity-Flat-AnymalD",
            "Isaac-Velocity-Rough-AnymalD",
            "Isaac-Humanoid",
            "Isaac-Ant",
        ),
    ),
    ExportFlowBackend(
        rl_library="rl_games",
        export_script=_LEAPP_ROOT / "rl_games" / "export.py",
        tasks=("Isaac-Cartpole",),
    ),
    ExportFlowBackend(
        rl_library="skrl",
        export_script=_LEAPP_ROOT / "skrl" / "export.py",
        tasks=("Isaac-Cartpole",),
    ),
    ExportFlowBackend(
        rl_library="sb3",
        export_script=_LEAPP_ROOT / "sb3" / "export.py",
        tasks=("Isaac-Cartpole",),
    ),
)

# Selected as the untyped ``presets=`` form rather than the typed ``physics=``
# form: tasks vary in where they declare the preset, and only some declare it on
# a ``PhysicsCfg`` (which is what ``physics=`` requires). Tasks that swap a whole
# ``SimulationCfg`` instead are only reachable through ``presets=``.
_SIM_PRESET = "newton_mjwarp"


def _ensure_text(output: str | bytes | None) -> str:
    """Return subprocess output as text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _run_checked(cmd: Sequence[str], *, label: str) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and fail the test on timeout or non-zero exit."""
    try:
        result = subprocess.run(
            list(cmd),
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)
        pytest.fail(
            f"{label} timed out after {_SUBPROCESS_TIMEOUT}s.\n"
            f"--- stdout tail ---\n{stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{stderr[-_OUTPUT_TAIL_CHARS:]}"
        )

    if result.returncode != 0:
        pytest.fail(
            f"{label} exited with code {result.returncode}.\n"
            f"--- stdout tail ---\n{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
        )
    return result


def _create_checkpoint(backend: ExportFlowBackend, task_name: str, checkpoint_root: Path) -> Path:
    """Create an initialized checkpoint and return its path."""
    result = _run_checked(
        [
            sys.executable,
            str(_CHECKPOINT_SCRIPT),
            "--backend",
            backend.rl_library,
            "--task",
            task_name,
            "--checkpoint_root",
            str(checkpoint_root),
            "--preset",
            _SIM_PRESET,
        ],
        label=f"checkpoint creation for {backend.rl_library}/{task_name}",
    )

    path_file = resolved_path_file(checkpoint_root)
    if not path_file.is_file():
        pytest.fail(
            f"checkpoint creation for {backend.rl_library}/{task_name} did not record a checkpoint path.\n"
            f"--- stdout tail ---\n{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
        )

    checkpoint_path = Path(path_file.read_text().strip())
    assert checkpoint_path.is_file(), f"Checkpoint was not written: {checkpoint_path}"
    return checkpoint_path


def _run_export(backend: ExportFlowBackend, task_name: str, checkpoint_path: Path, export_root: Path) -> None:
    """Run the backend export.py CLI against *checkpoint_path*."""
    _run_checked(
        [
            sys.executable,
            str(backend.export_script),
            "--task",
            task_name,
            "--checkpoint",
            str(checkpoint_path),
            "--export_save_path",
            str(export_root),
            "--disable_graph_visualization",
            f"presets={_SIM_PRESET}",
        ],
        label=f"export.py for {backend.rl_library}/{task_name}",
    )


def _assert_leapp_artifacts(export_root: Path, task_name: str) -> None:
    """Assert the expected LEAPP export artifacts exist."""
    export_dir = export_root / task_name
    assert (export_dir / f"{task_name}.onnx").is_file(), f"Missing .onnx export in {export_dir}"
    assert (export_dir / f"{task_name}.yaml").is_file(), f"Missing .yaml export in {export_dir}"
    assert (export_dir / "log.txt").is_file(), f"Missing log.txt in {export_dir}"


def _tasks_for_backend(backend: ExportFlowBackend) -> tuple[str, ...]:
    """Return manager-based tasks to export for *backend*.

    Direct workflow tasks are excluded. They are covered separately by
    ``test_rsl_rl_direct_export_flow.py``.
    """
    tasks = backend.tasks if backend.tasks else discover_backend_tasks(backend.agent_cfg_entry_points)
    return tuple(task for task in tasks if "Direct" not in task)


def _export_cases() -> list[pytest.ParameterSet]:
    """Build one pytest case per backend/task pair."""
    cases: list[pytest.ParameterSet] = []
    for backend in _EXPORT_BACKENDS:
        for task_name in _tasks_for_backend(backend):
            cases.append(pytest.param(backend, task_name, id=f"{backend.rl_library}-{task_name}"))
    return cases


@pytest.mark.parametrize(("backend", "task_name"), _export_cases())
def test_leapp_export_flow(backend: ExportFlowBackend, task_name: str):
    """Create a checkpoint, run export.py, and assert LEAPP artifacts are created."""
    with tempfile.TemporaryDirectory(prefix=f"isaaclab-leapp-{backend.rl_library}-") as tmp_dir:
        checkpoint_root = Path(tmp_dir) / "checkpoint"
        export_root = Path(tmp_dir) / "export"

        checkpoint_path = _create_checkpoint(backend, task_name, checkpoint_root)
        _run_export(backend, task_name, checkpoint_path, export_root)
        _assert_leapp_artifacts(export_root, task_name)
