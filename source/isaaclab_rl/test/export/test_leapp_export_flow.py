# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-backend LEAPP export integration tests.

All initialized checkpoints are created in one Kit subprocess first. Each
backend/task export then runs in its own subprocess against those checkpoints.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from leapp_initialized_checkpoints import discover_backend_tasks, resolved_path_file, task_checkpoint_dir

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEAPP_ROOT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp"
_CHECKPOINT_SCRIPT = Path(__file__).resolve().parent / "leapp_initialized_checkpoints.py"
_SUBPROCESS_TIMEOUT = 600
_CHECKPOINT_BATCH_TIMEOUT = 1200
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


# Representative RSL-RL tasks chosen for export-path diversity, not coverage
# breadth. Prefer one example of each distinct observation / command / action
# pattern over near-duplicates of the same pattern.
#
# Default physics pin is Newton MJWarp. Tasks listed in ``_TASKS_WITHOUT_PRESET``
# declare no ``newton_mjwarp`` selector and run with the task's built-in default.
_EXPORT_BACKENDS = (
    ExportFlowBackend(
        rl_library="rsl_rl",
        export_script=_LEAPP_ROOT / "rsl_rl" / "export.py",
        tasks=(
            # joint-effort locomotion, no commands
            "Isaac-Cartpole",
            "Isaac-Humanoid",
            # pose command + joint-position arm
            "Isaac-Reach-Franka",
            "Isaac-Reach-UR10",
            # OperationalSpaceController is not exportable: the action term and the
            # controller build their buffers with partial-slice assignment, which
            # either emits a raw __setitem__ node that torch.export rejects or drops
            # tracing into a plain buffer. Expect failure until both are functional.
            # "Isaac-Reach-Franka-OSC",
            # binary gripper + cabinet articulation
            "Isaac-Open-Drawer-Franka",
            # classic lift: object_position_in_robot_root_frame + quat math
            "IsaacContrib-Lift-Cube-Franka",
            # history buffers + Object-cloud / contact observations
            "Isaac-Lift-KukaAllegro",
            # manager-based locomotion with contact-rich body observations
            "Isaac-Ant",
            # EMAJointPositionToLimitsAction is not exportable: it blends the new
            # target with ``_prev_applied_actions``, a plain buffer carried across
            # steps that the tracer inlines as a constant, freezing the recurrence
            # at its trace-time value. Expect failure until it is LEAPP state.
            # "Isaac-Reorient-Cube-Allegro",
            # velocity command + projected_gravity; rough adds height_scan
            "Isaac-Velocity-Flat-AnymalD",
            "Isaac-Velocity-Rough-AnymalD",
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

# Untyped ``presets=`` form: tasks vary in where they declare the preset, and only
# some declare it on a ``PhysicsCfg`` (which is what ``physics=`` requires).
_SIM_PRESET = "newton_mjwarp"

# These tasks reject any preset token; export uses their authored default backend.
_TASKS_WITHOUT_PRESET = frozenset({"IsaacContrib-Lift-Cube-Franka"})


def _ensure_text(output: str | bytes | None) -> str:
    """Return subprocess output as text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _run_checked(
    cmd: Sequence[str],
    *,
    label: str,
    timeout: int = _SUBPROCESS_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and fail the test on timeout or non-zero exit."""
    try:
        result = subprocess.run(
            list(cmd),
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)
        pytest.fail(
            f"{label} timed out after {timeout}s.\n"
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


def _tasks_for_backend(backend: ExportFlowBackend) -> tuple[str, ...]:
    """Return manager-based tasks to export for *backend*.

    Direct workflow tasks are excluded. They are covered separately by
    ``test_rsl_rl_direct_export_flow.py``.
    """
    tasks = backend.tasks if backend.tasks else discover_backend_tasks(backend.agent_cfg_entry_points)
    return tuple(task for task in tasks if "Direct" not in task)


def _export_cases() -> list[object]:
    """Build one pytest case per backend/task pair."""
    cases: list[object] = []
    for backend in _EXPORT_BACKENDS:
        for task_name in _tasks_for_backend(backend):
            cases.append(pytest.param(backend, task_name, id=f"{backend.rl_library}-{task_name}"))
    return cases


def _checkpoint_specs() -> list[tuple[str, str]]:
    """Return every backend/task pair that needs an initialized checkpoint."""
    specs: list[tuple[str, str]] = []
    for backend in _EXPORT_BACKENDS:
        for task_name in _tasks_for_backend(backend):
            specs.append((backend.rl_library, task_name))
    return specs


def _preset_for_task(task_name: str) -> str | None:
    """Return the Hydra preset token for *task_name*, or ``None`` if none apply."""
    if task_name in _TASKS_WITHOUT_PRESET:
        return None
    return _SIM_PRESET


def _read_checkpoint_path(checkpoint_root: Path, backend_id: str, task_name: str) -> Path:
    """Load the recorded checkpoint path for one backend/task pair."""
    task_dir = task_checkpoint_dir(checkpoint_root, backend_id, task_name)
    path_file = resolved_path_file(task_dir)
    assert path_file.is_file(), f"Missing checkpoint path file for {backend_id}/{task_name}: {path_file}"
    checkpoint_path = Path(path_file.read_text(encoding="utf-8").strip())
    assert checkpoint_path.is_file(), f"Checkpoint was not written: {checkpoint_path}"
    return checkpoint_path


def _run_export(backend: ExportFlowBackend, task_name: str, checkpoint_path: Path, export_root: Path) -> None:
    """Run the backend export.py CLI against *checkpoint_path*."""
    command = [
        sys.executable,
        str(backend.export_script),
        "--task",
        task_name,
        "--checkpoint",
        str(checkpoint_path),
        "--export_save_path",
        str(export_root),
        "--disable_graph_visualization",
    ]
    preset = _preset_for_task(task_name)
    if preset is not None:
        command.append(f"presets={preset}")
    _run_checked(command, label=f"export.py for {backend.rl_library}/{task_name}")


def _assert_leapp_artifacts(export_root: Path, task_name: str) -> None:
    """Assert the expected LEAPP export artifacts exist."""
    export_dir = export_root / task_name
    assert (export_dir / f"{task_name}.onnx").is_file(), f"Missing .onnx export in {export_dir}"
    assert (export_dir / f"{task_name}.yaml").is_file(), f"Missing .yaml export in {export_dir}"
    assert (export_dir / "log.txt").is_file(), f"Missing log.txt in {export_dir}"


@pytest.fixture(scope="module")
def initialized_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create every export-flow checkpoint in one Kit subprocess."""
    checkpoint_root = tmp_path_factory.mktemp("isaaclab-leapp-checkpoints")
    command = [
        sys.executable,
        str(_CHECKPOINT_SCRIPT),
        "--checkpoint_root",
        str(checkpoint_root),
        "--preset",
        _SIM_PRESET,
    ]
    for backend_id, task_name in _checkpoint_specs():
        preset = _preset_for_task(task_name)
        if preset is None:
            # ``_`` clears the default ``--preset`` for this spec.
            command.extend(("--spec", backend_id, task_name, "_"))
        else:
            command.extend(("--spec", backend_id, task_name))
    _run_checked(
        command,
        label="batched initialized checkpoint creation",
        timeout=_CHECKPOINT_BATCH_TIMEOUT,
    )
    return checkpoint_root


def test_initialized_checkpoints(initialized_checkpoints: Path):
    """Assert the shared Kit subprocess created every expected checkpoint.
    This task is purely for generating mock checkpoints for the export flow.
    """
    missing = [
        f"{backend_id}/{task_name}"
        for backend_id, task_name in _checkpoint_specs()
        if not resolved_path_file(task_checkpoint_dir(initialized_checkpoints, backend_id, task_name)).is_file()
    ]
    assert not missing, f"Missing initialized checkpoints for: {', '.join(missing)}"


@pytest.mark.parametrize(("backend", "task_name"), _export_cases())
def test_leapp_export_flow(backend: ExportFlowBackend, task_name: str, initialized_checkpoints: Path):
    """Export one backend/task pair using the shared initialized checkpoint."""
    checkpoint_path = _read_checkpoint_path(initialized_checkpoints, backend.rl_library, task_name)
    with tempfile.TemporaryDirectory(prefix=f"isaaclab-leapp-export-{backend.rl_library}-") as tmp_dir:
        export_root = Path(tmp_dir) / "export"
        _run_export(backend, task_name, checkpoint_path, export_root)
        _assert_leapp_artifacts(export_root, task_name)
