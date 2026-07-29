# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for LEAPP export integration tests."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPORT_BATCH_TIMEOUT = 600
OUTPUT_TAIL_CHARS = 5000
PROCESS_FAILURE_PATTERNS = (
    "Traceback (most recent call last):",
    "FileNotFoundError:",
    "[ERROR]",
    "Segmentation fault",
)


@dataclass(frozen=True)
class ExportFlowBackend:
    """Configuration for one RL-library LEAPP export flow."""

    id: str
    rl_library: str
    export_script: Path
    module_name: str
    export_fn_name: str
    cache_marker: str
    tasks: tuple[str, ...]
    include_headless_arg: bool = True
    strip_play_suffix: bool = True
    use_skrl_agent_entry_point: bool = False
    allow_missing_export: bool = False
    stub_isaaclab_on_load: bool = True
    subprocess_env: dict[str, str] | None = None
    skip_output_patterns: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def export_dir(backend: ExportFlowBackend, task_name: str) -> str:
    """Return the directory where export writes artifacts for *task_name*."""
    if backend.strip_play_suffix:
        train_task = task_name.replace("-Play", "")
        return os.path.join(
            REPO_ROOT,
            ".pretrained_checkpoints",
            backend.rl_library,
            train_task,
            task_name,
        )
    return os.path.join(REPO_ROOT, ".pretrained_checkpoints", backend.rl_library, task_name, task_name)


def ensure_text(output: str | bytes | None) -> str:
    """Return subprocess output as text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def leapp_log_tail(export_dir_path: str) -> str:
    """Return the tail of the LEAPP log when it exists."""
    log_txt_path = os.path.join(export_dir_path, "log.txt")
    if not os.path.isfile(log_txt_path):
        return ""
    with open(log_txt_path) as file:
        last_lines = file.readlines()[-50:]
    return f"\n--- leapp log.txt (last 50 lines) ---\n{''.join(last_lines)}"


def fail_on_process_error(result: subprocess.CompletedProcess[str], task_names: Sequence[str]) -> None:
    """Fail when Isaac Sim reports an error but exits with a successful status."""
    output = f"{result.stdout}\n{result.stderr}"
    for pattern in PROCESS_FAILURE_PATTERNS:
        if pattern in output:
            pytest.fail(
                f"export batch reported {pattern!r} for {list(task_names)}.\n"
                f"--- stdout tail ---\n{result.stdout[-OUTPUT_TAIL_CHARS:]}\n"
                f"--- stderr tail ---\n{result.stderr[-OUTPUT_TAIL_CHARS:]}"
            )


@contextlib.contextmanager
def clean_hydra_argv() -> Iterator[None]:
    """Temporarily hide pytest arguments from Hydra config resolution."""
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        yield
    finally:
        sys.argv = original_argv


@contextlib.contextmanager
def stub_isaaclab_cli_imports() -> Iterator[None]:
    """Stub Isaac Lab CLI imports so export scripts can be loaded without Kit."""
    original_modules = {
        name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.app", "isaaclab_tasks", "isaaclab_tasks.utils")
    }
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")
    isaaclab_tasks_module = types.ModuleType("isaaclab_tasks")
    isaaclab_tasks_utils_module = types.ModuleType("isaaclab_tasks.utils")

    class _AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            return None

    setattr(isaaclab_app_module, "AppLauncher", _AppLauncher)
    setattr(isaaclab_tasks_utils_module, "fold_preset_tokens", lambda args: args)
    setattr(
        isaaclab_tasks_utils_module,
        "setup_preset_cli",
        lambda parser, argv=None, **kwargs: parser.parse_known_args(argv),
    )
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module
    sys.modules["isaaclab_tasks"] = isaaclab_tasks_module
    sys.modules["isaaclab_tasks.utils"] = isaaclab_tasks_utils_module
    try:
        yield
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def load_export_module(backend: ExportFlowBackend, *, torch_module=None) -> ModuleType:
    """Load an export script as an importable module."""
    module = sys.modules.get(backend.module_name)
    if module is not None and hasattr(module, backend.cache_marker):
        return module

    sys.modules.pop(backend.module_name, None)
    spec = importlib.util.spec_from_file_location(backend.module_name, backend.export_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {backend.export_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[backend.module_name] = module
    if backend.stub_isaaclab_on_load:
        with stub_isaaclab_cli_imports():
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)

    if torch_module is not None:
        setattr(module, "torch", torch_module)
    return module


def export_argv(backend: ExportFlowBackend, task_name: str) -> list[str]:
    """Build CLI tokens passed to :func:`parse_export_args`."""
    argv = [
        "--task",
        task_name,
        "--use_pretrained_checkpoint",
        "--disable_graph_visualization",
    ]
    if backend.include_headless_arg:
        argv.append("--headless")
    return argv


def assert_leapp_artifacts(export_dir_path: str, task_name: str) -> None:
    """Assert the expected LEAPP export artifacts exist."""
    assert os.path.isfile(os.path.join(export_dir_path, f"{task_name}.onnx")), "Missing .onnx export"
    assert os.path.isfile(os.path.join(export_dir_path, f"{task_name}.yaml")), "Missing .yaml export"
    assert os.path.isfile(os.path.join(export_dir_path, "log.txt")), "Missing log.txt"


def resolve_configs(
    backend: ExportFlowBackend,
    export_module: ModuleType,
    task_name: str,
    args_cli,
    resolve_task_config: Callable,
) -> tuple:
    """Resolve Hydra environment and agent configs for one export task."""
    if backend.use_skrl_agent_entry_point:
        agent_cfg_entry_point, _ = export_module._agent_cfg_entry_point(args_cli)
        return resolve_task_config(task_name, agent_cfg_entry_point)
    return resolve_task_config(task_name, args_cli.agent)


def run_export_task(
    backend: ExportFlowBackend,
    task_name: str,
    *,
    simulation_app,
    sim_utils,
    get_settings_manager,
    resolve_task_config,
    export_module: ModuleType | None = None,
) -> None:
    """Export one task inside an already running Isaac Sim process."""
    export_dir_path = export_dir(backend, task_name)
    module = export_module if export_module is not None else load_export_module(backend)

    try:
        sim_utils.create_new_stage()
        get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)

        args_cli, _ = module.parse_export_args(export_argv(backend, task_name))
        with clean_hydra_argv():
            env_cfg, agent_cfg = resolve_configs(backend, module, task_name, args_cli, resolve_task_config)
        export_fn = getattr(module, backend.export_fn_name)
        exported = export_fn(args_cli, env_cfg, agent_cfg, simulation_app)

        if backend.allow_missing_export and not exported:
            return
        assert exported, "Expected export to produce LEAPP artifacts"
        assert_leapp_artifacts(export_dir_path, task_name)
    except Exception as exc:
        raise RuntimeError(f"export.py failed for {task_name}: {exc!r}{leapp_log_tail(export_dir_path)}") from exc
    finally:
        shutil.rmtree(export_dir_path, ignore_errors=True)


def run_export_batch(backend: ExportFlowBackend, task_names: Sequence[str]) -> None:
    """Run a batch of exports inside a single Isaac Sim process."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.app.settings_manager import get_settings_manager

    from isaaclab_tasks.utils.hydra import resolve_task_config

    export_module = load_export_module(backend)
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    try:
        for task_name in task_names:
            run_export_task(
                backend,
                task_name,
                simulation_app=simulation_app,
                sim_utils=sim_utils,
                get_settings_manager=get_settings_manager,
                resolve_task_config=resolve_task_config,
                export_module=export_module,
            )
    finally:
        simulation_app.close()


def export_batch_command(entry_script: Path, backend: ExportFlowBackend) -> list[str]:
    """Build the subprocess command for one backend export batch."""
    return [sys.executable, str(entry_script), "--export-flow-batch", backend.id, *backend.tasks]


def run_export_flow_subprocess(entry_script: Path, backend: ExportFlowBackend) -> subprocess.CompletedProcess[str]:
    """Run one backend export batch in a subprocess and apply skip rules."""
    env = os.environ.copy()
    if backend.subprocess_env is not None:
        env.update(backend.subprocess_env)

    try:
        result = subprocess.run(
            export_batch_command(entry_script, backend),
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=EXPORT_BATCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = ensure_text(exc.stdout)
        stderr = ensure_text(exc.stderr)
        pytest.fail(
            f"export batch timed out after {EXPORT_BATCH_TIMEOUT}s for {backend.id} {list(backend.tasks)}.\n"
            f"--- stdout tail ---\n{stdout[-OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{stderr[-OUTPUT_TAIL_CHARS:]}"
        )

    output = f"{result.stdout}\n{result.stderr}"
    for needle, reason in backend.skip_output_patterns:
        if needle in output:
            pytest.skip(reason)

    if result.returncode != 0:
        pytest.fail(
            f"export batch exited with code {result.returncode} for {backend.id} {list(backend.tasks)}.\n"
            f"--- stdout tail ---\n{result.stdout[-OUTPUT_TAIL_CHARS:]}\n"
            f"--- stderr tail ---\n{result.stderr[-OUTPUT_TAIL_CHARS:]}"
        )

    fail_on_process_error(result, backend.tasks)
    return result


def export_flow_batch_main(backends: Sequence[ExportFlowBackend], argv: Sequence[str]) -> None:
    """Subprocess entrypoint: ``python test_leapp_export_flow.py --export-flow-batch <backend_id> ...``."""
    if len(argv) < 3:
        raise ValueError("Expected backend id and tasks for --export-flow-batch")

    backend_id = argv[2]
    task_names = argv[3:]
    backend = next((item for item in backends if item.id == backend_id), None)
    if backend is None:
        raise ValueError(f"Unknown export backend: {backend_id}")
    if not task_names:
        task_names = list(backend.tasks)
    run_export_batch(backend, task_names)
