# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-env export integration test with subprocess-side gym re-registration."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import gymnasium as gym
import pytest

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = str(_THIS_FILE.parents[4])
_EXPORT_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "reinforcement_learning", "leapp", "rsl_rl", "export.py")
_THIS_SCRIPT = str(_THIS_FILE)
_TASK_NAME = "Isaac-Velocity-Flat-Anymal-C-Direct-v0"
_PACKAGE_NAME = "_isaaclab_test_tutorial_anymal_c"
_MODULE_NAME = f"{_PACKAGE_NAME}.anymal_c_env"
_CFG_MODULE_NAME = f"{_PACKAGE_NAME}.anymal_c_env_cfg"
_RUNTIME_MODULE_NAME = "_isaaclab_test_tutorial_anymal_c_runtime"
_TUTORIAL_ENV_PATH = Path(_REPO_ROOT) / "scripts" / "tutorials" / "06_deploy" / "anymal_c_env.py"


def _export_command(task_name: str, export_dir: str) -> list[str]:
    """Build a subprocess command that runs this file in helper mode."""
    return [
        sys.executable,
        _THIS_SCRIPT,
        "--task",
        task_name,
        "--use_pretrained_checkpoint",
        "--export_save_path",
        export_dir,
        "--disable_graph_visualization",
        "--headless",
    ]


def _artifact_dir(export_dir: str, task_name: str) -> str:
    """Return the LEAPP artifact directory for the exported task."""
    return os.path.join(export_dir, task_name)


def _load_tutorial_env_class():
    """Load the tutorial env through a synthetic package for relative imports."""
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module.AnymalCEnv

    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault(_PACKAGE_NAME, package)

    cfg_module = importlib.import_module("isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg")
    sys.modules[_CFG_MODULE_NAME] = cfg_module

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _TUTORIAL_ENV_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for tutorial env: {_TUTORIAL_ENV_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module.AnymalCEnv


class _LazyTutorialEnvModule(types.ModuleType):
    """Resolve the tutorial env class only when gym imports the entrypoint."""

    def __getattr__(self, name: str):
        if name != "AnymalCEnv":
            raise AttributeError(name)
        env_class = _load_tutorial_env_class()
        setattr(self, name, env_class)
        return env_class


def _install_lazy_runtime_module() -> str:
    """Install a lazy module so gym can defer tutorial env imports."""
    module = sys.modules.get(_RUNTIME_MODULE_NAME)
    if module is None:
        sys.modules[_RUNTIME_MODULE_NAME] = _LazyTutorialEnvModule(_RUNTIME_MODULE_NAME)
    return _RUNTIME_MODULE_NAME


def _reregister_task(task_name: str) -> None:
    """Override the direct task registration to point at the tutorial env."""
    import isaaclab_tasks.direct.anymal_c  # noqa: F401

    original_spec = gym.spec(task_name)
    original_kwargs = dict(original_spec.kwargs)
    runtime_module_name = _install_lazy_runtime_module()

    gym.registry.pop(task_name, None)
    gym.register(
        id=task_name,
        entry_point=f"{runtime_module_name}:AnymalCEnv",
        disable_env_checker=original_spec.disable_env_checker,
        kwargs=original_kwargs,
    )


def _run_export_subprocess_entrypoint() -> None:
    """Run export.py after re-registering the direct task in-process."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    args, remaining_args = parser.parse_known_args()

    _reregister_task(args.task)
    export_script_dir = os.path.dirname(_EXPORT_SCRIPT)
    sys.argv = [_EXPORT_SCRIPT, "--task", args.task, *remaining_args]
    if export_script_dir not in sys.path:
        sys.path.insert(0, export_script_dir)
    runpy.run_path(_EXPORT_SCRIPT, run_name="__main__")


def _build_failure_context(result: subprocess.CompletedProcess[str], artifact_dir: str) -> str:
    """Return debug context for subprocess and export artifacts."""
    export_dir = os.path.dirname(artifact_dir)
    log_txt_path = os.path.join(artifact_dir, "log.txt")
    leapp_tail = ""
    if os.path.isfile(log_txt_path):
        with open(log_txt_path) as file:
            last_lines = file.readlines()[-50:]
        leapp_tail = f"\n--- leapp log.txt (last 50 lines) ---\n{''.join(last_lines)}"

    try:
        export_dir_contents = sorted(os.listdir(export_dir))
    except FileNotFoundError:
        export_dir_contents = ["<missing directory>"]

    try:
        artifact_dir_contents = sorted(os.listdir(artifact_dir))
    except FileNotFoundError:
        artifact_dir_contents = ["<missing directory>"]

    return (
        f"--- export_dir ---\n{export_dir}\n"
        f"--- export_dir contents ---\n{export_dir_contents}\n"
        f"--- artifact_dir ---\n{artifact_dir}\n"
        f"--- artifact_dir contents ---\n{artifact_dir_contents}\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
        f"{leapp_tail}"
    )


def test_direct_env_export_flow():
    """Run export.py against the tutorial direct env and assert artifacts are created."""
    export_dir = tempfile.mkdtemp(prefix="isaaclab-direct-export-")
    artifact_dir = _artifact_dir(export_dir, _TASK_NAME)
    shutil.rmtree(artifact_dir, ignore_errors=True)

    result = subprocess.run(
        _export_command(_TASK_NAME, export_dir),
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=6000,
    )

    if "pre-trained checkpoint is currently unavailable" in result.stdout:
        pytest.skip(f"No pretrained checkpoint available for {_TASK_NAME}")

    if result.returncode != 0:
        pytest.fail(f"export.py exited with code {result.returncode}.\n{_build_failure_context(result, artifact_dir)}")

    onnx_path = os.path.join(artifact_dir, f"{_TASK_NAME}.onnx")
    yaml_path = os.path.join(artifact_dir, f"{_TASK_NAME}.yaml")
    log_path = os.path.join(artifact_dir, "log.txt")

    if not os.path.isfile(onnx_path):
        pytest.fail(f"Missing .onnx export at {onnx_path}.\n{_build_failure_context(result, artifact_dir)}")
    if not os.path.isfile(yaml_path):
        pytest.fail(f"Missing .yaml export at {yaml_path}.\n{_build_failure_context(result, artifact_dir)}")
    if not os.path.isfile(log_path):
        pytest.fail(f"Missing log.txt at {log_path}.\n{_build_failure_context(result, artifact_dir)}")


if __name__ == "__main__":
    _run_export_subprocess_entrypoint()
