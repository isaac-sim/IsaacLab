# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatch typed and command-line benchmark requests to workflow implementations."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import BenchmarkLauncherConfig, BenchmarkRequest, BenchmarkResult

_WORKFLOW_MODULES = {
    "runtime": "isaaclab.benchmark.entrypoints.runtime",
    "startup": "isaaclab.benchmark.entrypoints.startup",
}
_RL_WORKFLOW_MODULES = {
    "training": {
        "rl_games": "isaaclab.benchmark.entrypoints.backends.rl_games.benchmark_train_rl_games",
        "rsl_rl": "isaaclab.benchmark.entrypoints.backends.rsl_rl.benchmark_train_rsl_rl",
        "sb3": "isaaclab.benchmark.entrypoints.backends.sb3.benchmark_train_sb3",
        "skrl": "isaaclab.benchmark.entrypoints.backends.skrl.benchmark_train_skrl",
    },
    "play": {
        "rl_games": "isaaclab.benchmark.entrypoints.backends.rl_games.benchmark_play_rl_games",
        "rsl_rl": "isaaclab.benchmark.entrypoints.backends.rsl_rl.benchmark_play_rsl_rl",
        "sb3": "isaaclab.benchmark.entrypoints.backends.sb3.benchmark_play_sb3",
        "skrl": "isaaclab.benchmark.entrypoints.backends.skrl.benchmark_play_skrl",
    },
}


def run_benchmark_request(request: BenchmarkRequest) -> BenchmarkResult:
    """Dispatch a typed benchmark request to its workflow implementation.

    Args:
        request: Typed benchmark request.

    Returns:
        Completed benchmark result.

    Raises:
        ValueError: If the workflow rejects the typed request.
    """
    module_name = _workflow_module(request.workflow, getattr(request, "backend", None))
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    request_argv = _request_argv(request)
    sys.argv = [original_argv[0], *request_argv]
    try:
        try:
            result = module.run(request_argv)
        except SystemExit as exc:
            raise ValueError(f"Benchmark workflow {module_name!r} rejected the benchmark request") from exc
    finally:
        sys.argv = original_argv
    if result is None:
        raise RuntimeError(f"Benchmark workflow {module_name!r} did not return a result")
    return result


def run_benchmark_cli(argv: list[str] | None = None) -> int:
    """Run a runtime, startup, training, or play benchmark from CLI arguments."""
    if argv is None:
        argv = sys.argv[1:]
    argv = _fuse_kit_args(argv)
    parser = argparse.ArgumentParser(description="Run an Isaac Lab benchmark.")
    parser.add_argument("workflow", choices=(*_WORKFLOW_MODULES, *_RL_WORKFLOW_MODULES))
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
    selected = parser.parse_args(argv[:1])
    if selected.workflow in _RL_WORKFLOW_MODULES:
        return _run_rl_cli(selected.workflow, argv[1:])
    module = importlib.import_module(_WORKFLOW_MODULES[selected.workflow])
    module.run(argv[1:])
    return 0


def run_training_cli(argv: list[str] | None = None) -> int:
    """Dispatch training benchmark CLI arguments to an RL backend."""
    return _run_rl_cli("training", argv)


def run_play_cli(argv: list[str] | None = None) -> int:
    """Dispatch play benchmark CLI arguments to an RL backend."""
    return _run_rl_cli("play", argv)


def _run_rl_cli(workflow: str, argv: list[str] | None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = _fuse_kit_args(argv)
    backends = _RL_WORKFLOW_MODULES[workflow]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rl_library", choices=sorted(backends))
    selected, backend_argv = parser.parse_known_args(argv)
    if selected.rl_library is None:
        help_parser = argparse.ArgumentParser(description=f"Run an RL {workflow} benchmark.")
        help_parser.add_argument("--rl_library", choices=sorted(backends), required=True)
        help_parser.add_argument("args", nargs=argparse.REMAINDER)
        help_parser.print_help()
        if "-h" in argv or "--help" in argv:
            return 0
        print(f"\n{workflow}: error: the following argument is required: --rl_library", file=sys.stderr)
        return 2
    module = importlib.import_module(backends[selected.rl_library])
    module.run(backend_argv)
    return 0


def _workflow_module(workflow: str, backend: str | None) -> str:
    if workflow in _WORKFLOW_MODULES:
        return _WORKFLOW_MODULES[workflow]
    if workflow not in _RL_WORKFLOW_MODULES:
        raise ValueError(f"Unsupported benchmark workflow: {workflow!r}")
    if backend not in _RL_WORKFLOW_MODULES[workflow]:
        raise ValueError(f"Unsupported {workflow} benchmark backend: {backend!r}")
    return _RL_WORKFLOW_MODULES[workflow][backend]


def _request_argv(request: BenchmarkRequest) -> list[str]:
    argv: list[str] = ["--task", request.task]
    _append_value(argv, "--num_envs", getattr(request, "num_envs", None))
    _append_value(argv, "--seed", getattr(request, "seed", None))

    if request.workflow == "runtime":
        _append_value(argv, "--num_frames", request.num_frames)
        _append_value(argv, "--warmup_frames", request.warmup_frames)
    elif request.workflow == "startup":
        _append_value(argv, "--top_n", request.top_n)
        _append_value(argv, "--whitelist_config", request.whitelist_config)
    elif request.workflow == "training":
        _append_value(argv, "--agent", request.agent)
        _append_value(argv, "--max_iterations", request.max_iterations)
        _append_value(argv, "--warmup_steps", request.warmup_steps)
        _append_value(argv, "--ray-proc-id", request.ray_proc_id)
        if request.video:
            argv.append("--video")
        _append_value(argv, "--video_length", request.video_length)
        _append_value(argv, "--video_interval", request.video_interval)
        if request.export_io_descriptors:
            argv.append("--export_io_descriptors")
        if request.capture_env_sensors > 0:
            _append_value(argv, "--capture_env_sensors", request.capture_env_sensors)
            _append_value(argv, "--capture_env_sensors_length", request.capture_env_sensors_length)
            _append_value(argv, "--capture_env_sensors_interval", request.capture_env_sensors_interval)
            _append_value(argv, "--capture_env_sensors_format", request.capture_env_sensors_format)
        _append_value(argv, "--ema_alpha", request.ema_alpha)
        if not request.keep_series:
            argv.append("--no_series")
        if request.check_success:
            if request.backend not in ("rl_games", "rsl_rl"):
                raise ValueError(f"check_success is not supported by the {request.backend!r} benchmark backend")
            argv.append("--check_success")
        _append_value(argv, "--success_threshold", request.success_threshold)
        _append_value(argv, "--success_window", request.success_window)
    elif request.workflow == "play":
        _append_value(argv, "--checkpoint", request.checkpoint)
        _append_value(argv, "--agent", request.agent)
        _append_value(argv, "--num_frames", request.num_frames)
        _append_value(argv, "--warmup_frames", request.warmup_frames)

    if getattr(request, "measure_synchronized_step_breakdown", False):
        argv.append("--measure_sync_step")
    argv.extend(_output_argv(request.output))
    argv.extend(_launcher_argv(request.launcher))
    argv.extend(getattr(request, "backend_args", ()))
    if request.presets:
        argv.append(f"presets={','.join(request.presets)}")
    argv.extend(request.hydra_args)
    return _fuse_kit_args(argv)


def _output_argv(output: Any) -> list[str]:
    return ["--output_path", str(output.path), "--benchmark_formatter", ",".join(output.formatters)]


def _launcher_argv(launcher: BenchmarkLauncherConfig) -> list[str]:
    argv: list[str] = []
    _append_value(argv, "--device", launcher.device)
    if launcher.enable_cameras:
        argv.append("--enable_cameras")
    if launcher.visualizers is not None:
        value = ",".join(launcher.visualizers) if launcher.visualizers else "none"
        argv.extend(("--visualizer", value))
    _append_value(argv, "--max_visible_envs", launcher.max_visible_envs)
    _append_value(argv, "--experience", launcher.experience)
    if launcher.animation_recording:
        argv.append("--anim_recording_enabled")
    _append_value(argv, "--anim_recording_start_time", launcher.animation_recording_start_time)
    _append_value(argv, "--anim_recording_stop_time", launcher.animation_recording_stop_time)
    if launcher.deterministic:
        argv.append("--deterministic")
    _append_value(argv, "--kit_args", launcher.kit_args)
    _append_value(argv, "--livestream", launcher.livestream)
    if launcher.xr:
        argv.append("--xr")
    if launcher.verbose:
        argv.append("--verbose")
    if launcher.info:
        argv.append("--info")
    return _fuse_kit_args(argv)


def _append_value(argv: list[str], option: str, value: object | None) -> None:
    if value is not None:
        argv.extend((option, str(value)))


def _fuse_kit_args(argv: list[str]) -> list[str]:
    """Fuse an option-like Kit argument value into argparse-compatible form."""
    fused: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        next_token = argv[index + 1] if index + 1 < len(argv) else None
        if token == "--kit_args" and next_token is not None and next_token.startswith("-") and " " not in next_token:
            fused.append(f"--kit_args={next_token}")
            index += 2
        else:
            fused.append(token)
            index += 1
    return fused
