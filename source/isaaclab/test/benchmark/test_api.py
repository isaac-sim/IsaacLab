# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import isaaclab
import isaaclab.benchmark as benchmark
from isaaclab.benchmark import (
    BenchmarkLauncherConfig,
    BenchmarkOutputConfig,
    BenchmarkPlayRequest,
    BenchmarkResult,
    BenchmarkRuntimeRequest,
    BenchmarkTrainingRequest,
    PlayBundle,
    RuntimeBundle,
    StartupBundle,
    TrainingBundle,
    dispatch,
)
from isaaclab.benchmark.entrypoints import startup
from isaaclab.benchmark.entrypoints.backends.rl_games.registry import register_scoped_rl_games_environment


def test_training_request_builds_complete_cli() -> None:
    request = BenchmarkTrainingRequest(
        backend="rsl_rl",
        task="Isaac-Cartpole-Direct",
        agent="custom_agent",
        num_envs=64,
        seed=7,
        max_iterations=10,
        warmup_steps=12,
        ray_proc_id=4,
        video=True,
        video_length=100,
        video_interval=500,
        ema_alpha=0.2,
        keep_series=False,
        check_success=True,
        success_threshold=0.8,
        success_window=5,
        measure_synchronized_step_breakdown=True,
        presets=("newton_mjwarp", "newton_renderer"),
        backend_args=("--resume",),
        hydra_args=("env.scene.num_envs=32",),
        output=BenchmarkOutputConfig(path=Path("/tmp/results"), formatters=("schema", "summary")),
        launcher=BenchmarkLauncherConfig(
            device="cuda:1",
            visualizers=(),
            animation_recording=True,
            animation_recording_start_time=1.5,
            animation_recording_stop_time=3.0,
            kit_args="--ext-folder=/tmp/extensions",
        ),
    )

    assert dispatch._request_argv(request) == [
        "--task",
        "Isaac-Cartpole-Direct",
        "--num_envs",
        "64",
        "--seed",
        "7",
        "--agent",
        "custom_agent",
        "--max_iterations",
        "10",
        "--warmup_steps",
        "12",
        "--ray-proc-id",
        "4",
        "--video",
        "--video_length",
        "100",
        "--video_interval",
        "500",
        "--ema_alpha",
        "0.2",
        "--no_series",
        "--check_success",
        "--success_threshold",
        "0.8",
        "--success_window",
        "5",
        "--measure_sync_step",
        "--output_path",
        "/tmp/results",
        "--benchmark_formatter",
        "schema,summary",
        "--device",
        "cuda:1",
        "--visualizer",
        "none",
        "--anim_recording_enabled",
        "--anim_recording_start_time",
        "1.5",
        "--anim_recording_stop_time",
        "3.0",
        "--kit_args=--ext-folder=/tmp/extensions",
        "--resume",
        "presets=newton_mjwarp,newton_renderer",
        "env.scene.num_envs=32",
    ]


def test_runtime_request_uses_runtime_defaults() -> None:
    request = BenchmarkRuntimeRequest(task="Isaac-Cartpole-Direct")

    assert dispatch._request_argv(request) == [
        "--task",
        "Isaac-Cartpole-Direct",
        "--num_frames",
        "1000",
        "--warmup_frames",
        "50",
        "--output_path",
        ".",
        "--benchmark_formatter",
        "schema",
    ]


@pytest.mark.parametrize("backend", ["rsl_rl", "rl_games", "skrl", "sb3"])
def test_play_request_uses_backend_warmup_frames_argument(backend: str, monkeypatch) -> None:
    request = BenchmarkPlayRequest(backend=backend, task="Isaac-Cartpole-Direct", warmup_frames=12)

    argv = dispatch._request_argv(request)
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])
    entrypoint = importlib.import_module(dispatch._workflow_module("play", backend))
    args, remaining_args = entrypoint._parse_args(argv)

    assert argv[argv.index("--warmup_frames") + 1] == "12"
    assert args.warmup_frames == 12
    assert remaining_args == []


@pytest.mark.parametrize("workflow", ["runtime", "startup"])
def test_task_workflow_help_does_not_require_task(workflow: str, monkeypatch, capsys) -> None:
    entrypoint = importlib.import_module(f"isaaclab.benchmark.entrypoints.{workflow}")
    monkeypatch.setattr(sys, "argv", ["isaaclab", "benchmark", workflow, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint._parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "--task" in capsys.readouterr().out


def test_request_dispatches_to_library_module(monkeypatch) -> None:
    request = BenchmarkRuntimeRequest(task="Isaac-Cartpole-Direct", num_frames=1, warmup_frames=0)
    expected = BenchmarkResult(bundle=object(), output_paths=(Path("result.json"),))
    received: list[str] = []
    caller_argv = ["caller.py", "--unrelated"]
    expected_argv = dispatch._request_argv(request)
    monkeypatch.setattr(dispatch.sys, "argv", caller_argv)

    def fake_import_module(module_name: str):
        assert module_name == "isaaclab.benchmark.entrypoints.runtime"

        def fake_run(argv: list[str]):
            assert argv == expected_argv
            assert dispatch.sys.argv == [caller_argv[0], *expected_argv]
            received.extend(argv)
            dispatch.sys.argv = ["benchmark.py", "hydra.override=true"]
            return expected

        return SimpleNamespace(run=fake_run)

    monkeypatch.setattr(dispatch.importlib, "import_module", fake_import_module)

    assert dispatch.run_benchmark_request(request) is expected
    assert dispatch.sys.argv is caller_argv
    assert received[:6] == [
        "--task",
        "Isaac-Cartpole-Direct",
        "--num_frames",
        "1",
        "--warmup_frames",
        "0",
    ]


def test_success_check_rejects_unsupported_backend() -> None:
    request = BenchmarkTrainingRequest(backend="sb3", task="Isaac-Cartpole-Direct", check_success=True)

    with pytest.raises(ValueError, match="check_success is not supported"):
        dispatch._request_argv(request)


def test_request_translates_parser_exit_to_value_error(monkeypatch) -> None:
    request = BenchmarkRuntimeRequest(task="Isaac-Cartpole-Direct")
    module = SimpleNamespace(run=lambda argv: (_ for _ in ()).throw(SystemExit(2)))
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda module_name: module)

    with pytest.raises(ValueError, match="rejected the benchmark request"):
        dispatch.run_benchmark_request(request)


def test_kit_args_fuses_option_like_value() -> None:
    assert dispatch._fuse_kit_args(["--kit_args", "--ext-folder=/tmp/extensions", "--device", "cpu"]) == [
        "--kit_args=--ext-folder=/tmp/extensions",
        "--device",
        "cpu",
    ]


def test_startup_source_prefixes_include_installed_package_root() -> None:
    package_root = Path(isaaclab.__file__).resolve().parent

    assert package_root in map(Path, startup._isaaclab_source_prefixes())


def test_rl_games_registry_cleanup_restores_previous_values() -> None:
    vecenv = SimpleNamespace(
        vecenv_config={"IsaacRlgWrapper": "previous-wrapper"},
        register=lambda name, value: vecenv.vecenv_config.__setitem__(name, value),
    )
    env_configurations = SimpleNamespace(
        configurations={"rlgpu": "previous-environment"},
        register=lambda name, value: env_configurations.configurations.__setitem__(name, value),
    )
    environment = object()
    factory = object()

    restore = register_scoped_rl_games_environment(vecenv, env_configurations, environment, factory)

    assert vecenv.vecenv_config["IsaacRlgWrapper"] is factory
    assert env_configurations.configurations["rlgpu"]["env_creator"]() is environment

    restore()

    assert vecenv.vecenv_config["IsaacRlgWrapper"] == "previous-wrapper"
    assert env_configurations.configurations["rlgpu"] == "previous-environment"


def test_rl_games_registry_cleanup_removes_new_values() -> None:
    vecenv = SimpleNamespace(vecenv_config={})
    vecenv.register = lambda name, value: vecenv.vecenv_config.__setitem__(name, value)
    env_configurations = SimpleNamespace(configurations={})
    env_configurations.register = lambda name, value: env_configurations.configurations.__setitem__(name, value)

    restore = register_scoped_rl_games_environment(vecenv, env_configurations, object(), object())
    restore()

    assert "IsaacRlgWrapper" not in vecenv.vecenv_config
    assert "rlgpu" not in env_configurations.configurations


def test_cli_dispatch_fuses_option_like_kit_args(monkeypatch) -> None:
    received: list[str] = []
    module = SimpleNamespace(run=lambda argv: received.extend(argv))
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda module_name: module)

    assert dispatch.run_benchmark_cli(["runtime", "--kit_args", "--ext-folder=/tmp/extensions"]) == 0
    assert received == ["--kit_args=--ext-folder=/tmp/extensions"]


def test_backend_entrypoints_register_environment_cleanup_before_wrapping() -> None:
    """All RL backend entrypoints compile and register cleanup before wrapping."""
    backends_root = Path(__file__).parents[2] / "isaaclab" / "benchmark" / "entrypoints" / "backends"
    entrypoints = [
        backends_root / backend / f"benchmark_{mode}_{backend}.py"
        for backend in ("rl_games", "rsl_rl", "sb3", "skrl")
        for mode in ("train", "play")
    ]

    for entrypoint in entrypoints:
        source = entrypoint.read_text()
        compile(source, str(entrypoint), "exec")
        creation_index = max(source.find("gym.make("), source.find("_common.create_isaaclab_env("))
        cleanup_index = source.find("cleanup.callback(lambda: env.close())", creation_index)
        wrapper_indices = [
            source.find(wrapper, creation_index) for wrapper in ("_common.wrap_record_video(env", "VecEnvWrapper(env")
        ]
        first_wrapper_index = min(index for index in wrapper_indices if index >= 0)

        assert creation_index >= 0, entrypoint
        assert cleanup_index > creation_index, entrypoint
        assert first_wrapper_index > cleanup_index, entrypoint


def test_late_bound_environment_cleanup_closes_base_when_wrapping_fails() -> None:
    """The cleanup callback retains the base environment when assignment fails."""

    class _Environment:
        closed = False

        def close(self) -> None:
            self.closed = True

    def fail_to_wrap(environment: _Environment) -> _Environment:
        raise RuntimeError("wrapper construction failed")

    base_environment = _Environment()
    with pytest.raises(RuntimeError, match="wrapper construction failed"):
        with contextlib.ExitStack() as cleanup:
            env = base_environment
            cleanup.callback(lambda: env.close())
            env = fail_to_wrap(env)

    assert base_environment.closed


@pytest.mark.parametrize(
    ("helper", "bundle_type"),
    [
        (benchmark.run_runtime_benchmark, RuntimeBundle),
        (benchmark.run_startup_benchmark, StartupBundle),
        (benchmark.run_training_benchmark, TrainingBundle),
        (benchmark.run_play_benchmark, PlayBundle),
    ],
)
def test_workflow_helpers_return_bundle_specific_results(helper, bundle_type) -> None:
    """Workflow helpers expose their concrete bundle type to static type checkers."""
    assert get_type_hints(helper)["return"] == BenchmarkResult[bundle_type]


def test_scoped_backend_state_restores_values_after_exception() -> None:
    """Backend-global settings are restored when an in-process benchmark fails."""
    import torch

    from isaaclab_rl.entrypoints.common import preserve_attribute, scoped_torch_backend_flags

    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    holder = SimpleNamespace(value="original")

    with pytest.raises(RuntimeError, match="failed"):
        with (
            scoped_torch_backend_flags(
                cuda_matmul_allow_tf32=True,
                cudnn_allow_tf32=True,
                cudnn_deterministic=False,
                cudnn_benchmark=False,
            ),
            preserve_attribute(holder, "value"),
        ):
            holder.value = "temporary"
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
            assert torch.backends.cudnn.deterministic is False
            assert torch.backends.cudnn.benchmark is False
            assert holder.value == "temporary"
            raise RuntimeError("failed")

    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    ) == original
    assert holder.value == "original"


def test_invalid_rsl_request_preserves_torch_backend_state() -> None:
    """A rejected in-process request does not mutate its caller's Torch configuration."""
    import torch

    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    caller_state = (False, False, True, True)
    try:
        torch.backends.cuda.matmul.allow_tf32 = caller_state[0]
        torch.backends.cudnn.allow_tf32 = caller_state[1]
        torch.backends.cudnn.deterministic = caller_state[2]
        torch.backends.cudnn.benchmark = caller_state[3]
        request = BenchmarkTrainingRequest(backend="rsl_rl", task="Isaac-Cartpole-Direct", warmup_steps=-1)

        with pytest.raises(ValueError, match="rejected the benchmark request"):
            dispatch.run_benchmark_request(request)

        assert (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        ) == caller_state
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original[0]
        torch.backends.cudnn.allow_tf32 = original[1]
        torch.backends.cudnn.deterministic = original[2]
        torch.backends.cudnn.benchmark = original[3]
