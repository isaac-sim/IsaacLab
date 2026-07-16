# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import isaaclab
import isaaclab.benchmark as benchmark
import isaaclab.benchmark.recorders as benchmark_recorders
from isaaclab.benchmark import (
    BenchmarkLauncherConfig,
    BenchmarkOutputConfig,
    BenchmarkResult,
    BenchmarkRuntimeRequest,
    BenchmarkTrainingRequest,
    RuntimeBundle,
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
        "--measure_synchronized_step_breakdown",
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


def test_request_dispatches_to_library_module(monkeypatch) -> None:
    request = BenchmarkRuntimeRequest(task="Isaac-Cartpole-Direct", num_frames=1, warmup_frames=0)
    expected = BenchmarkResult(bundle=object(), output_paths=(Path("result.json"),))
    received: list[str] = []
    caller_argv = ["caller.py", "--unrelated"]
    monkeypatch.setattr(dispatch.sys, "argv", caller_argv)

    def fake_import_module(module_name: str):
        assert module_name == "isaaclab.benchmark.entrypoints.runtime"

        def fake_run(argv: list[str]):
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


def test_legacy_namespace_warns_and_aliases_public_modules() -> None:
    legacy_modules = (
        "isaaclab.test.benchmark",
        "isaaclab.test.benchmark.schema",
        "isaaclab.test.benchmark.recorders",
        "isaaclab.test.benchmark.recorders.record_cpu_info",
        "isaaclab.test.benchmark.recorders.record_gpu_info",
    )
    for module_name in legacy_modules:
        sys.modules.pop(module_name, None)
    sys.modules.pop("isaaclab.benchmark.recorders.record_cpu_info", None)
    sys.modules.pop("isaaclab.benchmark.recorders.record_gpu_info", None)

    with pytest.warns(
        DeprecationWarning,
        match="removed in Isaac Lab 3.1.*Import isaaclab.benchmark instead",
    ):
        legacy_benchmark = importlib.import_module("isaaclab.test.benchmark")

    assert legacy_benchmark.BenchmarkTrainingRequest is BenchmarkTrainingRequest
    assert legacy_benchmark.RuntimeBundle is RuntimeBundle
    assert legacy_benchmark.__all__ == benchmark.__all__
    assert "warnings" not in legacy_benchmark.__all__
    assert "isaaclab.benchmark.recorders.record_cpu_info" not in sys.modules
    assert "isaaclab.benchmark.recorders.record_gpu_info" not in sys.modules

    legacy_schema = importlib.import_module("isaaclab.test.benchmark.schema")
    assert legacy_schema.RuntimeBundle is RuntimeBundle

    legacy_cpu_info = importlib.import_module("isaaclab.test.benchmark.recorders.record_cpu_info")
    assert "isaaclab.benchmark.recorders.record_cpu_info" not in sys.modules
    public_cpu_info = importlib.import_module("isaaclab.benchmark.recorders.record_cpu_info")
    assert legacy_cpu_info.CPUInfoRecorder is public_cpu_info.CPUInfoRecorder

    legacy_recorders = importlib.import_module("isaaclab.test.benchmark.recorders")
    assert legacy_recorders.CPUInfoRecorder is benchmark_recorders.CPUInfoRecorder

    importlib.import_module("isaaclab.test.benchmark.recorders.record_gpu_info")
    assert "isaaclab.benchmark.recorders.record_gpu_info" not in sys.modules


def test_legacy_wildcard_import_reexports_public_symbols() -> None:
    sys.modules.pop("isaaclab.test.benchmark", None)
    sys.modules.pop("isaaclab.test.benchmark.schema", None)
    namespace: dict[str, object] = {}

    with pytest.warns(DeprecationWarning, match="removed in Isaac Lab 3.1"):
        exec("from isaaclab.test.benchmark.schema import *", namespace)

    assert namespace["RuntimeBundle"] is RuntimeBundle


def test_legacy_submodules_retain_type_stubs() -> None:
    legacy_benchmark = importlib.import_module("isaaclab.test.benchmark")
    legacy_root = Path(legacy_benchmark.__file__).parent

    for submodule in legacy_benchmark._LEGACY_SUBMODULES:
        if submodule == "recorders":
            stub_path = legacy_root / "recorders" / "__init__.pyi"
        else:
            stub_path = legacy_root.joinpath(*submodule.split(".")).with_suffix(".pyi")
        assert stub_path.is_file(), submodule
        assert f"from isaaclab.benchmark.{submodule} import *" in stub_path.read_text()


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
        backends_root / backend / f"{mode}.py"
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
