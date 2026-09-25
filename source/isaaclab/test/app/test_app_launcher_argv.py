# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for how :class:`AppLauncher` resolves arguments, environment variables, and Kit settings."""

import argparse
import logging
import sys
from types import SimpleNamespace

import pytest

import isaaclab.app as app_module
import isaaclab.app.app_launcher as app_launcher_module
import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils as utils_module
from isaaclab.app.app_launcher import AppLauncher, _sanitize_sys_argv_for_kit
from isaaclab.app.sim_launcher import Scan, _ensure_livestream_kit_visualizer, _get_kit_runtime_sources
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING


def test_sanitize_sys_argv_removes_trailing_pytest_verbosity(monkeypatch):
    """Remove a pytest verbosity flag even when it is the final argument."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "--capture=no", "-vv"])

    assert result == ["test_script.py"]


def test_sanitize_sys_argv_preserves_user_verbosity_outside_pytest(monkeypatch):
    """Preserve application verbosity flags when pytest is not running."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    argv = ["script.py", "-v"]

    result = _sanitize_sys_argv_for_kit(argv)

    assert result is argv


def test_sanitize_sys_argv_removes_pytest_marker_pair(monkeypatch):
    """Remove a pytest marker option together with its expression."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "-m", "not isaacsim_ci", "--keep"])

    assert result == ["test_script.py", "--keep"]


def _resolve_devices_and_kit_args(launcher_args: dict, monkeypatch, *, xr: bool = False) -> tuple[dict, list[str]]:
    """Resolve device settings and Kit arguments without constructing an ``AppLauncher``.

    ``_resolve_kit_args`` extends ``sys.argv``, so the caller's argv is isolated.
    """
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher.device_id = 0
    launcher._deferred_cuda_device_id = None
    launcher._xr = xr
    AppLauncher._resolve_device_settings(launcher, launcher_args)
    AppLauncher._resolve_kit_args(launcher, launcher_args)
    return launcher_args, launcher._kit_args


@pytest.mark.parametrize(
    ("launcher_args", "expected_renderer_args"),
    [
        pytest.param(
            {"device": "cuda:1", "multi_gpu": False},
            ["--/renderer/multiGpu/activeCudaGpus=1,"],
            id="single-gpu",
        ),
        pytest.param(
            {"device": "cuda:1", "multi_gpu": False, "kit_args": "--/renderer/multiGpu/activeCudaGpus=3,"},
            ["--/renderer/multiGpu/activeCudaGpus=3,"],
            id="explicit-kit-arg",
        ),
        pytest.param({"device": "cuda:1"}, [], id="multi-gpu"),
    ],
)
def test_devices_selected_by_cuda_index(launcher_args, expected_renderer_args, monkeypatch):
    """Select physics and single-GPU rendering devices by CUDA index."""
    args, kit_args = _resolve_devices_and_kit_args(launcher_args, monkeypatch)

    renderer_args = [arg for arg in kit_args if arg.startswith("--/renderer/multiGpu/activeCudaGpus=")]
    assert renderer_args == expected_renderer_args
    assert args["physics_gpu"] == 1
    assert "active_gpu" not in args


@pytest.mark.parametrize(
    ("launcher_args", "expected_renderer_args", "expected_physics_gpu"),
    [
        pytest.param(
            {"device": "cuda:1", "device_explicit": True},
            ["--/renderer/multiGpu/activeCudaGpus=1,"],
            1,
            id="xr-explicit-cuda-device",
        ),
        pytest.param({}, [], 0, id="xr-default-cpu-device"),
    ],
)
def test_xr_pins_the_renderer_only_for_a_cuda_device(
    launcher_args, expected_renderer_args, expected_physics_gpu, monkeypatch
):
    """Pin the renderer under XR when a CUDA device is selected, and only then.

    XR streams one stereo swapchain that the CloudXR compositor imports, so the renderer and
    the compositor have to agree on a device. A bare ``--xr`` resolves to ``cpu``, where there
    is no simulation GPU to align to, so Kit keeps its own choice -- forcing CUDA 0 there would
    break hosts whose display is not attached to GPU 0.
    """
    args, kit_args = _resolve_devices_and_kit_args(launcher_args, monkeypatch, xr=True)

    renderer_args = [arg for arg in kit_args if arg.startswith("--/renderer/multiGpu/activeCudaGpus=")]
    assert renderer_args == expected_renderer_args
    assert args["physics_gpu"] == expected_physics_gpu


@pytest.mark.parametrize(
    ("launcher_state", "expected_enabled"),
    [
        pytest.param({}, False, id="headless-training"),
        pytest.param({"_cfg_has_kit_visualizer": True}, True, id="config-kit-visualizer"),
        pytest.param(
            {"_cli_visualizer_explicit": True, "_cli_visualizer_types": ["kit"]},
            True,
            id="cli-kit-visualizer",
        ),
        pytest.param({"_render_viewport": True}, True, id="viewport"),
        pytest.param(
            {
                "_cfg_has_kit_visualizer": True,
                "_cli_visualizer_explicit": True,
                "_cli_visualizer_types": ["rerun"],
            },
            False,
            id="cli-non-kit-overrides-config",
        ),
        pytest.param({"_video_enabled": True}, True, id="video"),
        pytest.param({"_livestream": 1}, True, id="livestream"),
        pytest.param({"_xr": True}, True, id="xr"),
    ],
)
def test_spectator_view_follows_visual_output_intent(launcher_state, expected_enabled, monkeypatch):
    """Enable all-partitions spectator mode only for visual output paths."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._cli_visualizer_explicit = False
    launcher._cli_visualizer_types = []
    launcher._cfg_has_kit_visualizer = False
    launcher._render_viewport = False
    launcher._video_enabled = False
    launcher._livestream = 0
    launcher._xr = False
    launcher.device = "cpu"
    for name, value in launcher_state.items():
        setattr(launcher, name, value)

    launcher._resolve_kit_args({})

    spectator_arg = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=true"
    assert (spectator_arg in launcher._kit_args) is expected_enabled


def test_explicit_spectator_setting_overrides_visualizer_default(monkeypatch):
    """Preserve an explicit Kit setting when a Kit visualizer is requested."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._cli_visualizer_explicit = True
    launcher._cli_visualizer_types = ["kit"]
    launcher._xr = False
    launcher.device = "cpu"
    explicit_arg = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=false"

    launcher._resolve_kit_args({"kit_args": explicit_arg})

    assert explicit_arg in launcher._kit_args
    assert f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=true" not in launcher._kit_args


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["--kit_args", "--foo=/bar"], ["--kit_args=--foo=/bar"], id="space-separated-option-like"),
        pytest.param(["--kit_args=--foo=/bar"], ["--kit_args=--foo=/bar"], id="equals-attached"),
        # a token containing a space cannot be mistaken for an option by argparse
        pytest.param(["--kit_args", "--foo=/bar --baz=1"], ["--kit_args", "--foo=/bar --baz=1"], id="value-with-space"),
        pytest.param(["--kit_args", "foo.txt"], ["--kit_args", "foo.txt"], id="non-option-value"),
        # argparse should still report the missing value normally
        pytest.param(
            ["--task", "Isaac-Cartpole-Direct", "--kit_args"],
            ["--task", "Isaac-Cartpole-Direct", "--kit_args"],
            id="trailing",
        ),
        pytest.param(
            ["--kit_args", "--foo=/a", "--task", "X", "--kit_args", "--bar=/b"],
            ["--kit_args=--foo=/a", "--task", "X", "--kit_args=--bar=/b"],
            id="multiple-occurrences",
        ),
        pytest.param(
            ["--task", "X", "--kit_args", "--foo=/bar", "--num_envs", "16"],
            ["--task", "X", "--kit_args=--foo=/bar", "--num_envs", "16"],
            id="surrounding-tokens",
        ),
    ],
)
def test_fuse_kit_args(argv: list[str], expected: list[str]):
    """Fuse only ``--kit_args`` pairs whose value argparse would mistake for an option."""
    assert AppLauncher._fuse_kit_args(argv) == expected


def test_add_app_launcher_args_registers_every_launcher_option():
    """Every launcher config key except the removed render flags is a command-line option."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    for name in AppLauncher._APPLAUNCHER_CFG_INFO.keys() - {"headless", "enable_cameras"}:
        assert parser._option_string_actions[f"--{name}"]


def test_visualizer_alias_parsing():
    """Test that --viz alias maps to visualizer values, resolving the deprecated ``newton`` name."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    with pytest.warns(DeprecationWarning, match="--viz 'newton' is deprecated"):
        args = parser.parse_args(["--viz", "kit,newton"])
    assert args.visualizer == ["kit", "newton_gl"]
    assert args.visualizer_explicit is True


@pytest.mark.parametrize("deprecated_arg", ["--headless", "--enable_cameras"])
def test_deprecated_render_flags_are_rejected(deprecated_arg: str):
    """Test that removed render flags are rejected by the parser."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([deprecated_arg])


def test_help_on_parser_with_required_positionals(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Launcher arguments reach the help output of a script that takes required positionals.

    ``add_app_launcher_args`` probes the command line to check for name collisions. That probe
    exits when a required argument is missing, which is the case for every tool script invoked
    with ``--help``, so it must not take the parser down before the arguments are added.
    """
    monkeypatch.setattr("sys.argv", ["convert_urdf.py", "--help"])
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    AppLauncher.add_app_launcher_args(parser)

    # the probe's own usage line must not leak to stderr ahead of the real help output
    assert capsys.readouterr().err == ""
    assert "--visualizer" in parser._option_string_actions
    assert "--device" in parser._option_string_actions

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "app_launcher arguments" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["none", "None"])
def test_visualizer_none_parsing(value: str):
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--viz", value])
    assert args.visualizer is None
    assert args.visualizer_explicit is True


@pytest.mark.parametrize(
    ("launcher_args", "livestream_env"),
    [
        pytest.param({"livestream": 1}, "0", id="kwarg"),
        pytest.param(["--livestream", "1"], "0", id="namespace"),
        pytest.param({}, "1", id="env"),
    ],
)
def test_livestream_request_resolves_headless_livestream_launch(
    launcher_args: dict | list[str], livestream_env: str, monkeypatch: pytest.MonkeyPatch
):
    """Each livestream channel enables headless WebRTC streaming; a keyword argument supersedes ``LIVESTREAM``."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("LIVESTREAM", livestream_env)
    monkeypatch.setenv("HEADLESS", "0")
    monkeypatch.delenv("XR", raising=False)
    if isinstance(launcher_args, list):
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        launcher_args = vars(parser.parse_args(launcher_args))
    launcher_args["device"] = "cpu"
    launcher = AppLauncher.__new__(AppLauncher)
    monkeypatch.setattr(launcher, "_resolve_experience_file", lambda _launcher_args: None)

    launcher._config_resolution(launcher_args)

    assert launcher._livestream == 1
    assert launcher._headless is True
    assert "omni.kit.livestream.app" in launcher._livestream_args


def _resolve_kit_args(
    monkeypatch: pytest.MonkeyPatch, launcher_args: dict, argv: list[str] | None = None
) -> tuple[list[str], list[str]]:
    monkeypatch.setattr(sys, "argv", ["pytest", *(argv or [])])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._xr = False
    launcher._resolve_kit_args(launcher_args)
    return sys.argv[1:], launcher._kit_args


@pytest.mark.parametrize("env_value, expected", [("0", "false"), ("1", "true")])
def test_fabric_gpu_interop_env_adds_override(monkeypatch: pytest.MonkeyPatch, env_value: str, expected: str):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, env_value)

    kit_args, _ = _resolve_kit_args(monkeypatch, {})

    assert kit_args == [
        f"--/physics/fabricUseGPUInterop={expected}",
        "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false",
    ]


def test_fabric_gpu_interop_env_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, "false")

    with pytest.raises(ValueError, match="Expected: 0 or 1"):
        _resolve_kit_args(monkeypatch, {})


def test_explicit_kit_setting_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, "0")
    explicit_arg = "--/physics/fabricUseGPUInterop=true"

    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {}, [explicit_arg])

    assert kit_args == [
        explicit_arg,
        "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false",
    ]
    assert resolved_args == ["--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false"]


def test_explicit_kit_args_setting_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, "0")
    explicit_arg = "--/physics/fabricUseGPUInterop=true"

    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {"kit_args": explicit_arg})

    assert kit_args == [
        explicit_arg,
        "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false",
    ]
    assert resolved_args == [
        explicit_arg,
        "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false",
    ]


def test_simulation_manager_default_callbacks_disabled(monkeypatch: pytest.MonkeyPatch):
    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {})

    assert kit_args == ["--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false"]
    assert resolved_args == ["--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false"]


def test_explicit_simulation_manager_callback_setting_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    explicit_arg = "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=true"

    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {"kit_args": explicit_arg})

    assert kit_args == [explicit_arg]
    assert resolved_args == [explicit_arg]


def test_make_physics_cfg_builds_core_vbd():
    from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

    physics_cfg = sim_launcher.make_physics_cfg("newton_vbd")

    assert isinstance(physics_cfg, NewtonCfg)
    assert isinstance(physics_cfg.solver_cfg, VBDSolverCfg)


def test_livestream_injects_kit_visualizer_when_missing():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=False)

    _ensure_livestream_kit_visualizer(args)

    assert args.visualizer == ["kit"]


def test_livestream_rejects_disabled_visualizers():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=True)

    with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
        _ensure_livestream_kit_visualizer(args)


def test_explicit_experience_requires_isaac_sim_runtime():
    """An explicit Kit experience must override a kitless physics configuration."""
    scan = Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=False,
        has_kit_physics=False,
        has_kitless_physics=True,
        has_ovphysx_physics=False,
        needs_kit=False,
    )
    args = argparse.Namespace(experience="isaaclab.python.kit", visualizer=None)

    assert _get_kit_runtime_sources(scan, args)


_XR_KIT = {"xr": True, "visualizer": ["kit"], "visualizer_explicit": True}


@pytest.mark.parametrize(
    "launcher_args, headless_env, livestream, expected_headless, expected_has_window",
    [
        # XR with no CLI visualizer: headless even though the task config asks for a window.
        ({"xr": True}, 0, 0, True, False),
        # An explicit '--viz kit' is the only way to get a viewport alongside XR.
        (_XR_KIT, 0, 0, False, True),
        ({"xr": True, "visualizer": ["none"], "visualizer_explicit": True}, 0, 0, True, False),
        # ...but an explicit '--viz kit' cannot override HEADLESS=1.
        (_XR_KIT, 1, 0, True, False),
        # Livestreaming forces headless yet still presents a window to the remote client.
        (_XR_KIT, 0, 1, True, True),
        ({}, 0, 1, True, True),
        # Without XR or livestream the task config still decides.
        ({}, 0, 0, False, True),
    ],
)
def test_xr_without_explicit_windowed_visualizer_forces_headless(
    launcher_args: dict,
    headless_env: int,
    livestream: int,
    expected_headless: bool,
    expected_has_window: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that enabling XR runs headless unless a windowed visualizer is explicitly requested.

    A task config declaring a windowed visualizer must not leave ``--xr`` opening a window nobody
    asked for, and ``HEADLESS=1`` / livestreaming must keep forcing headless even when one was
    explicitly requested. Resolution is exercised directly to avoid launching Isaac Sim; what the
    resolved state then publishes is asserted by
    :func:`test_load_extensions_publishes_has_gui_setting`.

    ``has_window`` is asserted alongside because it is deliberately *not* the negation of
    headless: livestreaming is headless but windowed.
    """
    monkeypatch.setenv("HEADLESS", str(headless_env))
    # XR is read from the environment too, and is routinely exported by people working on this
    # feature -- pin it so the parametrization is what decides.
    monkeypatch.delenv("XR", raising=False)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = livestream
    args = {
        "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        **launcher_args,
    }

    launcher._resolve_visualizer_settings(args)
    launcher._resolve_xr_settings(args)
    launcher._resolve_headless_settings(args, livestream_arg=-1, livestream_env=0)

    assert launcher._headless is expected_headless
    assert launcher.has_window is expected_has_window


def test_launch_simulation_preserves_failure_exit_code(monkeypatch: pytest.MonkeyPatch):
    close_args = {}

    class _FakeApp:
        def close(self, *, exit_code: int = 0) -> None:
            close_args["exit_code"] = exit_code

    class _FakeAppLauncher:
        def __init__(self, _launcher_args):
            self.app = _FakeApp()

    scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=False,
        has_kit_physics=True,
        has_kitless_physics=False,
        has_ovphysx_physics=False,
        needs_kit=True,
    )
    monkeypatch.setattr(sim_launcher, "scan", lambda cfg, physics: scan)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(app_module, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(utils_module, "has_kit", lambda: False)

    with pytest.raises(RuntimeError, match="sentinel"):
        with sim_launcher.launch_simulation(object(), argparse.Namespace()):
            raise RuntimeError("sentinel")

    assert close_args == {"exit_code": 1}


def test_launch_simulation_auto_enables_kit_camera_without_launcher_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LIVESTREAM", raising=False)
    received_args = {}

    class _FakeApp:
        def close(self) -> None:
            pass

    class _FakeAppLauncher:
        def __init__(self, launcher_args):
            received_args.update(launcher_args)
            self.app = _FakeApp()

    scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=True,
        has_kit_physics=False,
        has_kitless_physics=False,
        has_ovphysx_physics=False,
        needs_kit=True,
    )

    def _scan(_cfg, launcher_args):
        assert launcher_args == {}
        return scan

    monkeypatch.setattr(sim_launcher, "scan", _scan)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(app_module, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(utils_module, "has_kit", lambda: False)

    with sim_launcher.launch_simulation(object()):
        pass

    assert received_args["enable_cameras"] is True


def test_deferred_cuda_device_synchronizes_torch_and_warp(monkeypatch: pytest.MonkeyPatch):
    """The post-Kit device hook must synchronize both CUDA runtimes."""
    devices = []
    monkeypatch.setattr(app_launcher_module, "set_cuda_device", devices.append)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deferred_cuda_device_id = 2

    launcher._set_deferred_cuda_device()

    assert devices == [2]


def test_preloaded_torch_cuda_is_initialized_before_kit(monkeypatch: pytest.MonkeyPatch):
    """Drain PyTorch's queued CUDA checks before Kit can change visible device indices."""
    init_calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda: init_calls.append(True),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deferred_cuda_device_id = 0

    launcher._initialize_preloaded_torch_cuda()

    assert init_calls == [True]


def test_cuda_initialization_remains_deferred_when_torch_is_not_loaded(monkeypatch: pytest.MonkeyPatch):
    """Do not import PyTorch solely to initialize CUDA before Kit."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deferred_cuda_device_id = 0

    launcher._initialize_preloaded_torch_cuda()

    assert "torch" not in sys.modules


def test_limit_cpu_threads_forwarded_to_simulation_app(monkeypatch: pytest.MonkeyPatch):
    """A SimulationApp thread limit must survive AppLauncher config resolution."""
    monkeypatch.setenv("HEADLESS", "0")
    monkeypatch.setenv("LIVESTREAM", "0")
    monkeypatch.setenv("XR", "0")

    launcher = AppLauncher.__new__(AppLauncher)
    monkeypatch.setattr(launcher, "_resolve_experience_file", lambda _launcher_args: None)

    launcher._config_resolution({"headless": True, "device": "cpu", "limit_cpu_threads": 1})

    assert launcher._sim_app_config["limit_cpu_threads"] == 1


class _DummySettings:
    def __init__(self):
        self.values = {}

    def set_string(self, path: str, value: str) -> None:
        self.values[path] = value

    def set_int(self, path: str, value: int) -> None:
        self.values[path] = value

    def set_bool(self, path: str, value: bool) -> None:
        self.values[path] = value


@pytest.mark.parametrize("deterministic", [True, False])
def test_load_extensions_publishes_deterministic_setting(monkeypatch: pytest.MonkeyPatch, deterministic: bool):
    """Publish ``/isaaclab/render/deterministic`` from ``_load_extensions``."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deterministic_rendering = deterministic
    launcher._python_logging_level = logging.ERROR
    launcher._headless = True
    launcher._livestream = 0
    launcher._enable_cameras = False
    launcher._offscreen_render = False
    launcher._render_viewport = False
    launcher._xr = False
    launcher._video_enabled = False

    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", lambda: None)
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)
    monkeypatch.setattr(app_launcher_module, "apply_python_logging_level", lambda _level: None)

    launcher._load_extensions()

    assert settings.values["/isaaclab/render/deterministic"] is deterministic


@pytest.mark.parametrize(
    ("headless", "livestream", "xr", "expected_has_gui", "expected_xr_auto_start"),
    [
        pytest.param(False, 0, False, True, False, id="local-window"),
        pytest.param(True, 0, False, False, False, id="headless"),
        pytest.param(True, 1, False, True, False, id="livestream"),
        pytest.param(True, 0, True, True, True, id="xr"),
        # XR with a window: the operator starts the session, so it must not auto-start.
        pytest.param(False, 0, True, True, False, id="xr-windowed"),
        # ...but a windowless XR run must, however that windowless state was reached.
        pytest.param(True, 1, True, True, True, id="xr-livestream"),
    ],
)
def test_load_extensions_publishes_has_gui_setting(
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
    livestream: int,
    xr: bool,
    expected_has_gui: bool,
    expected_xr_auto_start: bool,
):
    """Publish the GUI and XR auto-start state consumed by SimulationContext and the teleop stack.

    Asserted at the publication site rather than by recomputing the expression, so that changing
    what is published fails here instead of silently agreeing with a restated formula.
    """
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deterministic_rendering = False
    launcher._python_logging_level = logging.ERROR
    launcher._headless = headless
    launcher._livestream = livestream
    launcher._enable_cameras = False
    launcher._offscreen_render = False
    launcher._render_viewport = False
    launcher._xr = xr
    launcher._video_enabled = False

    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", lambda: None)
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)
    monkeypatch.setattr(app_launcher_module, "apply_python_logging_level", lambda _level: None)

    launcher._load_extensions()

    assert settings.values["/isaaclab/has_gui"] is expected_has_gui
    assert settings.values["/isaaclab/xr/auto_start"] is expected_xr_auto_start


def test_set_visualizer_settings_stores_values(monkeypatch: pytest.MonkeyPatch):
    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser", "rerun"], "max_visible_envs": 0})

    assert settings.values == {
        "/isaaclab/visualizer/types": "viser rerun",
        "/isaaclab/visualizer/explicit": False,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": 0,
    }


def test_set_visualizer_settings_rejects_negative_max_visible_envs(
    monkeypatch: pytest.MonkeyPatch,
):
    def _unexpected_settings_manager():
        raise AssertionError("settings manager should not be queried for invalid values")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _unexpected_settings_manager)

    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="Invalid value for --max_visible_envs: -5"):
        launcher._set_visualizer_settings({"visualizer": ["viser"], "max_visible_envs": -5})


def test_set_visualizer_settings_suppresses_settings_manager_errors(monkeypatch: pytest.MonkeyPatch):
    def _raise_settings_error():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _raise_settings_error)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser"], "max_visible_envs": 3})


def test_parse_visualizer_csv_rejects_spaces_between_entries():
    with pytest.raises(argparse.ArgumentTypeError, match="spaces are not allowed"):
        app_launcher_module.AppLauncher._parse_visualizer_csv("kit, newton_gl")


def test_resolve_visualizer_settings_rejects_none_with_others():
    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="'none' cannot be combined"):
        launcher._resolve_visualizer_settings(
            {"visualizer": ["none", "kit"], "visualizer_explicit": True},
        )


def test_visualizer_csv_does_not_swallow_hydra_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    app_launcher_module.AppLauncher.add_app_launcher_args(parser)

    args, hydra_args = parser.parse_known_args(
        ["--visualizer", "kit,newton_gl,rerun,viser", "presets=newton_mjwarp", "env.episode_length=10"]
    )

    assert args.visualizer == ["kit", "newton_gl", "rerun", "viser"]
    assert hydra_args == ["presets=newton_mjwarp", "env.episode_length=10"]


def _resolve_headless_for_case(monkeypatch: pytest.MonkeyPatch, launcher_args: dict) -> tuple[bool, AppLauncher]:
    monkeypatch.setenv("HEADLESS", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = 0
    launcher._resolve_visualizer_settings(launcher_args)
    launcher._resolve_headless_settings(launcher_args, livestream_arg=-1, livestream_env=0)
    return launcher._headless, launcher


def test_matrix_cli_kit_newton_gl_with_custom_kit_cfg_intent_non_headless(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": ["kit", "newton_gl"],
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is False
    assert launcher._cli_visualizer_types == ["kit", "newton_gl"]


def test_matrix_cli_rerun_with_custom_kit_cfg_intent_headless(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": ["rerun"],
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_types == ["rerun"]


def test_matrix_empty_dict_resolves_headless(monkeypatch: pytest.MonkeyPatch):
    # tools that launch Kit only to reach an extension API pass no visualizer, and must stay headless
    headless, _ = _resolve_headless_for_case(monkeypatch, {})
    assert headless is True


def test_matrix_viz_kit_dict_resolves_windowed(monkeypatch: pytest.MonkeyPatch):
    # a Kit viewport only exists when the launcher is told to create it
    headless, launcher = _resolve_headless_for_case(monkeypatch, {"visualizer": ["kit"]})
    assert headless is False
    assert launcher._cli_visualizer_types == ["kit"]


@pytest.mark.parametrize("visualizer", [None, ["none"]])
def test_matrix_viz_none_disables_all_and_headless(monkeypatch: pytest.MonkeyPatch, visualizer):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": visualizer,
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_disable_all is True
    assert launcher._cli_visualizer_types == []


def test_matrix_headless_flag_deprecated_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "headless": True,
            "headless_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_types == []


def test_no_cli_and_non_kit_cfg_visualizers_defaults_headless(monkeypatch: pytest.MonkeyPatch):
    headless, _ = _resolve_headless_for_case(
        monkeypatch,
        {"visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": False}},
    )
    assert headless is True


def test_invalid_visualizer_intent_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEADLESS", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="visualizer_intent"):
        launcher._resolve_visualizer_settings({"visualizer_intent": {"has_any_visualizers": "yes"}})


def _new_launcher_for_experience_check():
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._enable_cameras = False
    launcher._headless = False
    launcher._xr = False
    launcher._deterministic_rendering = False
    launcher.is_isaac_sim_version_5 = lambda: False
    return launcher


def test_rejects_custom_experience_with_isaacsim_full_dependency_and_livestream(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    experience = tmp_path / "merged.kit"
    experience.write_text('[dependencies]\n"isaaclab.python" = {}\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 2

    with pytest.raises(ValueError, match="depends on 'isaacsim.exp.full'"):
        launcher._resolve_experience_file({"experience": str(experience)})


def test_allows_isaacsim_full_streaming_experience_when_livestream_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    experience = tmp_path / "isaacsim.exp.full.streaming.kit"
    experience.write_text('[dependencies]\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 0

    launcher._resolve_experience_file({"experience": str(experience)})

    assert launcher._sim_experience_file == str(experience)


def test_constructor_reports_missing_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_launcher_module, "SimulationApp", None)

    with pytest.raises(ImportError, match="requires the full Isaac Sim runtime"):
        AppLauncher()


def test_is_available_reflects_simulation_app_presence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_launcher_module, "SimulationApp", None)
    assert AppLauncher.is_available() is False

    monkeypatch.setattr(app_launcher_module, "SimulationApp", object())
    assert AppLauncher.is_available() is True


def test_has_gui_reads_published_setting():
    from isaaclab.app.settings_manager import get_settings_manager

    settings = get_settings_manager()
    original = settings.get("/isaaclab/has_gui")
    try:
        settings.set_bool("/isaaclab/has_gui", True)
        assert AppLauncher.has_gui() is True

        settings.set_bool("/isaaclab/has_gui", False)
        assert AppLauncher.has_gui() is False
    finally:
        settings.set_bool("/isaaclab/has_gui", bool(original))
