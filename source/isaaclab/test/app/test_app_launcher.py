# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for :class:`~isaaclab.app.AppLauncher` argument parsing and launch-state resolution.

Kit is never started: the resolution steps run on a bare ``AppLauncher`` instance, and the settings
they publish are read back from the standalone :class:`SettingsManager`.
"""

import argparse
import logging
import sys
from types import SimpleNamespace

import pytest

import isaaclab.app.app_launcher as app_launcher_module
from isaaclab.app import AppLauncher
from isaaclab.app.app_launcher import _sanitize_sys_argv_for_kit
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING

pytestmark = pytest.mark.unit

_KIT_INTENT = {"has_any_visualizers": True, "has_kit_visualizer": True}
_XR_KIT = {"xr": True, "visualizer": ["kit"], "visualizer_explicit": True}
_SPECTATOR_ARG = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=true"
_DEFAULT_CALLBACKS_ARG = "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false"


@pytest.fixture
def launcher(monkeypatch):
    """A bare launcher with the state every resolution step reads, and a clean ``sys.argv``."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.delenv("XR", raising=False)
    monkeypatch.delenv("LIVESTREAM", raising=False)
    monkeypatch.setenv("HEADLESS", "0")
    instance = AppLauncher.__new__(AppLauncher)
    instance._livestream = 0
    instance._xr = False
    instance._headless = False
    instance._enable_cameras = False
    instance._offscreen_render = False
    instance._render_viewport = False
    instance._video_enabled = False
    instance._deterministic_rendering = False
    instance._cli_visualizer_explicit = False
    instance._cli_visualizer_types = []
    instance._cfg_has_kit_visualizer = False
    instance._cfg_has_any_visualizers = False
    instance._python_logging_level = logging.ERROR
    instance._deferred_cuda_device_id = None
    instance.device = "cpu"
    instance.device_id = 0
    return instance


@pytest.fixture
def parser():
    """An argument parser extended with the launcher arguments."""
    instance = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(instance)
    return instance


# ---------------------------------------------------------------------------
# Command-line parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pytest_loaded", "argv", "expected"),
    [
        (True, ["test_script.py", "--capture=no", "-vv"], ["test_script.py"]),
        (True, ["test_script.py", "-m", "not isaacsim_ci", "--keep"], ["test_script.py", "--keep"]),
        (False, ["script.py", "-v"], ["script.py", "-v"]),
    ],
    ids=["verbosity", "marker-pair", "outside-pytest"],
)
def test_sanitize_sys_argv_for_kit(monkeypatch, pytest_loaded, argv, expected):
    """pytest's own options are stripped before Kit parses the command line, and only under pytest."""
    if pytest_loaded:
        monkeypatch.setitem(sys.modules, "pytest", object())
    else:
        monkeypatch.delitem(sys.modules, "pytest", raising=False)

    assert _sanitize_sys_argv_for_kit(argv) == expected


def test_add_app_launcher_args_declares_the_launcher_options(parser):
    """Every launcher option except the removed render flags is available, and those are rejected."""
    for name in AppLauncher._APPLAUNCHER_CFG_INFO.keys() - {"headless", "enable_cameras"}:
        assert f"--{name}" in parser._option_string_actions
    for removed in ("--headless", "--enable_cameras"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])


def test_help_on_parser_with_required_positionals(monkeypatch, capsys):
    """Launcher arguments reach the help output of a script that takes required positionals.

    ``add_app_launcher_args`` probes the parser for name collisions; that probe must not parse the
    command line, which would exit on the missing positionals before the help text is complete.
    """
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "--help"])
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    AppLauncher.add_app_launcher_args(parser)

    assert capsys.readouterr().err == ""
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "app_launcher arguments" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("value", "expected"),
    [("kit,newton_gl,rerun,viser", ["kit", "newton_gl", "rerun", "viser"]), ("none", None), ("None", None)],
)
def test_visualizer_option_parses_csv(parser, value, expected):
    """``--viz`` accepts a comma-separated list, records an explicit choice, and maps ``none`` to ``None``."""
    args = parser.parse_args(["--viz", value])
    assert args.visualizer == expected
    assert args.visualizer_explicit is True


def test_visualizer_option_warns_on_deprecated_alias(parser):
    """The deprecated ``newton`` name resolves to ``newton_gl`` with a warning."""
    with pytest.warns(DeprecationWarning, match="newton_gl"):
        args = parser.parse_args(["--viz", "kit,newton"])
    assert args.visualizer == ["kit", "newton_gl"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("kit, newton_gl", "spaces are not allowed"),
        ("kit,,rerun", "empty visualizer entry"),
        ("none,kit", "'none' cannot"),
    ],
)
def test_visualizer_option_rejects_malformed_values(value, message):
    """Malformed visualizer lists fail with a message naming the problem."""
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        AppLauncher._parse_visualizer_csv(value)


def test_visualizer_csv_does_not_swallow_hydra_overrides():
    """The visualizer option consumes exactly one token so Hydra overrides stay unparsed."""
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)

    args, hydra_args = parser.parse_known_args(["--visualizer", "kit,rerun", "presets=newton_mjwarp", "env.n=10"])

    assert args.visualizer == ["kit", "rerun"]
    assert hydra_args == ["presets=newton_mjwarp", "env.n=10"]


# ---------------------------------------------------------------------------
# Kit arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("launcher_args", "xr", "expected_renderer_args", "expected_physics_gpu"),
    [
        pytest.param(
            {"device": "cuda:1", "multi_gpu": False},
            False,
            ["--/renderer/multiGpu/activeCudaGpus=1,"],
            1,
            id="single-gpu",
        ),
        pytest.param(
            {"device": "cuda:1", "multi_gpu": False, "kit_args": "--/renderer/multiGpu/activeCudaGpus=3,"},
            False,
            ["--/renderer/multiGpu/activeCudaGpus=3,"],
            1,
            id="explicit-kit-arg",
        ),
        pytest.param({"device": "cuda:1"}, False, [], 1, id="multi-gpu"),
        # XR pins the renderer to the simulation GPU so the CloudXR compositor imports the right swapchain
        pytest.param(
            {"device": "cuda:1", "device_explicit": True},
            True,
            ["--/renderer/multiGpu/activeCudaGpus=1,"],
            1,
            id="xr-cuda",
        ),
        # ...but a bare --xr resolves to cpu, where there is no simulation GPU to align to
        pytest.param({}, True, [], 0, id="xr-default-cpu"),
    ],
)
def test_devices_selected_by_cuda_index(launcher, launcher_args, xr, expected_renderer_args, expected_physics_gpu):
    """Physics uses the CUDA index; the renderer is pinned only for single-GPU runs and CUDA-backed XR."""
    launcher._xr = xr
    launcher._resolve_device_settings(launcher_args)
    launcher._resolve_kit_args(launcher_args)

    renderer_args = [arg for arg in launcher._kit_args if arg.startswith("--/renderer/multiGpu/activeCudaGpus=")]
    assert renderer_args == expected_renderer_args
    assert launcher_args["physics_gpu"] == expected_physics_gpu
    assert "active_gpu" not in launcher_args


@pytest.mark.parametrize(
    ("launcher_state", "expected_enabled"),
    [
        pytest.param({}, False, id="headless-training"),
        pytest.param({"_cfg_has_kit_visualizer": True}, True, id="config-kit-visualizer"),
        pytest.param({"_cli_visualizer_explicit": True, "_cli_visualizer_types": ["kit"]}, True, id="cli-kit"),
        pytest.param({"_render_viewport": True}, True, id="viewport"),
        pytest.param(
            {"_cfg_has_kit_visualizer": True, "_cli_visualizer_explicit": True, "_cli_visualizer_types": ["rerun"]},
            False,
            id="cli-non-kit-overrides-config",
        ),
        pytest.param({"_video_enabled": True}, True, id="video"),
        pytest.param({"_livestream": 1}, True, id="livestream"),
        pytest.param({"_xr": True}, True, id="xr"),
    ],
)
def test_spectator_view_follows_visual_output_intent(launcher, launcher_state, expected_enabled):
    """All-partitions spectator mode is requested only for launches that produce visual output."""
    for name, value in launcher_state.items():
        setattr(launcher, name, value)

    launcher._resolve_kit_args({})

    assert (_SPECTATOR_ARG in launcher._kit_args) is expected_enabled


@pytest.mark.parametrize(
    ("env_value", "argv", "kit_args", "expected"),
    [
        pytest.param("0", [], "", ["--/physics/fabricUseGPUInterop=false", _DEFAULT_CALLBACKS_ARG], id="interop-off"),
        pytest.param("1", [], "", ["--/physics/fabricUseGPUInterop=true", _DEFAULT_CALLBACKS_ARG], id="interop-on"),
        pytest.param(None, [], "", [_DEFAULT_CALLBACKS_ARG], id="defaults"),
        # explicit settings on the command line or in --kit_args take precedence over the defaults
        pytest.param("0", ["--/physics/fabricUseGPUInterop=true"], "", [_DEFAULT_CALLBACKS_ARG], id="argv-wins"),
        pytest.param(
            "0",
            [],
            "--/physics/fabricUseGPUInterop=true",
            ["--/physics/fabricUseGPUInterop=true", _DEFAULT_CALLBACKS_ARG],
            id="kit-args-win",
        ),
        pytest.param(
            None,
            [],
            "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=true",
            ["--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=true"],
            id="explicit-callbacks",
        ),
    ],
)
def test_default_kit_settings_yield_to_explicit_ones(launcher, monkeypatch, env_value, argv, kit_args, expected):
    """Isaac Lab's default Kit settings are appended unless the same setting was given explicitly."""
    if env_value is None:
        monkeypatch.delenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, raising=False)
    else:
        monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, env_value)
    sys.argv.extend(argv)

    launcher._resolve_kit_args({"kit_args": kit_args})

    assert launcher._kit_args == expected
    assert sys.argv[1:] == [*argv, *expected]


def test_explicit_spectator_setting_overrides_visualizer_default(launcher):
    """An explicit spectator setting is preserved even when a Kit visualizer would enable it."""
    launcher._cli_visualizer_explicit = True
    launcher._cli_visualizer_types = ["kit"]
    explicit_arg = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=false"

    launcher._resolve_kit_args({"kit_args": explicit_arg})

    assert launcher._kit_args == [explicit_arg, _DEFAULT_CALLBACKS_ARG]


def test_fabric_gpu_interop_env_rejects_invalid_value(launcher, monkeypatch):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, "false")

    with pytest.raises(ValueError, match="Expected: 0 or 1"):
        launcher._resolve_kit_args({})


# ---------------------------------------------------------------------------
# Visualizer, XR, and headless resolution
# ---------------------------------------------------------------------------


def _resolve_headless(launcher, launcher_args: dict, livestream: int = 0) -> bool:
    """Run the visualizer, XR, and headless resolution steps in launch order."""
    launcher._livestream = livestream
    launcher._resolve_visualizer_settings(launcher_args)
    launcher._resolve_xr_settings(launcher_args)
    launcher._resolve_headless_settings(launcher_args, livestream_arg=-1, livestream_env=0)
    return launcher._headless


@pytest.mark.parametrize(
    ("launcher_args", "expected_headless", "expected_types", "expected_disable_all"),
    [
        # explicit CLI selection decides: only kit implies a window
        (
            {"visualizer": ["kit", "newton_gl"], "visualizer_explicit": True, "visualizer_intent": _KIT_INTENT},
            False,
            ["kit", "newton_gl"],
            False,
        ),
        (
            {"visualizer": ["rerun"], "visualizer_explicit": True, "visualizer_intent": _KIT_INTENT},
            True,
            ["rerun"],
            False,
        ),
        ({"visualizer": ["kit"]}, False, ["kit"], False),
        ({"visualizer": None, "visualizer_explicit": True, "visualizer_intent": _KIT_INTENT}, True, [], True),
        ({"visualizer": ["none"], "visualizer_explicit": True, "visualizer_intent": _KIT_INTENT}, True, [], True),
        # without a CLI selection the config intent decides, and only a Kit visualizer opens a window
        ({"visualizer_intent": _KIT_INTENT}, False, [], False),
        ({"visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": False}}, True, [], False),
        ({}, True, [], False),
        # the deprecated headless flag still wins
        ({"headless": True, "headless_explicit": True, "visualizer_intent": _KIT_INTENT}, True, [], False),
    ],
    ids=[
        "cli-kit-and-newton",
        "cli-rerun",
        "kwargs-kit",
        "cli-none-value",
        "cli-none-name",
        "cfg-kit",
        "cfg-non-kit",
        "nothing",
        "headless-flag",
    ],
)
def test_headless_follows_visualizer_selection(
    launcher, launcher_args, expected_headless, expected_types, expected_disable_all
):
    """A Kit viewport only exists when the launcher is told to create it, by CLI or by config intent."""
    assert _resolve_headless(launcher, launcher_args) is expected_headless
    assert launcher._cli_visualizer_types == expected_types
    assert launcher._cli_visualizer_disable_all is expected_disable_all


@pytest.mark.parametrize(
    ("launcher_args", "headless_env", "livestream", "expected_headless", "expected_has_window"),
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
def test_xr_and_livestream_headless_resolution(
    launcher, monkeypatch, launcher_args, headless_env, livestream, expected_headless, expected_has_window
):
    """XR runs headless unless a windowed visualizer is explicit; ``has_window`` is not the negation of headless."""
    monkeypatch.setenv("HEADLESS", str(headless_env))
    args = {"visualizer_intent": _KIT_INTENT, **launcher_args}

    assert _resolve_headless(launcher, args, livestream) is expected_headless
    assert launcher.has_window is expected_has_window


@pytest.mark.parametrize(
    ("launcher_args", "message"),
    [
        ({"visualizer": ["none", "kit"], "visualizer_explicit": True}, "'none' cannot be combined"),
        ({"visualizer": ["holodeck"]}, "Invalid value"),
        ({"visualizer_intent": {"has_any_visualizers": "yes"}}, "visualizer_intent"),
        ({"visualizer_intent": {"has_any_visualizers": False, "has_kit_visualizer": True}}, "visualizer_intent"),
    ],
)
def test_resolve_visualizer_settings_rejects_invalid_input(launcher, launcher_args, message):
    with pytest.raises(ValueError, match=message):
        launcher._resolve_visualizer_settings(launcher_args)


def test_resolve_visualizer_settings_warns_on_deprecated_alias(launcher):
    with pytest.warns(DeprecationWarning, match="newton_gl"):
        launcher._resolve_visualizer_settings({"visualizer": ["newton"]})
    assert launcher._cli_visualizer_types == ["newton_gl"]


# ---------------------------------------------------------------------------
# Experience files, devices, and forwarded config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headless", "livestream", "enable_cameras", "xr", "expected"),
    [
        (True, 0, False, False, "isaaclab.python.headless.kit"),
        (False, 0, False, False, "isaaclab.python.kit"),
        (True, 1, False, False, "isaaclab.python.kit"),
        (True, 0, True, False, "isaaclab.python.headless.rendering.kit"),
        (False, 0, True, False, "isaaclab.python.rendering.kit"),
        (True, 0, True, True, "isaaclab.python.xr.openxr.headless.kit"),
        (False, 0, False, True, "isaaclab.python.xr.openxr.kit"),
    ],
)
def test_default_experience_follows_launch_state(
    launcher, monkeypatch, tmp_path, headless, livestream, enable_cameras, xr, expected
):
    """Without ``--experience`` the Isaac Lab experience is chosen from headless, camera, and XR state."""
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    monkeypatch.setattr(app_launcher_module, "ISAACLAB_ROOT", tmp_path)
    # livestreaming inspects the chosen experience file, so it has to exist
    (tmp_path / "apps").mkdir()
    (tmp_path / "apps" / expected).write_text('[dependencies]\n"isaaclab.python" = {}\n', encoding="utf-8")
    launcher.is_isaac_sim_version_5 = lambda: False
    launcher._headless, launcher._livestream, launcher._enable_cameras, launcher._xr = (
        headless,
        livestream,
        enable_cameras,
        xr,
    )

    launcher._resolve_experience_file({})

    assert launcher._sim_experience_file == str(tmp_path / "apps" / expected)


@pytest.mark.parametrize(
    ("dependencies", "livestream", "rejected"),
    [
        ('"isaacsim.exp.full" = {}', 2, True),
        ('"isaaclab.python" = {}\n"isaacsim.exp.full" = {}', 2, True),
        ('"isaacsim.exp.full" = {}', 0, False),
        ('"isaaclab.python" = {}', 2, False),
    ],
)
def test_explicit_experience_depending_on_isaacsim_full_is_rejected_with_livestream(
    launcher, monkeypatch, tmp_path, dependencies, livestream, rejected
):
    """``isaacsim.exp.full`` hangs or breaks PhysX tensor views under livestreaming, so it is refused there."""
    experience = tmp_path / "custom.kit"
    experience.write_text(f"[dependencies]\n{dependencies}\n", encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher.is_isaac_sim_version_5 = lambda: False
    launcher._livestream = livestream

    if rejected:
        with pytest.raises(ValueError, match="depends on 'isaacsim.exp.full'"):
            launcher._resolve_experience_file({"experience": str(experience)})
    else:
        launcher._resolve_experience_file({"experience": str(experience)})
        assert launcher._sim_experience_file == str(experience)


def test_relative_experience_is_resolved_against_the_apps_folders(launcher, monkeypatch, tmp_path):
    """A relative experience is looked up in Isaac Sim's apps folder first, then Isaac Lab's, else it fails."""
    (tmp_path / "kit_apps").mkdir()
    (tmp_path / "apps").mkdir()
    (tmp_path / "apps" / "custom.kit").touch()
    monkeypatch.setenv("EXP_PATH", str(tmp_path / "kit_apps"))
    monkeypatch.setattr(app_launcher_module, "ISAACLAB_ROOT", tmp_path)
    launcher.is_isaac_sim_version_5 = lambda: False

    launcher._resolve_experience_file({"experience": "custom.kit"})
    assert launcher._sim_experience_file == str(tmp_path / "apps" / "custom.kit")

    with pytest.raises(FileNotFoundError, match="missing.kit"):
        launcher._resolve_experience_file({"experience": "missing.kit"})


def test_deferred_cuda_device_is_set_after_kit(launcher, monkeypatch):
    """The post-Kit device hook synchronizes both CUDA runtimes through the shared helper."""
    devices = []
    monkeypatch.setattr(app_launcher_module, "set_cuda_device", devices.append)
    launcher._deferred_cuda_device_id = 2

    launcher._set_deferred_cuda_device()

    assert devices == [2]


@pytest.mark.parametrize("torch_loaded", [True, False])
def test_preloaded_torch_cuda_is_initialized_before_kit(launcher, monkeypatch, torch_loaded):
    """PyTorch's queued CUDA checks are drained before Kit changes the visible devices, without importing it."""
    init_calls = []
    if torch_loaded:
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: False, init=lambda: init_calls.append(True))
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    launcher._deferred_cuda_device_id = 0

    launcher._initialize_preloaded_torch_cuda()

    assert init_calls == [True] * torch_loaded
    assert ("torch" in sys.modules) is torch_loaded


def test_config_resolution_forwards_simulation_app_settings(launcher, monkeypatch):
    """Only the keys SimulationApp understands are forwarded to it."""
    monkeypatch.setattr(launcher, "_resolve_experience_file", lambda _launcher_args: None)

    launcher._config_resolution({"headless": True, "device": "cpu", "limit_cpu_threads": 1, "custom": 3})

    assert launcher._sim_app_config == {"headless": True, "hide_ui": True, "physics_gpu": 0, "limit_cpu_threads": 1}


@pytest.mark.parametrize(
    ("config", "message"),
    [({"device": "cpu"}, "already has the field"), ({"headless": True, "width": "wide"}, "already has the field")],
)
def test_check_argparser_config_params_rejects_conflicts(config, message):
    with pytest.raises(ValueError, match=message):
        AppLauncher._check_argparser_config_params(config)


def test_check_argparser_config_params_rejects_wrong_simulation_app_types():
    with pytest.raises(ValueError, match="Invalid value type"):
        AppLauncher._check_argparser_config_params({"width": "wide"})
    AppLauncher._check_argparser_config_params({"width": 640, "task": "Cartpole"})


# ---------------------------------------------------------------------------
# Published settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headless", "livestream", "xr", "deterministic", "expected_has_gui", "expected_xr_auto_start"),
    [
        pytest.param(False, 0, False, True, True, False, id="local-window"),
        pytest.param(True, 0, False, False, False, False, id="headless"),
        pytest.param(True, 1, False, False, True, False, id="livestream"),
        pytest.param(True, 0, True, False, True, True, id="xr"),
        # XR with a window: the operator starts the session, so it must not auto-start.
        pytest.param(False, 0, True, False, True, False, id="xr-windowed"),
        # ...but a windowless XR run must, however that windowless state was reached.
        pytest.param(True, 1, True, False, True, True, id="xr-livestream"),
    ],
)
def test_load_extensions_publishes_launch_state(
    launcher, monkeypatch, headless, livestream, xr, deterministic, expected_has_gui, expected_xr_auto_start
):
    """The GUI, XR auto-start, and deterministic-rendering state is published for SimulationContext to read."""
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", lambda: None)
    monkeypatch.setattr(app_launcher_module, "apply_python_logging_level", lambda _level: None)
    launcher._headless, launcher._livestream, launcher._xr = headless, livestream, xr
    launcher._deterministic_rendering = deterministic

    launcher._load_extensions()

    settings = get_settings_manager()
    assert settings.get("/isaaclab/has_gui") is expected_has_gui
    assert settings.get("/isaaclab/xr/auto_start") is expected_xr_auto_start
    assert settings.get("/isaaclab/render/deterministic") is deterministic
    assert AppLauncher.has_gui() is expected_has_gui


def test_set_visualizer_settings_stores_values(launcher):
    launcher._set_visualizer_settings({"visualizer": ["viser", "rerun"], "max_visible_envs": 0})

    settings = get_settings_manager()
    assert settings.get("/isaaclab/visualizer/types") == "viser rerun"
    assert settings.get("/isaaclab/visualizer/explicit") is False
    assert settings.get("/isaaclab/visualizer/disable_all") is False
    assert settings.get("/isaaclab/visualizer/max_visible_envs") == 0


def test_set_visualizer_settings_rejects_negative_max_visible_envs(launcher, monkeypatch):
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: pytest.fail("must validate first"))

    with pytest.raises(ValueError, match="Invalid value for --max_visible_envs: -5"):
        launcher._set_visualizer_settings({"visualizer": ["viser"], "max_visible_envs": -5})


def test_set_visualizer_settings_suppresses_settings_manager_errors(launcher, monkeypatch):
    def _raise():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _raise)
    launcher._set_visualizer_settings({"visualizer": ["viser"], "max_visible_envs": 3})


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_is_available_reflects_simulation_app_presence(monkeypatch):
    """Launchability follows the Isaac Sim import, and constructing without it reports that clearly."""
    monkeypatch.setattr(app_launcher_module, "SimulationApp", None)
    assert AppLauncher.is_available() is False
    with pytest.raises(ImportError, match="requires the full Isaac Sim runtime"):
        AppLauncher()

    monkeypatch.setattr(app_launcher_module, "SimulationApp", object())
    assert AppLauncher.is_available() is True
