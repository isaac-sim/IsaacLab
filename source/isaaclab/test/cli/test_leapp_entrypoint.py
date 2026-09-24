# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed LEAPP commands."""

import contextlib
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

import isaaclab.cli as cli

pytestmark = pytest.mark.unit


def test_deploy_module_does_not_import_pxr_before_app_launch():
    """Importing deploy must not load pxr before SimulationApp starts."""
    env = os.environ.copy()
    env.update({"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "Y"})
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import isaaclab.cli.commands.deploy; assert 'pxr' not in sys.modules"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_export_dispatches_in_process():
    """``isaaclab leapp export`` forwards arguments to the library dispatcher."""
    args = ["--rl_library", "rsl_rl", "--task", "Isaac-Cartpole"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "leapp", "export", *args]),
        mock.patch("isaaclab_rl.entrypoints.run_export_cli", return_value=0) as run_export,
    ):
        cli.cli()

    run_export.assert_called_once_with(args)


def test_export_propagates_nonzero_dispatch_status():
    """Export failures become the CLI process status."""
    with mock.patch("isaaclab_rl.entrypoints.run_export_cli", return_value=1):
        with pytest.raises(SystemExit) as exc_info:
            cli.leapp(["export", "--rl_library", "rsl_rl"])

    assert exc_info.value.code == 1


def test_deploy_dispatches_in_process():
    """``isaaclab leapp deploy`` forwards LEAPP deployment arguments."""
    args = [
        "--task",
        "Isaac-Cartpole",
        "--pipeline",
        "exported/Isaac-Cartpole.yaml",
        "physics=newton_mjwarp",
    ]

    deploy = mock.Mock(return_value=0)
    deploy_module = mock.Mock(command_deploy_leapp=deploy)
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "leapp", "deploy", *args]),
        mock.patch.dict(sys.modules, {"isaaclab.cli.commands.deploy": deploy_module}),
    ):
        cli.cli()

    deploy.assert_called_once_with(args)


def test_deploy_resolves_play_mode_and_injects_simulation_controller():
    """Deployment should use play config, forward the seed, and supply simulator capabilities."""
    from isaaclab.cli.commands import deploy as deploy_module

    calls = []

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default=None)

        @staticmethod
        def _fuse_kit_args(argv):
            return argv

    class FakeDeploymentEnv:
        @classmethod
        def simulated_controller_owned_write_handlers(cls):
            return {"gravity_compensation": "simulated"}

        def __init__(self, cfg, pipeline, *, controller_owned_write_handlers):
            calls.append(("env_init", cfg.seed, pipeline, controller_owned_write_handlers))
            self.cfg = cfg
            self.num_envs = 1
            self.step_dt = cfg.sim.dt * cfg.decimation
            self.sim = SimpleNamespace(is_headless_or_exist_active_visualizer=lambda: True)

        def reset(self):
            calls.append("env_reset")

        def step(self):
            calls.append("env_step")

        def close(self):
            calls.append("env_close")

    cfg = SimpleNamespace(seed=None, sim=SimpleNamespace(device="cpu", dt=0.01), decimation=2)
    fake_envs_module = ModuleType("isaaclab.envs")
    fake_envs_module.LeappDeploymentEnv = FakeDeploymentEnv
    resolve = mock.Mock(return_value=(cfg, None))

    @contextlib.contextmanager
    def fake_launch_simulation(launch_cfg, launch_args):
        calls.append(("launch", launch_cfg, launch_args.device))
        yield
        calls.append("launch_close")

    with (
        mock.patch.object(deploy_module, "AppLauncher", FakeAppLauncher),
        mock.patch.object(deploy_module, "launch_simulation", fake_launch_simulation),
        mock.patch.object(deploy_module, "resolve_task_config", resolve),
        mock.patch.dict(sys.modules, {"isaaclab.envs": fake_envs_module}),
    ):
        status = deploy_module.command_deploy_leapp(
            ["--task", "Isaac-Test-v0", "--pipeline", "policy.yaml", "--seed", "29", "--max_steps", "2"]
        )

    assert status == 0
    resolve.assert_called_once_with("Isaac-Test-v0", "", play_mode=True)
    assert calls[0] == ("launch", cfg, None)
    assert calls[1] == (
        "env_init",
        29,
        "policy.yaml",
        {"gravity_compensation": "simulated"},
    )
    assert calls[2:] == ["env_reset", "env_step", "env_step", "env_close", "launch_close"]
