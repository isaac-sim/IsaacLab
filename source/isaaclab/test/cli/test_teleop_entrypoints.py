# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified teleoperation console entry point."""

import argparse
import asyncio
import contextlib
import importlib
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import tomllib
import torch

import isaaclab.cli as cli

pytestmark = pytest.mark.unit

# Each teleop workflow and the repository-relative script it forwards to.
TELEOP_WORKFLOWS = {
    "run": ("scripts", "environments", "teleoperation", "teleop_se3_agent.py"),
    "record": ("scripts", "tools", "record_demos.py"),
    "replay": ("scripts", "tools", "replay_demos.py"),
}
# Each teleop workflow's isaaclab_teleop entrypoint module and function, which its script wraps.
TELEOP_ENTRYPOINTS = {
    "run": ("isaaclab_teleop.entrypoints.teleop_se3_agent", "run_teleop_se3_agent_cli"),
    "record": ("isaaclab_teleop.entrypoints.record_demos", "run_record_demos_cli"),
    "replay": ("isaaclab_teleop.entrypoints.replay_demos", "run_replay_demos_cli"),
}


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_help_exposes_task_preset_selectors(source_checkout_root: Path, command, script_parts):
    """Every teleop workflow accepts the task preset selectors documented for teleoperation."""
    script = source_checkout_root.joinpath(*script_parts)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=source_checkout_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "preset selection:" in result.stdout
    assert "physics=NAME" in result.stdout
    assert "renderer=NAME" in result.stdout
    assert "presets=NAME[,NAME,...]" in result.stdout


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_dispatches_to_its_workflow(source_checkout_root: Path, command, script_parts, monkeypatch):
    """``isaaclab teleop`` runs each workflow's entrypoint in process with the remaining arguments."""
    args = [command, "--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "--xr"]
    module_name, entrypoint = TELEOP_ENTRYPOINTS[command]
    workflow = mock.Mock(return_value=0)
    monkeypatch.setattr(importlib.import_module(module_name), entrypoint, workflow)

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    workflow.assert_called_once_with(args[1:])
    run_python.assert_not_called()
    assert source_checkout_root.joinpath(*script_parts).is_file()


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_callback_preserves_selector_aliases(source_checkout_root: Path, command, script_parts, monkeypatch):
    """Callback-owned flags are removed without losing task or viewer selections."""
    callback = ModuleType("_teleop_callback")

    def register():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--callback_option")
        args, remaining = parser.parse_known_args()
        assert args.callback_option == "42"
        return remaining

    callback.register = register
    monkeypatch.setitem(sys.modules, callback.__name__, callback)
    script = source_checkout_root.joinpath(*script_parts)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--task", "Isaac-Cartpole-Direct", "--external_callback", "_teleop_callback.register",
        "--callback_option", "42", "--viz", "newton_gl", "--physics=newton_mjwarp", "--renderer", "newton_renderer",
        "env.sim.dt=0.01",
    ])  # fmt: skip
    args_cli = importlib.import_module(TELEOP_ENTRYPOINTS[command][0])._parse_args()
    assert args_cli.task == "Isaac-Cartpole-Direct"
    assert set(sys.argv[1:]) == {
        "visualizer=newton_gl",
        "physics=newton_mjwarp",
        "renderer=newton_renderer",
        "env.sim.dt=0.01",
    }


def _requirement_names(requirements: list[str]) -> set[str]:
    """Return the importable module names for a list of requirement strings."""
    names = set()
    for requirement in requirements:
        name = re.split(r"[\s<>=!~\[;@]", requirement, maxsplit=1)[0]
        if name.startswith("isaaclab"):
            names.add(name.replace("-", "_"))
    return names


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_isaaclab_imports_are_covered_by_the_extras(source_checkout_root: Path, command, script_parts):
    """Every ``isaaclab_*`` package a teleop workflow imports must ship with the extras.

    A package missing from ``--extra teleop`` fails with ``ModuleNotFoundError`` only after
    Isaac Sim has finished booting. This catches that class of gap without needing an install.
    """
    with (source_checkout_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject["project"]

    available = _requirement_names(project["dependencies"])
    available |= _requirement_names(project["optional-dependencies"]["teleop"])

    module_path = TELEOP_ENTRYPOINTS[command][0].replace(".", "/") + ".py"
    source = (source_checkout_root / "source" / "isaaclab_teleop" / module_path).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+(isaaclab_\w+)", source, re.MULTILINE))

    missing = imported - available
    assert not missing, f"'isaaclab teleop {command}' imports {sorted(missing)}, absent from the teleop extra"


def test_teleop_rejects_unknown_workflow():
    """An unknown workflow name fails at the sub-parser instead of launching a script."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", "bogus"]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
        pytest.raises(SystemExit),
    ):
        cli.cli()

    run_python.assert_not_called()


@pytest.mark.parametrize("task", ["Reach", "Lift"])
def test_mimic_teleop_holds_the_gripper_while_capture_starts_or_pauses(source_checkout_root, monkeypatch, task):
    """Capture's flat command reaches the action queue; missing/paused input stops motion without opening a grip."""
    script = source_checkout_root / "scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--task", task])
    namespace = runpy.run_path(str(script))
    run = namespace["run_teleop_robot"]
    handler = mock.Mock()
    monkeypatch.setitem(run.__globals__, "HDF5DatasetFileHandler", lambda: handler)
    commands = iter(
        [
            (None, None),
            (torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]), True),
            (torch.ones(7), False),
            (None, None),
            (torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.0]), True),
        ]
    )
    teleop = mock.Mock(last_control_events=SimpleNamespace(is_active=None))

    def advance():
        action, active = next(commands)
        teleop.last_control_events.is_active = active
        return action

    teleop.advance.side_effect = advance
    actions = []

    class Finished(Exception):
        pass

    class ActionQueue:
        async def put(self, entry):
            env_id, action = entry
            assert env_id == 2
            actions.append(action.clone())

        async def join(self):
            if len(actions) == 5:
                raise Finished

    env = SimpleNamespace(device="cpu", cfg=SimpleNamespace(env_name=task))
    env.unwrapped = env
    with pytest.raises(Finished):
        asyncio.run(run(env, 2, ActionQueue(), None, None, "output.hdf5", teleop))
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1],
        ]
    )
    torch.testing.assert_close(torch.cat(actions), expected[:, :6] if task == "Reach" else expected)


@pytest.mark.parametrize("has_gui", [False, True])
def test_pick_and_place_waits_for_capture_and_closes_it_before_the_environment(
    source_checkout_root, monkeypatch, has_gui
):
    """Windowless demos need no device; GUI input waits through startup/pause and closes before its windows."""
    import isaaclab_teleop

    script = source_checkout_root / "examples/demos/pick_and_place.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--max_steps", "1"])
    run = runpy.run_path(str(script))["main"]
    closed = []
    sim = mock.Mock(has_gui=has_gui, visualizers=[])
    env = mock.Mock(sim=sim, num_envs=1, device="cpu")
    env.close.side_effect = lambda: closed.append("environment")
    monkeypatch.setitem(run.__globals__, "PickAndPlaceEnv", lambda cfg: env)
    monkeypatch.setitem(run.__globals__, "launch_simulation", lambda *args: contextlib.nullcontext())
    teleop = mock.MagicMock(last_control_events=SimpleNamespace(is_active=None))
    teleop.__enter__.return_value = teleop
    teleop.__exit__.side_effect = lambda *args: closed.append("capture")
    commands = iter([(None, None), (torch.ones(7), False), (torch.tensor([0.25, -0.5, 1, 0, 0, 0, -1]), True)])

    def advance():
        action, active = next(commands)
        teleop.last_control_events.is_active = active
        return action

    teleop.advance.side_effect = advance
    factory = mock.Mock(return_value=teleop)
    monkeypatch.setattr(isaaclab_teleop, "create_isaac_teleop_device", factory)
    run()
    if has_gui:
        torch.testing.assert_close(env.step.call_args.args[0], torch.tensor([[0.25, -0.5, -200.0, 1.0]]))
        assert sim.render.call_count == 2
        assert closed == ["capture", "environment"]
    else:
        factory.assert_not_called()
        torch.testing.assert_close(env.step.call_args.args[0], torch.zeros(1, 4))
        assert closed == ["environment"]
