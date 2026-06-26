# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_add_launcher_args_does_not_import_sim_runtime_before_launch():
    """Test that launcher arg registration does not import Isaac Sim runtime modules."""
    program = textwrap.dedent(
        """
        import argparse
        import json
        import sys
        import traceback
        from pathlib import Path

        repo_root = Path(sys.argv[1])
        for rel_path in [
            "source/isaaclab",
            "source/isaaclab_tasks",
            "source/isaaclab_assets",
            "source/isaaclab_rl",
            "source/isaaclab_newton",
            "source/isaaclab_ovphysx",
            "source/isaaclab_physx",
        ]:
            sys.path.insert(0, str(repo_root / rel_path))

        forbidden = {"pxr", "omni", "carb", "isaacsim", "usdrt"}
        violations = {}
        original_import = __builtins__.__import__

        def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
            root_name = name.split(".")[0]
            if root_name in forbidden and root_name not in violations:
                violations[root_name] = "".join(traceback.format_stack(limit=18))
            return original_import(name, globals, locals, fromlist, level)

        __builtins__.__import__ = import_hook
        try:
            from isaaclab.app import add_launcher_args

            parser = argparse.ArgumentParser(add_help=False)
            add_launcher_args(parser)
        finally:
            __builtins__.__import__ = original_import

        print(json.dumps(violations, sort_keys=True))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program, str(REPO_ROOT)],
        capture_output=True,
        check=True,
        text=True,
    )

    violations = json.loads(result.stdout)
    assert violations == {}


def test_simulation_context_import_does_not_import_pxr_before_launch():
    """Test that resolving SimulationContext does not import USD modules."""
    program = textwrap.dedent(
        """
        import json
        import sys
        import traceback
        from pathlib import Path

        repo_root = Path(sys.argv[1])
        for rel_path in [
            "source/isaaclab",
            "source/isaaclab_tasks",
            "source/isaaclab_assets",
            "source/isaaclab_rl",
            "source/isaaclab_newton",
            "source/isaaclab_ovphysx",
            "source/isaaclab_physx",
        ]:
            sys.path.insert(0, str(repo_root / rel_path))

        violations = {}
        original_import = __builtins__.__import__

        def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] == "pxr" and "pxr" not in violations:
                violations["pxr"] = "".join(traceback.format_stack(limit=18))
            return original_import(name, globals, locals, fromlist, level)

        __builtins__.__import__ = import_hook
        try:
            from isaaclab.sim import SimulationContext

            assert SimulationContext.__name__ == "SimulationContext"
        finally:
            __builtins__.__import__ = original_import

        print(json.dumps(violations, sort_keys=True))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program, str(REPO_ROOT)],
        capture_output=True,
        check=True,
        text=True,
    )

    violations = json.loads(result.stdout)
    assert violations == {}


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_argparser(mocker):
    """Test launching with argparser arguments."""
    # Mock the parse_args method
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(livestream=1))
    # create argparser
    parser = argparse.ArgumentParser()
    # add app launcher arguments
    AppLauncher.add_app_launcher_args(parser)
    # check that argparser has the mandatory arguments
    for name in AppLauncher._APPLAUNCHER_CFG_INFO:
        assert parser._option_string_actions[f"--{name}"]
    # parse args
    mock_args = parser.parse_args()
    # everything defaults to None
    app_launcher = AppLauncher(mock_args)
    app = app_launcher.app
    assert app_launcher._livestream == 1
    assert app_launcher._headless is True

    # close the app on exit
    app.close()


def test_visualizer_alias_parsing():
    """Test that --viz alias maps to visualizer values."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(["--viz", "kit,newton"])
    assert args.visualizer == ["kit", "newton"]
    assert args.visualizer_explicit is True


def test_headless_deprecated_arg_parsing():
    """Test that deprecated --headless is still accepted by the parser."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(["--headless"])
    assert args.headless is True
    assert args.headless_explicit is True


@pytest.mark.parametrize("value", ["none", "None"])
def test_visualizer_none_parsing(value: str):
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--viz", value])
    assert args.visualizer is None
    assert args.visualizer_explicit is True
