# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Argument-order handling for ``docker/container.py``."""

import argparse
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

DOCKER_DIR = Path(__file__).resolve().parents[1]


def _container_module():
    """Import ``docker/container.py`` by path (``docker`` is not an importable package here).

    The script falls back to ``from utils import ...`` when it has no package, so ``docker/``
    has to be importable for the duration of the load.
    """
    spec = importlib.util.spec_from_file_location("container_cli", DOCKER_DIR / "container.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DOCKER_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DOCKER_DIR))
    return module


COMMANDS = ("build", "start", "enter", "config", "copy", "stop")
"""Expected subcommands. The CLI derives its own set from argparse; this pins the contract."""


@pytest.mark.parametrize(
    "argv, expected",
    [
        # profile first is rewritten to command first
        (["kitless", "build"], ["build", "kitless"]),
        (["ros2", "start"], ["start", "ros2"]),
        (["kitless", "start", "--suffix", "custom"], ["start", "kitless", "--suffix", "custom"]),
        # command first is already correct and left alone
        (["build", "kitless"], ["build", "kitless"]),
        (["start"], ["start"]),
        (["config", "kitless", "--output-yaml", "out.yaml"], ["config", "kitless", "--output-yaml", "out.yaml"]),
        # nothing to reorder
        ([], []),
        (["build"], ["build"]),
        # flags are never treated as a profile
        (["--help"], ["--help"]),
        (["-h", "build"], ["-h", "build"]),
    ],
)
def test_reorder_profile_first(argv: list[str], expected: list[str]):
    assert _container_module().reorder_profile_first(argv, COMMANDS) == expected


@pytest.mark.parametrize("argv", [["kitless", "build"], ["build", "kitless"]])
def test_both_orders_parse_to_the_same_arguments(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    """End-to-end through the real parser, so the derived command set is exercised."""
    monkeypatch.setattr("sys.argv", ["container.py", *argv])
    args = _container_module().parse_cli_args()

    assert (args.command, args.profile) == ("build", "kitless")


X11_OVERLAY = (["x11.yaml"], {"DISPLAY": ":1"})
"""What ``x11_check`` returns when X11 forwarding is enabled: compose files and environment variables."""


@pytest.mark.parametrize(
    "running, x11_outputs, overlay_at_start",
    [
        (False, None, ([], {})),
        (False, X11_OVERLAY, X11_OVERLAY),
        (True, X11_OVERLAY, None),
    ],
)
def test_enter_starts_the_container_only_when_it_is_down(
    running: bool, x11_outputs, overlay_at_start, monkeypatch: pytest.MonkeyPatch
):
    """``enter`` brings a stopped container up with the X11 overlay ``start`` would apply, and leaves a
    running one alone."""
    module = _container_module()

    interface = MagicMock()
    interface.is_container_running.return_value = running
    interface.add_yamls = []
    interface.environ = {}
    started_with = []
    interface.start.side_effect = lambda: started_with.append((list(interface.add_yamls), dict(interface.environ)))
    monkeypatch.setattr(module, "ContainerInterface", MagicMock(return_value=interface))
    monkeypatch.setattr(module.x11_utils, "x11_check", MagicMock(return_value=x11_outputs))
    monkeypatch.setattr(module.x11_utils, "x11_refresh", MagicMock())
    monkeypatch.setattr(module.shutil, "which", MagicMock(return_value="/usr/bin/docker"))

    module.main(
        argparse.Namespace(command="enter", profile="kitless", files=None, env_files=None, suffix=None, info=False)
    )

    assert started_with == ([] if overlay_at_start is None else [overlay_at_start])
    assert interface.enter.called


def test_profile_names_do_not_collide_with_commands():
    """The reorder is only unambiguous while the two name sets stay disjoint."""
    profiles = {path.name.split(".", 1)[1] for path in DOCKER_DIR.glob("Dockerfile.*")}

    assert profiles, "no Dockerfile.<profile> found"
    assert set(COMMANDS).isdisjoint(profiles)
