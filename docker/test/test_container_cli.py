# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line handling for ``docker/container.py``."""

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


@pytest.mark.parametrize(
    "argv, expected",
    [
        # positional profile, unchanged from before the option existed
        ("start", ("start", "base", None)),
        ("start kitless", ("start", "kitless", None)),
        ("start kitless --suffix custom", ("start", "kitless", "custom")),
        # option before the command, so only the command changes between calls
        ("-p kitless build", ("build", "kitless", None)),
        ("--profile kitless start --suffix custom", ("start", "kitless", "custom")),
        # option after the command, and agreeing duplicates
        ("enter -p kitless", ("enter", "kitless", None)),
        ("stop kitless -p kitless", ("stop", "kitless", None)),
    ],
)
def test_profile_is_accepted_positionally_or_as_an_option(argv: str, expected, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["container.py", *argv.split()])
    args = _container_module().parse_cli_args()

    assert (args.command, args.profile, args.suffix) == expected


@pytest.mark.parametrize(
    "argv",
    [
        "-p kitless start base",  # option and positional disagree
        "-p kitless start -p base",  # option before and after the command disagree
        "kitless start",  # a bare profile is never read as the command
    ],
)
def test_ambiguous_profile_is_rejected(argv: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["container.py", *argv.split()])

    with pytest.raises(SystemExit):
        _container_module().parse_cli_args()


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
