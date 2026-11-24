# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line handling for ``docker/container.py``."""

import argparse
from unittest.mock import MagicMock, call

import pytest

from docker import container as container_cli

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize(
    "argv, expected",
    [
        # positional profile, unchanged from before the option existed
        ("start", ("start", "base", None)),
        ("start kitless --suffix custom", ("start", "kitless", "custom")),
        # option before the command, so only the command changes between calls
        ("-p kitless build", ("build", "kitless", None)),
        # option after the command, and agreeing duplicates; exercise both aliases in both positions
        ("enter --profile kitless", ("enter", "kitless", None)),
        ("--profile kitless stop kitless -p kitless", ("stop", "kitless", None)),
    ],
)
def test_profile_is_accepted_positionally_or_as_an_option(argv: str, expected, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["container.py", *argv.split()])
    args = container_cli.parse_cli_args()

    assert (args.command, args.profile, args.suffix) == expected


@pytest.mark.parametrize(
    "argv",
    [
        "-p kitless start base",  # option and positional disagree
        "-p kitless start -p base",  # option before and after the command disagree
    ],
)
def test_conflicting_profiles_are_rejected(argv: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr("sys.argv", ["container.py", *argv.split()])

    with pytest.raises(SystemExit) as exc_info:
        container_cli.parse_cli_args()

    assert exc_info.value.code == 2
    assert "conflicting profiles" in capsys.readouterr().err


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
    interface = MagicMock()
    interface.is_container_running.return_value = running
    interface.add_yamls = []
    interface.environ = {}
    started_with = []
    interface.start.side_effect = lambda: started_with.append((list(interface.add_yamls), dict(interface.environ)))
    lifecycle = MagicMock()
    lifecycle.x11_check.return_value = x11_outputs
    lifecycle.attach_mock(interface.start, "start")
    lifecycle.attach_mock(interface.enter, "enter")
    monkeypatch.setattr(container_cli, "ContainerInterface", MagicMock(return_value=interface))
    monkeypatch.setattr(container_cli.x11_utils, "x11_check", lifecycle.x11_check)
    monkeypatch.setattr(container_cli.x11_utils, "x11_refresh", lifecycle.x11_refresh)
    monkeypatch.setattr(container_cli.shutil, "which", MagicMock(return_value="/usr/bin/docker"))

    container_cli.main(
        argparse.Namespace(command="enter", profile="kitless", files=None, env_files=None, suffix=None, info=False)
    )

    assert started_with == ([] if overlay_at_start is None else [overlay_at_start])
    expected_calls = [] if running else [call.x11_check(interface.statefile), call.start()]
    assert lifecycle.mock_calls == [*expected_calls, call.x11_refresh(interface.statefile), call.enter()]
