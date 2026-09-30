# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the console logging setup shared by Isaac Lab entry points and launchers."""

import contextlib
import logging
import sys

import pytest

from isaaclab.app import logging_utils


@contextlib.contextmanager
def _isolated_root(monkeypatch: pytest.MonkeyPatch):
    """Run with a bare root logger and no ``--verbose`` / ``--info`` flags, then restore it.

    Entered inside the test body because pytest attaches its capture handlers to the root logger for
    each test phase, and those would count as another configured handler.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    monkeypatch.setattr(logging_utils, "_requested_level", None)
    monkeypatch.setattr(sys, "argv", ["prog"])
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_console_handlers_print_isaaclab_info_and_all_warnings(monkeypatch, capsys):
    """Isaac Lab INFO goes to stdout, other INFO is dropped, and warnings from any logger reach stderr."""
    with _isolated_root(monkeypatch) as root:
        # entry points and launch_simulation both configure logging in one process
        logging_utils.configure_console_logging()
        logging_utils.configure_console_logging()

        logging.getLogger("isaaclab.envs").info("env created")
        logging.getLogger("third_party").info("noise")
        logging.getLogger("third_party").warning("careful")

        # the root is lowered to create INFO records, but the requested level is still reported
        assert root.level == logging.INFO
        assert logging_utils.resolve_python_logging_level() == logging.WARNING

    out, err = capsys.readouterr()
    assert out == "[INFO]: env created\n"
    assert err == "[WARNING]: careful\n"


@pytest.mark.parametrize("other_handler_first", [True, False], ids=["user_config", "kit_bridge"])
def test_other_handler_prints_warnings_once(monkeypatch, capsys, other_handler_first):
    """When another handler prints warnings, the stderr fallback is not added (or is removed) to avoid duplicates."""
    with _isolated_root(monkeypatch) as root:
        other = logging.StreamHandler(sys.stderr)
        other.setFormatter(logging.Formatter("other: %(message)s"))
        if other_handler_first:
            root.addHandler(other)
            logging_utils.configure_console_logging()
        else:
            # Kit flow: configured before Kit starts, then Kit's log bridge appears
            logging_utils.configure_console_logging()
            root.addHandler(other)
            logging_utils.apply_python_logging_level(logging.WARNING)
            logging_utils.ensure_console_handlers(logging.WARNING, fallback=False)

        logging.getLogger("isaaclab.envs").info("env created")
        logging.getLogger("third_party").warning("careful")

    out, err = capsys.readouterr()
    assert out == "[INFO]: env created\n"
    assert err == "other: careful\n"
