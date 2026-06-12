# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI entry point for the per-file test runner.

This conftest does not run tests in-process. It hands the whole run to the
:mod:`test_runner` framework, which executes each test file in its own pytest
subprocess (so a crash or device lock in one file cannot affect the next). All
behavior is in the framework; this file is just the two pytest hooks that wire
it in.
"""

import os

import pytest
from test_runner.planning import RunnerConfig
from test_runner.session import Session


def pytest_ignore_collect(collection_path, config):
    """PYTEST HOOK — skip pytest's own collection; the runner drives files itself.

    Args:
        collection_path: candidate path pytest is about to collect (unused).
        config: the session ``pytest.Config`` (unused).

    Returns:
        ``True`` always, so nothing is collected in this process; the per-file
        subprocesses launched by :class:`~test_runner.session.Session` do the
        real collection.
    """
    return True


def pytest_sessionstart(session):
    """PYTEST HOOK — build the runner from the environment and run the lifecycle.

    Called by pytest at session start. It exits pytest with the runner's code so
    normal pytest does not run (and overwrite the aggregate report) afterward.

    Args:
        session: the ``pytest.Session`` (only its startup timing is used).

    Returns:
        None — terminates the process via :func:`pytest.exit` with 0 when every
        file passed, else 1.
    """
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = RunnerConfig.from_env(workspace_root)
    print(
        f"test_runner config: mask={config.runtime_mask} queue={'on' if config.queue_path else 'off'}"
        f" isaacsim_ci={config.isaacsim_ci} filter={config.filter_pattern!r}"
        f" exclude={config.exclude_pattern!r} include={sorted(config.include_files) or 'none'}"
    )
    pytest.exit("Custom test execution completed", returncode=Session(config).run())
