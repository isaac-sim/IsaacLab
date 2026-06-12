# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-file CI test runner framework.

``tools/conftest.py`` runs each test file as its own pytest subprocess rather
than collecting everything into one process. This package is that runner, split
into the stages of a test-run lifecycle:

* :mod:`test_runner.planning` — what to run: the :class:`~test_runner.planning.RunnerConfig`
  knobs, the :class:`~test_runner.planning.Unit` work item, and the
  :class:`~test_runner.planning.Planner` that turns a file + device mask into units.
* :mod:`test_runner.execution` — how to run one unit: subprocess capture,
  timeout/hang handling, report parsing, retries.
* :mod:`test_runner.selector` — an in-process pytest plugin the executor injects
  into a unit's subprocess to keep only that unit's device variants.
* :mod:`test_runner.session` — the lifecycle: collect files, plan units, run
  them, aggregate and report.

The whole runner is driven by one :class:`~test_runner.planning.RunnerConfig`, so
its behavior (timeouts, retries, cold-cache budget, device mask, filters) is
configuration rather than scattered constants.
"""
