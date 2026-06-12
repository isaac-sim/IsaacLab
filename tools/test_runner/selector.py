# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process pytest plugin: keep only the tests a unit's device mask allows.

The executor injects this into a unit's subprocess with ``-p test_runner.selector``
for any unit that must run on a single device — both halves of a ``device_isolated``
split and every mgpu shard. It runs inside that subprocess, at collection time, so
a process locked to one device by a backend (e.g. ovphysx) never also tries another.

It is registered as a class instance (see :func:`pytest_configure`): the hooks are
methods of :class:`DeviceSelect`, so the device mask is read once into instance
state instead of from the environment on every call.
"""

from __future__ import annotations

import os

import pytest

_RUNTIME_ENV = "ISAACLAB_TEST_DEVICES"


class DeviceSelect:
    """Decide, per collected test, whether it belongs to this unit's device mask.

    Device-parametrized tests are kept iff their device is active in the mask
    (this catches literal ``["cuda:0", "cpu"]`` variants the mask cannot narrow at
    parametrize time); agnostic tests (no ``device`` param) are kept only when cpu
    is in the mask, i.e. the default unit — never on a cuda split unit or a shard.
    """

    def __init__(self, mask: str):
        self._mask = mask

    @staticmethod
    def _position(device) -> int | None:
        """Mask position for a device: cpu -> 0, cuda:k -> k + 1, else None (agnostic/unknown)."""
        if device == "cpu":
            return 0
        if isinstance(device, str) and device.startswith("cuda:") and device[len("cuda:") :].isdigit():
            return int(device[len("cuda:") :]) + 1
        return None

    def _active(self, position: int | None) -> bool:
        """Whether a mask position is included; a trailing ``X`` includes all remaining."""
        if position is None:
            return False
        body, fill = (self._mask[:-1], True) if self._mask.endswith("X") else (self._mask, False)
        return body[position] == "1" if 0 <= position < len(body) else fill

    def keeps(self, device: str | None) -> bool:
        """Whether a test on ``device`` (or ``None`` if agnostic) runs in this unit."""
        if device is None:
            return self._active(0)  # agnostic: only in a cpu-inclusive unit
        return self._active(self._position(device))

    # -- pytest hooks (methods of the registered instance) --

    def pytest_collection_modifyitems(self, config, items):
        """PYTEST HOOK — filter the collected tests in place to this unit's devices.

        Called by pytest after collection, before any test runs.

        Args:
            config: the session ``pytest.Config``; ``config.hook.pytest_deselected``
                reports the dropped tests so they count as deselected, not missing.
            items: the live, ordered ``list[pytest.Item]`` of collected tests. This
                hook mutates it in place (``items[:] = keep``) — that is how pytest
                learns what to run; reassigning a new list would have no effect.

        Returns:
            None. Its effect is the in-place edit of ``items`` plus the
            ``pytest_deselected`` report.
        """
        keep, drop = [], []
        for item in items:
            callspec = getattr(item, "callspec", None)
            device = callspec.params.get("device") if callspec is not None else None
            (keep if self.keeps(device) else drop).append(item)
        if drop:
            items[:] = keep
            config.hook.pytest_deselected(items=drop)

    def pytest_sessionfinish(self, session, exitstatus):
        """PYTEST HOOK — treat a unit that selected nothing as a clean pass.

        Called by pytest once the whole session ends.

        Args:
            session: the ``pytest.Session``; reassign ``session.exitstatus`` to
                change the process exit code.
            exitstatus: the proposed exit code (a ``pytest.ExitCode``).

        Returns:
            None. When this unit deselected every test (NO_TESTS_COLLECTED), it
            rewrites the exit to OK so the runner does not read a non-zero exit as
            a failure — "nothing in scope for this file" is success here.
        """
        if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
            session.exitstatus = pytest.ExitCode.OK


def pytest_configure(config):
    """PYTEST HOOK — the plugin's entry point: register the device selector.

    Loading this module with ``-p test_runner.selector`` makes pytest call this
    module-level hook during config init. It builds a :class:`DeviceSelect` from
    the unit's mask (the ``ISAACLAB_TEST_DEVICES`` the executor set) and registers
    the instance, so the instance's hook methods become active for the session.

    Args:
        config: the session ``pytest.Config`` (provides the plugin manager).

    Returns:
        None.
    """
    config.pluginmanager.register(DeviceSelect(os.environ.get(_RUNTIME_ENV, "")))
