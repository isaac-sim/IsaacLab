# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pytest plugin: keep only the tests a unit's device mask allows.

The executor (``tools/conftest.py`` via ``_device_exec``) injects this with
``-p _device_select`` for any unit that must run on a single device — both halves
of a ``device_isolated`` split (cpu unit, cuda unit) and every mgpu shard — so a
process that is locked to one device by a backend (e.g. ovphysx<=0.3.7) never
also tries another device.

Selection reads ``ISAACLAB_TEST_DEVICES`` (the unit's mask). For each collected
test:

* parametrized over ``device``: keep iff that device is active in the mask. This
  is what catches a literal ``["cuda:0", "cpu"]`` variant whose device the mask
  alone cannot narrow (only ``test_devices()`` reads the mask at parametrize
  time; a literal list does not).
* not parametrized over ``device`` (agnostic): keep only when cpu is in the mask
  — i.e. the default unit. On a cuda split unit or a shard it is dropped, since
  single-GPU CI already covers it on cpu/cuda:0.

A unit that deselects to empty (no in-scope tests in this file) exits
``NO_TESTS_COLLECTED``; the runner treats any non-zero per-file exit as failure,
so report "nothing in scope" as success.
"""

import os

import pytest

_RUNTIME_ENV = "ISAACLAB_TEST_DEVICES"


def _position(device):
    """Mask position for a device value: cpu -> 0, cuda:k -> k + 1 (else None)."""
    if device == "cpu":
        return 0
    if isinstance(device, str) and device.startswith("cuda:") and device[len("cuda:") :].isdigit():
        return int(device[len("cuda:") :]) + 1
    return None  # not device-parametrized, or an unrecognized device value


def _active(position, mask):
    """Whether a mask position is included. A trailing ``X`` includes all remaining."""
    if position is None:
        return False
    body, fill = (mask[:-1], True) if mask.endswith("X") else (mask, False)
    return body[position] == "1" if 0 <= position < len(body) else fill


def _device_of(item):
    """The ``device`` parametrize value of a collected item, or ``None`` if agnostic."""
    callspec = getattr(item, "callspec", None)
    return callspec.params.get("device") if callspec is not None else None


def pytest_collection_modifyitems(config, items):
    mask = os.environ.get(_RUNTIME_ENV, "")
    cpu_in_mask = _active(0, mask)
    keep, drop = [], []
    for item in items:
        device = _device_of(item)
        if device is None:
            (keep if cpu_in_mask else drop).append(item)  # agnostic: only in a cpu-inclusive unit
        elif _active(_position(device), mask):
            keep.append(item)  # this device variant is in scope for the unit
        else:
            drop.append(item)  # out-of-scope variant (e.g. a literal cpu variant in a cuda unit)
    if drop:
        items[:] = keep
        config.hook.pytest_deselected(items=drop)


def pytest_sessionfinish(session, exitstatus):
    # A file with nothing in scope for this unit deselects to zero and exits
    # NO_TESTS_COLLECTED (5). That is success here, not a failure.
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
