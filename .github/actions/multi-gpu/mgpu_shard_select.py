# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pytest plugin: select which tests a non-default GPU shard runs.

Loaded only by the multi-GPU lane: ``tools/conftest.py`` injects it via
``-p mgpu_shard_select`` into each per-file pytest subprocess (and prepends this
directory to ``PYTHONPATH`` so it is importable). It lives next to the lane
scripts rather than as a repo-root ``conftest.py`` so it only affects the lane.

Signal: ``ISAACLAB_TEST_DEVICES`` -- the runtime device mask. This is the SAME
env var ``isaaclab.test.utils.devices.test_devices()`` reads to decide a test's
device parametrization, so this plugin's keep/drop decision agrees with what was
parametrized. Kit-backed tests derive their AppLauncher device from the same mask.
The plugin reads the mask string directly -- no isaaclab import -- matching the
lib-free convention of every other conftest in the repo. The only shared knowledge
is the mask grammar below, mirrored from ``devices.py`` (position 0 = cpu,
position k = cuda:(k-1), trailing ``X`` = include all remaining); keep the two in sync.

Selection rule, on a shard whose mask excludes cpu + cuda:0:

    keep a test iff it is parametrized over ``device`` AND that variant's device
    is active in the mask; deselect everything else.

That single rule is both criteria at once: candidate = device-parametrized (a
test with no ``device`` param is device-agnostic, covered by single-GPU CI on
cuda:0); scope = only this shard's device variant runs (cpu / cuda:0 / other-index
variants are dropped). Out-of-scope items are deselected (not skipped) so the
shard report shows only what ran.
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
    """Whether a mask position is included. Trailing ``X`` includes all remaining."""
    if position is None:
        return False
    body, fill = (mask[:-1], True) if mask.endswith("X") else (mask, False)
    return body[position] == "1" if 0 <= position < len(body) else fill


def _shard_mask():
    """Return the runtime mask iff this is a multi-GPU shard run, else None.

    A shard targets non-default GPUs only, so cpu (position 0) and cuda:0
    (position 1) are excluded. Anything else -- unset, or single-GPU CI's default
    ``"110"`` -- is not a shard and the plugin is a no-op.
    """
    mask = os.environ.get(_RUNTIME_ENV, "")
    return mask if mask and not _active(0, mask) and not _active(1, mask) else None


def pytest_collection_modifyitems(config, items):
    mask = _shard_mask()
    if mask is None:
        return
    keep, drop = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        device = callspec.params.get("device") if callspec is not None else None
        if _active(_position(device), mask):
            keep.append(item)  # device-parametrized AND in this shard's scope
        else:
            drop.append(item)  # non-device, or cpu / cuda:0 / other-index variant
    if drop:
        items[:] = keep
        config.hook.pytest_deselected(items=drop)


def pytest_sessionfinish(session, exitstatus):
    # A file whose tests are all out of scope deselects to zero, so pytest exits
    # NO_TESTS_COLLECTED (5). The lane orchestrator treats any non-zero per-file
    # exit as a failure (tools/conftest.py), so report "nothing in scope for this
    # file" as success rather than a false failure.
    if _shard_mask() is not None and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
