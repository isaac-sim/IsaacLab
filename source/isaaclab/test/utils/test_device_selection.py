# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the test-device selection helpers.

These tests mock the host's device list, so they need no GPU and run on the
single-GPU CI lane. The helper is imported under an alias (``resolve_devices``)
so the multi-GPU workflow's auto-discovery does not mistake this file for an
opted-in device test. The helper itself sets ``__test__ = False`` to prevent
pytest from collecting it when imported under its public name.
"""

import pytest

from isaaclab.test.utils import devices as devices_mod
from isaaclab.test.utils.devices import DeviceScope, resolve_test_sim_device
from isaaclab.test.utils.devices import test_devices as resolve_devices

# Representative hosts, in mask order (cpu first, then cuda:0, cuda:1, ...).
SINGLE_GPU = ["cpu", "cuda:0"]
MULTI_GPU = ["cpu", "cuda:0", "cuda:1", "cuda:2"]


@pytest.fixture
def host(monkeypatch):
    """Return a setter that pins the available device list and the runtime env var."""

    def _set(available: list[str], runtime: str | None = None) -> None:
        monkeypatch.setattr(devices_mod, "_list_available_devices", lambda: available)
        if runtime is None:
            monkeypatch.delenv(devices_mod._RUNTIME_DEVICES_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(devices_mod._RUNTIME_DEVICES_ENV_VAR, runtime)

    return _set


# ---------------------------------------------------------------------------
# scope ∩ runtime resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "available, runtime, scope, expected",
    [
        # single-GPU CI lane: runtime unset -> default "110" (cpu + cuda:0).
        (SINGLE_GPU, None, "11X", ["cpu", "cuda:0"]),  # argless default is device-agnostic
        (SINGLE_GPU, None, "110", ["cpu", "cuda:0"]),  # pure-math scope
        (SINGLE_GPU, None, "100", ["cpu"]),  # cpu-only scope
        (SINGLE_GPU, None, "00X", []),  # non-default-only -> nothing here, skip
        # multi-GPU shard: runtime pins exactly one non-default GPU.
        (MULTI_GPU, "001", "11X", ["cuda:1"]),  # cuda:1 shard
        (MULTI_GPU, "0001", "11X", ["cuda:2"]),  # cuda:2 shard
        (MULTI_GPU, "0001", "00X", ["cuda:2"]),  # non-default regression on its shard
        (MULTI_GPU, "0001", "110", []),  # math skips the shard (cuda:0 != this shard)
        (MULTI_GPU, "0001", "100", []),  # cpu test skips the shard
        # a runtime that lists several GPUs hands back all in-scope ones.
        (MULTI_GPU, "111", "11X", ["cpu", "cuda:0", "cuda:1"]),
    ],
)
def test_resolves_scope_intersect_runtime(host, available, runtime, scope, expected):
    host(available, runtime)
    assert resolve_devices(scope) == expected


def test_argless_equals_default_scope(host):
    # The common case: argless must equal the explicit default mask everywhere.
    for available, runtime in [(SINGLE_GPU, None), (MULTI_GPU, "0001"), (MULTI_GPU, "001")]:
        host(available, runtime)
        assert resolve_devices() == resolve_devices("11X")


def test_argless_runs_once_on_one_non_default_gpu(host):
    # Run-once property: with a one-device-per-shard runtime, an argless test
    # resolves to exactly one non-default GPU.
    host(MULTI_GPU, "0001")
    result = resolve_devices()
    assert result == ["cuda:2"]
    assert all(d not in ("cpu", "cuda:0") for d in result)


@pytest.mark.parametrize(
    "scope, mask",
    [
        (DeviceScope.ALL, "11X"),
        (DeviceScope.CUDA, "01X"),
        (DeviceScope.CPU_AND_DEFAULT_CUDA, "110"),
        (DeviceScope.NON_DEFAULT_CUDA, "00X"),
        (DeviceScope.CPU, "100"),
        (DeviceScope.DEFAULT_CUDA, "010"),
        (DeviceScope.CPU | DeviceScope.NON_DEFAULT_CUDA, "10X"),
    ],
)
def test_named_scope_matches_mask(host, scope, mask):
    host(MULTI_GPU, "1111")
    assert scope.mask == mask
    assert resolve_devices(scope) == resolve_devices(mask)


def test_named_combination_matches_runtime_composition():
    assert DeviceScope.CPU_AND_DEFAULT_CUDA == DeviceScope.CPU | DeviceScope.DEFAULT_CUDA
    assert DeviceScope.CUDA == DeviceScope.DEFAULT_CUDA | DeviceScope.NON_DEFAULT_CUDA
    assert DeviceScope.ALL == DeviceScope.CPU | DeviceScope.DEFAULT_CUDA | DeviceScope.NON_DEFAULT_CUDA


def test_helper_is_not_collected_by_pytest():
    assert resolve_devices.__test__ is False


# ---------------------------------------------------------------------------
# AppLauncher device resolution from the runtime mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runtime, expected",
    [
        (None, "cuda:0"),
        ("110", "cuda:0"),
        ("100", "cpu"),
        ("001", "cuda:1"),
        ("0001", "cuda:2"),
        ("101", "cuda:1"),
    ],
)
def test_resolve_test_sim_device(host, runtime, expected):
    host(MULTI_GPU, runtime)
    assert resolve_test_sim_device() == expected


@pytest.mark.parametrize("runtime", ["", "000", "011", "01X", "0X1", "0A1"])
def test_resolve_test_sim_device_rejects_invalid_or_ambiguous_runtime(host, runtime):
    host(MULTI_GPU, runtime)
    with pytest.raises(ValueError, match="ISAACLAB_TEST_DEVICES"):
        resolve_test_sim_device()


# ---------------------------------------------------------------------------
# skip vs raise: legitimate skips never raise; only a missing runtime device does
# ---------------------------------------------------------------------------


def test_unset_runtime_never_raises(host):
    # On a dev box / single-GPU lane (runtime unset), an out-of-scope result is a
    # silent skip, not an error.
    host(SINGLE_GPU, None)
    assert resolve_devices("00X") == []


def test_runtime_naming_absent_device_raises(host):
    # A run that explicitly asked for cuda:2 on a host that only has cuda:0/cuda:1
    # is misconfigured -> fail loudly instead of a vacuous green.
    host(["cpu", "cuda:0", "cuda:1"], "0001")
    with pytest.raises(ValueError, match="no device available"):
        resolve_devices("11X")


def test_in_range_runtime_with_empty_scope_does_not_raise(host):
    # The runtime device exists (cuda:2 is present), the scope just doesn't include
    # it -> skip, not raise. This is what keeps "110"/"100" tests green on a shard.
    host(MULTI_GPU, "0001")
    assert resolve_devices("110") == []  # would raise if the guard keyed off scope


# ---------------------------------------------------------------------------
# skip= : gate a specific device visibly
# ---------------------------------------------------------------------------


def test_skip_wraps_named_device_as_skipped_param(host):
    host(SINGLE_GPU, None)
    result = resolve_devices("11X", skip={"cuda:0": "known broken"})
    assert result[0] == "cpu"  # untouched devices stay plain strings
    param = result[1]
    assert param.values == ("cuda:0",)
    assert param.marks[0].name == "skip"
    assert param.marks[0].kwargs["reason"] == "known broken"


def test_skip_ignores_out_of_result_devices(host):
    # Skipping a device that isn't in the resolved set is a no-op (no stray params).
    host(SINGLE_GPU, None)
    assert resolve_devices("11X", skip={"cuda:3": "n/a"}) == ["cpu", "cuda:0"]


# ---------------------------------------------------------------------------
# _expand mask grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask, count, expected",
    [
        ("110", 2, [True, True]),  # exact length, no wildcard
        ("100", 3, [True, False, False]),  # short mask padded with False
        ("11X", 4, [True, True, True, True]),  # trailing X fills the rest True
        ("00X", 4, [False, False, True, True]),  # X spans the non-default GPUs
        ("11X", 2, [True, True]),  # X with nothing left to fill
        ("0001", 3, [False, False, False]),  # mask longer than host -> truncated
    ],
)
def test_expand(mask, count, expected):
    assert devices_mod._expand(mask, count) == expected
