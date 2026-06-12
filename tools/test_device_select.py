# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the per-unit device selector (``tools/_device_select.py``).

These pin the rule that broke when device selection was left to the mask alone:
a literal ``["cuda:0", "cpu"]`` variant must still be dropped when it is out of
the unit's mask, or it leaks into a device-locked process. The plugin reads the
structured ``device`` param, so it catches both literal and ``test_devices()``
variants.
"""

from __future__ import annotations

import _device_select
from _device_select import _active, _position, pytest_collection_modifyitems


class _FakeCallspec:
    def __init__(self, params):
        self.params = params


class _FakeItem:
    """Stand-in for a pytest item. Omitting ``params`` means no callspec (agnostic)."""

    def __init__(self, name, params=None):
        self.name = name
        if params is not None:
            self.callspec = _FakeCallspec(params)


class _FakeHook:
    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class _FakeConfig:
    def __init__(self):
        self.hook = _FakeHook()


def _select(mask, items, monkeypatch):
    monkeypatch.setenv("ISAACLAB_TEST_DEVICES", mask)
    config = _FakeConfig()
    pytest_collection_modifyitems(config, items)
    return items, config.hook.deselected


# ---- position / active helpers ------------------------------------------------


def test_position_maps_cpu_and_cuda():
    assert _position("cpu") == 0
    assert _position("cuda:0") == 1
    assert _position("cuda:2") == 3
    assert _position(None) is None
    assert _position("weird") is None


def test_active_reads_mask_position_and_trailing_x():
    assert _active(0, "110") is True  # cpu
    assert _active(1, "110") is True  # cuda:0
    assert _active(2, "110") is False  # cuda:1 not set
    assert _active(3, "00X") is True  # trailing X includes the rest
    assert _active(None, "110") is False


# ---- selection: the bug this guards against -----------------------------------


def test_cpu_unit_drops_literal_cuda_variant(monkeypatch):
    # The bug: a literal ["cuda:0","cpu"] test's cuda variant leaked into the cpu
    # unit and tripped the device lock. The cpu unit (mask 100) must drop it.
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    cuda = _FakeItem("t[cuda:0]", {"device": "cuda:0"})
    items, dropped = _select("100", [cpu, cuda], monkeypatch)
    assert items == [cpu]
    assert dropped == [cuda]


def test_cuda_unit_drops_literal_cpu_variant_and_agnostic(monkeypatch):
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    cuda = _FakeItem("t[cuda:0]", {"device": "cuda:0"})
    agnostic = _FakeItem("t_logic")
    items, dropped = _select("010", [cpu, cuda, agnostic], monkeypatch)
    assert items == [cuda]
    assert set(dropped) == {cpu, agnostic}


def test_shard_keeps_its_gpu_variant_drops_others_and_agnostic(monkeypatch):
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    cuda0 = _FakeItem("t[cuda:0]", {"device": "cuda:0"})
    cuda2 = _FakeItem("t[cuda:2]", {"device": "cuda:2"})
    agnostic = _FakeItem("t_logic")
    items, dropped = _select("0001", [cpu, cuda0, cuda2, agnostic], monkeypatch)  # cuda:2 shard
    assert items == [cuda2]
    assert set(dropped) == {cpu, cuda0, agnostic}


def test_cpu_unit_keeps_agnostic(monkeypatch):
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    agnostic = _FakeItem("t_logic")
    items, dropped = _select("100", [cpu, agnostic], monkeypatch)
    assert items == [cpu, agnostic]  # cpu in mask -> agnostic kept
    assert dropped == []


def test_default_mask_keeps_cpu_and_cuda0_and_agnostic(monkeypatch):
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    cuda = _FakeItem("t[cuda:0]", {"device": "cuda:0"})
    agnostic = _FakeItem("t_logic")
    items, dropped = _select("110", [cpu, cuda, agnostic], monkeypatch)
    assert items == [cpu, cuda, agnostic]
    assert dropped == []


def test_sessionfinish_maps_no_tests_collected_to_ok():
    import pytest

    class _Session:
        exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED

    session = _Session()
    _device_select.pytest_sessionfinish(session, pytest.ExitCode.NO_TESTS_COLLECTED)
    assert session.exitstatus == pytest.ExitCode.OK
