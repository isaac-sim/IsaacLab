# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for test_runner.selector.DeviceSelect.

The class form is directly testable: construct DeviceSelect(mask) and call keeps()
or the hooks with fakes — no env var, no monkeypatch.
"""

from __future__ import annotations

import pytest

from test_runner.selector import DeviceSelect


class _Callspec:
    def __init__(self, params):
        self.params = params


class _Item:
    def __init__(self, name, params=None):
        self.name = name
        if params is not None:
            self.callspec = _Callspec(params)


class _Hook:
    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class _Config:
    def __init__(self):
        self.hook = _Hook()


# ---- pure mask logic ----


def test_position_maps_cpu_and_cuda():
    assert DeviceSelect._position("cpu") == 0
    assert DeviceSelect._position("cuda:0") == 1
    assert DeviceSelect._position("cuda:2") == 3
    assert DeviceSelect._position(None) is None


def test_active_handles_trailing_x():
    assert DeviceSelect("00X")._active(3) is True  # cuda:2 via the trailing fill
    assert DeviceSelect("110")._active(2) is False


def test_keeps_cpu_unit():
    s = DeviceSelect("100")
    assert s.keeps("cpu") is True
    assert s.keeps("cuda:0") is False  # literal cuda variant dropped from the cpu unit
    assert s.keeps(None) is True  # agnostic kept (cpu in mask)


def test_keeps_cuda_unit():
    s = DeviceSelect("010")
    assert s.keeps("cuda:0") is True
    assert s.keeps("cpu") is False  # literal cpu variant dropped from the cuda unit
    assert s.keeps(None) is False  # agnostic dropped (no cpu)


def test_keeps_shard():
    s = DeviceSelect("0001")  # cuda:2 shard
    assert s.keeps("cuda:2") is True
    assert s.keeps("cuda:0") is False
    assert s.keeps("cpu") is False
    assert s.keeps(None) is False


# ---- the collection hook ----


def test_modifyitems_drops_out_of_scope_and_agnostic_in_cuda_unit():
    cpu = _Item("t[cpu]", {"device": "cpu"})
    cuda = _Item("t[cuda:0]", {"device": "cuda:0"})
    agnostic = _Item("t_logic")
    items = [cpu, cuda, agnostic]
    config = _Config()
    DeviceSelect("010").pytest_collection_modifyitems(config, items)
    assert items == [cuda]
    assert set(config.hook.deselected) == {cpu, agnostic}


def test_modifyitems_keeps_everything_in_scope_for_default_mask():
    cpu = _Item("t[cpu]", {"device": "cpu"})
    cuda = _Item("t[cuda:0]", {"device": "cuda:0"})
    agnostic = _Item("t_logic")
    items = [cpu, cuda, agnostic]
    config = _Config()
    DeviceSelect("110").pytest_collection_modifyitems(config, items)
    assert items == [cpu, cuda, agnostic]
    assert config.hook.deselected == []


def test_sessionfinish_maps_no_tests_collected_to_ok():
    class _Session:
        exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED

    session = _Session()
    DeviceSelect("010").pytest_sessionfinish(session, pytest.ExitCode.NO_TESTS_COLLECTED)
    assert session.exitstatus == pytest.ExitCode.OK
