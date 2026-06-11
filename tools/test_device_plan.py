# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure test-run planner (``tools/_device_plan.py``).

The planner turns (files, runtime device mask, marker predicate) into a list of
``(file, mask)`` work units. It is pure: no I/O, no subprocess, no collection.
"""

from __future__ import annotations

import pytest
from _device_plan import is_isolated, plan_units

# ---- plan_units: the per-runner planning logic --------------------------------


def test_mix_ok_file_is_a_single_unit():
    # A file that does not need device isolation runs once, covering the whole mask.
    assert plan_units(["a.py"], "110", is_isolated=lambda f: False) == [("a.py", "110")]


def test_cant_mix_file_splits_into_one_unit_per_set_bit():
    # An isolated file on a multi-device runtime splits into one process per device.
    assert plan_units(["a.py"], "110", is_isolated=lambda f: True) == [("a.py", "100"), ("a.py", "010")]


def test_cant_mix_on_single_device_runtime_does_not_split():
    # An mgpu shard's runtime is one device, so even an isolated file is one unit.
    assert plan_units(["a.py"], "0001", is_isolated=lambda f: True) == [("a.py", "0001")]


def test_cant_mix_on_cpu_only_runtime_does_not_split():
    assert plan_units(["a.py"], "100", is_isolated=lambda f: True) == [("a.py", "100")]


def test_split_masks_use_each_set_bit_position():
    # Set bits at positions 0 and 2 -> two single-bit masks of the same width.
    assert plan_units(["a.py"], "1010", is_isolated=lambda f: True) == [("a.py", "1000"), ("a.py", "0010")]


def test_multiple_files_preserve_order_and_decide_per_file():
    def isolated(f):
        return f == "b.py"

    assert plan_units(["a.py", "b.py"], "110", is_isolated=isolated) == [
        ("a.py", "110"),
        ("b.py", "100"),
        ("b.py", "010"),
    ]


def test_runtime_mask_with_trailing_x_is_rejected():
    # Runtime masks are always concrete; the open-ended ``X`` form is a scope-only
    # construct and must never reach the planner.
    with pytest.raises(ValueError):
        plan_units(["a.py"], "11X", is_isolated=lambda f: True)


# ---- is_isolated: the device-isolation marker detector ------------------------


def test_single_mark_is_detected(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("import pytest\npytestmark = pytest.mark.device_isolated\ndef test_x(): pass\n")
    assert is_isolated(f) is True


def test_list_form_mark_is_detected(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("import pytest\npytestmark = [pytest.mark.device_isolated, pytest.mark.slow]\ndef test_x(): pass\n")
    assert is_isolated(f) is True


def test_preloaded_source_is_used():
    assert is_isolated("does_not_exist.py", source="pytestmark = pytest.mark.device_isolated\n") is True


def test_no_mark_is_not_detected(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("import pytest\ndef test_x(): pass\n")
    assert is_isolated(f) is False


def test_word_in_comment_does_not_match(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("# device_isolated explains the lock\ndef test_x(): pass\n")
    assert is_isolated(f) is False


def test_unrelated_pytestmark_does_not_match(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("import pytest\npytestmark = pytest.mark.slow\ndef test_x(): pass\n")
    assert is_isolated(f) is False


def test_missing_file_returns_false(tmp_path):
    assert is_isolated(tmp_path / "nope.py") is False
