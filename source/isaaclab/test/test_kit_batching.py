# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Kit batching grouping and JUnit demultiplexing.

Both are pure functions over paths and strings, so they run anywhere; the process machinery
they feed is POSIX-only and only exercisable in CI.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from junitparser import JUnitXml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from _kit_batching import (  # noqa: E402
    Batch,
    batch_size,
    batching_enabled,
    file_profile,
    group_test_files,
    split_batch_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


KIT = "pytestmark = pytest.mark.kit\n"
CAMERAS = "pytestmark = [pytest.mark.kit_cameras, pytest.mark.integration]\n"
SOLO = "pytestmark = [pytest.mark.kit, pytest.mark.kit_solo]\n"
KITLESS = "pytestmark = pytest.mark.kitless\n"
LEGACY = "simulation_app = AppLauncher(headless=True).app\n"


class TestFileProfile:
    """`file_profile` classifies a file from its marker text."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            (KIT, "kit"),
            (CAMERAS, "kit_cameras"),
            (SOLO, None),
            (KITLESS, None),
            (LEGACY, None),
            ("", None),
        ],
    )
    def test_profile_matches_markers(self, source: str, expected: str | None):
        assert file_profile(source) == expected

    def test_kit_pattern_does_not_swallow_the_longer_markers(self):
        """A bare `kit` match must not claim kit_cameras or kit_solo files."""
        assert file_profile("pytest.mark.kit_cameras") == "kit_cameras"
        assert file_profile("pytest.mark.kit_solo") is None
        assert file_profile("pytest.mark.kitless") is None


class TestGrouping:
    """`group_test_files` batches same-profile files and isolates everything else."""

    def test_same_profile_files_share_one_batch(self):
        files = ["a.py", "b.py", "c.py"]
        batches = group_test_files(files, dict.fromkeys(files, KIT))
        assert len(batches) == 1
        assert batches[0].profile == "kit"
        assert batches[0].files == files

    def test_profiles_never_mix(self):
        sources = {"a.py": KIT, "b.py": CAMERAS, "c.py": KIT}
        batches = group_test_files(list(sources), sources)
        by_profile = {b.profile: b.files for b in batches}
        assert by_profile["kit"] == ["a.py", "c.py"]
        assert by_profile["kit_cameras"] == ["b.py"]

    @pytest.mark.parametrize("source", [SOLO, KITLESS, LEGACY])
    def test_unbatchable_files_get_their_own_batch(self, source: str):
        sources = {"a.py": KIT, "b.py": source, "c.py": KIT}
        batches = group_test_files(list(sources), sources)
        solo = [b for b in batches if b.files == ["b.py"]]
        assert solo and solo[0].profile is None
        assert not solo[0].is_batched

    def test_explicit_unbatchable_overrides_the_marker(self):
        sources = {"a.py": KIT, "b.py": KIT}
        batches = group_test_files(list(sources), sources, unbatchable={"b.py"})
        assert Batch(profile=None, files=["b.py"]) in batches

    def test_missing_source_is_treated_as_unbatchable(self):
        """An unreadable file must not be assumed safe to share a process."""
        batches = group_test_files(["a.py", "b.py"], {"a.py": KIT})
        assert any(b.files == ["b.py"] and b.profile is None for b in batches)

    def test_batches_are_capped(self):
        files = [f"f{i}.py" for i in range(7)]
        batches = group_test_files(files, dict.fromkeys(files, KIT), max_size=3)
        assert [len(b.files) for b in batches] == [3, 3, 1]

    def test_labels_are_unique_across_batches(self):
        """A label becomes a JUnit report filename, so two batches must never collide.

        Two same-profile batches of equal size are the case that matters: without the index
        they would produce the same label and the second would overwrite the first's report.
        """
        files = [f"f{i}.py" for i in range(6)]
        batches = group_test_files(files, dict.fromkeys(files, KIT), max_size=3)
        labels = [b.label for b in batches]
        assert len(batches) == 2
        assert len(labels) == len(set(labels)), f"colliding labels: {labels}"

    def test_every_file_appears_exactly_once(self):
        sources = {"a.py": KIT, "b.py": CAMERAS, "c.py": SOLO, "d.py": KIT, "e.py": LEGACY}
        batches = group_test_files(list(sources), sources)
        covered = [f for b in batches for f in b.files]
        assert sorted(covered) == sorted(sources)
        assert len(covered) == len(set(covered))


def _report(*cases: tuple[str, str, str, float]) -> JUnitXml:
    """Build a JUnit report from ``(classname, name, outcome, time)`` tuples."""
    body = "".join(
        f'<testcase classname="{cls}" name="{name}" time="{t}">'
        + {"pass": "", "fail": "<failure message='boom'/>", "error": "<error message='boom'/>", "skip": "<skipped/>"}[
            outcome
        ]
        + "</testcase>"
        for cls, name, outcome, t in cases
    )
    xml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites><testsuite name="pytest" tests="{len(cases)}">{body}</testsuite></testsuites>
        """)
    return JUnitXml.fromstring(xml.encode("utf-8"))


class TestSplitBatchStatus:
    """`split_batch_status` attributes a batch's report back to individual files."""

    def test_counts_are_attributed_per_file(self):
        report = _report(
            ("source.sim.test_a", "test_one", "pass", 1.0),
            ("source.sim.test_a", "test_two", "fail", 2.0),
            ("source.sim.test_b", "test_three", "pass", 3.0),
        )
        status = split_batch_status(
            report, ["source/sim/test_a.py", "source/sim/test_b.py"], wall_time=60.0, batch_result="CRASHED"
        )
        a = status["source/sim/test_a.py"]
        b = status["source/sim/test_b.py"]
        assert (a["tests"], a["failures"], a["result"]) == (2, 1, "FAILED")
        assert (b["tests"], b["failures"], b["result"]) == (1, 0, "passed")
        assert a["time_elapsed"] == pytest.approx(3.0)
        assert b["time_elapsed"] == pytest.approx(3.0)

    def test_files_that_never_ran_take_the_batch_result(self):
        """A file with no testcases means the shared process died before reaching it."""
        report = _report(("source.sim.test_a", "test_one", "pass", 1.0))
        status = split_batch_status(
            report, ["source/sim/test_a.py", "source/sim/test_b.py"], wall_time=10.0, batch_result="CRASHED"
        )
        assert status["source/sim/test_a.py"]["result"] == "passed"
        assert status["source/sim/test_b.py"]["result"] == "CRASHED"
        assert status["source/sim/test_b.py"]["errors"] == 1

    def test_wall_time_is_shared_only_between_files_that_ran(self):
        report = _report(
            ("source.sim.test_a", "t", "pass", 1.0),
            ("source.sim.test_b", "t", "pass", 1.0),
        )
        files = ["source/sim/test_a.py", "source/sim/test_b.py", "source/sim/test_c.py"]
        status = split_batch_status(report, files, wall_time=90.0, batch_result="CRASHED")
        assert status["source/sim/test_a.py"]["wall_time"] == pytest.approx(45.0)
        assert status["source/sim/test_b.py"]["wall_time"] == pytest.approx(45.0)
        assert status["source/sim/test_c.py"]["wall_time"] == 0.0

    def test_errors_and_skips_are_counted_separately(self):
        report = _report(
            ("source.sim.test_a", "t1", "error", 0.5),
            ("source.sim.test_a", "t2", "skip", 0.0),
        )
        status = split_batch_status(report, ["source/sim/test_a.py"], wall_time=5.0, batch_result="CRASHED")
        a = status["source/sim/test_a.py"]
        assert (a["errors"], a["skipped"], a["result"]) == (1, 1, "FAILED")

    def test_ambiguous_stems_are_not_misattributed(self):
        """Two members sharing a basename cannot be told apart, so neither claims the case."""
        report = _report(("pkg.one.test_dup", "t", "pass", 1.0))
        files = ["pkg/one/test_dup.py", "pkg/two/test_dup.py"]
        status = split_batch_status(report, files, wall_time=10.0, batch_result="CRASHED")
        assert all(status[f]["result"] == "CRASHED" for f in files)


class TestEnvironmentToggles:
    """Batching stays off unless explicitly enabled."""

    @pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("0", False), ("", False)])
    def test_enable_flag(self, value: str, expected: bool):
        assert batching_enabled({"ISAACLAB_TEST_BATCH_KIT": value}) is expected

    def test_disabled_when_unset(self):
        assert batching_enabled({}) is False

    @pytest.mark.parametrize("value,expected", [("5", 5), ("", 12), ("nonsense", 12), ("0", 12), ("-3", 12)])
    def test_batch_size_override(self, value: str, expected: int):
        assert batch_size({"ISAACLAB_TEST_BATCH_SIZE": value}) == expected
