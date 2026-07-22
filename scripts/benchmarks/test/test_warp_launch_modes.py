# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight tests for the mixed Warp launch-mode benchmark."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "benchmarks" / "warp_launch_modes.py"


@pytest.fixture(scope="module")
def modes_module():
    """Load the standalone benchmark as a module without running its CLI."""
    spec = importlib.util.spec_from_file_location("warp_launch_modes_test_module", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_builds_deterministic_cycles_and_launch_counts(modes_module, capsys: pytest.CaptureFixture[str]):
    """The dry-run matrix should preserve the four-kernel cycle structure."""
    arguments = [
        "--dry_run",
        "--threads",
        "256,1024",
        "--stage_repeats",
        "1,4",
        "--modes",
        "eager,cache_eager",
        "--case_order",
        "randomized",
        "--case_seed",
        "17",
    ]
    assert modes_module.run(arguments) == 0
    first_output = capsys.readouterr().out
    assert modes_module.run(arguments) == 0
    second_output = capsys.readouterr().out

    assert first_output == second_output
    assert "Matrix cases: 8" in first_output
    assert "eager__threads_256__nodes_4" in first_output
    assert "cache_eager__threads_1024__nodes_16" in first_output


@pytest.mark.parametrize(
    ("mode", "total_executions", "expected_action_checksum"),
    [
        ("eager", 3, 2.0),
        ("bare_dynamic_normal", 2, 2.0),
        ("bare_dynamic_normal", 3, -2.0),
        ("bare_dynamic_ctype_constructed", 3, -2.0),
        ("bare_dynamic_ctype_prepacked", 3, -2.0),
    ],
)
def test_semantic_expectations_track_dynamic_action_bank(
    modes_module, mode: str, total_executions: int, expected_action_checksum: float
):
    """Analytical checksums should account for alternating dynamic action pointers."""
    case = modes_module._Case(mode=mode, threads=8, stage_repeats=1)
    expected = modes_module._semantic_expectations(case, total_executions)

    assert expected == {
        "action_checksum": expected_action_checksum,
        "observation_checksum": 4.8,
        "reward_checksum": 7.896,
        "reset_count": 0,
    }


def test_mode_parser_exposes_explicit_no_conversion_variants(modes_module):
    """Descriptor construction and prepacking must remain distinct CLI modes."""
    modes = modes_module._mode_list("bare_dynamic_ctype_constructed,bare_dynamic_ctype_prepacked,cache_eager")
    assert modes == (
        "bare_dynamic_ctype_constructed",
        "bare_dynamic_ctype_prepacked",
        "cache_eager",
    )
    with pytest.raises(argparse.ArgumentTypeError, match="unknown modes"):
        modes_module._mode_list("bare_dynamic_ctype")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown modes"):
        modes_module._mode_list("cache_dynamic_changed")


def test_default_case_order_is_randomized(modes_module):
    """The benchmark should randomize cases by default to limit time-order bias."""
    assert modes_module._parse_args([]).case_order == "randomized"


def test_dynamic_probe_rejects_a_noop_setter(modes_module):
    """Two same-sign outputs must not pass the explicit dynamic-pointer probe."""
    case = modes_module._Case(mode="bare_dynamic_normal", threads=8, stage_repeats=1)

    modes_module._validate_dynamic_probe(case, (-2.0, 2.0))
    with pytest.raises(RuntimeError, match="Dynamic action probe failed"):
        modes_module._validate_dynamic_probe(case, (2.0, 2.0))


def test_graph_activity_selection_prefers_graph_events(modes_module):
    """Graph modes should prefer one graph event when child kernel events are also available."""
    case = modes_module._Case(mode="graph_eager", threads=8, stage_repeats=1)
    kernel_results = [SimpleNamespace(elapsed=0.25) for _ in range(case.nodes_per_step)]
    graph_results = [SimpleNamespace(elapsed=0.5)]

    selected = modes_module._select_gpu_activity(case, kernel_results, graph_results)

    assert selected == (1.0, 0.5, 0.5, "graph", 1)


def test_graph_activity_selection_accepts_child_kernel_events(modes_module):
    """Graph modes should fall back to child kernels when no graph event is reported."""
    case = modes_module._Case(mode="graph_replay", threads=8, stage_repeats=1)
    kernel_results = [SimpleNamespace(elapsed=0.25) for _ in range(case.nodes_per_step)]

    selected = modes_module._select_gpu_activity(case, kernel_results, [])

    assert selected == (1.0, None, 1.0, "kernel", case.nodes_per_step)
