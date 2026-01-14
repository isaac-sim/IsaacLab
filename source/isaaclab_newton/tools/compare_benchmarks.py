# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark comparison tool for ArticulationData performance analysis.

This script compares two benchmark JSON files and identifies performance
regressions and improvements.

Usage:
    python compare_benchmarks.py baseline.json current.json [--threshold 10]

Example:
    python compare_benchmarks.py articulation_data_2025-12-01.json articulation_data_2025-12-11.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    """Type of performance change."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    UNCHANGED = "unchanged"
    NEW = "new"
    REMOVED = "removed"


@dataclass
class PropertyComparison:
    """Comparison result for a single property."""

    name: str
    baseline_mean_us: float | None
    current_mean_us: float | None
    baseline_std_us: float | None
    current_std_us: float | None
    absolute_change_us: float | None
    percent_change: float | None
    change_type: ChangeType
    combined_std_us: float | None = None
    """Combined standard deviation: sqrt(baseline_std^2 + current_std^2)."""
    sigma_change: float | None = None
    """Change expressed in units of combined standard deviation."""


def load_benchmark(filepath: str) -> dict:
    """Load a benchmark JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(filepath) as f:
        return json.load(f)


def extract_results(benchmark_data: dict) -> dict[str, dict]:
    """Extract results from benchmark data into a lookup dict.

    Args:
        benchmark_data: Parsed benchmark JSON data.

    Returns:
        Dictionary mapping property names to their result data.
    """
    results = {}
    for result in benchmark_data.get("results", []):
        name = result.get("name")
        if name:
            results[name] = result
    return results


def compare_benchmarks(
    baseline_data: dict,
    current_data: dict,
    regression_threshold: float = 10.0,
    improvement_threshold: float = 10.0,
    sigma_threshold: float = 1.0,
) -> list[PropertyComparison]:
    """Compare two benchmark results.

    A change is only considered significant if it exceeds BOTH:
    1. The percentage threshold (regression_threshold or improvement_threshold)
    2. The sigma threshold (change must be > sigma_threshold * combined_std)

    This prevents flagging changes that are within statistical noise.

    Args:
        baseline_data: Baseline benchmark JSON data.
        current_data: Current benchmark JSON data.
        regression_threshold: Percent increase to consider a regression.
        improvement_threshold: Percent decrease to consider an improvement.
        sigma_threshold: Number of combined standard deviations the change must
            exceed to be considered significant. Default 1.0 means changes within
            1 std dev of combined uncertainty are considered unchanged.

    Returns:
        List of PropertyComparison objects.
    """
    baseline_results = extract_results(baseline_data)
    current_results = extract_results(current_data)

    all_properties = set(baseline_results.keys()) | set(current_results.keys())
    comparisons = []

    for prop_name in sorted(all_properties):
        baseline = baseline_results.get(prop_name)
        current = current_results.get(prop_name)

        if baseline is None:
            # New property in current (current must exist since prop is in all_properties)
            assert current is not None
            comparisons.append(
                PropertyComparison(
                    name=prop_name,
                    baseline_mean_us=None,
                    current_mean_us=current["mean_time_us"],
                    baseline_std_us=None,
                    current_std_us=current["std_time_us"],
                    absolute_change_us=None,
                    percent_change=None,
                    change_type=ChangeType.NEW,
                )
            )
        elif current is None:
            # Property removed in current
            comparisons.append(
                PropertyComparison(
                    name=prop_name,
                    baseline_mean_us=baseline["mean_time_us"],
                    current_mean_us=None,
                    baseline_std_us=baseline["std_time_us"],
                    current_std_us=None,
                    absolute_change_us=None,
                    percent_change=None,
                    change_type=ChangeType.REMOVED,
                )
            )
        else:
            # Both exist - compare
            baseline_mean = baseline["mean_time_us"]
            current_mean = current["mean_time_us"]
            baseline_std = baseline["std_time_us"]
            current_std = current["std_time_us"]
            absolute_change = current_mean - baseline_mean

            # Compute combined standard deviation
            combined_std = math.sqrt(baseline_std**2 + current_std**2)

            # Compute change in units of sigma
            if combined_std > 0:
                sigma_change = absolute_change / combined_std
            else:
                sigma_change = float("inf") if absolute_change != 0 else 0.0

            if baseline_mean > 0:
                percent_change = (absolute_change / baseline_mean) * 100
            else:
                percent_change = 0.0 if current_mean == 0 else float("inf")

            # Determine change type:
            # A change is significant only if it exceeds BOTH the percentage threshold
            # AND the sigma threshold (i.e., the change is outside statistical noise)
            is_statistically_significant = abs(sigma_change) > sigma_threshold

            if percent_change > regression_threshold and is_statistically_significant:
                change_type = ChangeType.REGRESSION
            elif percent_change < -improvement_threshold and is_statistically_significant:
                change_type = ChangeType.IMPROVEMENT
            else:
                change_type = ChangeType.UNCHANGED

            comparisons.append(
                PropertyComparison(
                    name=prop_name,
                    baseline_mean_us=baseline_mean,
                    current_mean_us=current_mean,
                    baseline_std_us=baseline_std,
                    current_std_us=current_std,
                    absolute_change_us=absolute_change,
                    percent_change=percent_change,
                    change_type=change_type,
                    combined_std_us=combined_std,
                    sigma_change=sigma_change,
                )
            )

    return comparisons


def print_metadata_comparison(baseline_data: dict, current_data: dict):
    """Print comparison of metadata between two benchmarks.

    Args:
        baseline_data: Baseline benchmark JSON data.
        current_data: Current benchmark JSON data.
    """
    print("\n" + "=" * 115)
    print("BENCHMARK COMPARISON")
    print("=" * 115)

    baseline_meta = baseline_data.get("metadata", {})
    current_meta = current_data.get("metadata", {})

    # Repository info
    baseline_repo = baseline_meta.get("repository", {})
    current_repo = current_meta.get("repository", {})

    print(f"\n{'':30} {'BASELINE':>30} {'CURRENT':>30}")
    print("-" * 100)
    print(f"{'Timestamp:':<30} {baseline_meta.get('timestamp', 'N/A'):>30} {current_meta.get('timestamp', 'N/A'):>30}")
    print(
        f"{'Commit:':<30} {baseline_repo.get('commit_hash_short', 'N/A'):>30} {current_repo.get('commit_hash_short', 'N/A'):>30}"
    )
    print(f"{'Branch:':<30} {baseline_repo.get('branch', 'N/A'):>30} {current_repo.get('branch', 'N/A'):>30}")

    # Config
    baseline_config = baseline_meta.get("config", {})
    current_config = current_meta.get("config", {})

    print(f"\n{'Configuration:':<30}")
    print(
        f"{'  Iterations:':<30} {baseline_config.get('num_iterations', 'N/A'):>30} {current_config.get('num_iterations', 'N/A'):>30}"
    )
    print(
        f"{'  Instances:':<30} {baseline_config.get('num_instances', 'N/A'):>30} {current_config.get('num_instances', 'N/A'):>30}"
    )
    print(
        f"{'  Bodies:':<30} {baseline_config.get('num_bodies', 'N/A'):>30} {current_config.get('num_bodies', 'N/A'):>30}"
    )
    print(
        f"{'  Joints:':<30} {baseline_config.get('num_joints', 'N/A'):>30} {current_config.get('num_joints', 'N/A'):>30}"
    )

    # Hardware
    baseline_hw = baseline_meta.get("hardware", {})
    current_hw = current_meta.get("hardware", {})

    baseline_gpu = baseline_hw.get("gpu", {})
    current_gpu = current_hw.get("gpu", {})

    baseline_gpu_name = "N/A"
    current_gpu_name = "N/A"
    if baseline_gpu.get("devices"):
        baseline_gpu_name = baseline_gpu["devices"][0].get("name", "N/A")
    if current_gpu.get("devices"):
        current_gpu_name = current_gpu["devices"][0].get("name", "N/A")

    print(f"\n{'Hardware:':<30}")
    print(f"{'  GPU:':<30} {baseline_gpu_name:>30} {current_gpu_name:>30}")


def print_comparison_results(
    comparisons: list[PropertyComparison],
    show_unchanged: bool = False,
):
    """Print comparison results.

    Args:
        comparisons: List of property comparisons.
        show_unchanged: Whether to show unchanged properties.
    """
    # Separate by change type
    regressions = [c for c in comparisons if c.change_type == ChangeType.REGRESSION]
    improvements = [c for c in comparisons if c.change_type == ChangeType.IMPROVEMENT]
    unchanged = [c for c in comparisons if c.change_type == ChangeType.UNCHANGED]
    new_props = [c for c in comparisons if c.change_type == ChangeType.NEW]
    removed_props = [c for c in comparisons if c.change_type == ChangeType.REMOVED]

    # Sort regressions by percent change (worst first)
    regressions.sort(key=lambda x: x.percent_change or 0, reverse=True)
    # Sort improvements by percent change (best first)
    improvements.sort(key=lambda x: x.percent_change or 0)

    # Print regressions
    if regressions:
        print("\n" + "=" * 115)
        print(f"🔴 REGRESSIONS ({len(regressions)} properties)")
        print("=" * 115)
        print(
            f"\n{'Property':<35} {'Baseline (µs)':>12} {'Current (µs)':>12} {'Change':>12} {'% Change':>10} {'σ Change':>10}"
        )
        print("-" * 115)
        for comp in regressions:
            change_str = f"+{comp.absolute_change_us:.2f}" if comp.absolute_change_us else "N/A"
            pct_str = f"+{comp.percent_change:.1f}%" if comp.percent_change else "N/A"
            sigma_str = f"+{comp.sigma_change:.1f}σ" if comp.sigma_change else "N/A"
            print(
                f"{comp.name:<35} {comp.baseline_mean_us:>12.2f} {comp.current_mean_us:>12.2f} "
                f"{change_str:>12} {pct_str:>10} {sigma_str:>10}"
            )

    # Print improvements
    if improvements:
        print("\n" + "=" * 115)
        print(f"🟢 IMPROVEMENTS ({len(improvements)} properties)")
        print("=" * 115)
        print(
            f"\n{'Property':<35} {'Baseline (µs)':>12} {'Current (µs)':>12} {'Change':>12} {'% Change':>10} {'σ Change':>10}"
        )
        print("-" * 115)
        for comp in improvements:
            change_str = f"{comp.absolute_change_us:.2f}" if comp.absolute_change_us else "N/A"
            pct_str = f"{comp.percent_change:.1f}%" if comp.percent_change else "N/A"
            sigma_str = f"{comp.sigma_change:.1f}σ" if comp.sigma_change else "N/A"
            print(
                f"{comp.name:<35} {comp.baseline_mean_us:>12.2f} {comp.current_mean_us:>12.2f} "
                f"{change_str:>12} {pct_str:>10} {sigma_str:>10}"
            )

    # Print unchanged (if requested)
    if show_unchanged and unchanged:
        print("\n" + "=" * 115)
        print(f"⚪ UNCHANGED ({len(unchanged)} properties)")
        print("=" * 115)
        print(
            f"\n{'Property':<35} {'Baseline (µs)':>12} {'Current (µs)':>12} {'Change':>12} {'% Change':>10} {'σ Change':>10}"
        )
        print("-" * 115)
        for comp in unchanged:
            change_str = f"{comp.absolute_change_us:+.2f}" if comp.absolute_change_us else "N/A"
            pct_str = f"{comp.percent_change:+.1f}%" if comp.percent_change else "N/A"
            sigma_str = f"{comp.sigma_change:+.1f}σ" if comp.sigma_change else "N/A"
            print(
                f"{comp.name:<35} {comp.baseline_mean_us:>12.2f} {comp.current_mean_us:>12.2f} "
                f"{change_str:>12} {pct_str:>10} {sigma_str:>10}"
            )

    # Print new properties
    if new_props:
        print("\n" + "=" * 115)
        print(f"🆕 NEW PROPERTIES ({len(new_props)} properties)")
        print("=" * 115)
        for comp in new_props:
            print(f"  - {comp.name}: {comp.current_mean_us:.2f} µs")

    # Print removed properties
    if removed_props:
        print("\n" + "=" * 115)
        print(f"❌ REMOVED PROPERTIES ({len(removed_props)} properties)")
        print("=" * 115)
        for comp in removed_props:
            print(f"  - {comp.name}: was {comp.baseline_mean_us:.2f} µs")

    # Print summary
    print("\n" + "=" * 115)
    print("SUMMARY")
    print("=" * 115)
    total = len(comparisons)
    print(f"\n  Total properties compared: {total}")
    print(f"  🔴 Regressions:  {len(regressions):>4} ({100 * len(regressions) / total:.1f}%)")
    print(f"  🟢 Improvements: {len(improvements):>4} ({100 * len(improvements) / total:.1f}%)")
    print(f"  ⚪ Unchanged:    {len(unchanged):>4} ({100 * len(unchanged) / total:.1f}%)")
    if new_props:
        print(f"  🆕 New:          {len(new_props):>4}")
    if removed_props:
        print(f"  ❌ Removed:      {len(removed_props):>4}")

    # Overall verdict
    print("\n" + "-" * 115)
    if regressions:
        print(f"  ⚠️  VERDICT: {len(regressions)} regression(s) detected!")
        return 1  # Exit code for CI
    else:
        print("  ✅ VERDICT: No regressions detected.")
        return 0


def export_comparison_json(
    comparisons: list[PropertyComparison],
    baseline_data: dict,
    current_data: dict,
    filename: str,
):
    """Export comparison results to JSON.

    Args:
        comparisons: List of property comparisons.
        baseline_data: Baseline benchmark data.
        current_data: Current benchmark data.
        filename: Output filename.
    """
    output = {
        "baseline": {
            "file": baseline_data.get("metadata", {}).get("timestamp", "Unknown"),
            "commit": baseline_data.get("metadata", {}).get("repository", {}).get("commit_hash_short", "Unknown"),
        },
        "current": {
            "file": current_data.get("metadata", {}).get("timestamp", "Unknown"),
            "commit": current_data.get("metadata", {}).get("repository", {}).get("commit_hash_short", "Unknown"),
        },
        "regressions": [],
        "improvements": [],
        "unchanged": [],
        "new": [],
        "removed": [],
    }

    for comp in comparisons:
        entry = {
            "name": comp.name,
            "baseline_mean_us": comp.baseline_mean_us,
            "current_mean_us": comp.current_mean_us,
            "baseline_std_us": comp.baseline_std_us,
            "current_std_us": comp.current_std_us,
            "absolute_change_us": comp.absolute_change_us,
            "percent_change": comp.percent_change,
            "combined_std_us": comp.combined_std_us,
            "sigma_change": comp.sigma_change,
        }

        if comp.change_type == ChangeType.REGRESSION:
            output["regressions"].append(entry)
        elif comp.change_type == ChangeType.IMPROVEMENT:
            output["improvements"].append(entry)
        elif comp.change_type == ChangeType.UNCHANGED:
            output["unchanged"].append(entry)
        elif comp.change_type == ChangeType.NEW:
            output["new"].append(entry)
        elif comp.change_type == ChangeType.REMOVED:
            output["removed"].append(entry)

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nComparison exported to {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two ArticulationData benchmark JSON files and find regressions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline benchmark JSON file.",
    )
    parser.add_argument(
        "current",
        type=str,
        help="Path to current benchmark JSON file.",
    )
    parser.add_argument(
        "--regression-threshold",
        "-r",
        type=float,
        default=10.0,
        help="Percent increase threshold to consider a regression.",
    )
    parser.add_argument(
        "--improvement-threshold",
        "-i",
        type=float,
        default=10.0,
        help="Percent decrease threshold to consider an improvement.",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        default=1.0,
        help=(
            "Number of standard deviations the change must exceed to be significant. "
            "Changes within this many std devs of combined uncertainty are considered noise."
        ),
    )
    parser.add_argument(
        "--show-unchanged",
        "-u",
        action="store_true",
        help="Show unchanged properties in output.",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        default=None,
        help="Export comparison results to JSON file.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with code 1 if regressions are found.",
    )

    args = parser.parse_args()

    # Load benchmark files
    try:
        baseline_data = load_benchmark(args.baseline)
        current_data = load_benchmark(args.current)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    # Print metadata comparison
    print_metadata_comparison(baseline_data, current_data)

    # Compare benchmarks
    comparisons = compare_benchmarks(
        baseline_data,
        current_data,
        regression_threshold=args.regression_threshold,
        improvement_threshold=args.improvement_threshold,
        sigma_threshold=args.sigma,
    )

    # Print results
    exit_code = print_comparison_results(comparisons, show_unchanged=args.show_unchanged)

    # Export if requested
    if args.export:
        export_comparison_json(comparisons, baseline_data, current_data, args.export)

    # Exit with appropriate code in CI mode
    if args.ci:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
