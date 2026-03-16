# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect per-test-file coverage and produce a source-to-test dependency mapping.

This script discovers all test files, runs each one with ``coverage run``, and
then combines the per-test coverage data into a JSON mapping suitable for use
by ``select_tests.py``.

Usage:
    python tools/collect_coverage_map.py [--output PATH] [--workers N] [--timeout SECS]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def discover_test_files(source_dirs: list[str]) -> list[str]:
    """Walk source directories and return all test_*.py file paths."""
    test_files = []
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
        for root, _, files in os.walk(source_dir):
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    test_files.append(os.path.join(root, f))
    return sorted(test_files)


def run_test_with_coverage(test_file: str, coverage_dir: str, timeout: int) -> str | None:
    """Run a single test file with coverage and return the coverage data file path.

    Args:
        test_file: Absolute path to the test file.
        coverage_dir: Directory to store .coverage.<hash> files.
        timeout: Maximum seconds to allow the test to run.

    Returns:
        Path to the coverage data file, or None if the test failed/timed out.
    """
    # Use a unique coverage data file per test
    test_hash = str(abs(hash(test_file)))
    data_file = os.path.join(coverage_dir, f".coverage.{test_hash}")

    cmd = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        f"--data-file={data_file}",
        "--source=source/,scripts/",
        "-m",
        "pytest",
        "--no-header",
        "-x",  # stop on first failure
        test_file,
    ]

    try:
        subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        # Even if the test failed, coverage data may have been written
        if os.path.exists(data_file):
            return data_file
        return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {test_file} (>{timeout}s)", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: {test_file}: {e}", file=sys.stderr)
        return None


def extract_covered_files(data_file: str) -> set[str]:
    """Extract the set of source files covered by a single test run.

    Args:
        data_file: Path to a .coverage data file.

    Returns:
        Set of source file paths (relative to repo root).
    """
    # Use coverage's JSON export to get the file list
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", f"--data-file={data_file}", "-o", tmp_path],
            capture_output=True,
            text=True,
            check=True,
        )
        with open(tmp_path) as f:
            cov_data = json.load(f)

        repo_root = str(Path(__file__).resolve().parent.parent)
        files = set()
        for abs_path in cov_data.get("files", {}):
            # Convert absolute paths to repo-relative
            if abs_path.startswith(repo_root):
                rel = os.path.relpath(abs_path, repo_root)
                files.add(rel)
        return files
    except Exception:
        return set()
    finally:
        os.unlink(tmp_path)


def build_mapping(
    test_files: list[str],
    coverage_dir: str,
    timeout: int,
    workers: int,
    repo_root: str,
) -> dict:
    """Run all tests with coverage and build the source-to-test mapping.

    Args:
        test_files: List of test file absolute paths.
        coverage_dir: Temp directory for coverage data files.
        timeout: Per-test timeout in seconds.
        workers: Number of parallel workers.
        repo_root: Repository root directory.

    Returns:
        The complete mapping dict ready for JSON serialization.
    """
    source_to_tests = defaultdict(set)
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for test_file in test_files:
            future = executor.submit(run_test_with_coverage, test_file, coverage_dir, timeout)
            futures[future] = test_file

        for future in as_completed(futures):
            test_file = futures[future]
            data_file = future.result()

            if data_file is None:
                failed += 1
                print(f"SKIP (no coverage): {test_file}", file=sys.stderr)
                continue

            covered_files = extract_covered_files(data_file)
            test_rel = os.path.relpath(test_file, repo_root)

            for src_file in covered_files:
                source_to_tests[src_file].add(test_rel)

            completed += 1
            print(f"[{completed}/{len(test_files)}] {test_file} -> {len(covered_files)} files", file=sys.stderr)

    total = len(test_files)
    print(f"\nCompleted: {completed}/{total}, Failed: {failed}/{total}", file=sys.stderr)

    # Check partial failure threshold
    if total > 0 and completed / total < 0.5:
        print("ERROR: Less than 50% of tests completed. Not writing mapping.", file=sys.stderr)
        sys.exit(1)

    # Convert sets to sorted lists for JSON
    mapping = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "commit": _get_current_commit(),
            "test_file_count": completed,
            "source_file_count": len(source_to_tests),
        },
        "source_to_tests": {k: sorted(v) for k, v in sorted(source_to_tests.items())},
    }

    return mapping


def _get_current_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Collect per-test coverage and produce dependency mapping.")
    parser.add_argument("--output", default="tools/test-dependency-map.json", help="Output mapping JSON path.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel test workers.")
    parser.add_argument("--timeout", type=int, default=2000, help="Per-test timeout in seconds.")

    args = parser.parse_args()

    repo_root = str(Path(__file__).resolve().parent.parent)
    source_dirs = [
        os.path.join(repo_root, "scripts"),
        os.path.join(repo_root, "source"),
    ]

    print("Discovering test files...", file=sys.stderr)
    test_files = discover_test_files(source_dirs)
    print(f"Found {len(test_files)} test files", file=sys.stderr)

    with tempfile.TemporaryDirectory() as coverage_dir:
        mapping = build_mapping(test_files, coverage_dir, args.timeout, args.workers, repo_root)

    output_path = os.path.join(repo_root, args.output)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Mapping written to {output_path}", file=sys.stderr)
    print(f"  {mapping['metadata']['test_file_count']} test files", file=sys.stderr)
    print(f"  {mapping['metadata']['source_file_count']} source files", file=sys.stderr)


if __name__ == "__main__":
    main()
