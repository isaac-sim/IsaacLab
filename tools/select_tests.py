# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select tests to run based on changed files and a coverage-based dependency mapping.

This script is used by CI to determine which test files need to run for a given PR.
It loads a pre-computed mapping of source files to test files (produced by
collect_coverage_map.py) and intersects it with the set of changed files.

Usage:
    python tools/select_tests.py [--mapping PATH] [--base-branch BRANCH]
                                 [--max-age-days N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import test settings for category lists
sys.path.insert(0, str(Path(__file__).parent))
import test_settings  # noqa: E402

# Directories whose changes are ignored (no tests needed)
_IGNORE_PREFIXES = ("docs/",)

# Directory prefixes that trigger full fallback (affect build/test infrastructure)
_INFRA_PREFIXES = (".github/", "docker/")

# Infrastructure files that trigger full fallback
_INFRASTRUCTURE_FILES = {
    "tools/conftest.py",
    "tools/test_settings.py",
    "tools/select_tests.py",
    "tools/collect_coverage_map.py",
    "pyproject.toml",
}

# Patterns for infrastructure files matched by suffix
_INFRASTRUCTURE_SUFFIXES = ("setup.py", "config/extension.toml")

# All CI job names (used for initializing the output dict)
_ALL_JOBS = [
    "test-physx",
    "test-newton",
    "test-general",
    "test-isaaclab-tasks",
    "test-isaaclab-tasks-2",
    "test-environments-training",
    "test-flaky",
    "test-slightly-flaky",
    "test-curobo",
]

# Classifications that trigger full fallback
_FALLBACK_CLASSIFICATIONS = {"unmapped", "non_python_source", "infrastructure", "apps", "deleted_source", "renamed"}

# Tasks 1/2 file list (mirrors build.yaml test-isaaclab-tasks include-files).
# Keep in sync with the include-files lists in .github/workflows/build.yaml.
_TASKS_1_FILES = {
    "test_multi_agent_environments.py",
    "test_pickplace_stack_environments.py",
    "test_environments.py",
    "test_factory_environments.py",
    "test_cartpole_showcase_environments.py",
    "test_teleop_environments.py",
}

# Tasks 2/2 file list (mirrors build.yaml test-isaaclab-tasks-2 include-files).
# Keep in sync with the include-files lists in .github/workflows/build.yaml.
_TASKS_2_FILES = {
    "test_teleop_environments_with_stage_in_memory.py",
    "test_lift_teddy_bear.py",
    "test_environment_determinism.py",
    "test_hydra.py",
    "test_rl_device_separation.py",
    "test_cartpole_showcase_environments_with_stage_in_memory.py",
    "test_environments_with_stage_in_memory.py",
}


def load_mapping(path: str) -> dict | None:
    """Load the test dependency mapping from a JSON file.

    Args:
        path: Path to the mapping JSON file.

    Returns:
        The parsed mapping dict, or None if the file is missing, empty, or invalid.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not data or "source_to_tests" not in data:
        return None

    return data


def is_mapping_stale(metadata: dict, max_age_days: int) -> bool:
    """Check if the mapping is too old to trust.

    Args:
        metadata: The "metadata" dict from the mapping file.
        max_age_days: Maximum age in days before considering the mapping stale.

    Returns:
        True if the mapping is stale or has no timestamp.
    """
    generated_at = metadata.get("generated_at")
    if not generated_at:
        return True

    try:
        generated_dt = datetime.fromisoformat(generated_at)
        if generated_dt.tzinfo is None:
            generated_dt = generated_dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - generated_dt
        return age.days > max_age_days
    except (ValueError, TypeError):
        return True


def _is_test_file(path: str) -> bool:
    """Check if a path looks like a test file.

    Matches test files directly in test/ or in any subdirectory of test/.
    """
    return bool(re.match(r"(source|scripts)/[^/]+/test(/.*)?/test_[^/]+\.py$", path))


def _is_under_source_or_scripts(path: str) -> bool:
    return path.startswith("source/") or path.startswith("scripts/")


def classify_file(
    path: str,
    status: str,
    mapping_keys: set[str],
) -> str:
    """Classify a changed file for test selection purposes.

    Args:
        path: File path relative to repo root.
        status: Git diff status character (M, A, D, R, etc.).
        mapping_keys: Set of source file paths present in the coverage mapping.

    Returns:
        One of: 'test', 'mapped', 'unmapped', 'non_python_source', 'infrastructure',
        'apps', 'deleted_source', 'renamed', 'ignore'.
    """
    # Ignored directories
    if any(path.startswith(p) for p in _IGNORE_PREFIXES):
        return "ignore"

    # CI/build infrastructure directories
    if any(path.startswith(p) for p in _INFRA_PREFIXES):
        return "infrastructure"

    # Renamed files
    if status == "R":
        if _is_under_source_or_scripts(path):
            return "renamed"
        return "ignore"

    # Deleted files
    if status == "D":
        if _is_under_source_or_scripts(path) and path.endswith(".py"):
            return "deleted_source"
        return "ignore"

    # Apps directory
    if path.startswith("apps/"):
        return "apps"

    # Infrastructure files (exact match)
    if path in _INFRASTRUCTURE_FILES:
        return "infrastructure"

    # Infrastructure files (suffix match)
    if any(path.endswith(s) for s in _INFRASTRUCTURE_SUFFIXES):
        return "infrastructure"

    # Test files
    if _is_test_file(path):
        return "test"

    # Source/scripts files
    if _is_under_source_or_scripts(path):
        if not path.endswith(".py"):
            return "non_python_source"
        if path in mapping_keys:
            return "mapped"
        return "unmapped"

    # Everything else (root files, etc.) — ignore
    return "ignore"


def assign_test_to_job(test_path: str) -> str:
    """Assign a test file to the correct CI job.

    Uses the file's full path and basename to determine which job should run it.
    Priority: special category lists > package path > default.

    Args:
        test_path: Full path to the test file, relative to repo root.

    Returns:
        Job name string (e.g. 'test-physx', 'test-general').
    """
    basename = test_path.rsplit("/", 1)[-1] if "/" in test_path else test_path

    # Special category lists take priority (these tests are excluded from normal jobs)
    if basename == "test_environments_training.py":
        return "test-environments-training"
    if basename in set(test_settings.FLAKY_TESTS):
        return "test-flaky"
    if basename in set(test_settings.SLIGHTLY_FLAKY_TESTS):
        return "test-slightly-flaky"
    if basename in set(test_settings.CUROBO_TESTS):
        return "test-curobo"

    # Package-based assignment using full path
    if "isaaclab_physx" in test_path:
        return "test-physx"
    if "isaaclab_newton" in test_path:
        return "test-newton"
    if "isaaclab_tasks" in test_path:
        if basename in _TASKS_1_FILES:
            return "test-isaaclab-tasks"
        if basename in _TASKS_2_FILES:
            return "test-isaaclab-tasks-2"
        # Tasks tests not in either split go to tasks-1 by default
        return "test-isaaclab-tasks"

    return "test-general"


def select_tests(
    mapping_path: str,
    changed_files: list[tuple[str, str]],
    max_age_days: int,
) -> dict:
    """Determine which tests to run based on changed files and the coverage mapping.

    Args:
        mapping_path: Path to the test-dependency-map.json file.
        changed_files: List of (status, path) tuples from git diff --name-status.
        max_age_days: Maximum mapping age in days before triggering full fallback.

    Returns:
        Dict with keys:
        - "run_all": bool indicating whether to run all tests.
        - "jobs": dict mapping job name to comma-separated test file paths.
        - "summary": human-readable summary string.
    """
    jobs = {job: set() for job in _ALL_JOBS}
    result = {"run_all": False, "jobs": {}, "summary": ""}

    # Load mapping
    mapping_data = load_mapping(mapping_path)
    if mapping_data is None:
        result["run_all"] = True
        result["jobs"] = {job: "" for job in _ALL_JOBS}
        result["summary"] = "Mapping not available. Falling back to all tests."
        return result

    # Check staleness
    if is_mapping_stale(mapping_data.get("metadata", {}), max_age_days):
        result["run_all"] = True
        result["jobs"] = {job: "" for job in _ALL_JOBS}
        result["summary"] = f"Mapping is older than {max_age_days} days. Falling back to all tests."
        return result

    source_to_tests = mapping_data["source_to_tests"]
    mapping_keys = set(source_to_tests.keys())

    # Classify each changed file and collect tests
    classifications = {}
    for status, path in changed_files:
        classification = classify_file(path, status, mapping_keys)
        classifications[path] = classification

        if classification in _FALLBACK_CLASSIFICATIONS:
            result["run_all"] = True
            result["jobs"] = {job: "" for job in _ALL_JOBS}
            result["summary"] = f"Full fallback triggered by {classification} file: {path}"
            return result

        if classification == "test":
            job = assign_test_to_job(path)
            jobs[job].add(path)
        elif classification == "mapped":
            for test_path in source_to_tests[path]:
                job = assign_test_to_job(test_path)
                jobs[job].add(test_path)
        # 'ignore' requires no action

    # Build output
    total_selected = sum(len(v) for v in jobs.values())
    total_tests = mapping_data.get("metadata", {}).get("test_file_count", "?")
    result["jobs"] = {job: ",".join(sorted(tests)) for job, tests in jobs.items()}

    n_changed = len(changed_files)
    n_mapped = sum(1 for c in classifications.values() if c == "mapped")
    n_ignored = sum(1 for c in classifications.values() if c == "ignore")
    n_test = sum(1 for c in classifications.values() if c == "test")

    if total_tests != "?" and total_tests > 0:
        pct = (1 - total_selected / total_tests) * 100
        result["summary"] = (
            f"Selected {total_selected}/{total_tests} tests ({pct:.0f}% reduction). "
            f"{n_changed} files changed: {n_mapped} mapped, {n_test} test files, {n_ignored} ignored."
        )
    else:
        result["summary"] = f"Selected {total_selected} tests. {n_changed} files changed."

    return result


def parse_git_diff_output(lines: list[str]) -> list[tuple[str, str]]:
    """Parse the output of ``git diff --name-status`` into (status, path) tuples.

    Handles rename lines (Rxx<TAB>old<TAB>new) by extracting only the new path.

    Args:
        lines: Lines of git diff --name-status output.

    Returns:
        List of (status, path) tuples. Status is a single character (M, A, D, R).
    """
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]

        if status.startswith("R"):
            # Rename: status is like R100, and there are two paths (old, new)
            new_path = parts[2] if len(parts) >= 3 else parts[1]
            results.append(("R", new_path))
        else:
            # M, A, D, C, etc.
            path = parts[1] if len(parts) >= 2 else ""
            results.append((status[0], path))

    return results


def get_changed_files(base_branch: str) -> list[tuple[str, str]]:
    """Get the list of changed files between HEAD and the base branch.

    Args:
        base_branch: The base branch to diff against (e.g. 'origin/main').

    Returns:
        List of (status, path) tuples.
    """
    cmd = ["git", "diff", "--name-status", f"{base_branch}...HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_git_diff_output(result.stdout.strip().split("\n"))


def main():
    parser = argparse.ArgumentParser(description="Select tests based on changed files and coverage mapping.")
    parser.add_argument("--mapping", default="tools/test-dependency-map.json", help="Path to the mapping JSON file.")
    parser.add_argument("--base-branch", default="origin/main", help="Base branch to diff against.")
    parser.add_argument("--max-age-days", type=int, default=7, help="Maximum mapping age in days.")
    parser.add_argument("--dry-run", action="store_true", help="Print selection rationale to stderr.")

    args = parser.parse_args()

    # Get changed files
    changed_files = get_changed_files(args.base_branch)

    # Run selection
    result = select_tests(args.mapping, changed_files, args.max_age_days)

    # Dry-run output to stderr
    if args.dry_run:
        print(result["summary"], file=sys.stderr)
        mapping_data = load_mapping(args.mapping)
        mapping_keys = set(mapping_data.get("source_to_tests", {}).keys()) if mapping_data else set()
        for status, path in changed_files:
            classification = classify_file(path, status, mapping_keys)
            print(f"  {status} {path} -> {classification}", file=sys.stderr)

    # Output JSON to stdout
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
