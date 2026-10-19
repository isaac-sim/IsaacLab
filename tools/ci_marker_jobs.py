# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report which workflow test jobs select at least one test file carrying a pytest marker.

``build.yaml`` runs this in its config job so that a marker run skips jobs with nothing to run
before they wait for a GPU runner and pull the CI image. It applies the same file selection as
``tools/conftest.py`` to each job's ``run-package-tests`` inputs.

Prints ``marker_jobs=<JSON object of job id to bool>`` for ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_selection import collect_test_files, files_with_marker  # isort: skip

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PACKAGE_TESTS = "./.github/actions/run-package-tests"
# The only expression the selection inputs use: a value that applies to push runs alone.
EVENT_EXPRESSION = re.compile(r"^\$\{\{\s*github\.event_name == '(\w+)' && '([^']*)' \|\| ''\s*\}\}$")

# run-package-tests inputs that decide which test files a job runs.
SELECTION_INPUTS = (
    "filter-pattern",
    "exclude-pattern",
    "include-files",
    "quarantined-only",
    "curobo-only",
    "shard-index",
    "shard-count",
    "test-node-ids-file",
    "test-node-ids-key",
)


def registered_markers() -> set[str]:
    with open(os.path.join(WORKSPACE_ROOT, "pyproject.toml"), "rb") as f:
        markers = tomllib.load(f)["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip() for entry in markers}


def resolve(value, event_name: str) -> str | None:
    """Return an input's value for this event, or None for an expression this script cannot evaluate."""
    value = str(value)
    if "${{" not in value:
        return value
    match = EVENT_EXPRESSION.match(value.strip())
    if match is None:
        return None
    return match[2] if match[1] == event_name else ""


def node_id_files(node_ids_file: str, node_ids_key: str) -> set[str]:
    with open(os.path.join(WORKSPACE_ROOT, node_ids_file), "rb") as f:
        node_ids = tomllib.load(f)[node_ids_key]
    return {os.path.normpath(os.path.join(WORKSPACE_ROOT, node_id.split("::", 1)[0])) for node_id in node_ids}


def marked_files(inputs: dict[str, str], marker: str) -> list[str]:
    """Select a job's test files the way ``run_tests.sh`` and ``tools/conftest.py`` do, then keep marked ones."""
    filter_pattern = inputs.get("filter-pattern", "")
    exclude_pattern = inputs.get("exclude-pattern", "")
    # run_tests.sh turns the legacy "not <pattern>" filter into an exclude entry.
    if filter_pattern.startswith("not "):
        exclude_pattern = ",".join(p for p in (exclude_pattern, filter_pattern.removeprefix("not ")) if p)
        filter_pattern = ""
    include_files = {f.strip() for f in inputs.get("include-files", "").split(",") if f.strip()}
    configured_files = set()
    if inputs.get("test-node-ids-file") and inputs.get("test-node-ids-key"):
        configured_files = node_id_files(inputs["test-node-ids-file"], inputs["test-node-ids-key"])
        include_files.update(os.path.basename(path) for path in configured_files)

    test_files = collect_test_files(
        [os.path.join(WORKSPACE_ROOT, "scripts"), os.path.join(WORKSPACE_ROOT, "source")],
        filter_pattern,
        exclude_pattern,
        include_files,
        inputs.get("quarantined-only", "false") == "true",
        inputs.get("curobo-only", "false") == "true",
        shard_index=inputs.get("shard-index", ""),
        shard_count=inputs.get("shard-count", ""),
    )
    if configured_files:
        test_files = [f for f in test_files if os.path.normpath(f) in configured_files]
    return files_with_marker(test_files, marker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workflow", required=True, help="Workflow file converted to JSON (yq -o=json).")
    parser.add_argument("--marker", required=True, help="Single registered pytest marker.")
    parser.add_argument("--event-name", required=True, help="github.event_name of the run.")
    args = parser.parse_args()

    if args.marker not in registered_markers():
        print(f"::error::Marker {args.marker!r} is not a single marker registered in pyproject.toml")
        return 1

    with open(args.workflow) as f:
        jobs = json.load(f)["jobs"]
    result = {}
    for job_id, job in jobs.items():
        for step in job.get("steps", []):
            inputs = step.get("with") or {}
            # Jobs that ignore the marker (e.g. performance-smoke) do not forward it and always run.
            if step.get("uses") != RUN_PACKAGE_TESTS or "inputs.marker" not in str(inputs.get("ci-marker", "")):
                continue
            resolved = {key: resolve(inputs[key], args.event_name) for key in SELECTION_INPUTS if key in inputs}
            unresolved = sorted(key for key, value in resolved.items() if value is None)
            if unresolved:
                # Running a job that turns out to be empty is cheaper than skipping one that is not.
                result[job_id] = True
                print(f"{job_id}: runs, cannot evaluate {', '.join(unresolved)}", file=sys.stderr)
                continue
            files = marked_files(resolved, args.marker)
            result[job_id] = bool(files)
            print(f"{job_id}: {len(files)} file(s) marked {args.marker}", file=sys.stderr)

    print(f"marker_jobs={json.dumps(result, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
