# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the test plan, the workflow generator, and the CI argument plumbing.

The plan is the single source of truth for what every CI job runs, so the things that can go
quietly wrong are: a workflow naming a job the plan does not define, checked-in YAML drifting
from what the generator produces, a job resolving to nothing, and the positional arguments
``run-tests/action.yml`` passes to ``run_tests.sh`` sliding out of alignment with the ``local``
bindings at the top of that script. Each has a test here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
for _package in sorted((REPO_ROOT / "source").iterdir()):
    if (_package / _package.name).is_dir():
        sys.path.insert(0, str(_package))

import generate_workflows  # noqa: E402
import testplan  # noqa: E402

pytestmark = pytest.mark.unit

_RUN_TESTS_SH = REPO_ROOT / ".github/actions/run-tests/run_tests.sh"
_RUN_TESTS_ACTION = REPO_ROOT / ".github/actions/run-tests/action.yml"


@pytest.fixture(scope="module")
def plan() -> list[testplan.Job]:
    return testplan.load_plan()


def test_every_job_resolves_to_at_least_one_file(plan: list[testplan.Job]):
    """A job that selects nothing is a lane that silently tests nothing."""
    empty = [job.name for job in plan if not testplan.resolve(job)]
    assert not empty, "these jobs resolve to no test files:\n  " + "\n  ".join(empty)


def test_shards_cover_the_job_exactly_once(plan: list[testplan.Job]):
    """Every file in a sharded job runs in exactly one shard."""
    for job in plan:
        if job.shards <= 1:
            continue
        whole = testplan.resolve(job)
        pieces = [path for shard in range(job.shards) for path in testplan.resolve(job, shard=shard)]
        assert sorted(pieces) == whole, f"{job.name}: shards do not partition the job"
        assert len(pieces) == len(set(pieces)), f"{job.name}: a file appears in more than one shard"


def test_workflows_only_reference_jobs_the_plan_defines(plan: list[testplan.Job]):
    """A workflow naming an unknown job fails at run time, in CI, after an image build."""
    known = {job.name for job in plan}
    offenders = []
    for path in sorted((REPO_ROOT / ".github/workflows").glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, definition in (data.get("jobs") or {}).items():
            for step in definition.get("steps") or []:
                name = (step.get("with") or {}).get("job")
                if name and name not in known:
                    offenders.append(f"{path.name}::{job_id} -> {name!r}")
    assert not offenders, "these workflow steps name a job absent from the test plan:\n  " + "\n  ".join(offenders)


def test_generated_workflow_blocks_are_up_to_date():
    """The checked-in YAML must match what the generator produces.

    Editing a generated block by hand is silently undone by the next regeneration, so the
    mismatch has to fail here instead.
    """
    stale = generate_workflows.check()
    assert not stale, (
        "these workflow files are out of date with tools/test_plan.toml:\n  "
        + "\n  ".join(stale)
        + "\n\nFix: run `python tools/generate_workflows.py`."
    )


def _shell_bindings() -> list[str]:
    """Return run_tests.sh's positional bindings, ordered by position."""
    text = _RUN_TESTS_SH.read_text(encoding="utf-8")
    found = {}
    for name, position in re.findall(r'^  local ([a-z_]+)="\$\{?(\d+)\}?"', text, re.M):
        found[int(position)] = name
    return [found[i] for i in sorted(found)]


def _action_arguments() -> list[str]:
    """Return the arguments the action passes to run_tests.sh, ordered."""
    text = _RUN_TESTS_ACTION.read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if "run_tests.sh" in ln and "bash" in ln)
    call = line.split("run_tests.sh", 1)[1]
    names = []
    for raw in re.findall(r'"([^"]*)"', call):
        match = re.search(r"inputs\.([a-z0-9-]+)", raw)
        names.append(match.group(1).replace("-", "_") if match else raw.lstrip("$").lower())
    return names


def test_run_tests_sh_arguments_line_up_with_the_action():
    """A positional interface this long silently misbinds when one side is edited alone.

    Names are compared rather than counts: an off-by-one that happens to preserve the count
    would otherwise pass while feeding, say, the container name in as the job.
    """
    bindings = _shell_bindings()
    arguments = _action_arguments()
    assert len(bindings) == len(arguments), (
        f"run_tests.sh binds {len(bindings)} positional arguments but action.yml passes"
        f" {len(arguments)}:\n  binds:  {bindings}\n  passes: {arguments}"
    )
    mismatched = [
        f"${i + 1}: script binds {b!r}, action passes {a!r}"
        for i, (b, a) in enumerate(zip(bindings, arguments))
        # The action spells a few values as env vars (PYTEST_OPTIONS) or literals rather than
        # `inputs.<name>`; those are matched loosely on the shared stem.
        if b not in a and a not in b
    ]
    assert not mismatched, "run_tests.sh and action.yml disagree on argument order:\n  " + "\n  ".join(mismatched)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash to parse the script")
def test_run_tests_sh_is_valid_shell():
    """``bash -n`` catches the quoting mistakes that are easy to make editing this by hand."""
    result = subprocess.run(["bash", "-n", str(_RUN_TESTS_SH)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
