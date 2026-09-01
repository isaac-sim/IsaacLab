# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve a job in ``tools/test_plan.toml`` to the test files it covers.

Selection used to be spread over four layers: a workflow job's inputs, two composite actions,
a bash translation into ``TEST_*`` environment variables, and a collector in the test runner
that read them back. A job's coverage could only be worked out by tracing all four. This
module is the whole of it: the plan says what a job runs, and :func:`resolve` turns that into
a file list.

Markers are read with :func:`isaaclab.test.kit.module_markers`, which parses the file rather
than searching its text, so a marker named in a docstring or a comment no longer pulls a file
into a lane that does not want it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import test_settings
import tomllib

from isaaclab.test.kit import module_markers

REPO_ROOT = Path(__file__).resolve().parent.parent

PLAN_PATH = REPO_ROOT / "tools" / "test_plan.toml"

# Has its own pytest.ini and conftest, and runs through .github/actions/install-ci-run rather
# than this plan. Skipping the whole subtree keeps its tests out of every job.
_EXCLUDED_DIRS = frozenset({"install_ci"})

_POOLS = {"quarantined": "QUARANTINED_TESTS", "curobo": "CUROBO_TESTS"}


@dataclass(frozen=True)
class Job:
    """One entry in the test plan.

    Attributes:
        name: Stable identifier, used on the command line and as the workflow job suffix.
        title: Display name; a sharded job renders as ``"<title> [i/n]"``.
        workflow: Which workflow file the generated job block belongs to.
        paths: Repo-relative directories walked for ``test_*.py``.
        exclude: Substrings; a file whose repo-relative path contains any of them is dropped.
        files: Basenames; when set, only these run, and they override ``TESTS_TO_SKIP``.
        shards: Number of jobs the resolved list is split across.
        pool: Named list in ``tools/test_settings.py`` to run instead of walking ``paths``.
        marker: Only files declaring this pytest marker are selected.
        k_expr: ``-k`` expression passed to each pytest invocation; narrows within a file.
        node_ids_key: Key in ``.github/test-subsets/`` selecting exact node IDs on push.
        container_name: Docker container name for the CI job.
        warp_cache: Warp cache mode for the CI job.
        generate: Whether ``tools/generate_workflows.py`` renders this job's workflow block.
            False for the lanes whose CI setup is bespoke -- extra build steps, wheelhouse
            expressions, artifact uploads -- which stay hand-written and are only checked
            against the plan.
        continue_on_error: Whether a failure in this lane leaves the run green.
        extra_pip_packages: Packages installed in the container before the tests start.
        timeout_minutes: Job timeout.
    """

    name: str
    title: str
    workflow: str
    paths: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    shards: int = 1
    pool: str | None = None
    marker: str | None = None
    k_expr: str | None = None
    node_ids_key: str | None = None
    container_name: str | None = None
    warp_cache: str | None = None
    generate: bool = False
    continue_on_error: bool = False
    extra_pip_packages: str | None = None
    timeout_minutes: int = 180


def load_plan(path: Path | None = None) -> list[Job]:
    """Read the plan file.

    Args:
        path: Plan to read; defaults to :data:`PLAN_PATH`.

    Returns:
        Every job, in file order.

    Raises:
        ValueError: If a job is missing a name or two jobs share one.
    """
    raw = tomllib.loads((path or PLAN_PATH).read_text(encoding="utf-8"))
    jobs = []
    seen = set()
    for entry in raw.get("job", []):
        name = entry.get("name")
        if not name:
            raise ValueError(f"a job in {path or PLAN_PATH} has no name: {entry}")
        if name in seen:
            raise ValueError(f"duplicate job name in the test plan: {name!r}")
        seen.add(name)
        jobs.append(
            Job(
                name=name,
                title=entry.get("title", name),
                workflow=entry["workflow"],
                paths=tuple(entry.get("paths", ())),
                exclude=tuple(entry.get("exclude", ())),
                files=tuple(entry.get("files", ())),
                shards=int(entry.get("shards", 1)),
                pool=entry.get("pool"),
                marker=entry.get("marker"),
                k_expr=entry.get("k-expr"),
                node_ids_key=entry.get("node-ids-key"),
                container_name=entry.get("container-name"),
                warp_cache=entry.get("warp-cache"),
                generate=bool(entry.get("generate", False)),
                continue_on_error=bool(entry.get("continue-on-error", False)),
                extra_pip_packages=entry.get("extra-pip-packages"),
                timeout_minutes=int(entry.get("timeout-minutes", 180)),
            )
        )
    return jobs


def get_job(name: str, path: Path | None = None) -> Job:
    """Return the named job.

    Args:
        name: Job name from the plan.
        path: Plan to read; defaults to :data:`PLAN_PATH`.

    Returns:
        The matching job.

    Raises:
        KeyError: If no job has that name.
    """
    for job in load_plan(path):
        if job.name == name:
            return job
    raise KeyError(f"no job named {name!r} in the test plan; try --list-jobs")


def _display_path(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or absolute when it lies outside the repository.

    The local runner accepts any directory, including one outside the checkout, so this cannot
    assume every result is repo-relative.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def walk_test_files(paths: tuple[str, ...] | list[str], root: Path | None = None) -> list[str]:
    """Return every ``test_*.py`` under ``paths``, sorted.

    Args:
        paths: Directories to walk, repo-relative or absolute.
        root: Repository root; defaults to :data:`REPO_ROOT`.

    Returns:
        Sorted paths, repo-relative where they lie inside the repository.

    Raises:
        FileNotFoundError: If a listed directory does not exist, which would otherwise show up
            as a job that silently runs nothing.
    """
    root = root or REPO_ROOT
    found = set()
    for entry in paths:
        base = root / entry
        if not base.is_dir():
            raise FileNotFoundError(f"test plan path does not exist: {entry}")
        for directory, _, names in os.walk(base):
            if not _EXCLUDED_DIRS.isdisjoint(Path(directory).parts):
                continue
            for name in names:
                if name.startswith("test_") and name.endswith(".py"):
                    found.add(_display_path(Path(directory) / name, root))
    return sorted(found)


def resolve(job: Job, *, shard: int | None = None, root: Path | None = None) -> list[str]:
    """Return the test files ``job`` covers, repo-relative and sorted.

    Args:
        job: Job to resolve.
        shard: Which shard to take, for a job with ``shards > 1``. None returns every shard.
        root: Repository root; defaults to :data:`REPO_ROOT`.

    Returns:
        Sorted repo-relative paths.

    Raises:
        ValueError: If ``shard`` is out of range for the job.
    """
    root = root or REPO_ROOT
    candidates = walk_test_files(job.paths, root=root)
    wanted = set(job.files)

    selected = []
    for path in candidates:
        name = os.path.basename(path)
        if job.pool is not None:
            if name not in getattr(test_settings, _POOLS[job.pool], ()):
                continue
        elif wanted:
            # An explicit file list is the job's whole point, so it overrides the skip list.
            if name not in wanted:
                continue
        elif name in test_settings.TESTS_TO_SKIP:
            continue
        if any(token in path for token in job.exclude):
            continue
        selected.append(path)

    if job.marker:
        selected = [
            path for path in selected if job.marker in module_markers((root / path).read_text(errors="replace"))
        ]

    if job.shards > 1 and shard is not None:
        if not 0 <= shard < job.shards:
            raise ValueError(f"shard {shard} out of range for job {job.name!r} with {job.shards} shards")
        selected = [path for index, path in enumerate(selected) if index % job.shards == shard]
    return selected
