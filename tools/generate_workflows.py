# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the uniform test lanes in the workflow files from ``tools/test_plan.toml``.

Most package lanes are the same twenty-odd lines of YAML differing only in a name, a container
and a package list, so they were maintained by copy-paste: adding a package meant adding a
block, and a change to the shape had to be applied to every copy by hand. Those lanes are
generated here, between sentinel comments, and :mod:`tools.test.test_test_plan` fails if the
checked-in YAML has drifted from what this produces.

Only jobs marked ``generate = true`` are rendered. The lanes with bespoke setup -- extra image
builds, wheelhouse expressions, artifact uploads -- stay hand-written; they are still required
to name a job the plan defines, which is what keeps the two from diverging.

Usage::

    python tools/generate_workflows.py           # rewrite the generated blocks
    python tools/generate_workflows.py --check   # report drift, change nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import testplan
from testplan import REPO_ROOT, Job

WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

BEGIN = "  # >>> generated from tools/test_plan.toml -- edit the plan, then run tools/generate_workflows.py"
END = "  # <<< end generated jobs"


def _job_id(job: Job, shard: int | None) -> str:
    """Workflow job id; sharded jobs are numbered from 1 to read like their display name."""
    return f"test-{job.name}" if shard is None else f"test-{job.name}-{shard + 1}"


def _title(job: Job, shard: int | None) -> str:
    """Display name; a sharded job carries its position."""
    return job.title if shard is None else f"{job.title} [{shard + 1}/{job.shards}]"


def _container(job: Job, shard: int | None) -> str:
    """Container name, kept distinct per shard so two shards cannot collide on one runner."""
    base = job.container_name or f"isaac-lab-{job.name}"
    if shard is None:
        return base
    return base[: -len("-test")] + f"-{shard + 1}-test" if base.endswith("-test") else f"{base}-{shard + 1}"


def render_job(job: Job, shard: int | None) -> str:
    """Render one workflow job block.

    Args:
        job: Job from the plan.
        shard: Shard index, or None for an unsharded job.

    Returns:
        The YAML block, ending in a blank line.
    """
    lines = [
        f"  {_job_id(job, shard)}:",
        f"    name: {_title(job, shard)}",
        "    runs-on: [self-hosted, gpu]",
        f"    timeout-minutes: {job.timeout_minutes}",
    ]
    if job.continue_on_error:
        lines.append("    continue-on-error: true")
    lines += [
        "    needs: [build, config]",
        "    if: >-",
        "      github.event_name != 'push' &&",
        "      needs.build.result == 'success'",
        "    steps:",
        "    - uses: actions/checkout@v6",
        "      with:",
        "        fetch-depth: 1",
        "        lfs: true",
        "    - uses: ./.github/actions/run-package-tests",
        "      with:",
        "        image-tag: ${{ needs.config.outputs.ci_image_tag }}",
        "        isaacsim-base-image: ${{ needs.config.outputs.isaacsim_image_name }}",
        "        isaacsim-version: ${{ needs.config.outputs.isaacsim_image_tag }}",
        f"        job: {job.name}",
    ]
    if shard is not None:
        lines.append(f'        shard: "{shard}"')
    if job.extra_pip_packages:
        lines.append(f'        extra-pip-packages: "{job.extra_pip_packages}"')
    if job.warp_cache:
        lines.append(f"        warp-cache: {job.warp_cache}")
    lines.append(f"        container-name: {_container(job, shard)}")
    return "\n".join(lines) + "\n"


def render(workflow: str) -> str:
    """Render every generated job for one workflow file, in plan order."""
    blocks = []
    for job in testplan.load_plan():
        if not job.generate or job.workflow != workflow:
            continue
        shards = range(job.shards) if job.shards > 1 else [None]
        blocks.extend(render_job(job, shard) for shard in shards)
    return "\n".join(blocks)


def _splice(text: str, body: str) -> str:
    """Replace the text between the sentinels, keeping everything outside untouched.

    Raises:
        ValueError: If the sentinels are missing or out of order, which would otherwise let
            the generator silently write nothing.
    """
    start, end = text.find(BEGIN), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"generated-block sentinels not found in order; expected\n{BEGIN}\n...\n{END}")
    return text[: start + len(BEGIN)] + "\n\n" + body + "\n" + text[end:]


def _workflow_path(workflow: str) -> Path:
    for suffix in (".yaml", ".yml"):
        candidate = WORKFLOW_DIR / f"{workflow}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no workflow file for {workflow!r}")


def _generated_workflows() -> list[str]:
    return sorted({job.workflow for job in testplan.load_plan() if job.generate})


def write() -> list[str]:
    """Rewrite the generated blocks. Returns the workflow files that changed."""
    changed = []
    for workflow in _generated_workflows():
        path = _workflow_path(workflow)
        text = path.read_text(encoding="utf-8")
        updated = _splice(text, render(workflow))
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.name)
    return changed


def check() -> list[str]:
    """Return the workflow files whose generated blocks are out of date."""
    stale = []
    for workflow in _generated_workflows():
        path = _workflow_path(workflow)
        text = path.read_text(encoding="utf-8")
        if _splice(text, render(workflow)) != text:
            stale.append(path.name)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report drift instead of rewriting")
    args = parser.parse_args(argv)

    if args.check:
        stale = check()
        for name in stale:
            print(f"out of date: {name}")
        return 1 if stale else 0

    changed = write()
    for name in changed:
        print(f"updated: {name}")
    if not changed:
        print("already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
