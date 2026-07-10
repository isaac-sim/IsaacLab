# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that seeded baselines will actually be used by the gate.

The gate keeps a baseline sample only when both hold:

1. Its fingerprint -- ``launch_config_hash``, ``benchmark_contract_hash``,
   ``runtime_contract_hash``, ``baseline_epoch`` (within a
   ``gpu_model``/``task_id``/``backend_key`` bucket) -- matches the run.
2. Its ``commit_sha`` is an ancestor of the run's ``base_sha`` (the target
   branch HEAD). See :func:`baseline_manager._sample_matches`.

This tool replays that selection against the ``perf-baselines`` branch for a
given target tip and reports, per bucket, how many samples would survive and
whether that clears ``MIN_BASELINE_SAMPLES`` (gate blocks) or leaves the gate
advisory (``< MIN_BASELINE_SAMPLES``). It answers the operational question
"did seeding actually populate baselines the gate will use?" without running a
benchmark.

With ``--require`` the process exits non-zero when any expected bucket has fewer
than ``MIN_BASELINE_SAMPLES`` usable samples, so a seed workflow can gate on it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_manager import _git_is_ancestor, _git_show_file, _samples_path, refresh_baseline_branch  # noqa: E402
from gate_config import MIN_BASELINE_SAMPLES  # noqa: E402
from gpu_identity import canonical_gpu_model, detect_gpu_model  # noqa: E402
from task_config import load_tasks  # noqa: E402

# Fields that (with the bucket key) form the gate's compatibility fingerprint.
_FINGERPRINT_FIELDS = (
    "launch_config_hash",
    "benchmark_contract_hash",
    "runtime_contract_hash",
    "baseline_epoch",
)


def _fingerprint(record: dict[str, Any]) -> tuple:
    return tuple(record.get(field) for field in _FINGERPRINT_FIELDS)


def usable_sample_count(
    records: Iterable[dict[str, Any]],
    is_ancestor: Callable[[str], bool] | None = None,
) -> tuple[int, int, int]:
    """Return ``(usable, total, num_fingerprints)`` for a bucket's samples.

    ``usable`` is the size of the largest single-fingerprint group among the
    ancestry-eligible samples -- the count the gate would actually compare
    against, since it never pools samples across fingerprints. ``num_fingerprints``
    is the number of distinct fingerprints among eligible samples (``> 1`` means
    the bucket mixes environments/configs and no single group may be large enough).

    Args:
        records: Sample dicts (as stored in ``samples.ndjson``).
        is_ancestor: Predicate ``commit_sha -> bool``; when provided, only samples
            with a ``commit_sha`` that satisfies it are eligible (mirrors the
            gate's ancestry filter). ``None`` disables the ancestry filter.
    """
    records = list(records)
    if is_ancestor is None:
        eligible = records
    else:
        eligible = [r for r in records if r.get("commit_sha") and is_ancestor(str(r["commit_sha"]))]
    by_fingerprint: dict[tuple, int] = defaultdict(int)
    for record in eligible:
        by_fingerprint[_fingerprint(record)] += 1
    return max(by_fingerprint.values(), default=0), len(records), len(by_fingerprint)


def _load_samples(ref: str, gpu_model: str, task_id: str, backend: str, *, repo_dir: Path | None = None) -> list[dict]:
    rel_path = str(_samples_path(Path(""), gpu_model, task_id, backend, None))
    content = _git_show_file(ref, rel_path, repo_dir=repo_dir)
    if not content:
        return []
    records: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def verify_bucket(
    ref: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    base_sha: str | None,
    *,
    repo_dir: Path | None = None,
) -> tuple[int, int, int]:
    """Load a bucket from git and report ``(usable, total, num_fingerprints)``."""
    records = _load_samples(ref, gpu_model, task_id, backend, repo_dir=repo_dir)
    predicate: Callable[[str], bool] | None = None
    if base_sha:

        def predicate(commit_sha: str) -> bool:
            return _git_is_ancestor(commit_sha, str(base_sha), repo_dir=repo_dir)

    return usable_sample_count(records, predicate)


def _resolve_tip(ref: str, repo_dir: Path | None = None) -> str | None:
    for candidate in (f"origin/{ref}", ref):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify seeded baselines are usable by the gate for a target tip.")
    parser.add_argument("--baseline_branch", default="perf-baselines", help="Branch storing baseline samples.")
    parser.add_argument("--baseline_remote", default="origin", help="Remote owning the baseline branch (empty=local).")
    parser.add_argument("--gpu_model", default="", help="GPU model label; auto-detected via nvidia-smi when empty.")
    parser.add_argument(
        "--base_sha",
        default="",
        help="Tip to check ancestry against. Empty resolves --target_branch (origin/<branch>).",
    )
    parser.add_argument("--target_branch", default="develop", help="Branch whose tip is used when --base_sha is empty.")
    parser.add_argument("--tasks", default="", help="Comma-separated task_id allowlist (empty = all tasks.json tasks).")
    parser.add_argument("--backends", default="", help="Comma-separated backend_key allowlist (empty = all backends).")
    parser.add_argument("--repo_dir", default=None, help="Repo root for git lookups (default: cwd).")
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero when any expected bucket has fewer than MIN_BASELINE_SAMPLES usable samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_dir = Path(args.repo_dir) if args.repo_dir else None
    gpu_model = canonical_gpu_model(detect_gpu_model(args.gpu_model))
    remote = args.baseline_remote or None

    ref = refresh_baseline_branch(args.baseline_branch, remote=remote, repo_dir=repo_dir, allow_missing=True)
    if not ref:
        print(f"::error::[verify] baseline branch {args.baseline_branch!r} not found; nothing seeded yet")
        return 1 if args.require else 0

    base_sha = args.base_sha.strip() or _resolve_tip(args.target_branch, repo_dir)
    if not base_sha:
        print(f"::warning::[verify] could not resolve target tip for {args.target_branch!r}; ancestry not checked")

    task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}
    backend_filter = {b.strip() for b in args.backends.split(",") if b.strip()}
    tasks = [
        t
        for t in load_tasks()
        if (not task_filter or t.task_id in task_filter) and (not backend_filter or t.backend_key in backend_filter)
    ]

    print(f"[verify] baseline={args.baseline_branch}@{ref[:12]} gpu={gpu_model} base_sha={(base_sha or 'n/a')[:12]}")
    print(f"[verify] MIN_BASELINE_SAMPLES={MIN_BASELINE_SAMPLES}")
    print("[verify] | task | backend | usable | total | fingerprints | verdict |")

    results: list[dict[str, Any]] = []
    all_ok = True
    for task in tasks:
        usable, total, fingerprints = verify_bucket(
            ref, gpu_model, task.task_id, task.backend_key, base_sha, repo_dir=repo_dir
        )
        ok = usable >= MIN_BASELINE_SAMPLES
        all_ok = all_ok and ok
        verdict = "BLOCKING" if ok else ("WARMING" if usable > 0 else "EMPTY")
        results.append(
            {
                "task_id": task.task_id,
                "backend_key": task.backend_key,
                "usable": usable,
                "total": total,
                "fingerprints": fingerprints,
                "verdict": verdict,
            }
        )
        print(f"[verify] | {task.task_id} | {task.backend_key} | {usable} | {total} | {fingerprints} | {verdict} |")

    print("[verify] summary: " + json.dumps({"gpu_model": gpu_model, "buckets": results}, sort_keys=True))
    if args.require and not all_ok:
        print("::error::[verify] one or more buckets have fewer usable samples than MIN_BASELINE_SAMPLES")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
