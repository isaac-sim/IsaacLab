# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline storage for the CI performance regression gate

The baseline store keeps immutable structured samples in ``samples.ndjson``.
Threshold stats are calculated over the newest compatible samples instead of
truncating history on write. Git-backed updates are append-only transactions:
the manager refetches the remote branch, reapplies queued samples, and retries
the push to prevent races across CI runners
"""

import contextlib
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .gate_config import BASELINE_PUSH_RETRIES, DEFAULT_K_BLOCK, DEFAULT_K_WARN, MAX_BASELINE_SAMPLES
    from .oracle import Baseline
except ImportError:  # pragma: no cover - supports direct script imports
    from gate_config import BASELINE_PUSH_RETRIES, DEFAULT_K_BLOCK, DEFAULT_K_WARN, MAX_BASELINE_SAMPLES
    from oracle import Baseline

SAMPLES_FILENAME = "samples.ndjson"
_REPO_DIR = Path(__file__).resolve().parent
_COMMIT_ENV_DEFAULTS = {
    "GIT_AUTHOR_NAME": "perf-regression-gate",
    "GIT_AUTHOR_EMAIL": "perf-regression-gate@localhost",
    "GIT_COMMITTER_NAME": "perf-regression-gate",
    "GIT_COMMITTER_EMAIL": "perf-regression-gate@localhost",
}


@dataclass(frozen=True)
class BaselineUpdateRecord:
    gpu_model: str
    task_id: str
    backend: str
    fps: float
    fingerprint: str | None = None
    sample_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BaselinePushResult:
    branch: str
    remote: str | None
    base_sha: str | None
    pushed_sha: str | None
    attempts: int
    update_count: int
    pushed: bool


def _bucket_dir(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Path:
    base = baselines_dir / gpu_model / task_id / backend
    return base if fingerprint is None else base / fingerprint


def _samples_path(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Path:
    return _bucket_dir(baselines_dir, gpu_model, task_id, backend, fingerprint) / SAMPLES_FILENAME


def _baseline_from_values(
    values: list[float], *, source: str, total_sample_count: int | None = None
) -> Baseline | None:
    if not values:
        return None
    selected = values[-MAX_BASELINE_SAMPLES:]
    median = statistics.median(selected)
    deviations = [abs(v - median) for v in selected]
    mad = statistics.median(deviations) if len(deviations) > 1 else 0.0
    return Baseline(
        median_fps=median,
        mad_fps=mad,
        k_warn=DEFAULT_K_WARN,
        k_block=DEFAULT_K_BLOCK,
        sample_count=len(selected),
        source=source,
        total_sample_count=total_sample_count if total_sample_count is not None else len(values),
    )


def _load_sample_records_from_text(content: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("fps"), (int, float)):
            records.append(record)
    return records


def _commit_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in _COMMIT_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    return env


def _git(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or _REPO_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ["git", *args], result.stdout, result.stderr)
    return result


def _git_error(result: subprocess.CompletedProcess) -> str:
    return (result.stderr or result.stdout or "unknown git error").strip()


def _remote_ref_missing(result: subprocess.CompletedProcess) -> bool:
    message = _git_error(result).lower()
    return "couldn't find remote ref" in message or "could not find remote ref" in message


def _resolve_git_ref(ref: str, *, repo_dir: Path | None = None) -> str | None:
    result = _git(["rev-parse", "--verify", ref], cwd=repo_dir)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def refresh_baseline_branch(
    branch: str,
    *,
    remote: str | None = "origin",
    repo_dir: Path | None = None,
    allow_missing: bool = True,
) -> str | None:
    """Fetch and return the exact baseline branch SHA to read.

    Returning a SHA instead of the branch name makes aggregate comparisons traceable
    and prevents a stale local branch from being used after the remote moved.
    """
    if not remote:
        return _resolve_git_ref(branch, repo_dir=repo_dir)

    remote_ref = f"refs/remotes/{remote}/{branch}"
    refspec = f"+refs/heads/{branch}:{remote_ref}"
    result = _git(["fetch", remote, refspec], cwd=repo_dir)
    if result.returncode != 0:
        if allow_missing and _remote_ref_missing(result):
            return None
        raise RuntimeError(f"Failed to fetch baseline branch {branch!r} from {remote!r}: {_git_error(result)}")
    return _resolve_git_ref(remote_ref, repo_dir=repo_dir)


def _git_is_ancestor(commit_sha: str, base_sha: str, *, repo_dir: Path | None = None) -> bool:
    result = _git(["merge-base", "--is-ancestor", commit_sha, base_sha], cwd=repo_dir)
    return result.returncode == 0


def _git_distance(commit_sha: str, base_sha: str, *, repo_dir: Path | None = None) -> int:
    result = _git(["rev-list", "--count", f"{commit_sha}..{base_sha}"], cwd=repo_dir)
    if result.returncode != 0:
        return 10**9
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 10**9


def _sample_matches(record: dict[str, Any], context: dict[str, Any] | None) -> bool:
    if context is None:
        return True
    exact_fields = (
        "gpu_model",
        "task_id",
        "backend_key",
        "launch_config_hash",
        "baseline_epoch",
        "benchmark_contract_hash",
        "runtime_contract_hash",
    )
    for field in exact_fields:
        expected = context.get(field)
        if expected is not None and record.get(field) != expected:
            return False
    base_sha = context.get("base_sha")
    commit_sha = record.get("commit_sha")
    repo_dir = context.get("_repo_dir")
    if base_sha:
        if not commit_sha:
            return False
        if not _git_is_ancestor(str(commit_sha), str(base_sha), repo_dir=repo_dir):
            return False
    return True


def _select_records(records: list[dict[str, Any]], context: dict[str, Any] | None) -> list[dict[str, Any]]:
    compatible = [r for r in records if _sample_matches(r, context)]
    base_sha = context.get("base_sha") if context else None
    if base_sha:
        repo_dir = context.get("_repo_dir") if context else None
        compatible.sort(
            key=lambda r: (
                _git_distance(str(r.get("commit_sha", "")), str(base_sha), repo_dir=repo_dir),
                r.get("timestamp", ""),
            )
        )
        return compatible[:MAX_BASELINE_SAMPLES]
    return compatible[-MAX_BASELINE_SAMPLES:]


def _load_baseline_from_contents(
    samples_content: str | None,
    *,
    match_context: dict[str, Any] | None = None,
) -> Baseline | None:
    if not samples_content:
        return None
    records = _load_sample_records_from_text(samples_content)
    selected = _select_records(records, match_context)
    return _baseline_from_values(
        [float(r["fps"]) for r in selected],
        source="samples",
        total_sample_count=len(records),
    )


def load_baseline(
    baselines_dir: Path,
    gpu_model: str,
    task_id: str,
    backend: str,
    fingerprint=None,
    match_context: dict[str, Any] | None = None,
) -> Baseline | None:
    """Load compatible baseline stats for a task/backend pair."""
    samples = _samples_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    return _load_baseline_from_contents(
        samples.read_text() if samples.exists() else None,
        match_context=match_context,
    )


def _stable_sample_id(metadata: dict[str, Any]) -> str:
    keys = (
        "ci_run_id",
        "ci_run_attempt",
        "commit_sha",
        "task_id",
        "backend_key",
        "launch_config_hash",
        "benchmark_contract_hash",
        "runtime_contract_hash",
        "baseline_epoch",
        "fps",
        "attempt",
        "was_retried",
    )
    payload = {key: metadata.get(key) for key in keys if metadata.get(key) is not None}
    if not payload.get("ci_run_id"):
        payload["timestamp"] = metadata.get("timestamp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _sample_id_exists(samples_path: Path, sample_id: str | None) -> bool:
    if not sample_id or not samples_path.exists():
        return False
    for record in _load_sample_records_from_text(samples_path.read_text()):
        if record.get("sample_id") == sample_id:
            return True
    return False


def make_sample_metadata(
    *,
    gpu_model: str,
    task_id: str,
    backend: str,
    fps: float,
    bench_result: dict | None = None,
    target_branch: str | None = None,
    source_branch: str | None = None,
    trusted_source: str = "protected_branch",
    commit_sha: str | None = None,
) -> dict[str, Any]:
    bench_result = bench_result or {}
    launch_config = bench_result.get("launch_config") or {}
    provenance = bench_result.get("provenance") or {}
    git_info = provenance.get("git") or {}
    software = provenance.get("software") or {}
    gpu_diag = bench_result.get("gpu_diag") or {}
    metadata = {
        "schema_version": 1,
        "fps": float(fps),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trusted_source": trusted_source,
        "gpu_model": gpu_model,
        "task_id": task_id,
        "backend_key": backend,
        "physics_backend": launch_config.get("physics_backend") or bench_result.get("physics_backend"),
        "render_backend": launch_config.get("render_backend") or bench_result.get("render_backend"),
        # An explicit ``commit_sha`` (e.g. from the seeder, which checks out each
        # commit by SHA) takes precedence over benchmark-captured provenance, which
        # can be empty when git cannot read the source tree inside the container.
        "commit_sha": commit_sha or git_info.get("commit_hash"),
        "branch": git_info.get("branch") or source_branch,
        "target_branch": target_branch,
        "launch_config_hash": bench_result.get("launch_config_hash") or launch_config.get("launch_config_hash"),
        "benchmark_contract_hash": bench_result.get("benchmark_contract_hash")
        or launch_config.get("benchmark_contract_hash"),
        "runtime_contract_hash": bench_result.get("runtime_contract_hash"),
        "runtime_contract": bench_result.get("runtime_contract"),
        "runtime_info": bench_result.get("runtime_info"),
        "baseline_epoch": bench_result.get("baseline_epoch") or launch_config.get("baseline_epoch", 1),
        "attempt": bench_result.get("attempt"),
        "was_retried": bench_result.get("was_retried"),
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "ci_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "ci_job": os.environ.get("GITHUB_JOB"),
        "launch_config": launch_config,
        "runtime": {
            "isaacsim": software.get("isaacsim"),
            "warp": software.get("warp"),
            "cuda": (gpu_diag or {}).get("cuda_version"),
            "driver": (gpu_diag or {}).get("nvidia_driver_version"),
        },
    }
    metadata["sample_id"] = _stable_sample_id(metadata)
    return metadata


def match_context_from_bench_result(
    bench_result: dict,
    *,
    gpu_model: str,
    base_sha: str | None = None,
    target_branch: str | None = None,
) -> dict[str, Any]:
    launch_config = bench_result.get("launch_config") or {}
    return {
        "gpu_model": gpu_model,
        "task_id": bench_result.get("task_id"),
        "backend_key": bench_result.get("backend_key") or bench_result.get("backend"),
        "launch_config_hash": bench_result.get("launch_config_hash") or launch_config.get("launch_config_hash"),
        "benchmark_contract_hash": bench_result.get("benchmark_contract_hash")
        or launch_config.get("benchmark_contract_hash"),
        "runtime_contract_hash": bench_result.get("runtime_contract_hash"),
        "baseline_epoch": bench_result.get("baseline_epoch") or launch_config.get("baseline_epoch", 1),
        "base_sha": base_sha,
        "target_branch": target_branch,
    }


def update_baseline(
    baselines_dir: Path,
    gpu_model: str,
    task_id: str,
    backend: str,
    fps: float,
    fingerprint=None,
    sample_metadata: dict[str, Any] | None = None,
) -> bool:
    """Append a structured baseline sample. Returns False when already present."""
    bucket = _bucket_dir(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    bucket.mkdir(parents=True, exist_ok=True)
    samples = _samples_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)

    metadata = dict(sample_metadata or {})
    metadata.setdefault("schema_version", 1)
    metadata.setdefault("fps", float(fps))
    metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    metadata.setdefault("gpu_model", gpu_model)
    metadata.setdefault("task_id", task_id)
    metadata.setdefault("backend_key", backend)
    metadata.setdefault("sample_id", _stable_sample_id(metadata))

    if _sample_id_exists(samples, metadata.get("sample_id")):
        return False

    with samples.open("a") as fh:
        fh.write(json.dumps(metadata, sort_keys=True) + "\n")
    return True


def delete_baseline_files(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> None:
    """Delete structured samples for a task/backend pair."""
    samples = _samples_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    if samples.exists():
        samples.unlink()


def seed_baseline_with_spread(
    baselines_dir: Path,
    gpu_model: str,
    task_id: str,
    backend: str,
    center_fps: float,
    noise_fps: float = 5.0,
    n_samples: int = 10,
    seed: int = 0,
    fingerprint=None,
) -> None:
    """Populate a deterministic structured baseline window for tests and demos."""
    import random as _random

    rng = _random.Random(seed)
    delete_baseline_files(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    for _ in range(n_samples):
        fps = max(1.0, rng.gauss(center_fps, noise_fps))
        update_baseline(baselines_dir, gpu_model, task_id, backend, fps, fingerprint=fingerprint)


def _git_show_file(ref: str, rel_path: str, *, repo_dir: Path | None = None) -> str | None:
    result = _git(["show", f"{ref}:{rel_path}"], cwd=repo_dir)
    return result.stdout if result.returncode == 0 else None


def load_baseline_git(
    ref: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    fingerprint: str | None,
    match_context: dict[str, Any] | None = None,
    *,
    repo_dir: Path | None = None,
) -> Baseline | None:
    """Load compatible baseline stats from an exact git ref or SHA."""
    samples = str(_samples_path(Path(""), gpu_model, task_id, backend, fingerprint))
    context = dict(match_context or {})
    if repo_dir is not None:
        context["_repo_dir"] = repo_dir
    return _load_baseline_from_contents(
        _git_show_file(ref, samples, repo_dir=repo_dir),
        match_context=context or None,
    )


@contextlib.contextmanager
def _baseline_update_worktree(base_ref: str | None, *, repo_dir: Path | None = None):
    repo_dir = repo_dir or _REPO_DIR
    tmpdir = tempfile.mkdtemp(prefix="perf-bl-wt-")
    orphan_branch = None
    try:
        if base_ref:
            _git(["worktree", "add", "--detach", tmpdir, base_ref], cwd=repo_dir, check=True)
        else:
            orphan_branch = f"perf-baseline-seed-{os.getpid()}-{Path(tmpdir).name}"
            _git(["worktree", "add", "--detach", tmpdir, "HEAD"], cwd=repo_dir, check=True)
            _git(["checkout", "--orphan", orphan_branch], cwd=Path(tmpdir), check=True)
            rm_result = _git(["rm", "-rf", "."], cwd=Path(tmpdir))
            if rm_result.returncode not in (0, 128):
                raise RuntimeError(f"Failed to clear orphan baseline worktree: {_git_error(rm_result)}")
        yield Path(tmpdir)
    finally:
        _git(["worktree", "remove", "--force", tmpdir], cwd=repo_dir)
        shutil.rmtree(tmpdir, ignore_errors=True)
        if orphan_branch:
            _git(["branch", "-D", orphan_branch], cwd=repo_dir)


def _commit_baseline_worktree(worktree: Path) -> str | None:
    status = _git(["status", "--porcelain"], cwd=worktree, check=True)
    if not status.stdout.strip():
        return None
    _git(["add", "-A"], cwd=worktree, check=True)
    _git(
        ["commit", "-m", "[baseline_manager] Append baseline samples"],
        cwd=worktree,
        check=True,
        env=_commit_env(),
    )
    commit = _git(["rev-parse", "HEAD"], cwd=worktree, check=True)
    return commit.stdout.strip()


def _apply_updates(root: Path, updates: list[BaselineUpdateRecord]) -> int:
    appended = 0
    for update in updates:
        if update_baseline(
            root,
            update.gpu_model,
            update.task_id,
            update.backend,
            update.fps,
            fingerprint=update.fingerprint,
            sample_metadata=update.sample_metadata,
        ):
            appended += 1
    return appended


def update_baselines_git(
    branch: str,
    updates: list[BaselineUpdateRecord],
    *,
    remote: str | None = "origin",
    max_retries: int = BASELINE_PUSH_RETRIES,
    repo_dir: Path | None = None,
) -> BaselinePushResult:
    """Append samples to a git-backed baseline branch and push safely.

    Each attempt starts from the latest remote branch SHA. If the push loses a
    race, the next attempt refetches and reapplies the same sample IDs, making
    retries idempotent.
    """
    if not updates:
        return BaselinePushResult(branch, remote, None, None, 0, 0, False)
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    last_error = "unknown push failure"
    for attempt in range(1, max_retries + 1):
        base_sha = refresh_baseline_branch(branch, remote=remote, repo_dir=repo_dir, allow_missing=True)
        with _baseline_update_worktree(base_sha, repo_dir=repo_dir) as worktree:
            appended = _apply_updates(worktree, updates)
            commit_sha = _commit_baseline_worktree(worktree)
            if commit_sha is None:
                return BaselinePushResult(branch, remote, base_sha, base_sha, attempt, appended, False)
            push_ref = f"HEAD:refs/heads/{branch}"
            if remote:
                push = _git(["push", remote, push_ref], cwd=worktree)
            else:
                push = _git(["branch", "--force", branch, "HEAD"], cwd=worktree)
            if push.returncode == 0:
                return BaselinePushResult(branch, remote, base_sha, commit_sha, attempt, appended, bool(remote))
            last_error = _git_error(push)
            print(f"[baseline_manager] baseline push attempt {attempt}/{max_retries} failed; refetching and retrying")

    raise RuntimeError(f"Failed to push baseline branch {branch!r} after {max_retries} attempts: {last_error}")


def update_baseline_git(
    branch: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    fps: float,
    fingerprint: str | None,
    sample_metadata: dict[str, Any] | None = None,
    *,
    remote: str | None = "origin",
    max_retries: int = BASELINE_PUSH_RETRIES,
    repo_dir: Path | None = None,
) -> BaselinePushResult:
    update = BaselineUpdateRecord(gpu_model, task_id, backend, fps, fingerprint, sample_metadata)
    return update_baselines_git(branch, [update], remote=remote, max_retries=max_retries, repo_dir=repo_dir)
