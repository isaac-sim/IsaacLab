# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline storage for the CI performance regression gate.

Baselines live per ``{gpu_model}/{task_id}/{backend}/[{fingerprint}/]`` bucket as a
``stats.json`` (median/MAD/sample_count) plus an append-only ``window.ndjson`` of
raw FPS samples. Two backends share the same layout: a flat directory (local /
offline testing) and a git **orphan branch** (production), the latter written
atomically through a temporary worktree.

Hardening over the original POC:

* **Actual rolling window** -- ``window.ndjson`` is capped at :data:`WINDOW_MAX`
  samples (oldest evicted on append), so the median+MAD track *recent* behavior
  instead of an unbounded all-time average.
* **Fingerprint fallback chain** -- a baseline is bucketed by an environment
  fingerprint (``{backend_version}/{runtime_hash}/{code_fingerprint}``). Loads
  resolve from the most specific bucket outward to looser ones (then the flat
  bucket), so a dependency/driver bump that creates a fresh empty bucket still
  gates against the nearest compatible history instead of silently seed-passing.
* **Concurrency-safe writes** -- git pushes use a bounded fetch -> rebase -> push
  retry loop so two near-simultaneous protected-branch runs cannot lose a sample.
"""

import contextlib
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from oracle import DEFAULT_K_BLOCK, DEFAULT_K_WARN, Baseline  # noqa: E402

WINDOW_MAX = 20  # rolling window length: keep only the most recent N samples
PUSH_RETRIES = 5  # bounded fetch->rebase->push attempts for the baselines branch


def _stats_path(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Path:
    if fingerprint is None:
        return baselines_dir / gpu_model / task_id / backend / "stats.json"
    return baselines_dir / gpu_model / task_id / backend / fingerprint / "stats.json"


def _window_path(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Path:
    if fingerprint is None:
        return baselines_dir / gpu_model / task_id / backend / "window.ndjson"
    return baselines_dir / gpu_model / task_id / backend / fingerprint / "window.ndjson"


# ---------------------------------------------------------------------------
# Fingerprint fallback chain
# ---------------------------------------------------------------------------


def fingerprint_candidates(fingerprint: str | None) -> list[str | None]:
    """Ordered fingerprint buckets to try, most specific first, flat (None) last.

    A fingerprint is a path like ``{backend_version}/{runtime_hash}/{code_fingerprint}``.
    We relax it one trailing segment at a time -- dropping ``code_fingerprint``, then
    ``runtime_hash``, then ``backend_version`` -- so a load falls back to the nearest
    compatible history when the exact bucket is empty (e.g. just after a dep bump).

    Args:
        fingerprint: The most-specific bucket path, or None for the flat bucket.

    Returns:
        Candidate buckets in resolution order; always ends with ``None``.
    """
    if not fingerprint:
        return [None]
    segments = [s for s in fingerprint.split("/") if s]
    candidates: list[str | None] = []
    for end in range(len(segments), 0, -1):
        candidates.append("/".join(segments[:end]))
    candidates.append(None)
    return candidates


def _baseline_from_stats(d: dict) -> Baseline:
    return Baseline(
        median_fps=d["median_fps"],
        mad_fps=d["mad_fps"],
        k_warn=d.get("k_warn", DEFAULT_K_WARN),
        k_block=d.get("k_block", DEFAULT_K_BLOCK),
        sample_count=d.get("sample_count", 0),
    )


# ---------------------------------------------------------------------------
# Flat-file backend (local / offline testing)
# ---------------------------------------------------------------------------


def load_baseline(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Baseline | None:
    """Load stats.json for a task/backend/fingerprint bucket, or None if it does not exist."""
    sp = _stats_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    if not sp.exists():
        return None
    with sp.open() as fh:
        return _baseline_from_stats(json.load(fh))


def load_baseline_resolved(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint: str | None
) -> tuple[Baseline | None, str | None]:
    """Load the baseline using the fingerprint fallback chain.

    Returns the first non-empty bucket walking from the exact fingerprint outward,
    along with the fingerprint that actually matched (None = flat bucket).
    """
    for candidate in fingerprint_candidates(fingerprint):
        bl = load_baseline(baselines_dir, gpu_model, task_id, backend, fingerprint=candidate)
        if bl is not None:
            return bl, candidate
    return None, None


def update_baseline(
    baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint=None
) -> None:
    """Append fps to the rolling window (capped at WINDOW_MAX), recompute stats, write stats.json.

    The window is an *actual* rolling window: the oldest samples are evicted so the
    retained set never exceeds :data:`WINDOW_MAX`, and the file is rewritten with the
    capped window (not blindly appended). Median + MAD are recomputed over the
    retained samples.
    """
    wp = _window_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    sp = _stats_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    wp.parent.mkdir(parents=True, exist_ok=True)

    fps_window: list[float] = []
    if wp.exists():
        with wp.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    fps_window.append(float(line))

    fps_window.append(fps)
    # Roll: keep only the most recent WINDOW_MAX samples.
    if len(fps_window) > WINDOW_MAX:
        fps_window = fps_window[-WINDOW_MAX:]

    # Rewrite the window file with the (capped) retained samples.
    with wp.open("w") as fh:
        for v in fps_window:
            fh.write(f"{v}\n")

    median = statistics.median(fps_window)
    deviations = [abs(v - median) for v in fps_window]
    mad = statistics.median(deviations) if len(deviations) > 1 else 0.0

    stats = {
        "median_fps": median,
        "mad_fps": mad,
        "k_warn": DEFAULT_K_WARN,
        "k_block": DEFAULT_K_BLOCK,
        "sample_count": len(fps_window),
    }
    with sp.open("w") as fh:
        json.dump(stats, fh, indent=2)


def delete_baseline_files(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> None:
    """Delete stats.json and window.ndjson for a task/backend/fingerprint bucket."""
    for p in (
        _stats_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint),
        _window_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint),
    ):
        if p.exists():
            p.unlink()


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
    """Populate the baseline window with n_samples of varied FPS data around center_fps and compute stats.json.
    For testing tasks/backends with no existing baseline or when a deterministic baseline is needed."""
    import random as _random

    rng = _random.Random(seed)
    delete_baseline_files(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    for _ in range(n_samples):
        fps = max(1.0, rng.gauss(center_fps, noise_fps))
        update_baseline(baselines_dir, gpu_model, task_id, backend, fps, fingerprint=fingerprint)


# ---------------------------------------------------------------------------
# Git orphan-branch backend (production)
# ---------------------------------------------------------------------------


def _git_show_file(branch: str, rel_path: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "show", f"{branch}:{rel_path}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).parent),
        )
        return r.stdout
    except subprocess.CalledProcessError:
        return None


@contextlib.contextmanager
def baseline_worktree(branch: str, *, remote: str | None = "origin", push: bool = True):
    """Check out ``branch`` in a temp worktree; commit and (optionally) push on exit.

    Pushes use a bounded fetch -> rebase -> push retry loop so two near-simultaneous
    protected-branch CI runs cannot lose a sample or wedge on a non-fast-forward
    rejection. A push that still fails after :data:`PUSH_RETRIES` attempts is logged
    (not raised): the run's verdict already stands; only the baseline append is lost.
    """
    tmpdir = tempfile.mkdtemp(prefix="perf-bl-wt-")
    committed = False
    cwd = str(Path(__file__).parent)
    try:
        subprocess.run(
            ["git", "worktree", "add", tmpdir, branch],
            check=True,
            capture_output=True,
            cwd=cwd,
        )
        yield Path(tmpdir)
        status = subprocess.run(
            ["git", "-C", tmpdir, "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            subprocess.run(["git", "-C", tmpdir, "add", "-A"], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", tmpdir, "commit", "-m", "[baseline_manager] Update baselines"],
                check=True,
                capture_output=True,
            )
            committed = True
            if push and remote:
                _push_with_retry(tmpdir, branch, remote)
    except subprocess.CalledProcessError as exc:
        if b"not found" in (exc.stderr or b"") or b"unknown" in (exc.stderr or b""):
            raise RuntimeError(
                f"Baseline branch {branch!r} not found. "
                "Create the orphan branch first, or use the flat-file backend (--baselines_dir)."
            ) from exc
        raise
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", tmpdir],
            cwd=cwd,
            capture_output=True,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)
    if committed:
        print(f"[baseline_manager]   -> committed baseline update to {branch!r}")


def _push_with_retry(wt_dir: str, branch: str, remote: str) -> None:
    """Fetch -> rebase -> push the baselines branch, retrying on non-fast-forward."""
    for attempt in range(1, PUSH_RETRIES + 1):
        push = subprocess.run(
            ["git", "-C", wt_dir, "push", remote, f"HEAD:{branch}"],
            capture_output=True,
            text=True,
        )
        if push.returncode == 0:
            if attempt > 1:
                print(f"[baseline_manager]   -> pushed {branch!r} on attempt {attempt}")
            return
        # Someone else advanced the branch: integrate their samples and retry.
        subprocess.run(["git", "-C", wt_dir, "fetch", remote, branch], capture_output=True, text=True)
        rebase = subprocess.run(
            ["git", "-C", wt_dir, "rebase", f"{remote}/{branch}"],
            capture_output=True,
            text=True,
        )
        if rebase.returncode != 0:
            # Window files are append/rewrite only; a content conflict is not expected,
            # but if it happens, abort the rebase and bail rather than wedge the gate.
            subprocess.run(["git", "-C", wt_dir, "rebase", "--abort"], capture_output=True, text=True)
            print(f"[baseline_manager] WARNING: baseline rebase conflict on {branch!r}; skipping push")
            return
        time.sleep(0.5 * attempt)  # small backoff before the next attempt
    print(f"[baseline_manager] WARNING: could not push {branch!r} after {PUSH_RETRIES} attempts; sample dropped")


def load_baseline_git(
    branch: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None
) -> Baseline | None:
    rel = str(_stats_path(Path(""), gpu_model, task_id, backend, fingerprint))
    content = _git_show_file(branch, rel)
    if content is None:
        return None
    return _baseline_from_stats(json.loads(content))


def load_baseline_git_resolved(
    branch: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None
) -> tuple[Baseline | None, str | None]:
    """Git equivalent of :func:`load_baseline_resolved` using the fingerprint fallback chain."""
    for candidate in fingerprint_candidates(fingerprint):
        bl = load_baseline_git(branch, gpu_model, task_id, backend, candidate)
        if bl is not None:
            return bl, candidate
    return None, None


def update_baseline_git(
    branch: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    fps: float,
    fingerprint: str | None,
    *,
    remote: str | None = "origin",
    push: bool = True,
) -> None:
    with baseline_worktree(branch, remote=remote, push=push) as wt:
        update_baseline(wt, gpu_model, task_id, backend, fps, fingerprint)


def seed_baseline_with_spread_git(
    branch: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    center_fps: float,
    noise_fps: float,
    n_samples: int,
    seed: int,
    fingerprint: str | None,
    *,
    remote: str | None = "origin",
    push: bool = True,
) -> None:
    with baseline_worktree(branch, remote=remote, push=push) as wt:
        seed_baseline_with_spread(wt, gpu_model, task_id, backend, center_fps, noise_fps, n_samples, seed, fingerprint)


def delete_baseline_files_git(
    branch: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    fingerprint: str | None,
    *,
    remote: str | None = "origin",
    push: bool = True,
) -> None:
    with baseline_worktree(branch, remote=remote, push=push) as wt:
        delete_baseline_files(wt, gpu_model, task_id, backend, fingerprint)
