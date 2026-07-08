#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seed the perf-baselines branch from real commit history.

This orchestrator simulates "the smoke test ran when each commit landed": for every
seed commit it checks that commit out into a throwaway clone, bind-mounts that
clone over the CI image's IsaacLab source, and runs the benchmark inside the
container. Because the container's ``.git`` then resolves to the seed commit,
:class:`VersionInfoRecorder` tags each ``perf_smoke_test_info.json`` with
that commit's hash, and the resulting baseline sample is keyed by the real SHA.
That is what lets the smoke test's merge-base/ancestry isolation work against a
populated baseline.

Why a separate clone (not an in-place ``git checkout``):

* Checking out an old commit in the main workspace would also revert this
  orchestrator and the rest of ``tools/perf_smoke_test``. We run the
  *current* tooling against *historical* source, so the historical source lives
  in an isolated clone that is bind-mounted into the container only.
* A ``git worktree`` cannot be used because its ``.git`` is a pointer file into
  the parent repo's ``.git/worktrees/...``; that path is not visible inside the
  container, so ``git rev-parse HEAD`` would fail and the commit tag would be
  lost. A real clone has a self-contained ``.git`` directory.

Faithful re-runs require the seed commit's Python dependencies to match the CI
image (the image installs IsaacLab editable, so the bind mount swaps the running
code without a reinstall). Commits outside that dependency window will fail to
import and are skipped (their samples never reach the baseline).

The FPS for each sample is computed via :func:`oracle.compare` with no baseline
and no hard floor, so it is byte-for-byte the same statistic the live smoke test
records. Samples are pushed with :func:`baseline_manager.update_baselines_git`
(append-only, idempotent), exactly like :mod:`aggregate`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aggregate import _bench_gpu_model  # noqa: E402
from baseline_manager import BaselineUpdateRecord, make_sample_metadata, update_baselines_git  # noqa: E402
from gpu_identity import detect_gpu_model  # noqa: E402
from launch_config import hydra_args_for_task  # noqa: E402
from oracle import compare  # noqa: E402
from task_config import TaskConfig, load_tasks  # noqa: E402

_TOOL_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed perf-baselines from real commit history.")
    parser.add_argument(
        "--branches",
        default="",
        help=(
            "Comma/space-separated branches to seed in one run (each stamped with its own target_branch). "
            "Use 'branch:target' to stamp a different target_branch, e.g. 'develop, unified-POC'. "
            "Takes precedence over --commits / --commit_branch."
        ),
    )
    parser.add_argument(
        "--commits",
        default="",
        help="Explicit commit refs/SHAs (space/comma/newline separated). Overrides --commit_branch.",
    )
    parser.add_argument(
        "--commit_branch",
        default="develop",
        help="Branch whose recent history is sampled when --branches and --commits are empty.",
    )
    parser.add_argument("--commit_count", type=int, default=5, help="Number of recent commits to seed from the branch.")
    parser.add_argument("--samples_per_commit", type=int, default=5, help="Benchmark repetitions per commit/backend.")
    parser.add_argument("--tasks", default="", help="Comma-separated task_id allowlist (empty = all tasks.json tasks).")
    parser.add_argument("--backends", default="", help="Comma-separated backend_key allowlist (empty = all backends).")
    parser.add_argument("--image", required=True, help="Local docker image tag to run (already pulled/tagged).")
    parser.add_argument("--gpu_model", default="", help="GPU model label; auto-detected via nvidia-smi when empty.")
    parser.add_argument("--target_branch", default="develop", help="Protected branch stamped onto each sample.")
    parser.add_argument("--baseline_branch", default="perf-baselines", help="Git branch storing baseline samples.")
    parser.add_argument("--baseline_remote", default="origin", help="Remote to push baselines to (empty = local only).")
    parser.add_argument("--baseline_push_retries", type=int, default=3)
    parser.add_argument("--workdir", default=".", type=Path, help="Repo root (source of historical commits).")
    parser.add_argument("--artifacts_root", default="seed-artifacts", type=Path, help="Where per-run outputs land.")
    parser.add_argument(
        "--seed_src_dir",
        default="",
        help="Throwaway clone path bind-mounted into the container (default: a temp dir).",
    )
    parser.add_argument(
        "--source_mount",
        default="true",
        help="Mount each commit's source into the container (true) or run the baked image source (false).",
    )
    parser.add_argument(
        "--dry_run",
        default="false",
        help="Run benchmarks and build records but skip the baseline push.",
    )
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _run(
    cmd: list[str], *, cwd: Path | None = None, check: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    print(f"[seed] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def _capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def _resolve_sha(token: str, workdir: Path) -> str:
    return _capture(["git", "rev-parse", "--verify", f"{token}^{{commit}}"], cwd=workdir)


def _branch_commits(branch: str, count: int, workdir: Path) -> list[str]:
    """Newest-first slice of a branch's history.

    Prefer the remote-tracking ref so a stale local branch cannot silently shadow
    what the runner fetched.
    """
    branch = branch.strip()
    rev_source = None
    for ref in (f"origin/{branch}", branch):
        check = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            rev_source = ref
            break
    if rev_source is None:
        raise RuntimeError(f"Could not resolve branch {branch!r} (tried origin/{branch}, {branch}).")
    out = _capture(["git", "rev-list", f"--max-count={count}", rev_source], cwd=workdir)
    return [line for line in out.splitlines() if line.strip()]


def _build_seed_plan(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Resolve what to seed as an ordered list of ``(target_branch, commit_sha)`` pairs.

    Precedence: ``--branches`` (multi-branch) > ``--commits`` (explicit SHAs) >
    ``--commit_branch`` (single branch). Each pair is de-duplicated so the same
    commit is not seeded twice under the same target branch.
    """
    workdir = args.workdir
    plan: list[tuple[str, str]] = []

    if args.branches.strip():
        for entry in args.branches.replace(",", " ").split():
            entry = entry.strip()
            if not entry:
                continue
            # "branch" stamps target=branch; "branch:target" overrides the stamp.
            branch, _, target = entry.partition(":")
            target = target.strip() or branch.strip()
            for sha in _branch_commits(branch.strip(), args.commit_count, workdir):
                plan.append((target, sha))
    elif args.commits.strip():
        target = args.target_branch.strip()
        for token in args.commits.replace(",", " ").split():
            if token.strip():
                plan.append((target, _resolve_sha(token.strip(), workdir)))
    else:
        target = args.target_branch.strip()
        for sha in _branch_commits(args.commit_branch.strip(), args.commit_count, workdir):
            plan.append((target, sha))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target, sha in plan:
        key = (target, sha)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    if not deduped:
        raise RuntimeError("No commits resolved to seed.")
    return deduped


def _select_tasks(args: argparse.Namespace) -> list[TaskConfig]:
    tasks = load_tasks()
    task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}
    backend_filter = {b.strip() for b in args.backends.split(",") if b.strip()}
    selected = [
        task
        for task in tasks
        if (not task_filter or task.task_id in task_filter)
        and (not backend_filter or task.backend_key in backend_filter)
    ]
    if not selected:
        raise RuntimeError(f"No tasks matched filters tasks={sorted(task_filter)} backends={sorted(backend_filter)}.")
    return selected


def _prepare_seed_source(workdir: Path, seed_src_dir: Path, sha: str) -> None:
    """Materialize ``sha`` into a self-contained clone at ``seed_src_dir``.

    The clone is created once and reused across commits; each commit is fetched
    from the (full-history) workspace by SHA and hard-checked-out so leftover
    files from a previous commit cannot leak in.
    """
    # Skip Git LFS smudge for every git op below. The benchmark only needs the
    # Python source, never the LFS-tracked binaries (e.g. rendering golden-image
    # test fixtures). A branch whose tip references a missing/orphaned LFS object
    # (common after a force-update) would otherwise abort ``checkout`` with a
    # smudge-filter failure and sink the whole seed run. With smudge skipped, LFS
    # files materialize as harmless pointer files instead.
    git_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not (seed_src_dir / ".git").exists():
        seed_src_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", str(workdir.resolve()), str(seed_src_dir)], env=git_env)
    _run(["git", "fetch", "--no-tags", str(workdir.resolve()), sha], cwd=seed_src_dir, env=git_env)
    _run(["git", "checkout", "-f", "--detach", sha], cwd=seed_src_dir, env=git_env)
    # Best-effort: a prior commit's container can leave behind files the runner
    # user cannot delete (e.g. root-owned ``__pycache__`` written into the
    # bind-mount). ``checkout -f`` already restored every tracked file, so leftover
    # untracked residue is harmless to the benchmark; never let cleanup abort the
    # whole seed run. Open up permissions first so files the runner *can* chmod
    # become removable.
    subprocess.run(["chmod", "-R", "a+rwX", str(seed_src_dir)], check=False)
    clean = _run(["git", "clean", "-fdx"], cwd=seed_src_dir, check=False)
    if clean.returncode != 0:
        print(f"[seed] warning: git clean left undeletable residue (exit {clean.returncode}); continuing", flush=True)
    # The in-container user is uid 1000 (isaaclab); the clone is created by the
    # runner user. Open it up so the benchmark can read the source and write the
    # throwaway _isaac_sim symlink, mirroring the smoke test's bind-mount chmod.
    subprocess.run(["chmod", "-R", "a+rwX", str(seed_src_dir)], check=False)


def _docker_run_benchmark(
    *,
    image: str,
    task: TaskConfig,
    artifact_dir: Path,
    jit_cache: Path,
    kit_cache: Path,
    seed_src_dir: Path | None,
    container_name: str,
) -> int:
    """Run one benchmark in the container, mirroring the smoke test's invocation."""
    hydra_args = " ".join(hydra_args_for_task(task))
    seed_token = f"--seed {task.seed}" if task.seed is not None else ""
    inner = (
        "set -e\n"
        "cd /workspace/isaaclab\n"
        "rm -f _isaac_sim\n"
        "ln -s /isaac-sim _isaac_sim\n"
        "./isaaclab.sh -p tools/perf_smoke_test/perf_runtime.py "
        f"--task '{task.task_id}' "
        f"--num_envs {task.num_envs} "
        f"--num_frames {task.num_frames} "
        f"--warmup_frames {task.warmup_frames} "
        "--benchmark_formatter schema "
        "--output_path /tmp/bench_out "
        f"{seed_token} {hydra_args}\n"
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--init",
        "--stop-timeout",
        "10",
        "--entrypoint",
        "bash",
        "--gpus",
        "all",
        "--network=host",
        "--security-opt=no-new-privileges:true",
        "--ulimit",
        "nofile=65536:65536",
        "--ulimit",
        "nproc=4096:4096",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=yes",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "OMNI_KIT_DISABLE_CUP=1",
        "-e",
        "ISAAC_SIM_HEADLESS=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        # Keep the bind-mounted source clean: writing __pycache__ here leaves
        # root-owned files the runner user cannot remove on the next git clean.
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "WARP_CACHE_PATH=/tmp/jit-cache/warp",
        "-e",
        "CUDA_CACHE_PATH=/tmp/jit-cache/nv",
        "-v",
        f"{artifact_dir}:/tmp/bench_out",
        "-v",
        f"{jit_cache}:/tmp/jit-cache",
        "-v",
        f"{kit_cache}:/isaac-sim/kit/cache",
    ]
    if seed_src_dir is not None:
        # Bind-mount the historical source over the image's IsaacLab tree so the
        # container runs (and git-tags) the seed commit, not the baked source.
        cmd += ["-v", f"{seed_src_dir}:/workspace/isaaclab"]
    cmd += [image, "-c", inner]

    result = subprocess.run(cmd, text=True)
    return result.returncode


def _build_bench_result(task: TaskConfig, artifact_dir: Path, exit_code: int, wall_time_s: int) -> None:
    _run(
        [
            "python3",
            str(_TOOL_DIR / "build_bench_result.py"),
            "--task_id",
            task.task_id,
            "--physics_backend",
            task.physics_backend,
            "--render_backend",
            task.render_backend or "",
            "--artifact_dir",
            str(artifact_dir),
            "--exit_code",
            str(exit_code),
            "--wall_time_s",
            str(wall_time_s),
            "--timeout_s",
            str(task.timeout_minutes * 60),
            "--log_file",
            str(artifact_dir / "benchmark.log"),
            "--launch_config",
            str(artifact_dir / "launch_config.json"),
            "--gate_config",
            str(_TOOL_DIR / "gate_config.json"),
        ],
        check=False,
    )


def _write_launch_config(task: TaskConfig, artifact_dir: Path, gpu_model: str) -> None:
    _run(
        [
            "python3",
            str(_TOOL_DIR / "write_launch_config.py"),
            "--task_id",
            task.task_id,
            "--physics_backend",
            task.physics_backend,
            "--render_backend",
            task.render_backend or "",
            "--gpu_model",
            gpu_model,
            "--artifact_dir",
            str(artifact_dir),
        ]
    )


def _record_from_result(
    artifact_dir: Path, gpu_model: str, target_branch: str, known_commit: str | None = None
) -> BaselineUpdateRecord | None:
    """Turn one built ``perf_smoke_test_result.json`` into a baseline sample.

    Returns ``None`` when the run was unhealthy (no benchmark info / no FPS) so the
    caller can skip it. The FPS is computed via :func:`oracle.compare` with no
    baseline and no hard floor, matching the live smoke test's statistic exactly.
    """
    result_path = artifact_dir / "perf_smoke_test_result.json"
    if not result_path.exists():
        print(f"[seed] skip @ {artifact_dir}: no result json (job crashed before build_bench_result)")
        return None
    bench_result = json.loads(result_path.read_text())
    task_id = bench_result.get("task_id")
    backend = bench_result.get("backend_key") or bench_result.get("backend")
    bench_gpu_model = _bench_gpu_model(bench_result, gpu_model)
    # The seeder checks out each commit by SHA, so it authoritatively knows the
    # commit even when the in-container benchmark cannot capture git provenance
    # (detached HEAD on a runner-owned bind mount). Prefer the known SHA so the
    # sample stays ancestry-selectable by the smoke test.
    commit_sha = known_commit or ((bench_result.get("provenance") or {}).get("git") or {}).get("commit_hash")

    if not bench_result.get("perf_smoke_test_info_present", False):
        print(f"[seed] skip {task_id}/{backend} @ {artifact_dir.name}: no benchmark info (run failed)")
        return None
    oracle_result = compare(
        bench_result=bench_result,
        baseline=None,
        fps_mean_thresholds=[],
        min_block_regression_pct=0.0,
    )
    if oracle_result.measured_fps is None:
        print(f"[seed] skip {task_id}/{backend} @ {artifact_dir.name}: no measurable FPS")
        return None
    if not commit_sha:
        print(
            f"[seed] WARNING {task_id}/{backend} @ {artifact_dir.name}: no commit_sha in provenance; "
            "sample will not be ancestry-selectable"
        )
    sample_metadata = make_sample_metadata(
        gpu_model=bench_gpu_model,
        task_id=task_id,
        backend=backend,
        fps=oracle_result.measured_fps,
        bench_result=bench_result,
        target_branch=target_branch,
        trusted_source="seed",
        commit_sha=commit_sha,
    )
    short = (commit_sha or "????????")[:8]
    print(
        f"[seed] sample {task_id}/{backend} commit={short} target={target_branch} fps={oracle_result.measured_fps:.1f}"
    )
    return BaselineUpdateRecord(
        gpu_model=bench_gpu_model,
        task_id=task_id,
        backend=backend,
        fps=oracle_result.measured_fps,
        sample_metadata=sample_metadata,
    )


def _safe_path_component(name: str) -> str:
    """Make a branch name safe to use as a single path segment (no nested dirs)."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)


# Log fragments that mean "this commit is incompatible with the baked container"
# rather than a genuine performance failure. A seeded historical commit can ship a
# kit experience / Python that references Isaac Sim extensions or APIs absent from
# the era-fixed image; in an offline image that surfaces as a dependency-solver
# failure at startup. We treat these as skips, not hard failures, so a divergent or
# pre-migration commit cannot sink a seed run that also has compatible commits.
_INCOMPATIBLE_ENV_SIGNATURES = (
    "registry cache path is not set",
    "because of dependency solver failure",
    "Can't pull extension",
    "Failed to resolve extension",
)


def _detect_incompatible_env(artifact_dir: Path) -> str | None:
    """Return a short reason if the benchmark log shows the run was incompatible with
    the container (an Isaac Sim extension/API the image does not contain), else ``None``.
    """
    try:
        text = (artifact_dir / "benchmark.log").read_text(errors="replace")
    except OSError:
        return None
    for sig in _INCOMPATIBLE_ENV_SIGNATURES:
        if sig in text:
            return sig
    return None


def main() -> int:
    import time

    args = _parse_args()
    source_mount = _as_bool(args.source_mount)
    dry_run = _as_bool(args.dry_run)
    gpu_model = detect_gpu_model(args.gpu_model)
    workdir = args.workdir.resolve()
    artifacts_root = (workdir / args.artifacts_root).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    seed_src_dir = (
        Path(args.seed_src_dir).resolve() if args.seed_src_dir else Path(tempfile.gettempdir()) / "perf-seed-src"
    )
    jit_cache = workdir / "jit-cache"
    kit_cache = workdir / "kit-cache"
    for sub in (jit_cache / "warp", jit_cache / "nv", kit_cache):
        sub.mkdir(parents=True, exist_ok=True)
    _run(["chmod", "-R", "0777", str(jit_cache), str(kit_cache)], check=False)

    plan = _build_seed_plan(args)
    tasks = _select_tasks(args)

    plan_by_target: dict[str, list[str]] = {}
    for target, sha in plan:
        plan_by_target.setdefault(target, []).append(sha[:8])

    print("[seed] ----------------------------------------------------------------")
    for target, shas in plan_by_target.items():
        print(f"[seed] target={target}: commits {shas}")
    print(f"[seed] tasks ({len(tasks)}): {[f'{t.task_id}/{t.backend_key}' for t in tasks]}")
    print(f"[seed] samples_per_commit={args.samples_per_commit}  source_mount={source_mount}  dry_run={dry_run}")
    print(f"[seed] baseline_branch={args.baseline_branch}  gpu_model={gpu_model}")
    print("[seed] ----------------------------------------------------------------")

    records: list[BaselineUpdateRecord] = []
    incompatible_skips: list[str] = []
    prep_skips: list[str] = []
    other_failures = 0
    total = 0
    for target_branch, commit in plan:
        short = commit[:8]
        target_dir = _safe_path_component(target_branch)
        if source_mount:
            try:
                _prepare_seed_source(workdir, seed_src_dir, commit)
            except (RuntimeError, OSError) as exc:
                # Don't let one un-checkout-able commit (e.g. a broken LFS object or
                # rewritten history) abort commits that would otherwise seed cleanly.
                print(f"::warning::[seed] skip commit {short} on {target_branch}: source prep failed: {exc}")
                prep_skips.append(f"{target_branch}/{short}")
                continue
        for task in tasks:
            for sample_idx in range(args.samples_per_commit):
                total += 1
                artifact_dir = (
                    artifacts_root / target_dir / short / task.task_id / task.backend_key / f"sample{sample_idx}"
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                _run(["chmod", "-R", "0777", str(artifact_dir)], check=False)
                container = f"perf-seed-{target_dir}-{short}-{task.task_id}-{task.backend_key}-{sample_idx}"
                container = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in container)
                print(
                    f"[seed] >>> target={target_branch} {short} {task.task_id}/{task.backend_key} "
                    f"sample {sample_idx + 1}/{args.samples_per_commit}"
                )

                _write_launch_config(task, artifact_dir, gpu_model)
                subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)
                start = time.time()
                exit_code = _docker_run_benchmark(
                    image=args.image,
                    task=task,
                    artifact_dir=artifact_dir,
                    jit_cache=jit_cache,
                    kit_cache=kit_cache,
                    seed_src_dir=seed_src_dir if source_mount else None,
                    container_name=container,
                )
                wall_time_s = int(time.time() - start)
                _build_bench_result(task, artifact_dir, exit_code, wall_time_s)

                record = _record_from_result(artifact_dir, gpu_model, target_branch, known_commit=commit)
                if record is not None:
                    records.append(record)
                    continue
                label = f"{target_branch}/{short} {task.task_id}/{task.backend_key}"
                reason = _detect_incompatible_env(artifact_dir)
                if reason is not None:
                    print(f"[seed] skip {label}: incompatible with container ({reason})")
                    incompatible_skips.append(label)
                else:
                    other_failures += 1

    print(
        f"[seed] collected {len(records)} baseline sample(s) from {total} benchmark run(s); "
        f"skipped {len(incompatible_skips)} incompatible, {len(prep_skips)} prep-failed, "
        f"{other_failures} other failure(s)"
    )

    summary_path = artifacts_root / "seed_records.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "gpu_model": r.gpu_model,
                    "task_id": r.task_id,
                    "backend": r.backend,
                    "fps": r.fps,
                    "commit_sha": (r.sample_metadata or {}).get("commit_sha"),
                    "target_branch": (r.sample_metadata or {}).get("target_branch"),
                }
                for r in records
            ],
            indent=2,
        )
    )
    print(f"[seed] wrote sample summary -> {summary_path}")

    if not records:
        if incompatible_skips and not other_failures:
            print(
                "::warning::No samples seeded: every commit was incompatible with this container. "
                "A baseline run only works against commits from the same simulator era as the image "
                "(one container = one era). Seed the branch the image was built from (e.g. develop / "
                "unified-POC), not a divergent or pre-migration branch such as main."
            )
        else:
            print("::warning::No healthy samples collected; nothing to push.")
        return 1
    if dry_run:
        print("[seed] dry_run=true; skipping baseline push.")
        return 0

    push = update_baselines_git(
        args.baseline_branch,
        records,
        remote=args.baseline_remote or None,
        max_retries=args.baseline_push_retries,
        repo_dir=workdir,
    )
    print(
        f"[seed] baseline push: pushed={push.pushed} branch={push.branch} "
        f"base={(push.base_sha or '-')[:8]} new={(push.pushed_sha or '-')[:8]} "
        f"appended={push.update_count} attempts={push.attempts}"
    )
    return 0 if push.pushed or args.baseline_remote == "" else 1


if __name__ == "__main__":
    raise SystemExit(main())
