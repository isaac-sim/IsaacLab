# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the seed-time ancestry preflight.

The gate drops baseline samples whose ``commit_sha`` is not an ancestor of the
run's ``base_sha`` (the target branch HEAD). The seeder's preflight refuses to
seed such commits so they cannot silently produce unusable baselines.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import seed_baselines  # noqa: E402


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _commit(work: Path, name: str) -> str:
    (work / name).write_text(f"{name}\n", encoding="utf-8")
    _git(["add", name], work)
    _git(["commit", "-m", name], work)
    return _git(["rev-parse", "HEAD"], work)


@pytest.fixture()
def repo(tmp_path: Path) -> dict[str, object]:
    """A repo where ``develop`` (c0->c1->c2) and ``feature`` (c0->f1) diverge."""
    work = tmp_path / "work"
    work.mkdir()
    _git(["init", "-b", "develop"], work)
    _git(["config", "user.email", "test@example.com"], work)
    _git(["config", "user.name", "test"], work)
    c0 = _commit(work, "c0")
    c1 = _commit(work, "c1")
    c2 = _commit(work, "c2")

    _git(["checkout", "-b", "feature", c0], work)
    f1 = _commit(work, "f1")
    _git(["checkout", "develop"], work)
    return {"work": work, "c0": c0, "c1": c1, "c2": c2, "f1": f1}


def test_is_ancestor_tracks_reachability(repo) -> None:
    """``_is_ancestor`` is true only for commits reachable from the tip."""
    work = repo["work"]
    assert seed_baselines._is_ancestor(repo["c1"], repo["c2"], work) is True
    assert seed_baselines._is_ancestor(repo["f1"], repo["c2"], work) is False


def test_resolve_branch_tip_prefers_local_when_no_remote(repo) -> None:
    """With no ``origin`` the resolver still finds the local branch tip."""
    work = repo["work"]
    assert seed_baselines._resolve_branch_tip("develop", work) == repo["c2"]
    assert seed_baselines._resolve_branch_tip("nope", work) is None


def test_filter_drops_non_ancestor_commits(repo) -> None:
    """A commit off the target branch is skipped; on-branch commits are kept."""
    work = repo["work"]
    plan = [("develop", repo["c1"]), ("develop", repo["c2"]), ("develop", repo["f1"])]

    kept = seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="")

    assert kept == [("develop", repo["c1"]), ("develop", repo["c2"])]


def test_filter_strict_raises_on_non_ancestor(repo) -> None:
    """Strict mode turns an off-branch commit into a hard error."""
    work = repo["work"]
    plan = [("develop", repo["c1"]), ("develop", repo["f1"])]

    with pytest.raises(RuntimeError, match="NOT ancestors"):
        seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="", strict=True)


def test_build_seed_plan_branches_ref_with_target_seeds_single_commit(repo) -> None:
    """The era-roll invocation ('<sha>:develop', count 1) seeds exactly that commit.

    The auto-era-roll orchestrator seeds a brand-new era from its boundary commit
    (HEAD) via ``branches: "<sha>:develop"``; this locks that the branches path
    resolves a raw SHA and stamps the overridden target.
    """
    work = repo["work"]
    args = argparse.Namespace(
        branches=f"{repo['c1']}:develop",
        commits="",
        commit_branch="develop",
        commit_count=1,
        target_branch="develop",
        workdir=work,
    )

    plan = seed_baselines._build_seed_plan(args)

    assert plan == [("develop", repo["c1"])]


def test_filter_uses_explicit_target_sha(repo) -> None:
    """An explicit target_sha is used verbatim for every plan entry."""
    work = repo["work"]
    plan = [("develop", repo["c1"]), ("develop", repo["f1"])]

    kept = seed_baselines._filter_plan_by_ancestry(plan, work, target_sha=repo["c2"])

    assert kept == [("develop", repo["c1"])]


def test_filter_unresolvable_tip_keeps_when_lenient_raises_when_strict(repo) -> None:
    """An unresolvable target tip warns+keeps by default but aborts under strict."""
    work = repo["work"]
    plan = [("ghost", repo["c1"])]

    kept = seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="")
    assert kept == plan

    with pytest.raises(RuntimeError, match="cannot resolve tip"):
        seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="", strict=True)


def test_prepare_jit_cache_opens_task_cache(tmp_path: Path) -> None:
    """Each task/backend bucket gets writable Warp/CUDA cache roots."""
    cache_dir = tmp_path / "jit-cache" / "seed" / "task" / "newton"

    prepared = seed_baselines._prepare_jit_cache(cache_dir)

    assert prepared == cache_dir
    assert (cache_dir / "warp").is_dir()
    assert (cache_dir / "nv").is_dir()
    assert cache_dir.stat().st_mode & 0o777 == 0o777


def test_only_failing_camera_backends_use_container_local_jit_cache() -> None:
    """Only camera backends that failed on the host mount bypass cache reuse."""
    camera_task = "Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct"

    assert seed_baselines._uses_container_local_jit_cache(SimpleNamespace(task_id=camera_task, backend_key="physx"))
    assert seed_baselines._uses_container_local_jit_cache(SimpleNamespace(task_id=camera_task, backend_key="newton"))
    assert not seed_baselines._uses_container_local_jit_cache(
        SimpleNamespace(task_id=camera_task, backend_key="physx_newton_renderer")
    )
    assert not seed_baselines._uses_container_local_jit_cache(
        SimpleNamespace(task_id="Isaac-Cartpole-Direct", backend_key="newton")
    )


def test_seed_source_directories_are_unique_and_disposable() -> None:
    """Independent seeder invocations do not reuse source-clone residue."""
    first = seed_baselines._create_seed_source_dir()
    second = seed_baselines._create_seed_source_dir()
    try:
        assert first != second
        assert first.is_dir()
        assert second.is_dir()
    finally:
        seed_baselines._cleanup_run_dir(first)
        seed_baselines._cleanup_run_dir(second)


def test_jit_cache_reuses_samples_and_isolates_backends(tmp_path: Path) -> None:
    """Repeated samples share compiled kernels, but different backends do not."""
    cache_root = tmp_path / "jit-cache" / "seed" / "run-1-attempt-1"
    newton_cache = seed_baselines._cache_bucket_path(
        cache_root, "perf-smoke/develop-staging", "abcdef123456", "Isaac-Cartpole-Direct", "newton"
    )
    repeated_newton_cache = seed_baselines._cache_bucket_path(
        cache_root, "perf-smoke/develop-staging", "abcdef123456", "Isaac-Cartpole-Direct", "newton"
    )
    physx_cache = seed_baselines._cache_bucket_path(
        cache_root, "perf-smoke/develop-staging", "abcdef123456", "Isaac-Cartpole-Direct", "physx"
    )

    _ = seed_baselines._prepare_jit_cache(newton_cache)
    compiled_kernel = newton_cache / "warp" / "compiled-kernel.so"
    compiled_kernel.write_bytes(b"compiled")
    _ = seed_baselines._prepare_jit_cache(repeated_newton_cache)

    assert repeated_newton_cache == newton_cache
    assert compiled_kernel.read_bytes() == b"compiled"
    assert physx_cache != newton_cache
    assert not physx_cache.exists()


def test_kit_cache_reuses_samples_and_isolates_backends(tmp_path: Path) -> None:
    """Repeated samples share Kit data, but different backends do not."""
    cache_root = tmp_path / "kit-cache" / "seed" / "run-1-attempt-1"
    newton_cache = seed_baselines._cache_bucket_path(
        cache_root, "perf-smoke/develop-staging", "abcdef123456", "Isaac-Camera-Direct", "newton"
    )
    repeated_newton_cache = seed_baselines._cache_bucket_path(
        cache_root, "perf-smoke/develop-staging", "abcdef123456", "Isaac-Camera-Direct", "newton"
    )
    physx_cache = seed_baselines._cache_bucket_path(
        cache_root, "perf-smoke/develop-staging", "abcdef123456", "Isaac-Camera-Direct", "physx"
    )

    _ = seed_baselines._prepare_kit_cache(newton_cache)
    shader_cache = newton_cache / "shader-cache.bin"
    shader_cache.write_bytes(b"compiled")
    _ = seed_baselines._prepare_kit_cache(repeated_newton_cache)

    assert repeated_newton_cache == newton_cache
    assert shader_cache.read_bytes() == b"compiled"
    assert newton_cache.stat().st_mode & 0o777 == 0o777
    assert physx_cache != newton_cache
    assert not physx_cache.exists()


def test_cleanup_run_dir_removes_run_tree(tmp_path: Path) -> None:
    """Seeder exit cleanup removes generated files from its run directory."""
    cache_dir = tmp_path / "jit-cache" / "seed" / "run-1-attempt-1"
    compiled_dir = cache_dir / "warp" / "version" / "module"
    compiled_dir.mkdir(parents=True)
    (compiled_dir / "kernel.so").write_bytes(b"compiled")
    compiled_dir.chmod(0o500)

    seed_baselines._cleanup_run_dir(cache_dir)

    assert not cache_dir.exists()


def test_docker_benchmark_copies_internal_output_after_exit(tmp_path: Path, monkeypatch) -> None:
    """Benchmark output is copied from the container instead of bind-mounted."""
    calls: list[list[str]] = []

    def _run(cmd: list[str], **_kwargs) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    task = SimpleNamespace(
        task_id="Isaac-Cartpole-Direct",
        num_envs=16,
        num_frames=10,
        warmup_frames=2,
        seed=42,
    )
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(seed_baselines, "hydra_args_for_task", lambda _task: [])
    monkeypatch.setattr(seed_baselines.subprocess, "run", _run)

    exit_code = seed_baselines._docker_run_benchmark(
        image="isaac-lab:test",
        task=task,
        artifact_dir=artifact_dir,
        jit_cache=tmp_path / "jit-cache",
        kit_cache=tmp_path / "kit-cache",
        seed_src_dir=tmp_path / "source",
        container_name="perf-seed-test",
    )

    docker_run = calls[0]
    assert exit_code == 0
    assert "--rm" not in docker_run
    assert f"{artifact_dir}:/tmp/bench_out" not in docker_run
    assert f"{tmp_path / 'jit-cache'}:/tmp/jit-cache" in docker_run
    assert f"{tmp_path / 'kit-cache'}:/isaac-sim/kit/cache" in docker_run
    assert "umask 000" in docker_run[-1]
    assert "mkdir -p /tmp/bench_out" in docker_run[-1]
    assert calls[1] == ["docker", "cp", "perf-seed-test:/tmp/bench_out/.", str(artifact_dir)]
    assert calls[2] == ["chmod", "-R", "a+rwX", str(artifact_dir)]
    assert calls[3] == ["docker", "rm", "-f", "perf-seed-test"]


def test_docker_benchmark_can_use_container_local_jit_cache(tmp_path: Path, monkeypatch) -> None:
    """A container-local cache creates JIT roots without a host bind mount."""
    calls: list[list[str]] = []

    def _run(cmd: list[str], **_kwargs) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    task = SimpleNamespace(
        task_id="Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct",
        num_envs=16,
        num_frames=10,
        warmup_frames=2,
        seed=42,
    )
    monkeypatch.setattr(seed_baselines, "hydra_args_for_task", lambda _task: [])
    monkeypatch.setattr(seed_baselines.subprocess, "run", _run)

    exit_code = seed_baselines._docker_run_benchmark(
        image="isaac-lab:test",
        task=task,
        artifact_dir=tmp_path / "artifacts",
        jit_cache=None,
        kit_cache=tmp_path / "kit-cache",
        seed_src_dir=tmp_path / "source",
        container_name="perf-seed-test",
    )

    docker_run = calls[0]
    assert exit_code == 0
    assert not any(arg.endswith(":/tmp/jit-cache") for arg in docker_run)
    assert "mkdir -p /tmp/bench_out /tmp/jit-cache/warp /tmp/jit-cache/nv" in docker_run[-1]
