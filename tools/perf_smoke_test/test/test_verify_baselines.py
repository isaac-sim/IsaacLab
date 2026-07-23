# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the seeded-baseline usability verifier.

Mirrors how the gate selects samples: a sample counts only if it shares a single
fingerprint bucket, targets the evaluated branch, and its ``commit_sha`` is an
ancestor of the target tip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import verify_baselines  # noqa: E402


def _sample(
    fps: float,
    commit: str,
    runtime_hash: str = "rt-a",
    epoch: int = 1,
    target_branch: str = "develop",
    sample_id: str | None = None,
) -> dict:
    return {
        "fps": fps,
        "commit_sha": commit,
        "target_branch": target_branch,
        "launch_config_hash": "lc-a",
        "benchmark_contract_hash": "bc-a",
        "runtime_contract_hash": runtime_hash,
        "baseline_epoch": epoch,
        "sample_id": sample_id or f"{commit}-{fps}",
    }


# --------------------------------------------------------------------------------------
# usable_sample_count (pure)
# --------------------------------------------------------------------------------------


def test_usable_count_single_fingerprint_no_ancestry() -> None:
    """Uniform samples all count when ancestry is not enforced."""
    records = [_sample(100.0 + i, f"c{i}") for i in range(5)]

    usable, total, fingerprints = verify_baselines.usable_sample_count(records)

    assert (usable, total, fingerprints) == (5, 5, 1)


def test_usable_count_reports_largest_fingerprint_group() -> None:
    """Mixed environments never pool; the largest single group is what the gate sees."""
    records = [_sample(100.0, f"a{i}", runtime_hash="rt-a") for i in range(3)]
    records += [_sample(100.0, f"b{i}", runtime_hash="rt-b") for i in range(2)]

    usable, total, fingerprints = verify_baselines.usable_sample_count(records)

    assert usable == 3
    assert total == 5
    assert fingerprints == 2


def test_usable_count_applies_ancestry_predicate() -> None:
    """Only samples whose commit passes the ancestry predicate are eligible."""
    records = [_sample(100.0, "in1"), _sample(100.0, "in2"), _sample(100.0, "out1")]
    ancestors = {"in1", "in2"}

    usable, total, _ = verify_baselines.usable_sample_count(records, lambda c: c in ancestors)

    assert usable == 2
    assert total == 3


def test_usable_count_excludes_samples_missing_commit_sha_under_ancestry() -> None:
    """A sample without a commit_sha cannot satisfy the gate's ancestry filter."""
    records = [_sample(100.0, "in1"), {"fps": 1.0, "runtime_contract_hash": "rt-a"}]

    usable, _, _ = verify_baselines.usable_sample_count(records, lambda c: True)

    assert usable == 1


def test_usable_count_requires_target_branch_match() -> None:
    """Staging samples cannot satisfy a develop verification."""
    records = [
        _sample(100.0, "develop", target_branch="develop"),
        _sample(100.0, "staging", target_branch="perf-smoke/develop-staging"),
    ]

    usable, total, _ = verify_baselines.usable_sample_count(records, target_branch="develop")

    assert usable == 1
    assert total == 2


def test_usable_count_requires_current_seeder_sample_ids() -> None:
    """Old matching fingerprints cannot hide missing samples from the current run."""
    records = [
        _sample(100.0, "old1", sample_id="old-1"),
        _sample(100.0, "old2", sample_id="old-2"),
        _sample(100.0, "new1", sample_id="new-1"),
    ]

    usable, total, _ = verify_baselines.usable_sample_count(
        records,
        target_branch="develop",
        expected_sample_ids={"new-1", "new-2"},
    )

    assert usable == 1
    assert total == 3


def test_expected_sample_ids_are_scoped_to_gpu_and_target_branch(tmp_path: Path) -> None:
    """The current-run verification ignores records for other gate contexts."""
    summary = tmp_path / "seed_records.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "gpu_model": "l40s",
                    "task_id": "task-a",
                    "backend": "physx",
                    "target_branch": "develop",
                    "sample_id": "expected",
                },
                {
                    "gpu_model": "l40s",
                    "task_id": "task-a",
                    "backend": "physx",
                    "target_branch": "perf-smoke/develop-staging",
                    "sample_id": "wrong-branch",
                },
                {
                    "gpu_model": "rtx6000",
                    "task_id": "task-a",
                    "backend": "physx",
                    "target_branch": "develop",
                    "sample_id": "wrong-gpu",
                },
            ]
        )
    )

    expected = verify_baselines._load_expected_sample_ids(summary, "l40s", "develop")

    assert expected == {("task-a", "physx"): {"expected"}}


# --------------------------------------------------------------------------------------
# verify_bucket (git-backed)
# --------------------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_verify_bucket_counts_ancestor_samples_from_git(tmp_path: Path) -> None:
    """verify_bucket reads samples.ndjson from a git ref and honors ancestry."""
    work = tmp_path / "work"
    work.mkdir()
    _git(["init", "-b", "develop"], work)
    _git(["config", "user.email", "test@example.com"], work)
    _git(["config", "user.name", "test"], work)

    # Build a short develop history and remember each commit SHA.
    shas: list[str] = []
    for i in range(5):
        (work / f"f{i}").write_text(f"{i}\n", encoding="utf-8")
        _git(["add", f"f{i}"], work)
        _git(["commit", "-m", f"c{i}"], work)
        shas.append(_git(["rev-parse", "HEAD"], work))
    develop_tip = shas[-1]

    # A diverged commit that is NOT on develop.
    _git(["checkout", "-b", "feature", shas[0]], work)
    (work / "ff").write_text("x\n", encoding="utf-8")
    _git(["add", "ff"], work)
    _git(["commit", "-m", "feature"], work)
    feature_sha = _git(["rev-parse", "HEAD"], work)
    _git(["checkout", "develop"], work)

    # Write the bucket samples: 5 on develop history + 1 off-branch.
    rel = str(verify_baselines._samples_path(Path(""), "l40s", "Isaac-Cartpole-Direct", "newton", None))
    sample_path = work / rel
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_sample(100.0 + i, shas[i])) for i in range(5)]
    lines.append(json.dumps(_sample(9.0, feature_sha)))
    sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(["add", rel], work)
    _git(["commit", "-m", "baselines"], work)
    ref = _git(["rev-parse", "HEAD"], work)

    usable, total, fingerprints = verify_baselines.verify_bucket(
        ref, "l40s", "Isaac-Cartpole-Direct", "newton", develop_tip, repo_dir=work
    )

    assert total == 6  # all samples present in the file
    assert usable == 5  # the off-branch sample is excluded by ancestry
    assert fingerprints == 1


def test_verify_bucket_missing_bucket_is_empty(tmp_path: Path) -> None:
    """A bucket with no samples.ndjson reports zero without raising."""
    work = tmp_path / "work"
    work.mkdir()
    _git(["init", "-b", "develop"], work)
    _git(["config", "user.email", "test@example.com"], work)
    _git(["config", "user.name", "test"], work)
    (work / "README").write_text("x\n", encoding="utf-8")
    _git(["add", "README"], work)
    _git(["commit", "-m", "init"], work)
    ref = _git(["rev-parse", "HEAD"], work)

    usable, total, fingerprints = verify_baselines.verify_bucket(ref, "l40s", "Nope", "newton", None, repo_dir=work)

    assert (usable, total, fingerprints) == (0, 0, 0)
