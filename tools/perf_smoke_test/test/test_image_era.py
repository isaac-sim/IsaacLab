# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the source-derivable image-era key and manifest resolver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import image_era  # noqa: E402

_ENV_BASE = """
# General settings
ACCEPT_EULA=Y
ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
# version comment
ISAACSIM_VERSION=6.0.0-dev2
DOCKER_USER_HOME=/root
DOCKER_NAME_SUFFIX=""
"""


def test_parse_env_file_strips_comments_and_quotes() -> None:
    """Comments, blank lines, and surrounding quotes must not leak into values."""
    env = image_era.parse_env_file(_ENV_BASE)

    assert env["ISAACSIM_BASE_IMAGE"] == "nvcr.io/nvidia/isaac-sim"
    assert env["ISAACSIM_VERSION"] == "6.0.0-dev2"
    assert env["DOCKER_NAME_SUFFIX"] == ""
    assert "# General settings" not in env


def test_era_key_is_deterministic() -> None:
    """The same parsed env must always hash to the same era key."""
    env = image_era.parse_env_file(_ENV_BASE)

    assert image_era.compute_era_key(env) == image_era.compute_era_key(env)


def test_era_key_changes_when_isaacsim_version_changes() -> None:
    """A different Isaac Sim version is a different era (under-sensitivity guard)."""
    old = image_era.compute_era_key(image_era.parse_env_file(_ENV_BASE))
    new = image_era.compute_era_key(image_era.parse_env_file(_ENV_BASE.replace("6.0.0-dev2", "5.0.0")))

    assert old != new


def test_era_key_ignores_cosmetic_and_non_era_fields() -> None:
    """Comment edits, reordering, and non-era fields must not split the era (over-sensitivity guard)."""
    base_key = image_era.compute_era_key(image_era.parse_env_file(_ENV_BASE))
    cosmetic = """
# a totally different comment
ISAACSIM_VERSION=6.0.0-dev2
ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
DOCKER_USER_HOME=/home/somethingelse
ACCEPT_EULA=N
"""

    assert image_era.compute_era_key(image_era.parse_env_file(cosmetic)) == base_key


def test_resolve_image_hit_returns_pinned_image() -> None:
    """A known era resolves to its immutable image and reports a match."""
    manifest = {
        "schema_version": 1,
        "fallback_image": "nvcr.io/nvidian/isaac-lab:latest-perf",
        "eras": {"era-abc": {"image": "nvcr.io/nvidian/isaac-lab:sha-deadbee"}},
    }

    image, matched = image_era.resolve_image("era-abc", manifest)

    assert matched is True
    assert image == "nvcr.io/nvidian/isaac-lab:sha-deadbee"


def test_resolve_image_miss_falls_back_to_latest_perf() -> None:
    """An unknown era degrades to the manifest fallback instead of failing."""
    manifest = {
        "schema_version": 1,
        "fallback_image": "nvcr.io/nvidian/isaac-lab:latest-perf",
        "eras": {},
    }

    image, matched = image_era.resolve_image("era-unknown", manifest)

    assert matched is False
    assert image == "nvcr.io/nvidian/isaac-lab:latest-perf"


def test_resolve_image_explicit_fallback_overrides_manifest() -> None:
    """An explicit fallback takes precedence over the manifest default."""
    image, matched = image_era.resolve_image(
        "era-unknown", {"eras": {}}, fallback_image="nvcr.io/nvidian/isaac-lab:sha-override"
    )

    assert matched is False
    assert image == "nvcr.io/nvidian/isaac-lab:sha-override"


def test_resolve_image_empty_manifest_uses_default_fallback() -> None:
    """A missing/empty manifest still yields the built-in default image."""
    image, matched = image_era.resolve_image("era-unknown", None)

    assert matched is False
    assert image == image_era.DEFAULT_FALLBACK_IMAGE


def test_era_key_from_commit_matches_tree_at_head() -> None:
    """The git-show path and the working-tree path must agree for the same source."""
    tree_key = image_era.era_key_from_tree(_REPO_ROOT)
    commit_key = image_era.era_key_from_commit("HEAD", _REPO_ROOT)

    assert tree_key == commit_key


# --------------------------------------------------------------------------------------
# manifest_upsert (pure)
# --------------------------------------------------------------------------------------


def test_manifest_upsert_adds_new_entry_with_metadata() -> None:
    """A new era records its image plus any provided metadata and reports a change."""
    updated, changed = image_era.manifest_upsert(
        image_era.empty_manifest(), "era-1", "repo:sha-a", extra={"isaacsim_version": "6.0.0", "dropped": None}
    )

    assert changed is True
    assert updated["eras"]["era-1"]["image"] == "repo:sha-a"
    assert updated["eras"]["era-1"]["isaacsim_version"] == "6.0.0"
    assert "dropped" not in updated["eras"]["era-1"]  # None-valued metadata is not stored


def test_manifest_upsert_same_image_is_noop() -> None:
    """Re-recording the identical era -> image mapping reports no change."""
    manifest, _ = image_era.manifest_upsert(None, "era-1", "repo:sha-a")
    _, changed = image_era.manifest_upsert(manifest, "era-1", "repo:sha-a")

    assert changed is False


def test_manifest_upsert_remap_changes_image_and_preserves_other_eras() -> None:
    """Remapping one era to a new image must not disturb sibling eras."""
    manifest, _ = image_era.manifest_upsert(None, "era-1", "repo:sha-a")
    manifest, _ = image_era.manifest_upsert(manifest, "era-2", "repo:sha-b")
    updated, changed = image_era.manifest_upsert(manifest, "era-1", "repo:sha-c")

    assert changed is True
    assert updated["eras"]["era-1"]["image"] == "repo:sha-c"
    assert updated["eras"]["era-2"]["image"] == "repo:sha-b"


def test_manifest_upsert_does_not_mutate_input() -> None:
    """Upsert returns a new manifest and leaves the caller's manifest untouched."""
    manifest = image_era.empty_manifest()
    image_era.manifest_upsert(manifest, "era-1", "repo:sha-a")

    assert manifest["eras"] == {}


# --------------------------------------------------------------------------------------
# git-backed manifest record / load round trips
# --------------------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def _init_repo_with_remote(tmp_path: Path) -> Path:
    """Create a bare ``origin`` and a working clone with one commit; return the clone."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)

    work = tmp_path / "work"
    work.mkdir()
    _git(["init"], work)
    _git(["remote", "add", "origin", str(remote)], work)
    _git(["config", "user.email", "test@example.com"], work)
    _git(["config", "user.name", "test"], work)
    (work / "README.md").write_text("seed\n", encoding="utf-8")
    _git(["add", "README.md"], work)
    _git(["commit", "-m", "init"], work)
    return work


def test_record_and_load_manifest_round_trips_over_git(tmp_path) -> None:
    """A recorded era resolves to its immutable image when read back from git."""
    work = _init_repo_with_remote(tmp_path)

    pushed = image_era.record_era_image(
        "era-1", "repo:sha-a", extra={"isaacsim_version": "6.0.0"}, remote="origin", repo_dir=work
    )
    assert pushed is True

    manifest = image_era.load_manifest_from_git(remote="origin", repo_dir=work)
    image, matched = image_era.resolve_image("era-1", manifest)

    assert matched is True
    assert image == "repo:sha-a"
    assert manifest["eras"]["era-1"]["isaacsim_version"] == "6.0.0"


def test_record_era_image_is_idempotent_and_supports_remap(tmp_path) -> None:
    """Re-recording the same mapping is a no-op; a remap pushes the new image."""
    work = _init_repo_with_remote(tmp_path)

    assert image_era.record_era_image("era-1", "repo:sha-a", remote="origin", repo_dir=work) is True
    assert image_era.record_era_image("era-1", "repo:sha-a", remote="origin", repo_dir=work) is False
    assert image_era.record_era_image("era-1", "repo:sha-b", remote="origin", repo_dir=work) is True

    manifest = image_era.load_manifest_from_git(remote="origin", repo_dir=work)
    assert manifest["eras"]["era-1"]["image"] == "repo:sha-b"


def test_record_era_image_preserves_existing_eras(tmp_path) -> None:
    """Recording a second era appends without clobbering the first."""
    work = _init_repo_with_remote(tmp_path)

    image_era.record_era_image("era-1", "repo:sha-a", remote="origin", repo_dir=work)
    image_era.record_era_image("era-2", "repo:sha-b", remote="origin", repo_dir=work)

    manifest = image_era.load_manifest_from_git(remote="origin", repo_dir=work)
    assert manifest["eras"]["era-1"]["image"] == "repo:sha-a"
    assert manifest["eras"]["era-2"]["image"] == "repo:sha-b"


def test_load_manifest_from_git_missing_branch_is_empty(tmp_path) -> None:
    """A repo without the era branch resolves to an empty manifest (fallback path)."""
    work = _init_repo_with_remote(tmp_path)

    manifest = image_era.load_manifest_from_git(remote="origin", repo_dir=work)

    assert manifest["eras"] == {}


def _resolve_matched(work: Path) -> bool:
    """Reproduce the era-roll detect job: is HEAD's era already recorded?"""
    manifest = image_era.load_manifest_from_git(remote="origin", repo_dir=work)
    _, matched = image_era.resolve_image(image_era.era_key_from_tree(work), manifest)
    return matched


def test_era_roll_policy_new_then_recorded_then_changed(tmp_path) -> None:
    """The auto-roll decision loop: new era rolls, recorded era skips, a changed era rolls again.

    ``should_roll == not matched``, so this asserts the ``matched`` signal the
    orchestrator keys off of across a full env-change cycle.
    """
    work = _init_repo_with_remote(tmp_path)
    (work / "docker").mkdir()
    (work / "docker" / ".env.base").write_text(_ENV_BASE, encoding="utf-8")

    # New era: nothing recorded -> should_roll.
    assert _resolve_matched(work) is False

    # Record this era's image -> subsequent pushes on the same era skip.
    era_key = image_era.era_key_from_tree(work)
    assert image_era.record_era_image(era_key, "repo:sha-a", remote="origin", repo_dir=work) is True
    assert _resolve_matched(work) is True

    # A change to the era inputs is a new era again -> should_roll, and the old era
    # stays recorded (pinnable) rather than being clobbered.
    (work / "docker" / ".env.base").write_text(_ENV_BASE.replace("6.0.0-dev2", "7.0.0"), encoding="utf-8")
    assert image_era.era_key_from_tree(work) != era_key
    assert _resolve_matched(work) is False
    assert image_era.load_manifest_from_git(remote="origin", repo_dir=work)["eras"][era_key]["image"] == "repo:sha-a"


def test_era_roll_seeds_full_task_matrix() -> None:
    """A new image era immediately fills every perf-smoke task bucket."""
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/perf-smoke-era-roll.yaml"

    assert 'tasks: "__ALL_TASKS__"' in workflow.read_text()
