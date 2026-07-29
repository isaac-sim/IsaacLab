# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dockerfile rendering, tagging, and the build/push driver."""

import subprocess
from pathlib import Path

import pytest

from tools.odin.config_file import ImageSpec
from tools.odin.image import BUNDLE_REF, ImageBuildError, build_image, image_tag_for, render_dockerfile, resolve_ref

_SPEC = ImageSpec(registry="nvcr.io/nvidian", repository="antoiner-isaac-lab", pull_credential=None)
_SHA = "abcdef1234567890abcdef1234567890abcdef12"


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeRun:
    """Records commands and replays results keyed by command prefix.

    Keying by prefix rather than call order keeps the tests readable and stops
    them breaking when the implementation adds or reorders a subprocess call.

    Args:
        overrides: Maps a space-joined command prefix (e.g. ``"docker build"``)
            to the result it returns. Anything unmatched succeeds, with
            ``git rev-parse`` yielding a SHA.
    """

    def __init__(self, overrides: dict[str, subprocess.CompletedProcess] | None = None) -> None:
        self._overrides = dict(overrides or {})
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        cmd = list(cmd)
        self.calls.append(cmd)
        joined = " ".join(cmd)
        for prefix, result in self._overrides.items():
            if joined.startswith(prefix):
                return result
        if cmd[:2] == ["git", "rev-parse"]:
            return _cp(stdout=f"{_SHA}\n")
        return _cp()

    def ran(self, *prefix: str) -> bool:
        return any(call[: len(prefix)] == list(prefix) for call in self.calls)


def test_tag_encodes_the_short_sha_and_profiles() -> None:
    assert image_tag_for(_SPEC, _SHA, ["isaacsim", "newton"]) == (
        "nvcr.io/nvidian/antoiner-isaac-lab:abcdef1-isaacsim-newton"
    )


def test_resolve_ref_returns_the_full_sha(tmp_path: Path) -> None:
    fake = _FakeRun()
    assert resolve_ref("HEAD", repo_root=tmp_path, run=fake) == _SHA
    assert fake.calls[0][:2] == ["git", "rev-parse"]


def test_resolve_ref_rejects_an_unknown_ref(tmp_path: Path) -> None:
    fake = _FakeRun({"git rev-parse": _cp(returncode=128, stderr="unknown revision")})
    with pytest.raises(ImageBuildError, match="nope"):
        resolve_ref("nope", repo_root=tmp_path, run=fake)


def test_dockerfile_clones_from_the_bundle_and_verifies_the_sha() -> None:
    text = render_dockerfile(commit_sha=_SHA, cuda_image="nvidia/cuda:12.8.1-runtime-ubuntu24.04", profiles=["newton"])
    assert "FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04" in text
    assert "isaaclab.bundle" in text
    assert f"git clone --branch {BUNDLE_REF}" in text
    assert f'test "$(git rev-parse HEAD)" = "{_SHA}"' in text


def test_dockerfile_syncs_once_per_profile() -> None:
    text = render_dockerfile(commit_sha=_SHA, cuda_image="nvidia/cuda:12.8.1", profiles=["isaacsim", "newton"])
    assert text.count("RUN uv sync") == 2


def test_dockerfile_pins_the_lockfile_on_sync() -> None:
    # Without --frozen a stale local uv could re-resolve and defeat the pinning.
    text = render_dockerfile(commit_sha=_SHA, cuda_image="nvidia/cuda:12.8.1", profiles=["newton"])
    assert "uv sync --frozen" in text


def test_isaacsim_profile_never_selects_conflicting_extras() -> None:
    # [tool.uv].conflicts declares isaacsim incompatible with these.
    text = render_dockerfile(commit_sha=_SHA, cuda_image="nvidia/cuda:12.8.1", profiles=["isaacsim"])
    sync_line = next(line for line in text.splitlines() if "uv sync" in line)
    for forbidden in ("--extra all", "--extra mimic", "--extra teleop", "--extra ovphysx", "--extra viser"):
        assert forbidden not in sync_line


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ImageBuildError, match="quantum"):
        render_dockerfile(commit_sha=_SHA, cuda_image="nvidia/cuda:12.8.1", profiles=["quantum"])


def test_dry_run_writes_the_context_but_invokes_no_docker(tmp_path: Path) -> None:
    fake = _FakeRun()
    tag = build_image(
        spec=_SPEC,
        ref="HEAD",
        repo_root=tmp_path,
        profiles=["newton"],
        context_dir=tmp_path / "ctx",
        dry_run=True,
        run=fake,
    )
    assert tag.endswith(":abcdef1-newton")
    assert (tmp_path / "ctx" / "Dockerfile").exists()
    assert not fake.ran("docker")
    assert not fake.ran("git", "bundle")


def test_build_bundles_one_ref_and_invokes_docker_build(tmp_path: Path) -> None:
    fake = _FakeRun()
    build_image(
        spec=_SPEC, ref="HEAD", repo_root=tmp_path, profiles=["newton"], context_dir=tmp_path / "ctx", run=fake
    )
    assert fake.ran("git", "update-ref", f"refs/heads/{BUNDLE_REF}")
    assert fake.ran("git", "bundle", "create")
    assert fake.ran("docker", "build")


def test_temp_ref_is_deleted_after_bundling(tmp_path: Path) -> None:
    fake = _FakeRun()
    build_image(
        spec=_SPEC, ref="HEAD", repo_root=tmp_path, profiles=["newton"], context_dir=tmp_path / "ctx", run=fake
    )
    assert fake.ran("git", "update-ref", "-d", f"refs/heads/{BUNDLE_REF}")


def test_temp_ref_is_deleted_even_when_bundling_fails(tmp_path: Path) -> None:
    fake = _FakeRun({"git bundle create": _cp(returncode=1, stderr="bundle boom")})
    with pytest.raises(ImageBuildError, match="bundle boom"):
        build_image(
            spec=_SPEC, ref="HEAD", repo_root=tmp_path, profiles=["newton"], context_dir=tmp_path / "ctx", run=fake
        )
    assert fake.ran("git", "update-ref", "-d", f"refs/heads/{BUNDLE_REF}")


def test_push_is_skipped_unless_requested(tmp_path: Path) -> None:
    fake = _FakeRun()
    build_image(
        spec=_SPEC, ref="HEAD", repo_root=tmp_path, profiles=["newton"], context_dir=tmp_path / "ctx", run=fake
    )
    assert not fake.ran("docker", "push")


def test_push_runs_when_requested(tmp_path: Path) -> None:
    fake = _FakeRun()
    build_image(
        spec=_SPEC,
        ref="HEAD",
        repo_root=tmp_path,
        profiles=["newton"],
        context_dir=tmp_path / "ctx",
        push=True,
        run=fake,
    )
    assert fake.ran("docker", "push")


def test_failed_docker_build_raises(tmp_path: Path) -> None:
    fake = _FakeRun({"docker build": _cp(returncode=1, stderr="no space left on device")})
    with pytest.raises(ImageBuildError, match="no space left"):
        build_image(
            spec=_SPEC, ref="HEAD", repo_root=tmp_path, profiles=["newton"], context_dir=tmp_path / "ctx", run=fake
        )
