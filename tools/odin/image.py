# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a commit-pinned Isaac Lab benchmark image and push it to a registry.

The Dockerfile is deliberately minimal: the repository's committed ``uv.lock``
and ``uv sync`` do the dependency work, so this module must not reimplement any
of it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.odin.config_file import ImageSpec

__all__ = [
    "BUNDLE_REF",
    "DEFAULT_CUDA_IMAGE",
    "PROFILES",
    "ImageBuildError",
    "build_image",
    "image_tag_for",
    "render_dockerfile",
    "resolve_ref",
]

# uv extras per profile. ``full`` is everything that co-resolves in one
# virtualenv, so a task never reasons about which extras a workload needs and a
# row never has to be routed to the right image.
#
# This became possible once the packaging cap was widened: ovphysx capped
# packaging at <24 while isaacsim-core pinned ==26.0, which made the two extras
# unresolvable together. The ``packaging>=20,<27`` override in the root
# pyproject collapses them onto 26.0.
#
# Still excluded, because they genuinely cannot co-resolve with isaacsim:
# teleop, viser, mimic, test, and the ``all`` aggregate.
PROFILES: dict[str, tuple[str, ...]] = {
    "full": ("isaacsim", "ovphysx", "ovrtx", "rsl-rl", "skrl", "rl-games", "sb3", "rerun"),
}

# x86_64 resolves torch from the cu128 index (see [tool.uv.sources] in
# pyproject.toml); aarch64 uses cu130 and needs a matching base override.
DEFAULT_CUDA_IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"

# Temporary branch the build bundle is published under. Created and deleted
# inside :func:`build_image`, so ``Versions.git_branch`` in emitted benchmark
# bundles reads ``odin-build``; ``git_commit`` remains the exact commit and is
# the authoritative provenance.
BUNDLE_REF = "odin-build"

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_Runner = Callable[..., subprocess.CompletedProcess]


class ImageBuildError(RuntimeError):
    """Raised when resolving a ref, rendering, building, or pushing fails."""


def resolve_ref(ref: str, *, repo_root: Path, run: _Runner = subprocess.run) -> str:
    """Resolve a git ref to a full commit SHA.

    Args:
        ref: Branch, tag, or SHA.
        repo_root: Repository to resolve within.
        run: Subprocess runner, injected for testing.

    Returns:
        The full commit SHA.

    Raises:
        ImageBuildError: If *ref* does not resolve.
    """
    completed = run(
        ["git", "rev-parse", f"{ref}^{{commit}}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ImageBuildError(f"could not resolve ref {ref!r}: {(completed.stderr or '').strip()}")
    return completed.stdout.strip()


def image_tag_for(spec: ImageSpec, commit_sha: str, profiles: list[str]) -> str:
    """Return the tag for a built image.

    Args:
        spec: Registry coordinates.
        commit_sha: Full commit SHA; the tag uses its first seven characters.
        profiles: Profiles baked into the image, in order.

    Returns:
        ``<registry>/<repository>:<short_sha>-<profile>[-<profile>...]``.
    """
    return f"{spec.registry}/{spec.repository}:{commit_sha[:7]}-{'-'.join(profiles)}"


def render_dockerfile(*, commit_sha: str, cuda_image: str, profiles: list[str]) -> str:
    """Render the Dockerfile for a commit-pinned benchmark image.

    Args:
        commit_sha: Commit to check out inside the image.
        cuda_image: Base image reference.
        profiles: Profile names to warm, each a key of :data:`PROFILES`.

    Returns:
        The rendered Dockerfile.

    Raises:
        ImageBuildError: If a profile is not a known profile name.
    """
    unknown = [profile for profile in profiles if profile not in PROFILES]
    if unknown:
        raise ImageBuildError(f"unknown profile(s) {unknown}; expected some of {sorted(PROFILES)}")
    environment = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return environment.get_template("Dockerfile.j2").render(
        commit_sha=commit_sha,
        cuda_image=cuda_image,
        profiles=profiles,
        profile_extras=PROFILES,
        bundle_ref=BUNDLE_REF,
    )


def build_image(
    *,
    spec: ImageSpec,
    ref: str,
    repo_root: Path,
    profiles: list[str],
    context_dir: Path,
    cuda_image: str = DEFAULT_CUDA_IMAGE,
    push: bool = False,
    dry_run: bool = False,
    run: _Runner = subprocess.run,
) -> str:
    """Build, and optionally push, a commit-pinned benchmark image.

    Args:
        spec: Registry coordinates.
        ref: Git ref to pin.
        repo_root: Repository the bundle is created from.
        profiles: Profile names to warm.
        context_dir: Directory the build context is written to.
        cuda_image: Base image reference.
        push: Whether to ``docker push`` after a successful build.
        dry_run: Write the context and return the tag without invoking git
            bundling or Docker.
        run: Subprocess runner, injected for testing.

    Returns:
        The image tag.

    Raises:
        ImageBuildError: If the ref does not resolve, or bundling, building, or
            pushing fails.
    """
    commit_sha = resolve_ref(ref, repo_root=repo_root, run=run)
    tag = image_tag_for(spec, commit_sha, profiles)

    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "Dockerfile").write_text(
        render_dockerfile(commit_sha=commit_sha, cuda_image=cuda_image, profiles=profiles)
    )
    if dry_run:
        return tag

    _bundle_commit(commit_sha, repo_root=repo_root, dest=context_dir / "isaaclab.bundle", run=run)
    # Docker output is streamed rather than captured. This build pulls Isaac Sim
    # and can run for tens of minutes; with output captured, a wedged step and a
    # merely slow one look identical until the process exits.
    _check(run(["docker", "build", "--network=host", "-t", tag, str(context_dir)], check=False), "docker build")
    if push:
        _check(run(["docker", "push", tag], check=False), "docker push")
    return tag


def _bundle_commit(commit_sha: str, *, repo_root: Path, dest: Path, run: _Runner) -> None:
    """Bundle exactly one commit under :data:`BUNDLE_REF`.

    ``git bundle`` requires a named ref, so a temporary branch is pointed at
    *commit_sha*, bundled, and removed. Bundling that one ref rather than
    ``--all`` keeps the build context small.

    Args:
        commit_sha: Commit to bundle.
        repo_root: Repository to bundle from.
        dest: Bundle path to write.
        run: Subprocess runner.

    Raises:
        ImageBuildError: If creating the ref or the bundle fails.
    """
    ref_path = f"refs/heads/{BUNDLE_REF}"
    _check(
        run(
            ["git", "update-ref", ref_path, commit_sha],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        ),
        "git update-ref",
    )
    try:
        _check(
            run(
                ["git", "bundle", "create", str(dest), BUNDLE_REF],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            ),
            "git bundle create",
        )
    finally:
        # Best effort: leaving the temp branch behind is untidy but harmless,
        # and failing here would mask a real bundling error.
        run(
            ["git", "update-ref", "-d", ref_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )


def _check(completed: subprocess.CompletedProcess, what: str) -> None:
    """Raise :class:`ImageBuildError` when *completed* failed.

    ``stderr`` is only present for the steps that capture it; streamed steps
    have already printed their diagnostics, so the message names the step and
    the exit code instead.
    """
    if completed.returncode == 0:
        return
    detail = (completed.stderr or "").strip()
    raise ImageBuildError(
        f"{what} failed: {detail}" if detail else f"{what} failed with exit code {completed.returncode}"
    )
