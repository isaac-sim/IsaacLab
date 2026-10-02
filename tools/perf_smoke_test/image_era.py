# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-derivable image-era key and manifest resolution for the perf smoke test.

The gate pulls a prebuilt CI image, but older commits may expect a different
dependency era than the moving ``latest-perf`` tag provides. The era key is a
hash computed *statically* from a commit's container-defining source (today:
``docker/.env.base``), so the gate can pick the right immutable image
(``sha-<short>``) *before* running anything -- unlike ``runtime_contract_hash``,
which is only known after a benchmark runs inside the container.

This module is the single authority for that key: both the publish side (which
records ``era_key -> image`` into the manifest) and the gate/seed/bisect side
(which reads it) MUST compute the key here so their values agree byte-for-byte.
A divergence would silently miss every lookup and fall back forever.

The era key deliberately captures only the *base/container* layer, not
IsaacLab's own source dependencies: the PR code is bind-mounted over the
container, so those deps are what we are testing, not part of the image era.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .hashing import stable_hash
except ImportError:  # pragma: no cover - supports direct script imports
    from hashing import stable_hash

# Bump when the selected field set changes; old keys stay valid under their version.
IMAGE_ERA_VERSION = 1

# Container-defining source file, relative to the repo root.
ENV_BASE_RELPATH = "docker/.env.base"

# v1 field set: the Isaac Sim base image + version are the dominant era drivers
# and the only inputs that change the base layer the image is built FROM.
ERA_ENV_FIELDS: tuple[str, ...] = ("ISAACSIM_BASE_IMAGE", "ISAACSIM_VERSION")

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_FALLBACK_IMAGE = "nvcr.io/nvidian/isaac-lab:latest-perf"

# The manifest is machine-managed mutable state, so it lives on the same
# ``perf-baselines`` branch as the rolling baselines (not in the source tree):
# the gate already fetches this branch, and CI already has ``contents: write``
# for it. Keeping era -> image mappings beside the baselines they pin means one
# branch carries all of the gate's out-of-tree state.
MANIFEST_BRANCH = "perf-baselines"
MANIFEST_RELPATH = "image_era_manifest.json"


def parse_env_file(text: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` env file, ignoring comments and surrounding quotes."""
    env: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def read_env_base_from_tree(source_root: str | Path) -> str:
    """Return ``docker/.env.base`` contents from a checked-out working tree."""
    path = Path(source_root) / ENV_BASE_RELPATH
    if not path.exists():
        raise FileNotFoundError(f"{ENV_BASE_RELPATH} not found under {source_root}")
    return path.read_text(encoding="utf-8")


def read_env_base_from_commit(commit: str, repo_root: str | Path = ".") -> str:
    """Return ``docker/.env.base`` contents at a git commit via ``git show``.

    Lets the gate compute an older commit's era key without checking it out.
    """
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{ENV_BASE_RELPATH}"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised via integration
        raise FileNotFoundError(f"{ENV_BASE_RELPATH} not found at commit {commit}: {exc.stderr.strip()}") from exc


def build_era_contract(env: Mapping[str, str]) -> dict[str, Any]:
    """Return the versioned ``{version, fields}`` contract that the key hashes."""
    fields = {field: env.get(field) for field in ERA_ENV_FIELDS}
    return {"image_era_version": IMAGE_ERA_VERSION, "fields": fields}


def compute_era_key(env: Mapping[str, str]) -> str:
    """Return the era key for a parsed ``.env.base`` mapping."""
    return stable_hash(build_era_contract(env))


def era_key_from_tree(source_root: str | Path) -> str:
    """Convenience: compute the era key from a working tree."""
    return compute_era_key(parse_env_file(read_env_base_from_tree(source_root)))


def era_key_from_commit(commit: str, repo_root: str | Path = ".") -> str:
    """Convenience: compute the era key for a git commit."""
    return compute_era_key(parse_env_file(read_env_base_from_commit(commit, repo_root)))


def resolve_image(
    era_key: str,
    manifest: Mapping[str, Any] | None,
    *,
    fallback_image: str | None = None,
) -> tuple[str, bool]:
    """Resolve an image reference for an era key.

    Returns ``(image_ref, matched)``. ``matched`` is ``True`` when the manifest
    has an entry for ``era_key``; otherwise the fallback is returned so the gate
    degrades to ``latest-perf`` instead of failing.
    """
    eras = (manifest or {}).get("eras") or {}
    entry = eras.get(era_key)
    if isinstance(entry, Mapping) and entry.get("image"):
        return str(entry["image"]), True
    fallback = fallback_image or (manifest or {}).get("fallback_image") or DEFAULT_FALLBACK_IMAGE
    return str(fallback), False


def empty_manifest() -> dict[str, Any]:
    """Return a fresh, empty manifest with the current schema and default fallback."""
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "fallback_image": DEFAULT_FALLBACK_IMAGE, "eras": {}}


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load an image-era manifest from disk; return an empty manifest if absent."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return empty_manifest()
    with manifest_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def manifest_upsert(
    manifest: Mapping[str, Any] | None,
    era_key: str,
    image: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return ``(new_manifest, changed)`` with ``era_key`` mapped to ``image``.

    ``changed`` is ``False`` when the era already resolves to ``image`` so callers
    can skip an empty commit/push (the mapping is the source of truth; identical
    remaps are no-ops even if ``extra`` metadata differs).

    Args:
        manifest: Existing manifest to update, or ``None`` to start from empty.
        era_key: The era key to map.
        image: The immutable image reference to record for the era.
        extra: Optional metadata to store alongside the image (e.g. the Isaac Sim
            version or publishing commit); ``None`` values are dropped.
    """
    result = json.loads(json.dumps(dict(manifest))) if manifest else empty_manifest()
    result.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    result.setdefault("fallback_image", DEFAULT_FALLBACK_IMAGE)
    eras = result.setdefault("eras", {})

    prior = eras.get(era_key)
    if isinstance(prior, Mapping) and prior.get("image") == image:
        return result, False

    entry: dict[str, Any] = {"image": image}
    if extra:
        entry.update({key: value for key, value in extra.items() if value is not None})
    eras[era_key] = entry
    return result, True


def load_manifest_from_git(
    *,
    branch: str = MANIFEST_BRANCH,
    remote: str | None = "origin",
    repo_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the manifest from ``branch`` in git; empty manifest if branch/file absent.

    A missing branch or file degrades to :func:`empty_manifest` so a first run
    resolves to the fallback image instead of failing.
    """
    # Lazy import: keeps the pure key/resolve path free of the baseline git deps.
    from baseline_manager import _git_show_file, refresh_baseline_branch

    repo_path = Path(repo_dir) if repo_dir is not None else None
    sha = refresh_baseline_branch(branch, remote=remote, repo_dir=repo_path, allow_missing=True)
    if not sha:
        return empty_manifest()
    content = _git_show_file(sha, MANIFEST_RELPATH, repo_dir=repo_path)
    if not content:
        return empty_manifest()
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        return empty_manifest()
    return loaded if isinstance(loaded, dict) else empty_manifest()


def record_era_image(
    era_key: str,
    image: str,
    *,
    extra: Mapping[str, Any] | None = None,
    branch: str = MANIFEST_BRANCH,
    remote: str | None = "origin",
    repo_dir: str | Path | None = None,
    max_retries: int = 3,
) -> bool:
    """Record ``era_key -> image`` in the manifest on ``branch`` and push.

    Returns ``True`` when a new/updated entry was pushed, ``False`` when the era
    already resolved to ``image`` (idempotent no-op). Mirrors the baseline
    manager's refetch-reapply-retry transaction so concurrent publishes cannot
    clobber each other's entries.
    """
    # Lazy import: reuse the baseline manager's git transaction machinery without
    # coupling the pure key/resolve path (its only importer today) to it.
    from baseline_manager import (
        _baseline_update_worktree,
        _commit_env,
        _git,
        _git_error,
        refresh_baseline_branch,
    )

    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    repo_path = Path(repo_dir) if repo_dir is not None else None
    last_error = "unknown push failure"
    for attempt in range(1, max_retries + 1):
        base_sha = refresh_baseline_branch(branch, remote=remote, repo_dir=repo_path, allow_missing=True)
        with _baseline_update_worktree(base_sha, repo_dir=repo_path) as worktree:
            manifest_path = worktree / MANIFEST_RELPATH
            updated, changed = manifest_upsert(load_manifest(manifest_path), era_key, image, extra=extra)
            if not changed:
                return False
            manifest_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _git(["add", MANIFEST_RELPATH], cwd=worktree, check=True)
            _git(
                ["commit", "-m", f"[image_era] Record {era_key[:12]} -> {image}"],
                cwd=worktree,
                check=True,
                env=_commit_env(),
            )
            push_ref = f"HEAD:refs/heads/{branch}"
            if remote:
                push = _git(["push", remote, push_ref], cwd=worktree)
            else:
                push = _git(["branch", "--force", branch, "HEAD"], cwd=worktree)
            if push.returncode == 0:
                return True
            last_error = _git_error(push)
            print(f"[image_era] manifest push attempt {attempt}/{max_retries} failed; refetching and retrying")

    raise RuntimeError(f"Failed to push era manifest to {branch!r} after {max_retries} attempts: {last_error}")


def _parse_extra(pairs: list[str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--extra expects KEY=VALUE, got {pair!r}")
        extra[key.strip()] = value.strip()
    return extra


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute the perf-smoke image-era key / resolve or record an image.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source_root", help="Working tree root to read docker/.env.base from.")
    source.add_argument("--commit", help="Git commit to read docker/.env.base from (via git show).")
    parser.add_argument("--repo_root", default=".", help="Repo root for --commit lookups and git manifest ops.")
    resolve_group = parser.add_mutually_exclusive_group()
    resolve_group.add_argument("--manifest", help="Resolve the key against this on-disk manifest file.")
    resolve_group.add_argument(
        "--manifest_from_git",
        action="store_true",
        help="Resolve against the manifest on the era branch (fetched from --remote).",
    )
    parser.add_argument("--fallback_image", help="Fallback image when the era is not in the manifest.")
    parser.add_argument("--branch", default=MANIFEST_BRANCH, help="Branch holding the era manifest.")
    parser.add_argument("--remote", default="origin", help="Git remote for --manifest_from_git / --record_image.")
    parser.add_argument("--record_image", help="Record era_key -> this image into the manifest on --branch and push.")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra metadata stored on the recorded era entry (repeatable).",
    )
    args = parser.parse_args(argv)

    if args.commit:
        era_key = era_key_from_commit(args.commit, args.repo_root)
    else:
        era_key = era_key_from_tree(args.source_root or args.repo_root)

    result: dict[str, Any] = {"era_key": era_key, "image_era_version": IMAGE_ERA_VERSION}
    if args.record_image:
        recorded = record_era_image(
            era_key,
            args.record_image,
            extra=_parse_extra(args.extra),
            branch=args.branch,
            remote=args.remote,
            repo_dir=args.repo_root,
        )
        result["recorded_image"] = args.record_image
        result["recorded"] = recorded
    elif args.manifest_from_git:
        manifest = load_manifest_from_git(branch=args.branch, remote=args.remote, repo_dir=args.repo_root)
        image, matched = resolve_image(era_key, manifest, fallback_image=args.fallback_image)
        result["image"] = image
        result["matched"] = matched
    elif args.manifest:
        image, matched = resolve_image(era_key, load_manifest(args.manifest), fallback_image=args.fallback_image)
        result["image"] = image
        result["matched"] = matched
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
