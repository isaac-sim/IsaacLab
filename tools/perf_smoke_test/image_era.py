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


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load an image-era manifest from disk; return an empty manifest if absent."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "fallback_image": DEFAULT_FALLBACK_IMAGE, "eras": {}}
    with manifest_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute the perf-smoke image-era key / resolve an image.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source_root", help="Working tree root to read docker/.env.base from.")
    source.add_argument("--commit", help="Git commit to read docker/.env.base from (via git show).")
    parser.add_argument("--repo_root", default=".", help="Repo root for --commit lookups.")
    parser.add_argument("--manifest", help="Optional image-era manifest to resolve the key against.")
    parser.add_argument("--fallback_image", help="Fallback image when the era is not in the manifest.")
    args = parser.parse_args(argv)

    if args.commit:
        era_key = era_key_from_commit(args.commit, args.repo_root)
    else:
        era_key = era_key_from_tree(args.source_root or args.repo_root)

    result: dict[str, Any] = {"era_key": era_key, "image_era_version": IMAGE_ERA_VERSION}
    if args.manifest:
        image, matched = resolve_image(era_key, load_manifest(args.manifest), fallback_image=args.fallback_image)
        result["image"] = image
        result["matched"] = matched
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
