# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute stable cache and run-scoped publication identifiers for CI wheelhouses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_DIGEST_PATTERN = re.compile(r"@(?P<digest>sha256:[0-9a-f]{64})(?:$|[?#])", re.IGNORECASE)
_CACHE_SCHEMA = 1


def _workflow_message(level: str, message: str) -> None:
    """Emit an escaped single-line GitHub Actions workflow command."""
    message = message.replace("\r", " ").replace("\n", " ").replace("%", "%25")
    print(f"::{level}::{message}", file=sys.stderr)


def _warning(message: str) -> None:
    """Emit a GitHub Actions warning."""
    _workflow_message("warning", message)


def _normalize_architecture(value: str) -> str:
    """Normalize runner and wheel architecture aliases."""
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "arm64v8": "aarch64",
        "aarch64": "aarch64",
    }
    return aliases.get(value, value or "unknown")


def _docker_architecture(value: str) -> str:
    """Translate a wheel architecture to an OCI platform architecture."""
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(value, value)


def _image_reference(base_image: str, base_version: str) -> str:
    """Compose an image tag when the image and version are supplied separately."""
    if not base_image or not base_version or "@" in base_image:
        return base_image
    final_component = base_image.rsplit("/", 1)[-1]
    if ":" in final_component:
        return base_image
    return f"{base_image}:{base_version}"


def _resolve_base_image_digest(base_image: str, architecture: str) -> str:
    """Resolve a pinned or registry-backed image to its platform manifest digest."""
    if not base_image:
        return ""

    match = _DIGEST_PATTERN.search(base_image)
    if match:
        return match.group("digest").lower()

    command = [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        base_image,
        "--format",
        "{{json .Manifest}}",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        _warning(f"Docker is unavailable; cache identity falls back to base image reference {base_image!r}")
        return ""
    except subprocess.TimeoutExpired:
        _warning(
            f"Timed out resolving digest for base image {base_image!r}; "
            "cache identity falls back to the image reference and base version"
        )
        return ""

    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        _warning(
            f"Could not resolve digest for base image {base_image!r} ({detail}); "
            "cache identity falls back to the image reference and base version"
        )
        return ""

    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        _warning(
            f"Could not parse the registry manifest for {base_image!r} ({exc}); "
            "cache identity falls back to the image reference and base version"
        )
        return ""

    expected_architecture = _docker_architecture(architecture)
    children = manifest.get("manifests")
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue
            platform = child.get("platform")
            if not isinstance(platform, dict):
                continue
            if platform.get("os") == "linux" and platform.get("architecture") == expected_architecture:
                digest = child.get("digest")
                if isinstance(digest, str) and digest.startswith("sha256:"):
                    return digest.lower()

    digest = manifest.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        return digest.lower()

    _warning(
        f"The registry manifest for {base_image!r} had no usable digest; "
        "cache identity falls back to the image reference and base version"
    )
    return ""


def _slug(value: str, fallback: str) -> str:
    """Make a stable cache/artifact-safe component without silent collisions."""
    original = value.strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", original).strip(".-_").lower()
    normalized = normalized or fallback
    if normalized != original.lower() or len(normalized) > 48:
        suffix = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        normalized = f"{normalized[:39].rstrip('.-_')}-{suffix}"
    return normalized


def _write_outputs(values: dict[str, str]) -> None:
    """Append action outputs, or print them when run outside GitHub Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            for name, value in values.items():
                output.write(f"{name}={value}\n")
    else:
        for name, value in values.items():
            print(f"{name}={value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--python-tag", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--base-image", default="")
    parser.add_argument("--base-version", default="")
    return parser.parse_args()


def main() -> int:
    """Compute and export wheelhouse cache and artifact identifiers."""
    args = _parse_args()
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
    lock_path = workspace / "uv.lock"
    profiles_path = workspace / ".github" / "ci-wheelhouse" / "profiles.toml"
    sys.path.insert(0, str(workspace))

    try:
        from tools.ci_wheelhouse.builder import load_profile

        lock_bytes = lock_path.read_bytes()
        profiles_bytes = profiles_path.read_bytes()
        profile = load_profile(
            profiles_path,
            args.profile,
            python_tag=args.python_tag or None,
            architecture=args.architecture or None,
            base_image=args.base_image or None,
            base_version=args.base_version or None,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        _workflow_message("error", str(exc))
        return 1

    profile_metadata = profile.to_manifest()
    python_tag = profile.python_tag
    architecture = _normalize_architecture(profile.architecture or os.environ.get("RUNNER_ARCH", ""))
    base_image = profile.base_image or ""
    base_version = profile.base_version or ""
    resolved_image = _image_reference(base_image, base_version)
    base_image_digest = _resolve_base_image_digest(resolved_image, architecture)

    identity = {
        "schema": _CACHE_SCHEMA,
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "profiles_sha256": hashlib.sha256(profiles_bytes).hexdigest(),
        "profile_name": args.profile,
        "profile_metadata": profile_metadata,
        "python_tag": python_tag,
        "architecture": architecture,
        "base_image": base_image,
        "base_version": base_version,
        "resolved_base_image": resolved_image,
        "base_image_digest": base_image_digest,
        "base_image_identity": base_image_digest or resolved_image,
    }
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    identity_hash = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()

    profile_slug = _slug(args.profile, "profile")
    python_slug = _slug(python_tag, "profile-python")
    architecture_slug = _slug(architecture, "unknown-arch")
    runner_os_slug = _slug(os.environ.get("RUNNER_OS", sys.platform), "unknown-os")
    collection_prefix = (
        f"ci-wheelhouse-v{_CACHE_SCHEMA}-{runner_os_slug}-{profile_slug}-{python_slug}-{architecture_slug}-"
    )
    collection = f"{collection_prefix}{identity_hash[:24]}"

    run_id = _slug(os.environ.get("GITHUB_RUN_ID", "local"), "local")
    run_attempt = _slug(os.environ.get("GITHUB_RUN_ATTEMPT", "1"), "1")
    artifact_name = (
        f"ci-wheelhouse-{profile_slug}-{python_slug}-{architecture_slug}-{identity_hash[:12]}-{run_id}-{run_attempt}"
    )
    runner_temp = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    cache_dir = runner_temp / "isaaclab-ci-wheelhouse-cache" / f"{profile_slug}-{python_slug}-{architecture_slug}"

    _write_outputs(
        {
            "collection": collection,
            "key": collection,
            "restore-keys": collection_prefix,
            "host-dir": str(cache_dir),
            "artifact-name": artifact_name,
        }
    )

    if base_image:
        resolution = base_image_digest or f"unresolved reference {resolved_image}"
        print(f"CI wheelhouse base image identity: {resolution}")
    print(f"CI wheelhouse collection: {collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
