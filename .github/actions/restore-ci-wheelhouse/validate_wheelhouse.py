# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate a restored CI wheelhouse and export its status."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_WORKSPACE = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
sys.path.insert(0, str(_WORKSPACE))

from tools.ci_wheelhouse.builder import load_profile, verify_wheelhouse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--python-tag", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--base-image", default="")
    parser.add_argument("--base-version", default="")
    parser.add_argument("--soft-fail", action="store_true")
    return parser.parse_args()


def _profile_errors(observed: Any, args: argparse.Namespace) -> list[str]:
    """Compare a manifest profile with the fully resolved requested profile."""
    if not isinstance(observed, dict):
        return ["manifest profile must be an object"]
    if not args.profile:
        # Artifact-only callers historically validate the profile hash embedded
        # in the manifest without selecting a local profile name.
        return []

    try:
        expected = load_profile(
            _WORKSPACE / ".github" / "ci-wheelhouse" / "profiles.toml",
            args.profile,
            python_tag=args.python_tag or None,
            architecture=args.architecture or None,
            base_image=args.base_image or None,
            base_version=args.base_version or None,
        ).to_manifest()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return [f"could not resolve expected profile {args.profile!r}: {exc}"]

    if observed == expected:
        return []
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    observed_json = json.dumps(observed, sort_keys=True, separators=(",", ":"))
    return [f"manifest profile mismatch: expected {expected_json}, observed {observed_json}"]


def _write_outputs(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as output:
        for name, value in values.items():
            output.write(f"{name}={value}\n")


def _workflow_message(level: str, message: str) -> None:
    message = message.replace("\r", " ").replace("\n", " ").replace("%", "%25")
    print(f"::{level}::{message}", file=sys.stderr)


def main() -> int:
    args = _parse_args()
    host_dir = args.host_dir.resolve()
    manifest_path = host_dir / "manifest.json"

    try:
        errors = verify_wheelhouse(
            host_dir,
            lock_path=_WORKSPACE / "uv.lock",
            profiles_path=_WORKSPACE / ".github" / "ci-wheelhouse" / "profiles.toml",
            require_complete=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        errors = [f"validator raised {type(exc).__name__}: {exc}"]
    manifest: Any = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if not any(error.startswith("invalid manifest.json") for error in errors):
                errors.append(f"invalid manifest.json: {exc}")

    if isinstance(manifest, dict):
        errors.extend(_profile_errors(manifest.get("profile"), args))

    errors = sorted(set(errors))
    complete = (
        not errors
        and isinstance(manifest, dict)
        and manifest.get("complete") is True
        and (host_dir / "complete").is_file()
    )
    status = "invalid" if errors else ("complete" if complete else "partial")
    _write_outputs(
        {
            "valid": str(not errors).lower(),
            "complete": str(complete).lower(),
            "manifest": str(manifest_path),
            "status": status,
        }
    )

    if errors:
        level = "warning" if args.soft_fail else "error"
        for error in errors:
            _workflow_message(level, f"CI wheelhouse validation failed: {error}")
        return 0 if args.soft_fail else 1

    print(f"Validated {status} CI wheelhouse at {host_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
