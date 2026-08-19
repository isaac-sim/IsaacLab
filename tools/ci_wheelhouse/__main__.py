# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build, inspect, and verify Isaac Lab CI wheelhouses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .builder import (
    DEFAULT_LOCK_PATH,
    DEFAULT_PROFILES_PATH,
    WheelhouseProfile,
    build_wheelhouse,
    inventory_wheelhouse,
    load_profile,
    verify_wheelhouse,
)


def main(argv: list[str] | None = None) -> int:
    """Run the CI wheelhouse command-line interface.

    Args:
        argv: Optional arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    build_parser = subparsers.add_parser("build", help="Fill a profile wheelhouse and write its manifest")
    _add_profile_arguments(build_parser)
    build_parser.add_argument(
        "--output_dir",
        "--output-dir",
        "--output",
        type=Path,
        required=True,
        help="Output root containing wheelhouse/, manifest.json, and complete",
    )
    build_parser.add_argument(
        "--target_python",
        "--target-python",
        default=sys.executable,
        help="Python executable used for target-aware pip download",
    )
    ci_root_group = build_parser.add_mutually_exclusive_group()
    ci_root_group.add_argument(
        "--include_ci_roots",
        "--include-ci-roots",
        action="store_true",
        dest="include_ci_roots",
        help="Download profile CI-only roots (the default)",
    )
    ci_root_group.add_argument(
        "--skip_ci_roots",
        "--skip-ci-roots",
        action="store_false",
        dest="include_ci_roots",
        help="Mirror only wheels already listed in uv.lock",
    )
    build_parser.set_defaults(include_ci_roots=True)
    build_parser.add_argument("--attempts", type=int, default=4, help="Maximum attempts per download")
    build_parser.add_argument(
        "--backoff_seconds",
        "--backoff-seconds",
        type=float,
        default=1.0,
        help="Initial retry delay",
    )
    build_parser.add_argument(
        "--max_backoff_seconds",
        "--max-backoff-seconds",
        type=float,
        default=8.0,
        help="Maximum retry delay",
    )
    build_parser.add_argument(
        "--timeout_seconds",
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request URL timeout",
    )
    build_parser.add_argument("--workers", type=int, default=4, help="Concurrent exact-URL downloads")

    profile_parser = subparsers.add_parser("profile", help="Print resolved profile metadata")
    _add_profile_arguments(profile_parser)

    inventory_parser = subparsers.add_parser("inventory", help="Inventory wheel METADATA and hashes")
    inventory_parser.add_argument("wheelhouse", type=Path, help="Directory containing wheel files")
    inventory_parser.add_argument("--output", type=Path, help="Optional JSON output file")

    verify_parser = subparsers.add_parser("verify", help="Verify a built wheelhouse and manifest")
    verify_parser.add_argument(
        "--output_dir",
        "--output-dir",
        "--output",
        type=Path,
        required=True,
        help="Output root containing wheelhouse/ and manifest.json",
    )
    verify_parser.add_argument(
        "--lock_file",
        "--lock-file",
        "--lock",
        type=Path,
        help="Optional current uv.lock for input-hash verification",
    )
    verify_parser.add_argument(
        "--profiles_file",
        "--profiles-file",
        "--profiles",
        type=Path,
        help="Optional current profiles.toml for input-hash verification",
    )
    verify_parser.add_argument(
        "--allow_partial",
        "--allow-partial",
        action="store_true",
        help="Validate a partial manifest without requiring completeness",
    )

    args = parser.parse_args(argv)
    try:
        if args.operation == "build":
            profile = _profile_from_arguments(args)
            manifest = build_wheelhouse(
                args.lock_file,
                args.profiles_file,
                profile,
                args.output_dir,
                include_ci_roots=args.include_ci_roots,
                target_python=args.target_python,
                attempts=args.attempts,
                backoff_seconds=args.backoff_seconds,
                max_backoff_seconds=args.max_backoff_seconds,
                timeout_seconds=args.timeout_seconds,
                workers=args.workers,
            )
            summary = {
                "complete": manifest["complete"],
                "excluded_packages": len(manifest["exclusions"]),
                "files": len(manifest["files"]),
                "output_dir": str(args.output_dir),
                "profile": profile.name,
            }
            print(json.dumps(summary, sort_keys=True))
            return 0 if manifest["complete"] else 1
        if args.operation == "profile":
            profile = _profile_from_arguments(args)
            print(json.dumps(profile.to_manifest(), indent=2, sort_keys=True))
            return 0
        if args.operation == "inventory":
            payload = json.dumps(inventory_wheelhouse(args.wheelhouse), indent=2, sort_keys=True) + "\n"
            if args.output is None:
                print(payload, end="")
            else:
                args.output.write_text(payload, encoding="utf-8")
            return 0
        if args.operation == "verify":
            errors = verify_wheelhouse(
                args.output_dir,
                lock_path=args.lock_file,
                profiles_path=args.profiles_file,
                require_complete=not args.allow_partial,
            )
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            print(f"Verified CI wheelhouse at {args.output_dir}")
            return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ci-wheelhouse: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled operation: {args.operation}")


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="Profile name from profiles.toml")
    parser.add_argument(
        "--profiles_file",
        "--profiles-file",
        "--profiles",
        type=Path,
        default=DEFAULT_PROFILES_PATH,
        help="Profile definition TOML",
    )
    parser.add_argument(
        "--lock_file",
        "--lock-file",
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Input uv.lock",
    )
    parser.add_argument("--python_tag", "--python-tag", help="Optional profile Python-tag override")
    parser.add_argument("--architecture", help="Optional profile architecture override")
    parser.add_argument("--base_image", "--base-image", help="Optional profile base-image override")
    parser.add_argument("--base_version", "--base-version", help="Compatibility profile version value")


def _profile_from_arguments(args: argparse.Namespace) -> WheelhouseProfile:
    return load_profile(
        args.profiles_file,
        args.profile,
        python_tag=args.python_tag,
        architecture=args.architecture,
        base_image=args.base_image,
        base_version=args.base_version,
    )


if __name__ == "__main__":
    sys.exit(main())
