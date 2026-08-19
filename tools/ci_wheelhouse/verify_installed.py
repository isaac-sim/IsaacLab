# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify wheelhouse-requested distributions against a CI manifest."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import shlex
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def verify_installed(
    manifest: dict[str, Any],
    required_specs: list[str],
    optional_specs: list[str],
    *,
    required_fallback_used: bool,
) -> list[str]:
    """Verify installed package versions requested from a wheelhouse.

    Args:
        manifest: Parsed CI wheelhouse manifest.
        required_specs: Requirements that must be represented by the manifest.
        optional_specs: Extra requirements that may use the online fallback.
        required_fallback_used: Whether required packages were installed online.

    Returns:
        Human-readable verification errors.
    """
    expected = _expected_versions(manifest)
    excluded = _excluded_names(manifest)
    legacy_manifest = "ovphysx_version" in manifest and not any(
        key in manifest for key in ("package_versions", "versions", "packages", "distributions", "inventory", "wheels")
    )
    errors: list[str] = []

    for spec, required in [*((spec, True) for spec in required_specs), *((spec, False) for spec in optional_specs)]:
        try:
            name = canonicalize_name(Requirement(spec).name)
        except InvalidRequirement:
            print(f"Skipping manifest verification for non-registry requirement: {spec}")
            continue

        manifest_versions = expected.get(name)
        if manifest_versions is None:
            if name in excluded:
                print(f"Manifest documents {name} as an exclusion; skipping version verification")
            elif required and not required_fallback_used and (not legacy_manifest or name == "ovphysx"):
                errors.append(f"CI wheelhouse manifest has no version for requested distribution {name}")
            elif required:
                print(f"No manifest version for online-fallback distribution {name}; skipping verification")
            else:
                print(f"No manifest version for optional distribution {name}; skipping verification")
            continue

        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Requested distribution {name} is not installed")
            continue

        print(f"Resolved {name} distribution version: {installed_version}")
        print(f"CI wheelhouse manifest {name} versions: {', '.join(sorted(manifest_versions))}")
        if not _version_matches(installed_version, manifest_versions):
            errors.append(
                f"{name} version mismatch: installed {installed_version}, "
                f"manifest {', '.join(sorted(manifest_versions))}"
            )
            continue

        if name == "ovphysx":
            ovphysx = importlib.import_module("ovphysx")
            runtime_version = str(getattr(ovphysx, "__version__", installed_version))
            print(f"Imported ovphysx runtime version: {runtime_version}")
            if not _version_matches(runtime_version, manifest_versions):
                errors.append(
                    f"ovphysx version mismatch: installed {installed_version}, import {runtime_version}, "
                    f"manifest {', '.join(sorted(manifest_versions))}"
                )

    return errors


def _expected_versions(manifest: dict[str, Any]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}

    def record(name: Any, value: Any) -> None:
        if isinstance(value, dict):
            name = value.get("name") or value.get("distribution") or name
            value = value.get("version") or value.get("resolved_version")
        if name and value:
            expected.setdefault(canonicalize_name(str(name)), set()).add(str(value))

    for key in ("package_versions", "versions"):
        values = manifest.get(key, {})
        if isinstance(values, dict):
            for name, value in values.items():
                record(name, value)

    for key in ("packages", "distributions", "inventory", "wheels"):
        values = manifest.get(key, {})
        if isinstance(values, dict):
            for name, value in values.items():
                record(name, value)
        elif isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    record(value.get("name") or value.get("distribution"), value)

    if manifest.get("ovphysx_version"):
        expected.setdefault("ovphysx", set()).add(str(manifest["ovphysx_version"]))
    return expected


def _excluded_names(manifest: dict[str, Any]) -> set[str]:
    exclusions = manifest.get("exclusions", [])
    if isinstance(exclusions, dict):
        exclusions = [{"name": name, "value": value} for name, value in exclusions.items()]

    excluded: set[str] = set()
    if not isinstance(exclusions, list):
        return excluded
    for exclusion in exclusions:
        if isinstance(exclusion, str):
            excluded.add(canonicalize_name(exclusion))
        elif isinstance(exclusion, dict):
            name = exclusion.get("name") or exclusion.get("distribution") or exclusion.get("package")
            if name:
                excluded.add(canonicalize_name(str(name)))
    return excluded


def _version_matches(actual: str, candidates: set[str]) -> bool:
    for candidate in candidates:
        try:
            if Version(actual) == Version(candidate):
                return True
        except InvalidVersion:
            if actual == candidate:
                return True
    return False


def main() -> int:
    """Run verification using the test-container environment contract."""
    manifest_path = Path(os.environ["TEST_WHEELHOUSE_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = verify_installed(
        manifest,
        shlex.split(os.environ.get("TEST_WHEELHOUSE_REQUIRED_PACKAGES", "")),
        shlex.split(os.environ.get("TEST_WHEELHOUSE_OPTIONAL_PACKAGES", "")),
        required_fallback_used=os.environ.get("TEST_WHEELHOUSE_REQUIRED_FALLBACK_USED") == "true",
    )
    if errors:
        raise SystemExit("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
