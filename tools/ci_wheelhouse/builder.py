# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build and verify profile-specific CI wheelhouses from ``uv.lock``."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import email.policy
import hashlib
import http.client
import json
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import tomllib

SCHEMA_VERSION = 1
DEFAULT_PROFILES_PATH = Path(".github/ci-wheelhouse/profiles.toml")
DEFAULT_LOCK_PATH = Path("uv.lock")
MANIFEST_NAME = "manifest.json"
COMPLETE_SENTINEL_NAME = "complete"
WHEELHOUSE_DIRECTORY_NAME = "wheelhouse"

_ARCHITECTURE_ALIASES = {
    "amd64": "x86_64",
    "arm64": "aarch64",
    "x64": "x86_64",
}
_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXACT_VERSION_PATTERN = re.compile(r"(?<![!<>=~])==\s*([^,;\s]+)")
_CPYTHON_TAG_PATTERN = re.compile(r"^cp([0-9])([0-9]+)$")


@dataclasses.dataclass(frozen=True)
class WheelhouseProfile:
    """Resolved wheelhouse profile.

    Attributes:
        name: Profile name from ``profiles.toml``.
        python_tag: Target CPython interpreter tag, such as ``cp312``.
        python_version: Target Python major/minor version, such as ``3.12``.
        architecture: Canonical target architecture.
        implementation: Pip target implementation abbreviation.
        platforms: Pip target platform tags, in preference order.
        abis: Pip target ABI tags, in preference order.
        ci_roots: CI-only requirement roots not necessarily present in ``uv.lock``.
        ci_roots_no_deps: CI-only roots downloaded without dependency resolution.
        lock_roots: Requirement names that must be served only from ``uv.lock``.
        exclude_package_prefixes: Normalized lock package-name prefixes to exclude.
        mirror_lock: Whether compatible registry wheels from ``uv.lock`` are mirrored.
        index_url: Optional primary index for CI-only roots.
        extra_index_urls: Additional indexes for CI-only roots.
        base_image: Optional consumer base image identity.
        base_version: Optional consumer base version used by compatibility profiles.
        template: Whether the profile requires runtime template values.
    """

    name: str
    python_tag: str
    python_version: str
    architecture: str
    implementation: str
    platforms: tuple[str, ...]
    abis: tuple[str, ...]
    ci_roots: tuple[str, ...]
    lock_roots: tuple[str, ...]
    ci_roots_no_deps: tuple[str, ...] = ()
    exclude_package_prefixes: tuple[str, ...] = ()
    mirror_lock: bool = True
    index_url: str | None = None
    extra_index_urls: tuple[str, ...] = ()
    base_image: str | None = None
    base_version: str | None = None
    template: bool = False

    def to_manifest(self) -> dict[str, Any]:
        """Return the deterministic profile representation stored in a manifest."""
        return {
            "abis": list(self.abis),
            "architecture": self.architecture,
            "base_image": self.base_image,
            "base_version": self.base_version,
            "ci_roots": list(self.ci_roots),
            "ci_roots_no_deps": list(self.ci_roots_no_deps),
            "exclude_package_prefixes": list(self.exclude_package_prefixes),
            "extra_index_urls": list(self.extra_index_urls),
            "implementation": self.implementation,
            "index_url": self.index_url,
            "lock_roots": list(self.lock_roots),
            "mirror_lock": self.mirror_lock,
            "name": self.name,
            "platforms": list(self.platforms),
            "python_tag": self.python_tag,
            "python_version": self.python_version,
            "template": self.template,
        }


@dataclasses.dataclass(frozen=True)
class LockedWheel:
    """One unique compatible wheel selected from the lock file."""

    package_name: str
    package_version: str
    filename: str
    sha256: str
    urls: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class LockSelection:
    """Compatible lock wheels and intentionally unsupported lock entries."""

    wheels: tuple[LockedWheel, ...]
    exclusions: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, str], ...]
    package_versions: dict[str, tuple[str, ...]]
    selected_versions: dict[str, tuple[str, ...]]


def load_profile(
    profiles_path: Path,
    profile_name: str,
    *,
    python_tag: str | None = None,
    architecture: str | None = None,
    base_image: str | None = None,
    base_version: str | None = None,
) -> WheelhouseProfile:
    """Load and resolve one profile from a TOML definition file.

    Args:
        profiles_path: Profile definition file.
        profile_name: Name under ``[profiles]``.
        python_tag: Optional target Python tag override.
        architecture: Optional target architecture override.
        base_image: Optional consumer base image override.
        base_version: Optional compatibility-version template value.

    Returns:
        Fully resolved profile.

    Raises:
        KeyError: If the profile or a referenced root group does not exist.
        TypeError: If the profile has an invalid field type.
        ValueError: If the profile schema or target values are invalid.
    """
    with profiles_path.open("rb") as profile_file:
        data = tomllib.load(profile_file)

    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported profiles schema {data.get('schema')!r}; expected {SCHEMA_VERSION}")

    definitions = data.get("profiles")
    if not isinstance(definitions, dict):
        raise TypeError("[profiles] must be a TOML table")
    resolved = _resolve_profile_definition(definitions, profile_name, ())

    resolved_python_tag = python_tag or _require_string(resolved, "python_tag")
    resolved_architecture = _normalize_architecture(architecture or _require_string(resolved, "architecture"))
    resolved_python_version = _python_version_from_tag(resolved_python_tag)
    configured_python_version = resolved.get("python_version")
    if python_tag is None and configured_python_version is not None:
        if not isinstance(configured_python_version, str):
            raise TypeError(f"Profile '{profile_name}' field 'python_version' must be a string")
        if configured_python_version != resolved_python_version:
            raise ValueError(
                f"Profile '{profile_name}' python_tag {resolved_python_tag!r} does not match "
                f"python_version {configured_python_version!r}"
            )

    resolved_base_image = base_image if base_image is not None else _optional_string(resolved, "base_image")
    resolved_base_version = base_version if base_version is not None else _optional_string(resolved, "base_version")
    is_template = bool(resolved.get("template", False))
    if is_template and not resolved_base_version:
        raise ValueError(f"Profile '{profile_name}' requires --base_version")

    context = {
        "architecture": resolved_architecture,
        "base_image": resolved_base_image or "",
        "base_version": resolved_base_version or "",
        "python_tag": resolved_python_tag,
        "python_version": resolved_python_version,
    }
    root_groups = data.get("root_groups", {})
    if not isinstance(root_groups, dict):
        raise TypeError("[root_groups] must be a TOML table")
    group_names = _string_list(resolved.get("ci_root_groups", []), "ci_root_groups", profile_name)
    ci_roots = _string_list(resolved.get("ci_roots", []), "ci_roots", profile_name)
    for group_name in group_names:
        if group_name not in root_groups:
            raise KeyError(f"Profile '{profile_name}' references unknown root group '{group_name}'")
        ci_roots.extend(_string_list(root_groups[group_name], f"root_groups.{group_name}", profile_name))

    formatted_roots = tuple(_deduplicate(_format_template(root, context) for root in ci_roots))
    formatted_roots_no_deps = tuple(
        _deduplicate(
            _format_template(root, context)
            for root in _string_list(resolved.get("ci_roots_no_deps", []), "ci_roots_no_deps", profile_name)
        )
    )
    exclude_package_prefixes = tuple(
        _deduplicate(
            _normalize_name(_format_template(prefix, context))
            for prefix in _string_list(
                resolved.get("exclude_package_prefixes", []),
                "exclude_package_prefixes",
                profile_name,
            )
        )
    )
    if any(not prefix for prefix in exclude_package_prefixes):
        raise ValueError(f"Profile '{profile_name}' field 'exclude_package_prefixes' cannot contain empty prefixes")
    platforms = tuple(
        _format_template(value, context)
        for value in _string_list(resolved.get("platforms", []), "platforms", profile_name)
    )
    abis = tuple(
        _format_template(value, context) for value in _string_list(resolved.get("abis", []), "abis", profile_name)
    )
    if not platforms:
        raise ValueError(f"Profile '{profile_name}' must define at least one target platform")
    if not abis:
        raise ValueError(f"Profile '{profile_name}' must define at least one target ABI")

    implementation = resolved.get("implementation", "cp")
    if not isinstance(implementation, str):
        raise TypeError(f"Profile '{profile_name}' field 'implementation' must be a string")
    mirror_lock = resolved.get("mirror_lock", True)
    if not isinstance(mirror_lock, bool):
        raise TypeError(f"Profile '{profile_name}' field 'mirror_lock' must be a boolean")
    lock_roots = tuple(
        _normalize_name(value) for value in _string_list(resolved.get("lock_roots", []), "lock_roots", profile_name)
    )
    extra_index_urls = tuple(
        _format_template(value, context)
        for value in _string_list(resolved.get("extra_index_urls", []), "extra_index_urls", profile_name)
    )
    index_url = _optional_string(resolved, "index_url")
    if index_url is not None:
        index_url = _format_template(index_url, context)

    return WheelhouseProfile(
        name=profile_name,
        python_tag=resolved_python_tag,
        python_version=resolved_python_version,
        architecture=resolved_architecture,
        implementation=implementation,
        platforms=platforms,
        abis=abis,
        ci_roots=formatted_roots,
        lock_roots=tuple(_deduplicate(lock_roots)),
        ci_roots_no_deps=formatted_roots_no_deps,
        exclude_package_prefixes=exclude_package_prefixes,
        mirror_lock=mirror_lock,
        index_url=index_url,
        extra_index_urls=extra_index_urls,
        base_image=resolved_base_image,
        base_version=resolved_base_version,
        template=is_template,
    )


def wheel_is_compatible(
    filename: str,
    python_tag: str,
    architecture: str,
    platforms: tuple[str, ...] | None = None,
) -> bool:
    """Return whether a wheel filename is compatible with a Linux CPython profile.

    Pure ``any`` wheels, matching CPython wheels, and older ``abi3`` CPython
    wheels are accepted. Platform wheels must carry a manylinux tag for the
    requested x86_64 or aarch64 architecture.

    Args:
        filename: Wheel basename.
        python_tag: Target CPython tag, such as ``cp312``.
        architecture: Target architecture.
        platforms: Supported target platform tags. Defaults to the Linux floors
            supported by the checked-in profiles.

    Returns:
        True when at least one compressed wheel tag is compatible.
    """
    parsed = _parse_wheel_tags(filename)
    if parsed is None:
        return False
    python_tags, abi_tags, platform_tags, _ = parsed
    target_architecture = _normalize_architecture(architecture)
    highest_floor = _highest_manylinux_floor(platforms, target_architecture)
    platform_compatible = any(
        platform_tag == "any"
        or ((floor := _manylinux_floor(platform_tag, target_architecture)) is not None and floor <= highest_floor)
        for platform_tag in platform_tags
    )
    if not platform_compatible:
        return False
    return any(
        _python_abi_is_compatible(candidate_python, candidate_abi, python_tag)
        for candidate_python in python_tags
        for candidate_abi in abi_tags
    )


def select_locked_wheels(lock_path: Path, profile: WheelhouseProfile) -> LockSelection:
    """Select every compatible registry wheel from a uv lock file.

    Args:
        lock_path: ``uv.lock``-format TOML file.
        profile: Target profile.

    Returns:
        Selected wheels, source exclusions, and lock-shape errors.
    """
    with lock_path.open("rb") as lock_file:
        lock = tomllib.load(lock_file)

    requires_python = lock.get("requires-python")
    if requires_python is not None and not isinstance(requires_python, str):
        raise TypeError("uv.lock requires-python must be a string")
    if (
        profile.mirror_lock
        and requires_python is not None
        and not _python_satisfies_specifier(profile.python_version, requires_python)
    ):
        raise ValueError(
            f"uv.lock requires-python {requires_python!r} excludes profile Python {profile.python_version}; "
            "set mirror_lock=false for this profile"
        )

    packages = lock.get("package")
    if not isinstance(packages, list):
        raise TypeError("uv.lock must contain a [[package]] array")

    selected_by_filename: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    package_versions: dict[str, set[str]] = defaultdict(set)
    selected_versions: dict[str, set[str]] = defaultdict(set)

    for package in packages:
        if not isinstance(package, dict):
            raise TypeError("Each uv.lock package entry must be a table")
        name = _require_string(package, "name")
        version = _require_string(package, "version")
        normalized_name = _normalize_name(name)
        if not _package_resolution_matches(package, profile):
            continue
        source = package.get("source", {})
        if not isinstance(source, dict):
            raise TypeError(f"Package '{name}' source must be a table")

        if "registry" not in source:
            source_kind = next(iter(source), "unknown")
            reason = {
                "editable": "editable-source",
                "git": "git-source",
                "virtual": "virtual-source",
            }.get(source_kind, "non-registry-source")
            exclusions.append(
                {
                    "name": normalized_name,
                    "reason": reason,
                    "source": source,
                    "version": version,
                }
            )
            continue

        package_versions[normalized_name].add(version)
        if any(normalized_name.startswith(prefix) for prefix in profile.exclude_package_prefixes):
            exclusions.append(
                {
                    "name": normalized_name,
                    "reason": "profile-excluded",
                    "source": source,
                    "version": version,
                }
            )
            continue

        wheels = package.get("wheels", [])
        if not isinstance(wheels, list):
            raise TypeError(f"Package '{name}' wheels must be an array")
        compatible_wheels: dict[str, dict[str, Any]] = {}
        candidate_filenames: list[str] = []
        for wheel in wheels:
            if not isinstance(wheel, dict):
                raise TypeError(f"Package '{name}' wheel entry must be a table")
            url = _require_string(wheel, "url")
            filename = _wheel_filename_from_url(url)
            candidate_filenames.append(filename)
            if not wheel_is_compatible(
                filename,
                profile.python_tag,
                profile.architecture,
                profile.platforms,
            ):
                continue
            digest = _sha256_from_lock_hash(_require_string(wheel, "hash"), name, filename)
            existing_candidate = compatible_wheels.get(filename)
            if existing_candidate is None:
                compatible_wheels[filename] = {
                    "filename": filename,
                    "sha256": digest,
                    "urls": {url},
                }
            elif existing_candidate["sha256"] != digest:
                errors.append(
                    {
                        "error": "wheel filename has different sha256 values within one package",
                        "filename": filename,
                        "package": normalized_name,
                    }
                )
            else:
                existing_candidate["urls"].add(url)

        if not compatible_wheels:
            reason = "sdist-only" if not wheels and "sdist" in package else "incompatible-wheel-tags"
            if not wheels and "sdist" not in package:
                reason = "no-wheel-artifact"
            exclusion: dict[str, Any] = {
                "name": normalized_name,
                "reason": reason,
                "source": source,
                "version": version,
            }
            if candidate_filenames:
                exclusion["wheel_filenames"] = sorted(candidate_filenames)
            exclusions.append(exclusion)
            continue

        selected_versions[normalized_name].add(version)
        preferred = max(
            compatible_wheels.values(),
            key=lambda item: (*_wheel_preference(item["filename"], profile), item["filename"]),
        )
        filename = preferred["filename"]
        existing = selected_by_filename.get(filename)
        if existing is None:
            selected_by_filename[filename] = {
                "package_name": normalized_name,
                "package_version": version,
                "sha256": preferred["sha256"],
                "urls": set(preferred["urls"]),
            }
            continue
        if existing["sha256"] != preferred["sha256"]:
            errors.append(
                {
                    "error": "filename collision has different sha256 values",
                    "filename": filename,
                    "package": normalized_name,
                }
            )
            continue
        if existing["package_name"] != normalized_name or existing["package_version"] != version:
            errors.append(
                {
                    "error": "filename collision belongs to different packages",
                    "filename": filename,
                    "package": normalized_name,
                }
            )
            continue
        existing["urls"].update(preferred["urls"])

    selected = tuple(
        LockedWheel(
            package_name=entry["package_name"],
            package_version=entry["package_version"],
            filename=filename,
            sha256=entry["sha256"],
            urls=tuple(sorted(entry["urls"])),
        )
        for filename, entry in sorted(selected_by_filename.items())
    )
    return LockSelection(
        wheels=selected,
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda item: (
                    item["name"],
                    item["version"],
                    item["reason"],
                    json.dumps(item["source"], sort_keys=True),
                ),
            )
        ),
        errors=tuple(sorted(errors, key=lambda item: (item.get("filename", ""), item["error"]))),
        package_versions={name: tuple(sorted(versions)) for name, versions in sorted(package_versions.items())},
        selected_versions={name: tuple(sorted(versions)) for name, versions in sorted(selected_versions.items())},
    )


def inventory_wheel(wheel_path: Path) -> dict[str, Any]:
    """Read distribution metadata and a digest from a wheel.

    Args:
        wheel_path: Wheel file to inspect.

    Returns:
        Deterministic file and core METADATA fields.

    Raises:
        ValueError: If the wheel does not contain exactly one valid METADATA file.
        zipfile.BadZipFile: If the wheel is not a valid ZIP archive.
    """
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_names = []
        for name in wheel.namelist():
            parts = name.split("/")
            if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == "METADATA":
                metadata_names.append(name)
        metadata_names.sort()
        if len(metadata_names) != 1:
            raise ValueError(f"{wheel_path.name} contains {len(metadata_names)} top-level .dist-info/METADATA files")
        message = BytesParser(policy=email.policy.compat32).parsebytes(wheel.read(metadata_names[0]))

    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError(f"{wheel_path.name} METADATA must contain Name and Version")
    return {
        "filename": wheel_path.name,
        "name": str(name),
        "requires_python": str(message["Requires-Python"]) if message["Requires-Python"] else None,
        "sha256": _sha256_file(wheel_path),
        "size": wheel_path.stat().st_size,
        "version": str(version),
    }


def inventory_wheelhouse(wheelhouse_path: Path) -> list[dict[str, Any]]:
    """Inventory every wheel in a directory.

    Args:
        wheelhouse_path: Directory containing wheel files.

    Returns:
        File inventories sorted by filename.
    """
    return [inventory_wheel(path) for path in sorted(wheelhouse_path.glob("*.whl"))]


def build_pip_download_command(
    profile: WheelhouseProfile,
    target_python: str,
    destination: Path,
    requirements: list[str] | tuple[str, ...],
    *,
    constraint_path: Path | None = None,
    no_deps: bool = False,
) -> list[str]:
    """Construct the target-aware pip command for CI-only roots.

    Args:
        profile: Target profile.
        target_python: Python executable that owns pip.
        destination: Wheel download directory.
        requirements: CI-only requirement roots.
        constraint_path: Optional lock-derived pip constraints file.
        no_deps: Whether to skip dependency resolution.

    Returns:
        Argument vector suitable for ``subprocess.run``.
    """
    command = [
        target_python,
        "-m",
        "pip",
        "download",
        "--dest",
        str(destination),
        "--only-binary=:all:",
        "--disable-pip-version-check",
        "--retries",
        "0",
        "--implementation",
        profile.implementation,
        "--python-version",
        profile.python_version,
    ]
    if no_deps:
        command.append("--no-deps")
    if constraint_path is not None:
        command.extend(["--constraint", str(constraint_path)])
    for platform in profile.platforms:
        command.extend(["--platform", platform])
    for abi in profile.abis:
        command.extend(["--abi", abi])
    if profile.index_url:
        command.extend(["--index-url", profile.index_url])
    for extra_index_url in profile.extra_index_urls:
        command.extend(["--extra-index-url", extra_index_url])
    command.extend(requirements)
    return command


def build_wheelhouse(
    lock_path: Path,
    profiles_path: Path,
    profile: WheelhouseProfile,
    output_dir: Path,
    *,
    include_ci_roots: bool = True,
    target_python: str | None = None,
    attempts: int = 4,
    backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 8.0,
    timeout_seconds: float = 60.0,
    workers: int = 4,
) -> dict[str, Any]:
    """Build a deterministic manifest and fill a profile wheelhouse.

    Profiles that mirror the lock fetch compatible wheels only from their exact
    lock URLs and accept them only after SHA-256 verification. Unsupported
    package source kinds and wheel tags become manifest exclusions. Profiles
    that do not mirror the lock download every CI root and its binary closure
    with target-aware pip. Their ``ci_roots_no_deps`` use a separate pip
    invocation that skips dependency resolution.

    Args:
        lock_path: Input uv lock file.
        profiles_path: Input profile definitions, used for manifest hashing.
        profile: Resolved target profile.
        output_dir: Root containing ``wheelhouse/``, ``manifest.json``, and ``complete``.
        include_ci_roots: Whether to invoke pip for roots absent from the lock.
        target_python: Python executable used for pip; required when unresolved roots exist.
        attempts: Maximum attempts per URL or pip invocation.
        backoff_seconds: Initial retry delay.
        max_backoff_seconds: Maximum retry delay.
        timeout_seconds: Per-request URL timeout.
        workers: Maximum concurrent exact-URL downloads.

    Returns:
        Manifest dictionary. ``manifest["complete"]`` reports build success.
    """
    _validate_retry_options(attempts, backoff_seconds, max_backoff_seconds, timeout_seconds, workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    wheelhouse_path = output_dir / WHEELHOUSE_DIRECTORY_NAME
    wheelhouse_path.mkdir(parents=True, exist_ok=True)
    sentinel_path = output_dir / COMPLETE_SENTINEL_NAME
    sentinel_path.unlink(missing_ok=True)

    if profile.mirror_lock:
        selection = select_locked_wheels(lock_path, profile)
    else:
        selection = LockSelection(
            wheels=(),
            exclusions=(),
            errors=(),
            package_versions={},
            selected_versions={},
        )
    errors: list[dict[str, Any]] = [dict(error, stage="selection") for error in selection.errors]
    download_errors = _download_locked_wheels(
        selection.wheels,
        wheelhouse_path,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
        timeout_seconds=timeout_seconds,
        workers=workers,
    )
    errors.extend(download_errors)
    required_by_filename = {wheel.filename: wheel for wheel in selection.wheels}

    root_resolution = _resolve_ci_roots(profile, selection)
    pip_requirements = root_resolution["pip"]
    constraint_lines = _lock_constraints(lock_path, profile)
    constraint_content = "".join(f"{constraint}\n" for constraint in constraint_lines)
    constraint_record = {
        "count": len(constraint_lines),
        "sha256": hashlib.sha256(constraint_content.encode("utf-8")).hexdigest(),
    }
    pip_succeeded = True
    staged_filenames: set[str] = set()
    root_download_groups = (
        ("ci-roots", pip_requirements, False, "<ci-root-staging>"),
        ("ci-roots-no-deps", root_resolution["pip_no_deps"], True, "<ci-root-no-deps-staging>"),
    )
    with tempfile.TemporaryDirectory(prefix=".constraints.", dir=output_dir) as constraint_directory:
        constraint_path: Path | None = None
        if constraint_lines:
            constraint_path = Path(constraint_directory) / "uv-lock-constraints.txt"
            constraint_path.write_text(constraint_content, encoding="utf-8")
        if include_ci_roots:
            for stage, requirements, no_deps, staging_placeholder in root_download_groups:
                if not requirements:
                    continue
                if target_python is None:
                    pip_succeeded = False
                    errors.append(
                        {
                            "error": "target_python is required for CI roots absent from uv.lock",
                            "requirements": requirements,
                            "stage": stage,
                        }
                    )
                    continue

                with tempfile.TemporaryDirectory(prefix=f".{stage}.", dir=output_dir) as staging_directory:
                    staging_path = Path(staging_directory)
                    command = build_pip_download_command(
                        profile,
                        target_python,
                        staging_path,
                        requirements,
                        constraint_path=constraint_path,
                        no_deps=no_deps,
                    )
                    pip_error = _run_pip_download(
                        command,
                        attempts=attempts,
                        backoff_seconds=backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                    if pip_error is not None:
                        pip_succeeded = False
                        display_command = list(command)
                        display_command[display_command.index("--dest") + 1] = staging_placeholder
                        if constraint_path is not None:
                            display_command[display_command.index("--constraint") + 1] = "<lock-constraints>"
                        display_error = pip_error.replace(str(staging_path), staging_placeholder)
                        if constraint_path is not None:
                            display_error = display_error.replace(str(constraint_path), "<lock-constraints>")
                        errors.append(
                            {
                                "command": shlex.join(display_command),
                                "error": display_error,
                                "requirements": requirements,
                                "stage": stage,
                            }
                        )
                    merged_filenames, merge_errors = _merge_staged_wheels(
                        staging_path,
                        wheelhouse_path,
                        stage=f"{stage}-merge",
                    )
                    staged_filenames.update(merged_filenames)
                    errors.extend(merge_errors)
                    if merge_errors:
                        pip_succeeded = False

    errors.extend(_prune_wheelhouse(wheelhouse_path, set(required_by_filename) | staged_filenames))
    file_records: list[dict[str, Any]] = []
    valid_required_files: set[str] = set()
    for wheel_path in sorted(wheelhouse_path.glob("*.whl")):
        if not wheel_is_compatible(
            wheel_path.name,
            profile.python_tag,
            profile.architecture,
            profile.platforms,
        ):
            errors.append(
                {
                    "error": "wheelhouse contains a wheel incompatible with the profile",
                    "filename": wheel_path.name,
                    "stage": "inventory",
                }
            )
            continue
        try:
            inventory = inventory_wheel(wheel_path)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            errors.append(
                {
                    "error": str(error),
                    "filename": wheel_path.name,
                    "stage": "inventory",
                }
            )
            continue

        locked_wheel = required_by_filename.get(wheel_path.name)
        record = dict(inventory)
        if locked_wheel is None:
            record["origin"] = "ci-root" if include_ci_roots else "existing"
        else:
            record["origin"] = "lock"
            record["urls"] = list(locked_wheel.urls)
            if inventory["sha256"] != locked_wheel.sha256:
                errors.append(
                    {
                        "error": "downloaded wheel sha256 does not match uv.lock",
                        "expected": locked_wheel.sha256,
                        "filename": wheel_path.name,
                        "observed": inventory["sha256"],
                        "stage": "inventory",
                    }
                )
            elif (
                _normalize_name(inventory["name"]) != locked_wheel.package_name
                or inventory["version"] != locked_wheel.package_version
            ):
                errors.append(
                    {
                        "error": "wheel METADATA does not match uv.lock package identity",
                        "expected": f"{locked_wheel.package_name}=={locked_wheel.package_version}",
                        "filename": wheel_path.name,
                        "observed": f"{_normalize_name(inventory['name'])}=={inventory['version']}",
                        "stage": "inventory",
                    }
                )
            else:
                valid_required_files.add(wheel_path.name)
        file_records.append(record)

    missing_required_files = sorted(set(required_by_filename) - {record["filename"] for record in file_records})
    for filename in missing_required_files:
        errors.append(
            {
                "error": "required lock wheel is missing",
                "filename": filename,
                "stage": "inventory",
            }
        )

    errors = sorted(
        errors,
        key=lambda item: (
            str(item.get("stage", "")),
            str(item.get("filename", "")),
            str(item.get("error", "")),
            str(item.get("command", "")),
        ),
    )
    complete = not errors and (not include_ci_roots or pip_succeeded)
    packages = _package_inventory(file_records)
    profile_manifest = profile.to_manifest()
    manifest = {
        "complete": complete,
        "completeness": {
            "failed": len(errors),
            "present_lock_files": len(valid_required_files),
            "required_lock_files": len(selection.wheels),
            "status": "complete" if complete else "partial",
        },
        "errors": errors,
        "exclusions": list(selection.exclusions),
        "files": file_records,
        "inputs": {
            "lock": {"filename": lock_path.name, "sha256": _sha256_file(lock_path)},
            "profile": {"sha256": _sha256_json(profile_manifest)},
            "profiles": {"filename": profiles_path.name, "sha256": _sha256_file(profiles_path)},
        },
        "packages": packages,
        "profile": profile_manifest,
        "roots": {
            "enabled": include_ci_roots,
            "excluded": root_resolution["excluded"],
            "locked": root_resolution["locked"],
            "pip": pip_requirements,
            "pip_no_deps": root_resolution["pip_no_deps"],
            "constraints": constraint_record,
        },
        "schema": SCHEMA_VERSION,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if complete:
        sentinel_path.write_text("complete\n", encoding="utf-8")
    return manifest


def verify_wheelhouse(
    output_dir: Path,
    *,
    lock_path: Path | None = None,
    profiles_path: Path | None = None,
    require_complete: bool = True,
) -> list[str]:
    """Verify a wheelhouse against its manifest.

    Args:
        output_dir: Wheelhouse output root.
        lock_path: Optional current lock file for input-hash verification.
        profiles_path: Optional current profiles file for input-hash verification.
        require_complete: Whether a partial manifest is an error.

    Returns:
        Human-readable validation errors. An empty list means verification passed.
    """
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing {MANIFEST_NAME}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid {MANIFEST_NAME}: {error}"]
    if not isinstance(manifest, dict):
        return [f"invalid {MANIFEST_NAME}: top level must be an object"]

    errors: list[str] = []
    if manifest.get("schema") != SCHEMA_VERSION:
        errors.append(f"unsupported manifest schema {manifest.get('schema')!r}")
    structure_errors, files, required_lock_files, present_lock_files = _verify_manifest_structure(
        manifest,
        output_dir,
        require_complete,
    )
    errors.extend(structure_errors)
    errors.extend(_verify_manifest_files(output_dir, files))
    errors.extend(_verify_root_package_inventory(manifest, files))
    errors.extend(
        _verify_manifest_inputs(
            manifest,
            files,
            required_lock_files,
            present_lock_files,
            lock_path,
            profiles_path,
        )
    )
    return sorted(errors)


def _verify_manifest_structure(
    manifest: dict[str, Any],
    output_dir: Path,
    require_complete: bool,
) -> tuple[list[str], list[Any], int | None, int | None]:
    errors: list[str] = []
    manifest_errors = manifest.get("errors")
    if not isinstance(manifest_errors, list):
        errors.append("manifest errors must be an array")
        manifest_errors = []
    completeness = manifest.get("completeness")
    if not isinstance(completeness, dict):
        errors.append("manifest completeness must be an object")
        completeness = {}
    counts: dict[str, int | None] = {}
    for key in ("failed", "present_lock_files", "required_lock_files"):
        value = completeness.get(key)
        if not _is_nonnegative_int(value):
            errors.append(f"manifest completeness {key} must be a non-negative integer")
            counts[key] = None
        else:
            counts[key] = value
    if counts["failed"] is not None and counts["failed"] != len(manifest_errors):
        errors.append("manifest completeness failed count does not match errors")

    complete_value = manifest.get("complete")
    if not isinstance(complete_value, bool):
        errors.append("manifest complete must be a boolean")
    is_complete = complete_value is True
    expected_status = "complete" if is_complete else "partial"
    if completeness.get("status") != expected_status:
        errors.append(f"manifest completeness status must be {expected_status!r}")
    if is_complete and manifest_errors:
        errors.append("complete manifest must not contain errors")
    present = counts["present_lock_files"]
    required = counts["required_lock_files"]
    if present is not None and required is not None and present > required:
        errors.append("manifest present lock file count exceeds required count")
    if is_complete and present is not None and required is not None and present != required:
        errors.append("complete manifest must represent every required lock file")

    sentinel_path = output_dir / COMPLETE_SENTINEL_NAME
    if require_complete and not is_complete:
        errors.append("manifest is partial")
    if is_complete and not sentinel_path.is_file():
        errors.append(f"complete manifest is missing {COMPLETE_SENTINEL_NAME} sentinel")
    if not is_complete and sentinel_path.exists():
        errors.append(f"partial manifest must not have a {COMPLETE_SENTINEL_NAME} sentinel")

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("manifest files must be an array")
        files = []
    roots = manifest.get("roots")
    requirements_exist = isinstance(roots, dict) and any(
        isinstance(roots.get(key), list) and bool(roots[key]) for key in ("locked", "pip", "pip_no_deps")
    )
    if not files and (requirements_exist or (required is not None and required > 0)):
        errors.append("manifest files must not be empty when requirements exist")
    return errors, files, required, present


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _verify_manifest_files(output_dir: Path, files: list[Any]) -> list[str]:
    errors: list[str] = []
    expected_filenames: set[str] = set()
    wheelhouse_path = output_dir / WHEELHOUSE_DIRECTORY_NAME
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            errors.append("manifest contains an invalid file record")
            continue
        filename = record["filename"]
        if Path(filename).name != filename or filename in expected_filenames:
            errors.append(f"invalid or duplicate manifest filename: {filename}")
            continue
        expected_filenames.add(filename)
        wheel_path = wheelhouse_path / filename
        if not wheel_path.is_file():
            errors.append(f"missing wheel: {filename}")
            continue
        if _sha256_file(wheel_path) != record.get("sha256"):
            errors.append(f"sha256 mismatch: {filename}")
            continue
        if wheel_path.stat().st_size != record.get("size"):
            errors.append(f"size mismatch: {filename}")
        try:
            inventory = inventory_wheel(wheel_path)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            errors.append(f"invalid wheel {filename}: {error}")
            continue
        if inventory["name"] != record.get("name") or inventory["version"] != record.get("version"):
            errors.append(f"METADATA mismatch: {filename}")

    actual_filenames = {path.name for path in wheelhouse_path.glob("*.whl")} if wheelhouse_path.is_dir() else set()
    errors.extend(f"unmanifested wheel: {filename}" for filename in sorted(actual_filenames - expected_filenames))
    return errors


def _verify_root_package_inventory(manifest: dict[str, Any], files: list[Any]) -> list[str]:
    if manifest.get("complete") is not True:
        return []
    errors: list[str] = []
    roots = manifest.get("roots")
    if not isinstance(roots, dict):
        return ["complete manifest roots must be an object"]
    if roots.get("enabled") is not True:
        return []
    requirements: list[str] = []
    for key in ("pip", "pip_no_deps"):
        values = roots.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            errors.append(f"complete manifest roots {key} must be an array of strings")
        else:
            requirements.extend(values)

    excluded_names: set[str] = set()
    excluded_roots = roots.get("excluded")
    if isinstance(excluded_roots, list):
        for value in excluded_roots:
            if not isinstance(value, str):
                continue
            try:
                excluded_names.add(_requirement_name(value))
            except ValueError:
                errors.append(f"manifest contains an invalid excluded root: {value!r}")
    manifest_exclusions = manifest.get("exclusions")
    if isinstance(manifest_exclusions, list):
        excluded_names.update(
            item["name"] for item in manifest_exclusions if isinstance(item, dict) and isinstance(item.get("name"), str)
        )

    packages = manifest.get("packages")
    if not isinstance(packages, list):
        return [*errors, "complete manifest packages must be an array"]
    inventory_entries = {
        (_normalize_name(item["name"]), item["version"])
        for item in packages
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("version"), str)
    }
    file_entries = {
        (_normalize_name(item["name"]), item["version"])
        for item in files
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("version"), str)
    }
    available_entries = inventory_entries & file_entries
    for requirement in requirements:
        try:
            name = _requirement_name(requirement)
        except ValueError:
            errors.append(f"manifest contains an invalid root requirement: {requirement!r}")
            continue
        if name in excluded_names:
            continue
        exact_match = _EXACT_VERSION_PATTERN.search(requirement)
        represented = (
            (name, exact_match.group(1)) in available_entries
            if exact_match is not None
            else any(package_name == name for package_name, _ in available_entries)
        )
        if not represented:
            errors.append(f"manifest root requirement is missing package inventory: {requirement}")
    return errors


def _verify_manifest_inputs(
    manifest: dict[str, Any],
    files: list[Any],
    required_lock_files: int | None,
    present_lock_files: int | None,
    lock_path: Path | None,
    profiles_path: Path | None,
) -> list[str]:
    errors: list[str] = []
    inputs = manifest.get("inputs", {})
    if lock_path is not None and _nested_value(inputs, "lock", "sha256") != _sha256_file(lock_path):
        errors.append("lock input hash mismatch")
    if profiles_path is not None and _nested_value(inputs, "profiles", "sha256") != _sha256_file(profiles_path):
        errors.append("profiles input hash mismatch")
    profile = manifest.get("profile")
    if isinstance(profile, dict) and _nested_value(inputs, "profile", "sha256") != _sha256_json(profile):
        errors.append("profile input hash mismatch")
    if lock_path is None or profiles_path is None or not isinstance(profile, dict):
        return errors
    try:
        resolved_profile = load_profile(
            profiles_path,
            str(profile["name"]),
            python_tag=str(profile["python_tag"]),
            architecture=str(profile["architecture"]),
            base_image=profile.get("base_image"),
            base_version=profile.get("base_version"),
        )
        current_selection = (
            select_locked_wheels(lock_path, resolved_profile)
            if resolved_profile.mirror_lock
            else LockSelection((), (), (), {}, {})
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        errors.append(f"could not reconstruct current lock selection: {error}")
        return errors
    errors.extend(
        _verify_current_lock_records(
            files,
            current_selection,
            required_lock_files,
            present_lock_files,
        )
    )
    return errors


def _verify_current_lock_records(
    files: list[Any],
    selection: LockSelection,
    required_lock_files: int | None,
    present_lock_files: int | None,
) -> list[str]:
    errors: list[str] = []
    records_by_filename = {
        record["filename"]: record
        for record in files
        if isinstance(record, dict) and isinstance(record.get("filename"), str)
    }
    represented_lock_files = 0
    for locked_wheel in selection.wheels:
        record = records_by_filename.get(locked_wheel.filename)
        if record is None:
            errors.append(f"manifest is missing required lock wheel: {locked_wheel.filename}")
        elif record.get("sha256") != locked_wheel.sha256:
            errors.append(f"manifest lock sha256 mismatch: {locked_wheel.filename}")
        elif record.get("origin") != "lock":
            errors.append(f"manifest lock origin mismatch: {locked_wheel.filename}")
        else:
            represented_lock_files += 1
    if required_lock_files != len(selection.wheels):
        errors.append("manifest required lock file count does not match current selection")
    if present_lock_files != represented_lock_files:
        errors.append("manifest present lock file count does not match current selection")
    return errors


def _resolve_profile_definition(
    definitions: dict[str, Any], profile_name: str, resolving: tuple[str, ...]
) -> dict[str, Any]:
    if profile_name in resolving:
        chain = " -> ".join((*resolving, profile_name))
        raise ValueError(f"Profile inheritance cycle: {chain}")
    definition = definitions.get(profile_name)
    if not isinstance(definition, dict):
        raise KeyError(f"Unknown wheelhouse profile '{profile_name}'")
    parent_name = definition.get("extends")
    if parent_name is None:
        resolved: dict[str, Any] = {}
    else:
        if not isinstance(parent_name, str):
            raise TypeError(f"Profile '{profile_name}' field 'extends' must be a string")
        resolved = _resolve_profile_definition(definitions, parent_name, (*resolving, profile_name))
    resolved.update({key: value for key, value in definition.items() if key != "extends"})
    return resolved


def _require_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Field '{key}' must be a non-empty string")
    return value


def _optional_string(values: dict[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"Field '{key}' must be a string")
    return value


def _string_list(value: Any, field: str, profile_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Profile '{profile_name}' field '{field}' must be an array of strings")
    return list(value)


def _format_template(value: str, context: dict[str, str]) -> str:
    try:
        return value.format_map(context)
    except KeyError as error:
        raise ValueError(f"Unknown profile template field {error.args[0]!r}") from error


def _deduplicate(values: Any) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_architecture(architecture: str) -> str:
    normalized = _ARCHITECTURE_ALIASES.get(architecture.lower(), architecture.lower())
    if normalized not in {"aarch64", "x86_64"}:
        raise ValueError(f"Unsupported architecture '{architecture}'")
    return normalized


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _package_resolution_matches(package: dict[str, Any], profile: WheelhouseProfile) -> bool:
    markers = package.get("resolution-markers")
    if markers is None:
        return True
    if not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
        raise TypeError(f"Package '{package.get('name', '<unknown>')}' resolution-markers must be strings")
    return any(_resolution_marker_matches(marker, profile) for marker in markers)


def _resolution_marker_matches(marker: str, profile: WheelhouseProfile) -> bool:
    values: dict[str, str] = {}
    for condition in re.split(r"\s+and\s+", marker.strip()):
        match = re.fullmatch(r"""(platform_machine|sys_platform)\s*==\s*(['"])([^'"]+)\2""", condition)
        if match is None:
            raise ValueError(f"Unsupported uv.lock resolution marker {marker!r}")
        values[match.group(1)] = match.group(3)
    if set(values) != {"platform_machine", "sys_platform"}:
        raise ValueError(f"Unsupported uv.lock resolution marker {marker!r}")
    marker_architecture = _ARCHITECTURE_ALIASES.get(
        values["platform_machine"].lower(),
        values["platform_machine"].lower(),
    )
    return values["sys_platform"] == "linux" and marker_architecture == profile.architecture


def _python_satisfies_specifier(version: str, specifier: str) -> bool:
    version_parts = _version_tuple(version)
    for clause in (part.strip() for part in specifier.split(",")):
        match = re.fullmatch(r"(===|==|!=|>=|<=|>|<|~=)\s*([0-9]+(?:\.[0-9]+)*(?:\.\*)?)", clause)
        if match is None:
            raise ValueError(f"Unsupported uv.lock requires-python specifier {specifier!r}")
        operator, expected = match.groups()
        if expected.endswith(".*"):
            expected_parts = _version_tuple(expected[:-2])
            matches = version_parts[: len(expected_parts)] == expected_parts
            if (operator in {"==", "==="} and not matches) or (operator == "!=" and matches):
                return False
            if operator not in {"==", "===", "!="}:
                raise ValueError(f"Unsupported wildcard uv.lock requires-python specifier {specifier!r}")
            continue

        expected_parts = _version_tuple(expected)
        width = max(len(version_parts), len(expected_parts))
        observed = version_parts + (0,) * (width - len(version_parts))
        required = expected_parts + (0,) * (width - len(expected_parts))
        if operator in {"==", "==="} and observed != required:
            return False
        if operator == "!=" and observed == required:
            return False
        if operator == ">=" and observed < required:
            return False
        if operator == "<=" and observed > required:
            return False
        if operator == ">" and observed <= required:
            return False
        if operator == "<" and observed >= required:
            return False
        if operator == "~=":
            upper = (
                (expected_parts[0] + 1,) if len(expected_parts) <= 2 else (*expected_parts[:-2], expected_parts[-2] + 1)
            )
            if observed < required or observed[: len(upper)] >= upper:
                return False
    return True


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _python_version_from_tag(python_tag: str) -> str:
    match = _CPYTHON_TAG_PATTERN.fullmatch(python_tag)
    if match is None:
        raise ValueError(f"Unsupported Python tag '{python_tag}'; expected a CPython tag such as cp312")
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _parse_cpython_tag(python_tag: str) -> tuple[int, int] | None:
    match = _CPYTHON_TAG_PATTERN.fullmatch(python_tag)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_wheel_tags(
    filename: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[int, str]] | None:
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) != 4 or not all(parts[1:]):
        return None
    name_version_build = parts[0].rsplit("-", 2)
    if len(name_version_build) < 2:
        return None
    build_tag = (-1, "")
    if len(name_version_build) == 3:
        match = re.fullmatch(r"(\d+)([0-9A-Za-z_]*)", name_version_build[2])
        if match is None:
            return None
        build_tag = (int(match.group(1)), match.group(2))
    return (
        tuple(parts[1].split(".")),
        tuple(parts[2].split(".")),
        tuple(parts[3].split(".")),
        build_tag,
    )


def _manylinux_floor(platform_tag: str, architecture: str) -> tuple[int, int] | None:
    if platform_tag == f"manylinux2014_{architecture}":
        return (2, 17)
    if platform_tag == f"manylinux2010_{architecture}":
        return (2, 12)
    match = re.fullmatch(rf"manylinux_(\d+)_(\d+)_{re.escape(architecture)}", platform_tag)
    if match is None or int(match.group(1)) != 2:
        return None
    return int(match.group(1)), int(match.group(2))


def _highest_manylinux_floor(
    platforms: tuple[str, ...] | None,
    architecture: str,
) -> tuple[int, int]:
    if platforms is None:
        return (2, 35)
    return max(
        (floor for platform in platforms if (floor := _manylinux_floor(platform, architecture)) is not None),
        default=(0, 0),
    )


def _wheel_preference(
    filename: str,
    profile: WheelhouseProfile,
) -> tuple[int, tuple[int, int], tuple[int, str]]:
    parsed = _parse_wheel_tags(filename)
    if parsed is None:
        return (0, (0, 0), (-1, ""))
    python_tags, abi_tags, platform_tags, build_tag = parsed
    python_rank = 0
    for candidate_python in python_tags:
        for candidate_abi in abi_tags:
            if not _python_abi_is_compatible(candidate_python, candidate_abi, profile.python_tag):
                continue
            if candidate_python == profile.python_tag and candidate_abi != "abi3":
                python_rank = max(python_rank, 3)
            elif candidate_abi == "abi3":
                python_rank = max(python_rank, 2)
            else:
                python_rank = max(python_rank, 1)
    highest_floor = _highest_manylinux_floor(profile.platforms, profile.architecture)
    floor = max(
        (
            candidate_floor
            for platform_tag in platform_tags
            if (candidate_floor := _manylinux_floor(platform_tag, profile.architecture)) is not None
            and candidate_floor <= highest_floor
        ),
        default=(0, 0),
    )
    return python_rank, floor, build_tag


def _python_abi_is_compatible(candidate_python: str, candidate_abi: str, target_python: str) -> bool:
    target = _parse_cpython_tag(target_python)
    if target is None:
        return False
    target_major, target_minor = target
    if candidate_python in {f"py{target_major}", f"py{target_major}{target_minor}"}:
        return candidate_abi == "none"

    candidate = _parse_cpython_tag(candidate_python)
    if candidate is None:
        return False
    candidate_major, candidate_minor = candidate
    if candidate_python == target_python:
        return candidate_abi in {target_python, "abi3", "none"}
    return candidate_abi == "abi3" and candidate_major == target_major and candidate_minor <= target_minor


def _wheel_filename_from_url(url: str) -> str:
    filename = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    if not filename.endswith(".whl"):
        raise ValueError(f"Lock wheel URL does not end in .whl: {url}")
    return filename


def _sha256_from_lock_hash(lock_hash: str, package_name: str, filename: str) -> str:
    algorithm, separator, digest = lock_hash.partition(":")
    if separator != ":" or algorithm != "sha256" or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError(f"Package '{package_name}' wheel '{filename}' must have a sha256 lock hash")
    return digest.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_retry_options(
    attempts: int, backoff_seconds: float, max_backoff_seconds: float, timeout_seconds: float, workers: int
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if backoff_seconds < 0 or max_backoff_seconds < 0:
        raise ValueError("retry delays must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")


def _merge_staged_wheels(
    staging_path: Path,
    wheelhouse_path: Path,
    *,
    stage: str,
) -> tuple[set[str], list[dict[str, str]]]:
    merged_filenames: set[str] = set()
    errors: list[dict[str, str]] = []
    for staged_wheel in sorted(staging_path.glob("*.whl")):
        destination = wheelhouse_path / staged_wheel.name
        try:
            expected_sha256 = _sha256_file(staged_wheel)
            staged_wheel.replace(destination)
            observed_sha256 = _sha256_file(destination)
            if observed_sha256 != expected_sha256:
                destination.unlink(missing_ok=True)
                raise OSError(f"atomic merge sha256 mismatch: expected {expected_sha256}, observed {observed_sha256}")
        except OSError as error:
            errors.append(
                {
                    "error": str(error),
                    "filename": staged_wheel.name,
                    "stage": stage,
                }
            )
        else:
            merged_filenames.add(staged_wheel.name)
    return merged_filenames, errors


def _prune_wheelhouse(wheelhouse_path: Path, required_filenames: set[str]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    candidates = [*wheelhouse_path.glob("*.whl"), *wheelhouse_path.glob("*.part")]
    for artifact_path in sorted(candidates):
        if artifact_path.suffix == ".whl" and artifact_path.name in required_filenames:
            continue
        try:
            artifact_path.unlink()
        except OSError as error:
            errors.append(
                {
                    "error": str(error),
                    "filename": artifact_path.name,
                    "stage": "prune",
                }
            )
    return errors


def _download_locked_wheels(
    wheels: tuple[LockedWheel, ...],
    wheelhouse_path: Path,
    *,
    attempts: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    timeout_seconds: float,
    workers: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_wheel = {
            executor.submit(
                _ensure_locked_wheel,
                wheel,
                wheelhouse_path,
                attempts,
                backoff_seconds,
                max_backoff_seconds,
                timeout_seconds,
            ): wheel
            for wheel in wheels
        }
        for future in concurrent.futures.as_completed(future_to_wheel):
            wheel = future_to_wheel[future]
            try:
                error = future.result()
            except Exception as exception:  # keep a partial manifest for unexpected per-file failures
                error = f"{type(exception).__name__}: {exception}"
            if error is not None:
                failures.append(
                    {
                        "error": error,
                        "filename": wheel.filename,
                        "stage": "download",
                        "urls": list(wheel.urls),
                    }
                )
    return sorted(failures, key=lambda item: item["filename"])


def _ensure_locked_wheel(
    wheel: LockedWheel,
    wheelhouse_path: Path,
    attempts: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    timeout_seconds: float,
) -> str | None:
    destination = wheelhouse_path / wheel.filename
    if destination.is_file() and _sha256_file(destination) == wheel.sha256:
        return None
    destination.unlink(missing_ok=True)

    last_error = ""
    for attempt in range(attempts):
        url = wheel.urls[attempt % len(wheel.urls)]
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "IsaacLab-CI-Wheelhouse/1"})
            with (
                urllib.request.urlopen(request, timeout=timeout_seconds) as response,
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=wheelhouse_path,
                    prefix=f".{wheel.filename}.",
                    suffix=".part",
                    delete=False,
                ) as temporary_file,
            ):
                temporary_path = Path(temporary_file.name)
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
                    digest.update(chunk)
            observed_digest = digest.hexdigest()
            if observed_digest != wheel.sha256:
                raise ValueError(f"sha256 mismatch: expected {wheel.sha256}, observed {observed_digest}")
            temporary_path.replace(destination)
            return None
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            last_error = f"{type(error).__name__}: {error}"
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                _sleep_before_retry(attempt, backoff_seconds, max_backoff_seconds)
    return last_error


def _sleep_before_retry(attempt: int, backoff_seconds: float, max_backoff_seconds: float) -> None:
    delay = min(backoff_seconds * (2**attempt), max_backoff_seconds)
    if delay:
        time.sleep(delay)


def _resolve_ci_roots(profile: WheelhouseProfile, selection: LockSelection) -> dict[str, list[str]]:
    if not profile.mirror_lock:
        return {
            "excluded": [],
            "locked": [],
            "pip": list(profile.ci_roots),
            "pip_no_deps": list(profile.ci_roots_no_deps),
        }

    locked: list[str] = []
    excluded: list[str] = []
    pip: list[str] = []
    exclusion_names = {item["name"] for item in selection.exclusions}
    lock_only_names = set(profile.lock_roots)
    for requirement in profile.ci_roots:
        name = _requirement_name(requirement)
        selected_versions = selection.selected_versions.get(name, ())
        if selected_versions and _exact_requirement_matches(requirement, selected_versions):
            locked.append(requirement)
        elif name in lock_only_names or name in exclusion_names:
            excluded.append(requirement)
        else:
            pip.append(requirement)
    return {"excluded": excluded, "locked": locked, "pip": pip, "pip_no_deps": []}


def _lock_constraints(lock_path: Path, profile: WheelhouseProfile) -> tuple[str, ...]:
    with lock_path.open("rb") as lock_file:
        lock = tomllib.load(lock_file)
    requires_python = lock.get("requires-python")
    if requires_python is not None:
        if not isinstance(requires_python, str):
            raise TypeError("uv.lock requires-python must be a string")
        if not _python_satisfies_specifier(profile.python_version, requires_python):
            return ()

    versions: dict[str, set[str]] = defaultdict(set)
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise TypeError("uv.lock must contain a [[package]] array")
    for package in packages:
        if not isinstance(package, dict):
            raise TypeError("Each uv.lock package entry must be a table")
        if not _package_resolution_matches(package, profile):
            continue
        source = package.get("source", {})
        if not isinstance(source, dict) or "registry" not in source:
            continue
        name = _normalize_name(_require_string(package, "name"))
        if any(name.startswith(prefix) for prefix in profile.exclude_package_prefixes):
            continue
        wheels = package.get("wheels", [])
        if not isinstance(wheels, list):
            raise TypeError(f"Package '{name}' wheels must be an array")
        if not any(
            isinstance(wheel, dict)
            and wheel_is_compatible(
                _wheel_filename_from_url(_require_string(wheel, "url")),
                profile.python_tag,
                profile.architecture,
                profile.platforms,
            )
            for wheel in wheels
        ):
            continue
        versions[name].add(_require_string(package, "version"))
    return tuple(
        f"{name}=={next(iter(package_versions))}"
        for name, package_versions in sorted(versions.items())
        if len(package_versions) == 1
    )


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME_PATTERN.match(requirement)
    if match is None:
        raise ValueError(f"Could not parse requirement name from {requirement!r}")
    return _normalize_name(match.group(1))


def _exact_requirement_matches(requirement: str, selected_versions: tuple[str, ...]) -> bool:
    match = _EXACT_VERSION_PATTERN.search(requirement)
    return match is None or match.group(1) in selected_versions


def _run_pip_download(
    command: list[str],
    *,
    attempts: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
) -> str | None:
    last_error = ""
    for attempt in range(attempts):
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as error:
            last_error = f"{type(error).__name__}: {error}"
        else:
            if result.returncode == 0:
                return None
            output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            last_error = f"pip exited with {result.returncode}: {output[-4000:]}"
        if attempt + 1 < attempts:
            _sleep_before_retry(attempt, backoff_seconds, max_backoff_seconds)
    return last_error


def _package_inventory(file_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packages: dict[tuple[str, str], dict[str, Any]] = {}
    for record in file_records:
        key = (_normalize_name(record["name"]), record["version"])
        package = packages.setdefault(
            key,
            {
                "files": [],
                "name": key[0],
                "version": key[1],
            },
        )
        package["files"].append(record["filename"])
    for package in packages.values():
        package["files"].sort()
    return [packages[key] for key in sorted(packages)]


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
