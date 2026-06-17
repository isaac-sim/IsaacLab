#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check installed dependency licenses against the Isaac Lab CI policy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_LICENSE_SUBSTRINGS = ("MIT", "Apache", "BSD", "ISC", "zlib")
DEFAULT_EXCLUDED_PACKAGE_PREFIXES = ("nvidia",)

_DEBIAN_LICENSE_RE = re.compile(r"^License:[^\S\r\n]*(\S.*)?$", re.MULTILINE)
_FALLBACK_LICENSE_PATTERNS = (
    ("Apache-2.0", re.compile(r"\bApache(?: License)?(?:,? Version)? 2\.0\b|\bApache-2\.0\b", re.IGNORECASE)),
    ("BSD", re.compile(r"\bBSD\b", re.IGNORECASE)),
    ("MIT", re.compile(r"\bMIT\b", re.IGNORECASE)),
    ("ISC", re.compile(r"\bISC\b", re.IGNORECASE)),
    ("zlib", re.compile(r"\bzlib\b", re.IGNORECASE)),
    ("AGPL", re.compile(r"\b(?:AGPL|Affero General Public License)\b", re.IGNORECASE)),
    ("LGPL", re.compile(r"\b(?:LGPL|Lesser General Public License)\b", re.IGNORECASE)),
    ("GPL", re.compile(r"\b(?:GPL|General Public License)\b", re.IGNORECASE)),
    ("MPL", re.compile(r"\b(?:MPL|Mozilla Public License)\b", re.IGNORECASE)),
    ("EPL", re.compile(r"\b(?:EPL|Eclipse Public License)\b", re.IGNORECASE)),
    ("CDDL", re.compile(r"\b(?:CDDL|Common Development and Distribution License)\b", re.IGNORECASE)),
    ("Proprietary", re.compile(r"\b(?:proprietary|commercial)\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class PackageLicense:
    """Installed package license metadata."""

    manager: str
    name: str
    license: str
    version: str | None = None


@dataclass(frozen=True)
class LicenseException:
    """Package-level license exception loaded from JSON."""

    manager: str
    package: str
    license: str | None = None


@dataclass(frozen=True)
class LicenseFailure:
    """A package that failed the license policy."""

    package: PackageLicense
    reason: str


def _run(command: list[str]) -> str:
    """Run a command and return stdout."""
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def _canonical_package_name(package: str) -> str:
    """Return a normalized package name for comparisons."""
    package = package.split(":", 1)[0]
    return package.lower().replace("_", "-")


def _load_exceptions(exceptions_file: Path) -> dict[tuple[str, str], LicenseException]:
    """Load package exceptions keyed by package manager and package name."""
    with exceptions_file.open(encoding="utf-8") as stream:
        entries = json.load(stream)

    exceptions: dict[tuple[str, str], LicenseException] = {}
    for entry in entries:
        package = entry["package"]
        manager = entry.get("source", "pip")
        exception = LicenseException(
            manager=manager,
            package=package,
            license=entry.get("license"),
        )
        exceptions[(manager, _canonical_package_name(package))] = exception
    return exceptions


def _find_exception(
    package: PackageLicense, exceptions: dict[tuple[str, str], LicenseException]
) -> LicenseException | None:
    """Return an exception matching the package manager and package name."""
    package_name = _canonical_package_name(package.name)
    for manager in (package.manager, "all", "*"):
        exception = exceptions.get((manager, package_name))
        if exception is not None:
            return exception
    return None


def _is_excluded_package(package: PackageLicense) -> bool:
    """Return whether package should be excluded from license checks."""
    package_name = package.name.lower()
    return any(package_name.startswith(prefix) for prefix in DEFAULT_EXCLUDED_PACKAGE_PREFIXES)


def _has_allowed_license_substring(license_text: str) -> bool:
    """Return whether the license text contains an allowed license substring."""
    normalized = license_text.lower()
    return any(allowed.lower() in normalized for allowed in ALLOWED_LICENSE_SUBSTRINGS)


def _is_allowed_pip_license(license_text: str) -> bool:
    """Return whether a pip license satisfies the permissive-license policy."""
    return _has_allowed_license_substring(license_text)


def _is_allowed_apt_license(license_text: str) -> bool:
    """Return whether all detected apt license expressions satisfy the policy."""
    expressions = [expression.strip() for expression in license_text.split(";") if expression.strip()]
    if not expressions:
        return False
    return all(_has_allowed_license_substring(expression) for expression in expressions)


def _is_allowed_license(package: PackageLicense) -> bool:
    """Return whether a package license satisfies the policy for its package manager."""
    if package.manager == "apt":
        return _is_allowed_apt_license(package.license)
    if package.manager == "pip":
        return _is_allowed_pip_license(package.license)
    raise ValueError(f"Unsupported package manager: {package.manager}")


def _check_packages(
    packages: list[PackageLicense],
    exceptions: dict[tuple[str, str], LicenseException],
) -> list[LicenseFailure]:
    """Check packages against the license allowlist and exceptions."""
    failures: list[LicenseFailure] = []
    for package in packages:
        if _is_excluded_package(package):
            continue
        if _is_allowed_license(package):
            continue

        exception = _find_exception(package, exceptions)
        if exception is None:
            failures.append(LicenseFailure(package=package, reason="no exception found"))
            continue
        if exception.license is not None and exception.license != package.license:
            failures.append(
                LicenseFailure(
                    package=package,
                    reason=f"exception license mismatch: expected {exception.license}",
                )
            )
    return failures


def _load_pip_license_json(license_json: Path | None) -> list[dict[str, Any]]:
    """Load pip license metadata from a file or pip-licenses."""
    if license_json is not None:
        with license_json.open(encoding="utf-8") as stream:
            return json.load(stream)

    output = _run([sys.executable, "-m", "piplicenses", "--from=mixed", "--format=json"])
    return json.loads(output)


def _collect_pip_packages(license_json: Path | None) -> list[PackageLicense]:
    """Collect installed pip package licenses."""
    records = _load_pip_license_json(license_json)
    packages: list[PackageLicense] = []
    for record in records:
        packages.append(
            PackageLicense(
                manager="pip",
                name=record["Name"],
                version=record.get("Version"),
                license=record.get("License") or "UNKNOWN",
            )
        )
    return packages


def _collect_apt_versions() -> dict[str, str]:
    """Collect installed apt package versions."""
    output = _run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"])
    versions: dict[str, str] = {}
    for line in output.splitlines():
        package, version = line.split("\t", 1)
        versions[_canonical_package_name(package)] = version
    return versions


def _collect_apt_package_names(manual_only: bool) -> list[str]:
    """Collect installed apt package names."""
    if manual_only:
        output = _run(["apt-mark", "showmanual"])
        return sorted(line.strip() for line in output.splitlines() if line.strip())

    output = _run(["dpkg-query", "-W", "-f=${binary:Package}\n"])
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _load_apt_baseline(baseline_file: Path | None) -> set[str]:
    """Load baseline apt packages to exclude from checking."""
    if baseline_file is None:
        return set()
    with baseline_file.open(encoding="utf-8") as stream:
        return {_canonical_package_name(line.strip()) for line in stream if line.strip()}


def _extract_debian_license_expressions(copyright_text: str) -> list[str]:
    """Extract license expressions from a Debian copyright file."""
    expressions = []
    for match in _DEBIAN_LICENSE_RE.finditer(copyright_text):
        expression = match.group(1)
        if expression is not None:
            expressions.append(re.sub(r"\s+", " ", expression.strip()))
    if expressions:
        return sorted(set(expressions))

    fallback_expressions = [
        license_name for license_name, pattern in _FALLBACK_LICENSE_PATTERNS if pattern.search(copyright_text)
    ]
    return sorted(set(fallback_expressions)) or ["UNKNOWN"]


def _read_apt_license(package_name: str) -> str:
    """Read and summarize license metadata for an apt package."""
    doc_package_name = package_name.split(":", 1)[0]
    copyright_file = Path("/usr/share/doc") / doc_package_name / "copyright"
    try:
        if not copyright_file.exists():
            return "UNKNOWN"
        copyright_text = copyright_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "UNKNOWN"
    return "; ".join(_extract_debian_license_expressions(copyright_text))


def _collect_apt_packages(manual_only: bool, baseline_file: Path | None) -> list[PackageLicense]:
    """Collect installed apt package licenses."""
    versions = _collect_apt_versions()
    baseline = _load_apt_baseline(baseline_file)
    package_names = _collect_apt_package_names(manual_only)

    packages: list[PackageLicense] = []
    for package_name in package_names:
        canonical_name = _canonical_package_name(package_name)
        if canonical_name in baseline:
            continue
        packages.append(
            PackageLicense(
                manager="apt",
                name=package_name,
                version=versions.get(canonical_name),
                license=_read_apt_license(package_name),
            )
        )
    return packages


def _print_results(label: str, packages: list[PackageLicense], failures: list[LicenseFailure]) -> None:
    """Print a concise license-check summary."""
    print(f"Checked {len(packages)} {label} packages.")
    if not failures:
        print(f"All {label} packages passed the license policy.")
        return

    for failure in failures:
        package = failure.package
        version = f"=={package.version}" if package.version else ""
        print(f"ERROR: {package.manager}:{package.name}{version} has license: {package.license}")
        print(f"       Reason: {failure.reason}")
    print(f"ERROR: {len(failures)} {label} packages were flagged.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common checker arguments to a subcommand parser."""
    parser.add_argument(
        "--exceptions-file",
        type=Path,
        default=Path(".github/workflows/license-exceptions.json"),
        help="Path to the package license exceptions JSON file.",
    )
    parser.add_argument("--report-label", default=None, help="Human-readable label used in summary output.")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="manager", required=True)

    pip_parser = subparsers.add_parser("pip", help="Check installed pip package licenses.")
    _add_common_arguments(pip_parser)
    pip_parser.add_argument(
        "--license-json",
        type=Path,
        default=None,
        help="Optional pip-licenses JSON report to check instead of invoking pip-licenses.",
    )

    apt_parser = subparsers.add_parser("apt", help="Check installed apt package licenses.")
    _add_common_arguments(apt_parser)
    apt_parser.add_argument(
        "--apt-baseline-file",
        type=Path,
        default=None,
        help="Optional file of baseline apt package names to exclude from checking.",
    )
    apt_parser.add_argument(
        "--apt-manual-only",
        action="store_true",
        help="Only check packages marked as manually installed by apt.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the dependency license checker."""
    args = _parse_args()
    exceptions = _load_exceptions(args.exceptions_file)

    if args.manager == "pip":
        packages = _collect_pip_packages(args.license_json)
        label = args.report_label or "pip"
    elif args.manager == "apt":
        packages = _collect_apt_packages(args.apt_manual_only, args.apt_baseline_file)
        label = args.report_label or "apt"
    else:
        raise ValueError(f"Unsupported package manager: {args.manager}")

    failures = _check_packages(packages, exceptions)
    _print_results(label, packages, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
