#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check installed dependency licenses against the Isaac Lab CI policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_LICENSE_SUBSTRINGS = ("MIT", "Apache", "BSD", "ISC", "zlib")
DEFAULT_EXCLUDED_PACKAGE_PREFIXES = ("nvidia",)
GENERIC_LICENSE_VALUES = ("", "dual license", "unknown")
MAX_LICENSE_DISPLAY_LENGTH = 500
COPYLEFT_LICENSE_RE = re.compile(
    r"\b(?:AGPL|Affero General Public License|LGPL|Lesser General Public License|GPL|General Public License)\b",
    re.IGNORECASE,
)

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


def _normalize_license_for_match(license_text: str) -> str:
    """Normalize license text for exception comparisons."""
    normalized = license_text.lower()
    normalized = normalized.replace("license :: osi approved ::", "")
    replacements = {
        "psf-2.0": "python software foundation license",
        "mpl-2.0": "mozilla public license 2.0",
        "lgplv3": "gnu lesser general public license v3",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _matches_exception_license(package_license: str, exception_license: str) -> bool:
    """Return whether package and exception license strings are compatible."""
    if exception_license == package_license:
        return True
    normalized_package = _normalize_license_for_match(package_license)
    normalized_exception = _normalize_license_for_match(exception_license)
    return normalized_exception in normalized_package or normalized_package in normalized_exception


def _is_excluded_package(package: PackageLicense) -> bool:
    """Return whether package should be excluded from license checks."""
    package_name = package.name.lower()
    return any(package_name.startswith(prefix) for prefix in DEFAULT_EXCLUDED_PACKAGE_PREFIXES)


def _has_allowed_license_substring(license_text: str) -> bool:
    """Return whether the license text contains an allowed license substring."""
    normalized = license_text.lower()
    return any(allowed.lower() in normalized for allowed in ALLOWED_LICENSE_SUBSTRINGS)


def _has_copyleft_license_substring(license_text: str) -> bool:
    """Return whether the license text contains a GPL-family license substring."""
    return COPYLEFT_LICENSE_RE.search(license_text) is not None


def _is_allowed_pip_license(license_text: str) -> bool:
    """Return whether a pip license satisfies the permissive-license policy."""
    return _has_allowed_license_substring(license_text) and not _has_copyleft_license_substring(license_text)


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
            reason = (
                "contains GPL-family license text"
                if _has_copyleft_license_substring(package.license)
                else "no exception found"
            )
            failures.append(LicenseFailure(package=package, reason=reason))
            continue
        if exception.license is not None and not _matches_exception_license(package.license, exception.license):
            failures.append(
                LicenseFailure(
                    package=package,
                    reason=f"exception license mismatch: expected {exception.license}",
                )
            )
    return failures


def _format_license_for_error(license_text: str) -> str:
    """Return a compact license string for CI error output."""
    license_text = re.sub(r"\s+", " ", license_text).strip()
    if len(license_text) <= MAX_LICENSE_DISPLAY_LENGTH:
        return license_text
    return f"{license_text[:MAX_LICENSE_DISPLAY_LENGTH].rstrip()} ... [truncated]"


def _load_pip_license_json(license_json: Path | None) -> list[dict[str, Any]]:
    """Load pip license metadata from a file or pip-licenses."""
    if license_json is not None:
        with license_json.open(encoding="utf-8") as stream:
            return json.load(stream)

    output = _run([sys.executable, "-m", "piplicenses", "--from=mixed", "--format=json"])
    return json.loads(output)


def _collect_pip_packages_from_records(records: list[dict[str, Any]]) -> list[PackageLicense]:
    """Collect installed pip package licenses from pip-licenses-style records."""
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
    return _deduplicate_packages(packages)


def _collect_importlib_license_text(metadata: importlib.metadata.PackageMetadata) -> str:
    """Collect the best available license text from package metadata."""
    classifiers = [c for c in metadata.get_all("Classifier") or [] if "License" in c]
    license_text = metadata.get("License-Expression") or metadata.get("License")
    if classifiers and (not license_text or license_text.strip().lower() in GENERIC_LICENSE_VALUES):
        return "; ".join(classifiers)
    return license_text or "UNKNOWN"


def _deduplicate_packages(packages: list[PackageLicense]) -> list[PackageLicense]:
    """Deduplicate package records from repeated prebundled distribution paths."""
    deduplicated: list[PackageLicense] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for package in packages:
        key = (package.manager, _canonical_package_name(package.name), package.version, package.license)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(package)
    return deduplicated


def _collect_pip_packages_from_importlib() -> tuple[list[PackageLicense], list[str]]:
    """Collect installed pip package licenses from importlib metadata."""
    packages: list[PackageLicense] = []
    skipped: list[str] = []
    for dist in importlib.metadata.distributions():
        metadata = dist.metadata
        name = metadata.get("Name")
        if not name:
            skipped.append(str(getattr(dist, "_path", "UNKNOWN")))
            continue

        license_text = _collect_importlib_license_text(metadata)

        packages.append(
            PackageLicense(
                manager="pip",
                name=name,
                version=dist.version,
                license=license_text,
            )
        )
    packages.sort(key=lambda package: package.name.lower())
    return _deduplicate_packages(packages), skipped


def _collect_pip_packages(license_json: Path | None, collector: str) -> tuple[list[PackageLicense], list[str]]:
    """Collect installed pip package licenses."""
    if license_json is not None:
        return _collect_pip_packages_from_records(_load_pip_license_json(license_json)), []
    if collector == "importlib":
        return _collect_pip_packages_from_importlib()
    if collector != "pip-licenses":
        raise ValueError(f"Unsupported pip license collector: {collector}")
    records = _load_pip_license_json(license_json)
    return _collect_pip_packages_from_records(records), []


def _write_pip_license_json(packages: list[PackageLicense], output: Path) -> None:
    """Write pip package license metadata in pip-licenses-compatible JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "Name": package.name,
            "Version": package.version,
            "License": package.license,
        }
        for package in packages
    ]
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _escape_markdown_cell(value: str | None) -> str:
    """Escape a value for a markdown table cell."""
    return str(value or "").replace("|", "&#124;").replace("\n", " ")


def _write_pip_license_markdown(packages: list[PackageLicense], output: Path) -> None:
    """Write pip package license metadata as a markdown table."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("| Name | Version | License |\n")
        stream.write("|---|---|---|\n")
        for package in packages:
            stream.write(
                f"| {_escape_markdown_cell(package.name)}"
                f" | {_escape_markdown_cell(package.version)}"
                f" | {_escape_markdown_cell(package.license)} |\n"
            )


def _write_skipped_metadata(skipped: list[str], output: Path) -> None:
    """Write skipped distribution metadata paths."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")


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
        print(
            f"ERROR: {package.manager}:{package.name}{version} has license: "
            f"{_format_license_for_error(package.license)}"
        )
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
    pip_parser.add_argument(
        "--collector",
        choices=("pip-licenses", "importlib"),
        default="pip-licenses",
        help="Collector to use when --license-json is not provided.",
    )
    pip_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write collected pip license metadata as JSON.",
    )
    pip_parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to write collected pip license metadata as markdown.",
    )
    pip_parser.add_argument(
        "--skipped-metadata-output",
        type=Path,
        default=None,
        help="Optional path to write importlib distributions skipped because metadata was incomplete.",
    )
    pip_parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only collect and write reports; do not fail on license policy violations.",
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
        packages, skipped = _collect_pip_packages(args.license_json, args.collector)
        if args.output_json is not None:
            _write_pip_license_json(packages, args.output_json)
        if args.output_markdown is not None:
            _write_pip_license_markdown(packages, args.output_markdown)
        if args.skipped_metadata_output is not None:
            _write_skipped_metadata(skipped, args.skipped_metadata_output)
        label = args.report_label or "pip"
    elif args.manager == "apt":
        packages = _collect_apt_packages(args.apt_manual_only, args.apt_baseline_file)
        label = args.report_label or "apt"
    else:
        raise ValueError(f"Unsupported package manager: {args.manager}")

    if getattr(args, "report_only", False):
        print(f"Collected {len(packages)} {label} packages.")
        if args.manager == "pip" and skipped:
            print(f"Skipped {len(skipped)} pip distributions with incomplete metadata.")
        return 0

    failures = _check_packages(packages, exceptions)
    _print_results(label, packages, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
