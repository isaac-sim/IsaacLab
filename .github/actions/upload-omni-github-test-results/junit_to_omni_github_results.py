# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a JUnit XML report into the omni-github test-result artifact format."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

_MANIFEST_NAME = "omni-github-test-results-upload.json"
_RESULT_PATH = "_testoutput/test_results.json"
_MAX_MESSAGE_CHARS = 500
_FLAG_PATTERNS = {
    "crash": ("abort", "core dumped", "crash", "process killed", "segmentation fault", "signal "),
    "timeout": ("deadline exceeded", "timed out", "timeout"),
}


def _local_name(tag: str) -> str:
    """Return an XML tag name without its namespace."""
    return tag.rsplit("}", maxsplit=1)[-1]


def _iter_testcases(root: ET.Element) -> list[ET.Element]:
    """Return all JUnit testcase elements from the parsed XML tree."""
    return [element for element in root.iter() if _local_name(element.tag) == "testcase"]


def _first_child(testcase: ET.Element, names: set[str]) -> ET.Element | None:
    """Return the first direct testcase child whose local tag name matches."""
    for child in testcase:
        if _local_name(child.tag) in names:
            return child
    return None


def _duration_seconds(element: ET.Element) -> float:
    """Return the element duration in seconds."""
    try:
        duration = float(element.attrib.get("time", "0"))
    except ValueError:
        duration = 0.0
    return max(duration, 0.0)


def _test_id(testcase: ET.Element) -> str:
    """Return a stable test identifier from JUnit classname and name fields."""
    parts = [testcase.attrib.get(key, "").strip() for key in ("classname", "name")]
    return "::".join(part for part in parts if part) or "unknown-testcase"


def _short_message(element: ET.Element | None) -> str | None:
    """Return a compact message from a failure, error, or skip element."""
    if element is None:
        return None
    raw_message = element.attrib.get("message") or element.attrib.get("type") or element.text or ""
    message = " ".join(raw_message.split())
    if not message:
        return None
    if len(message) > _MAX_MESSAGE_CHARS:
        return message[: _MAX_MESSAGE_CHARS - 3].rstrip() + "..."
    return message


def _convert_testcase(
    testcase: ET.Element,
    test_type: str,
    group_id: str,
    retries: int,
    log_paths: list[str],
) -> dict[str, object]:
    """Convert one JUnit testcase element into an omni-github test row."""
    failure_or_error = _first_child(testcase, {"error", "failure"})
    skipped = _first_child(testcase, {"skipped"})
    message = _short_message(failure_or_error)
    skip_reason = _short_message(skipped)
    row: dict[str, object] = {
        "test_id": _test_id(testcase),
        "passed": failure_or_error is None and skipped is None,
        "duration": _duration_seconds(testcase),
        "test_type": test_type,
        "group_id": group_id,
        "retries": retries,
        "log_paths": log_paths,
    }
    if testcase.attrib.get("name"):
        row["test_name"] = testcase.attrib["name"]
    if skipped is not None:
        row["skipped"] = True
    if skip_reason is not None:
        row["skip_reason"] = skip_reason
        row.setdefault("message", skip_reason)
    if message is not None:
        row["message"] = message

    if failure_or_error is not None:
        detail_text = " ".join(
            filter(None, (message, failure_or_error.attrib.get("type"), failure_or_error.text))
        ).lower()
        for flag, patterns in _FLAG_PATTERNS.items():
            if any(pattern in detail_text for pattern in patterns):
                row[flag] = True

    return row


def convert_junit(
    junit_file: Path,
    output_dir: Path,
    test_tool_id: str,
    test_type: str,
    app_platform: str,
    app_config: str,
    group_name: str,
    junit_log_url: str,
    comparison_images_url: str,
    retries: int = 0,
) -> None:
    """Convert a JUnit XML report and write the omni-github artifact directory.

    Args:
        junit_file: Path to the source JUnit XML report.
        output_dir: Directory where the artifact root should be written.
        test_tool_id: Identifier for the test tool that produced the report.
        test_type: Suite category to store on each converted test row.
        app_platform: Platform label for the result app metadata.
        app_config: Configuration label for the result app metadata.
        group_name: Human-readable test group label to store on each test row.
        retries: Within-job retry count to store on each test row.
        junit_log_url: URL of the uploaded source JUnit XML artifact.
        comparison_images_url: URL of the uploaded comparison images artifact.
    """
    root = ET.parse(junit_file).getroot()
    log_paths: list[str] = []
    if junit_log_url:
        log_paths.append(junit_log_url)

    if comparison_images_url:
        log_paths.append(comparison_images_url)

    tests = []
    for testcase in _iter_testcases(root):
        tests.append(
            _convert_testcase(
                testcase,
                test_type,
                group_name,
                retries,
                log_paths=log_paths,
            )
        )
    result: dict[str, object] = {
        "result_schema_version": 1,
        "test_tool_id": test_tool_id,
        "app": {
            "platform": app_platform,
            "config": app_config,
        },
        "tests": tests,
    }
    manifest = {
        "schema_version": 1,
        "result_paths": [_RESULT_PATH],
    }

    result_path = output_dir / _RESULT_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    (output_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit-file", type=Path, required=True, help="Path to the JUnit XML report.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Artifact root directory to write.")
    parser.add_argument("--test-tool-id", required=True, help="Identifier for the test tool.")
    parser.add_argument("--test-type", required=True, help="Suite category for each test row.")
    parser.add_argument("--app-platform", required=True, help="App platform label.")
    parser.add_argument("--app-config", required=True, help="App configuration label.")
    parser.add_argument("--group-name", required=True, help="Human-readable group label for each test row.")
    parser.add_argument(
        "--junit-log-url", required=True, type=str, help="URL of the uploaded source JUnit XML artifact."
    )
    parser.add_argument(
        "--comparison-images-url", required=True, type=str, help="URL of the uploaded comparison images artifact."
    )
    parser.add_argument("--retries", type=int, default=0, help="Within-job retry count for each test row.")
    return parser.parse_args()


def main() -> None:
    """Run the converter."""
    args = parse_args()
    convert_junit(
        junit_file=args.junit_file,
        output_dir=args.output_dir,
        test_tool_id=args.test_tool_id,
        test_type=args.test_type,
        app_platform=args.app_platform,
        app_config=args.app_config,
        group_name=args.group_name,
        junit_log_url=args.junit_log_url,
        comparison_images_url=args.comparison_images_url,
        retries=max(args.retries, 0),
    )


if __name__ == "__main__":
    main()
