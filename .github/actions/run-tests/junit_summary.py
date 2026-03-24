#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parse a JUnit XML report and print a markdown summary (for $GITHUB_STEP_SUMMARY)."""

import sys
import xml.etree.ElementTree as ET

if len(sys.argv) < 2:
    print("Usage: junit_summary.py <report.xml>", file=sys.stderr)
    sys.exit(1)

try:
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()
except ET.ParseError as exc:
    print(f"🔴 Failed to parse test report: {exc}")
    sys.exit(0)  # non-fatal so the step summary still renders
except OSError as exc:
    print(f"🔴 Failed to read test report: {exc}")
    sys.exit(0)

passed, failed, errored, skipped = [], [], [], []
total_time = 0.0


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


for tc in root.iter("testcase"):
    name = tc.get("classname", tc.get("name", "unknown"))
    t = safe_float(tc.get("time", 0))
    total_time += t
    if tc.find("failure") is not None:
        failed.append((name, t, tc.find("failure").get("message", "")))
    elif tc.find("error") is not None:
        errored.append((name, t, tc.find("error").get("message", "")))
    elif (skip_el := tc.find("skipped")) is not None:
        skipped.append((name, t, skip_el.get("message", "")))
    else:
        passed.append((name, t))

mins, secs = divmod(total_time, 60)
time_str = f"{int(mins)}m:{secs:.0f}s"


def sanitize_msg(msg, max_len=300):
    """Collapse newlines, escape pipe characters, and truncate for markdown tables."""
    return msg.replace("\n", " ").replace("\r", "").replace("|", "\\|")[:max_len]


if failed or errored:
    print(f"🔴 {len(failed) + len(errored)} FAILED, {len(passed)} PASSED ({time_str})")
elif not passed and not skipped:
    print("🟠 No test cases found in report")

if failed or errored:
    print("")
    print("| Status | Test | Time | Message |")
    print("|--------|------|------|---------|")
    for name, t, msg in failed:
        print(f"| ASSERTION | `{name}` | {t:.1f}s | {sanitize_msg(msg)} |")
    for name, t, msg in errored:
        print(f"| ERROR | `{name}` | {t:.1f}s | {sanitize_msg(msg)} |")

if passed:
    print(f"\n<details><summary>🟢 {len(passed)} PASSED ({time_str})</summary>")
    print("")
    print("<br>")
    print("")
    print("| Test | Time |")
    print("|------|------|")
    for name, t in passed:
        print(f"| `{name}` | {t:.1f}s |")
    print("")
    print("</details>")

if skipped:
    print(f"\n<details><summary>🟠 {len(skipped)} SKIPPED</summary>")
    print("")
    print("<br>")
    print("")
    print("| Test | Reason |")
    print("|------|--------|")
    for name, t, msg in skipped:
        print(f"| `{name}` | {sanitize_msg(msg)} |")
    print("")
    print("</details>")
