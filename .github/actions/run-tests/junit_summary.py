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
# Keyed by (test_name, label) -> {"diff_pct": str, "ssim": str, "passed": bool}
comparison_scores = {}
total_time = 0.0


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


for tc in root.iter("testcase"):
    classname = tc.get("classname", "")
    tc_name = tc.get("name", "unknown")
    name = f"{classname}.{tc_name}" if classname else tc_name
    t = safe_float(tc.get("time", 0))
    total_time += t
    tc_failed = tc.find("failure") is not None
    tc_errored = tc.find("error") is not None
    if tc_failed:
        failed.append((name, t, tc.find("failure").get("message", "")))
    elif tc_errored:
        errored.append((name, t, tc.find("error").get("message", "")))
    elif (skip_el := tc.find("skipped")) is not None:
        skipped.append((name, t, skip_el.get("message", "")))
    else:
        passed.append((name, t))

    # Collect diff_pct:*, ssim:*, and img_*:* properties emitted by test_rendering_correctness.
    props = tc.find("properties")
    if props is not None:
        for prop in props.findall("property"):
            prop_name = prop.get("name", "")
            for prefix in ("diff_pct:", "ssim:", "threshold:", "img_result:", "img_golden:"):
                if prop_name.startswith(prefix):
                    label = prop_name[len(prefix) :]
                    key = (name, label)
                    if key not in comparison_scores:
                        comparison_scores[key] = {
                            "diff_pct": "",
                            "ssim": "",
                            "threshold": "",
                            "passed": not (tc_failed or tc_errored),
                            "img_result": "",
                            "img_golden": "",
                        }
                    field = prefix.rstrip(":")
                    comparison_scores[key][field] = prop.get("value", "")

mins, secs = divmod(total_time, 60)
time_str = f"{int(mins)}m:{secs:.0f}s"


def sanitize_msg(msg, max_len=300):
    """Collapse newlines, escape pipe characters, and truncate for markdown tables."""
    return msg.replace("\n", " ").replace("\r", "").replace("|", "\\|")[:max_len]


def fmt_name(name):
    """Format a test name for markdown: strip ``source.`` prefix, allow word-breaking."""
    if name.startswith("source."):
        name = name[len("source.") :]
    # Insert zero-width spaces after dots and brackets so tables can wrap.
    return name.replace(".", ".\u200b").replace("[", "[\u200b").replace("]", "]\u200b")


if failed or errored:
    print(f"🔴 {len(failed) + len(errored)} FAILED, {len(passed)} PASSED ({time_str})")
elif not passed and not skipped:
    print("🟠 No test cases found in report")

if failed or errored:
    print("")
    print("| Status | Test | Time | Message |")
    print("|--------|------|------|---------|")
    for name, t, msg in sorted(failed, key=lambda x: x[1], reverse=True):
        print(f"| ASSERTION | {fmt_name(name)} | {t:.1f}s | {sanitize_msg(msg)} |")
    for name, t, msg in sorted(errored, key=lambda x: x[1], reverse=True):
        print(f"| ERROR | {fmt_name(name)} | {t:.1f}s | {sanitize_msg(msg)} |")

if passed:
    print(f"\n<details><summary>🟢 {len(passed)} PASSED ({time_str})</summary>")
    print("")
    print("<br>")
    print("")
    print("| Test | Time |")
    print("|------|------|")
    for name, t in sorted(passed, key=lambda x: x[1], reverse=True):
        print(f"| {fmt_name(name)} | {t:.1f}s |")
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
        print(f"| {fmt_name(name)} | {sanitize_msg(msg)} |")
    print("")
    print("</details>")

if comparison_scores:
    print("\n")
    print("🔵 GOLDEN vs ACTUAL report is attached as artifact (at the bottom of this page).")
