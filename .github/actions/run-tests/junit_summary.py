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
    name = tc.get("classname", tc.get("name", "unknown"))
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
            for prefix in ("diff_pct:", "ssim:", "img_result:", "img_golden:"):
                if prop_name.startswith(prefix):
                    label = prop_name[len(prefix) :]
                    key = (name, label)
                    if key not in comparison_scores:
                        comparison_scores[key] = {
                            "diff_pct": "",
                            "ssim": "",
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

if comparison_scores:
    print(f"\n<details><summary>🔵 Image Comparison Scores ({len(comparison_scores)})</summary>")
    print("")
    print("<br>")
    print("")
    print("| Test | AOV | Diff % | SSIM | Status |")
    print("|------|-----|--------|------|--------|")
    for (name, label), scores in comparison_scores.items():
        status = "PASS" if scores["passed"] else "FAIL"
        print(f"| `{name}` | {label} | {scores['diff_pct']} | {scores['ssim']} | {status} |")
    print("")
    print("</details>")

# Show side-by-side golden vs result images for failed comparisons.
failed_images = {k: v for k, v in comparison_scores.items() if v["img_result"] and v["img_golden"]}
if failed_images:
    print(f"\n<details><summary>🔵 Failed Image Comparisons ({len(failed_images)})</summary>")
    print("")
    for (name, label), scores in failed_images.items():
        print(f"**`{name}`** — {label} (Diff {scores['diff_pct']}%, SSIM {scores['ssim']})")
        print("")
        print("| Golden (expected) | Result (actual) |")
        print("|:-:|:-:|")
        golden = f'<img src="data:image/png;base64,{scores["img_golden"]}">'
        result = f'<img src="data:image/png;base64,{scores["img_result"]}">'
        print(f"| {golden} | {result} |")
        print("")
    print("</details>")
