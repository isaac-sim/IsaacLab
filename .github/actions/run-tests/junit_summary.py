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

# List saved comparison images (available as workflow artifacts).
diff_images = {k: v for k, v in comparison_scores.items() if v["img_result"] and v["img_golden"]}
if diff_images:
    import os

    run_url = ""
    server = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if server and repo and run_id:
        run_url = f"{server}/{repo}/actions/runs/{run_id}"

    artifacts_link = f"[artifacts]({run_url}#artifacts)" if run_url else "artifacts"
    print(f"\n<details><summary>🔵 Image Comparison Artifacts ({len(diff_images)})</summary>")
    print("")
    print(f"Download **comparison-images** from {artifacts_link} to view golden vs result PNGs.")
    print("")
    print("| Test | AOV | Diff % | SSIM | Status | Result | Golden |")
    print("|------|-----|--------|------|--------|--------|--------|")
    for (name, label), scores in diff_images.items():
        status = "PASS" if scores["passed"] else "FAIL"
        result_file = scores["img_result"].rsplit("/", 1)[-1] if "/" in scores["img_result"] else scores["img_result"]
        golden_file = scores["img_golden"].rsplit("/", 1)[-1] if "/" in scores["img_golden"] else scores["img_golden"]
        diff_pct, ssim = scores["diff_pct"], scores["ssim"]
        print(f"| `{name}` | {label} | {diff_pct} | {ssim} | {status} | {result_file} | {golden_file} |")
    print("")
    print("</details>")
