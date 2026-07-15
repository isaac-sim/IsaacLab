#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate per-shard JUnit XML reports from the multi-GPU pytest lane.

Reads ``RUNTIME_DIR`` (the host launcher's work-queue root, holding
``queue/done/<shard>/<slug>``) and the ``tests/test-reports-*.xml`` JUnit files,
then prints a per-file + per-shard table to stdout and, when
``GITHUB_STEP_SUMMARY`` is set, writes the same content as markdown so it
renders at the top of the run page.
"""

import glob
import os
import pathlib
import xml.etree.ElementTree as ET

runtime_dir = os.environ.get("RUNTIME_DIR", "")
done_root = pathlib.Path(runtime_dir) / "queue" / "done" if runtime_dir else None

# Map slug -> shard from done/<shard>/<slug>; slug uses "__" for "/".
slug_to_shard = {}
if done_root and done_root.is_dir():
    for shard_dir in done_root.iterdir():
        if not shard_dir.is_dir():
            continue
        shard = shard_dir.name  # e.g. "cuda-1"
        for entry in shard_dir.iterdir():
            slug_to_shard[entry.name] = shard


def unslug(name: str) -> str:
    # report file pattern: test-reports-<slug>.xml
    if name.startswith("test-reports-"):
        name = name[len("test-reports-") :]
    if name.endswith(".xml"):
        name = name[:-4]
    return name  # this is the slug; the file path is slug.replace("__", "/")


# Per-file rows from JUnit XMLs.
rows = []
for path in sorted(glob.glob("tests/test-reports-*.xml")):
    slug = unslug(os.path.basename(path))
    file_path = slug.replace("__", "/")
    shard = slug_to_shard.get(slug, "?")
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        rows.append((file_path, shard, "PARSE-ERROR", 0, 0, 0, 0, 0.0))
        continue
    tests = failures = errors = skipped = 0
    total_time = 0.0
    for ts in root.iter("testsuite"):
        tests += int(ts.get("tests", 0))
        failures += int(ts.get("failures", 0))
        errors += int(ts.get("errors", 0))
        skipped += int(ts.get("skipped", 0))
        total_time += float(ts.get("time", 0) or 0)
    passed = tests - failures - errors - skipped
    status = "PASS"
    if failures + errors > 0:
        status = "FAIL"
    elif tests == 0:
        status = "EMPTY"
    rows.append((file_path, shard, status, passed, tests, skipped, failures + errors, total_time))

# Per-shard aggregate.
shards = {}
for fp, sh, st, p, t, sk, f, tm in rows:
    if sh not in shards:
        shards[sh] = {"files": 0, "passed": 0, "tests": 0, "skipped": 0, "failed": 0, "time": 0.0}
    s = shards[sh]
    s["files"] += 1
    s["passed"] += p
    s["tests"] += t
    s["skipped"] += sk
    s["failed"] += f
    s["time"] += tm


# ---------- Output: plain text to stdout ----------
def fmt_row(cols, widths):
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))


out = []
out.append("PER-FILE RESULTS")
out.append("=" * 110)
hdr = ["File", "Shard", "Status", "Pass/Total", "Skip", "Fail", "Time(s)"]
widths = [60, 7, 8, 12, 5, 5, 8]
out.append(fmt_row(hdr, widths))
out.append("-" * 110)
for fp, sh, st, p, t, sk, f, tm in sorted(rows, key=lambda r: (r[1], r[0])):
    out.append(fmt_row([fp[-60:], sh, st, f"{p}/{t}", str(sk), str(f), f"{tm:.1f}"], widths))
out.append("")
out.append("PER-SHARD AGGREGATE")
out.append("=" * 80)
shard_hdr = ["Shard", "Files", "Pass/Total", "Skipped", "Failed", "Test time (s)"]
shard_widths = [10, 7, 12, 9, 8, 14]
out.append(fmt_row(shard_hdr, shard_widths))
out.append("-" * 80)
grand = {"files": 0, "passed": 0, "tests": 0, "skipped": 0, "failed": 0, "time": 0.0}
for sh in sorted(shards):
    s = shards[sh]
    for k in grand:
        grand[k] += s[k]
    out.append(
        fmt_row(
            [
                sh,
                str(s["files"]),
                f"{s['passed']}/{s['tests']}",
                str(s["skipped"]),
                str(s["failed"]),
                f"{s['time']:.1f}",
            ],
            shard_widths,
        )
    )
out.append("-" * 80)
out.append(
    fmt_row(
        [
            "TOTAL",
            str(grand["files"]),
            f"{grand['passed']}/{grand['tests']}",
            str(grand["skipped"]),
            str(grand["failed"]),
            f"{grand['time']:.1f}",
        ],
        shard_widths,
    )
)
print("\n".join(out))

# ---------- Output: markdown to $GITHUB_STEP_SUMMARY ----------
summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_path:
    md = [
        "## Multi-GPU CI: Aggregated Test Summary",
        "",
        "### Per-shard totals",
        "",
        "| Shard | Files | Pass/Total | Skipped | Failed | Test time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sh in sorted(shards):
        s = shards[sh]
        md.append(
            f"| `{sh}` | {s['files']} | {s['passed']}/{s['tests']} | {s['skipped']} | {s['failed']} | {s['time']:.1f} |"
        )
    md.append(
        f"| **TOTAL** | **{grand['files']}** | **{grand['passed']}/{grand['tests']}** | "
        f"**{grand['skipped']}** | **{grand['failed']}** | **{grand['time']:.1f}** |"
    )
    md.append("")
    md.append("### Per-file results")
    md.append("")
    md.append("| File | Shard | Status | Pass/Total | Skipped | Failed | Test time (s) |")
    md.append("|---|---|---|---:|---:|---:|---:|")
    for fp, sh, st, p, t, sk, f, tm in sorted(rows, key=lambda r: (r[1], r[0])):
        status_md = {
            "PASS": ":white_check_mark: PASS",
            "FAIL": ":x: FAIL",
            "EMPTY": ":white_circle: EMPTY",
            "PARSE-ERROR": ":warning: PARSE-ERROR",
        }.get(st, st)
        md.append(f"| `{fp}` | `{sh}` | {status_md} | {p}/{t} | {sk} | {f} | {tm:.1f} |")
    md.append("")
    with open(summary_path, "a") as fh:
        fh.write("\n".join(md) + "\n")
