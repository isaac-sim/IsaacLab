# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Markdown and JSON rendering for individual and aggregate performance reports."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

from .compare import ERROR, FAIL, PASS, SKIP, WARN, Report
from .metrics import METRICS

_ICONS = {PASS: "✅", WARN: "⚠️", FAIL: "❌", SKIP: "⏭️", ERROR: "🚫"}


def _num(value: float | None, digits: int = 6) -> str:
    """Format a number for a table cell, or ``-`` when absent."""
    return "-" if value is None else f"{value:.{digits}g}"


def _pct(value: float | None) -> str:
    """Format a signed percentage, or ``-`` when absent."""
    return "-" if value is None else f"{value:+.2f}%"


def _icon(verdict: str) -> str:
    return f"{_ICONS.get(verdict, '')} {verdict}".strip()


def render(report: Report) -> str:
    """Render measured values, baselines and thresholds as Markdown."""
    lines = [
        f"## Performance smoke: {_icon(report.verdict)}",
        "",
        report.message,
        "",
        "| Metric | Measured | Baseline median | Change | Warn | Fail | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in report.metrics:
        label = metric.label if metric.gating else f"{metric.label} _(advisory)_"
        verdict = _icon(metric.verdict) if metric.gating else f"{metric.verdict} _(advisory)_"
        note = f" — {metric.note}" if metric.note else ""
        lines.append(
            f"| {label} | {_num(metric.measured)} | {_num(metric.reference)} | {_pct(metric.regression_pct)} | "
            f"{_pct(metric.warn_pct)} | {_pct(metric.fail_pct)} | "
            f"{verdict}{note} |"
        )

    samples = max((metric.sample_count for metric in report.metrics), default=0)
    gating = ", ".join(f"**{metric.label}**" for metric in METRICS if metric.gating)
    lines += [
        "",
        f"Baseline: {samples} comparable run(s), contract `{report.contract_hash}`.",
        "",
        f"Only {gating} gates. The other metrics are recorded and compared so their noise can be "
        "characterised before any of them is trusted to fail a pull request.",
    ]
    return "\n".join(lines) + "\n"


def render_aggregate(reports: list[tuple[str, Report]]) -> str:
    """Render every workload's rolling-history result in one Markdown table."""
    if not reports:
        return "### Rolling-history CI gate: no results\n\nNo comparison artifacts were produced.\n"

    # SKIP distinguishes runs with no comparisons from runs that passed.
    order = {SKIP: 0, PASS: 1, ERROR: 2, WARN: 3, FAIL: 4}
    worst = max((report.verdict for _, report in reports), key=lambda verdict: order.get(verdict, 0))

    lines = [
        f"### Rolling-history CI gate: {_icon(worst)}",
        "",
    ]
    counts = Counter(report.verdict for _, report in reports)
    labels = {PASS: "passed", WARN: "with warnings", FAIL: "failed", SKIP: "skipped", ERROR: "with errors"}
    lines += [
        "Workload checks: "
        + ", ".join(f"{counts[verdict]} {label}" for verdict, label in labels.items() if counts[verdict])
        + ".",
        "",
    ]
    reports = sorted(reports, key=lambda item: item[0])
    lines += [
        "| Combination | Total FPS | Baseline | FPS regression % | Startup [s] | GPU mem [GB] | RSS [GB] | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, report in reports:
        by_name = {metric.name: metric for metric in report.metrics}
        fps = by_name.get("total_fps")
        lines.append(
            f"| {_build_text(name)} | {_num(fps.measured) if fps else '-'} | {_num(fps.reference) if fps else '-'} | "
            f"{_pct(fps.regression_pct) if fps else '-'} | "
            f"{_num(by_name['startup_time_s'].measured, 4) if 'startup_time_s' in by_name else '-'} | "
            f"{_num(by_name['gpu_mem_peak_gb'].measured, 4) if 'gpu_mem_peak_gb' in by_name else '-'} | "
            f"{_num(by_name['ram_peak_gb'].measured, 4) if 'ram_peak_gb' in by_name else '-'} | "
            f"{_icon(report.verdict)} |"
        )
    lines += ["", "For the rolling-history gate, positive FPS regression % means lower FPS.", ""]
    for name, report in reports:
        reasons = [report.message] if report.message else []
        reasons.extend(f"{metric.label}: {metric.note}" for metric in report.metrics if metric.note)
        if not reasons and report.verdict in (SKIP, ERROR):
            reasons.append("No reason was recorded in this comparison artifact.")
        if reasons:
            lines.append(
                f"- **{_build_text(name)} — {_build_text(report.verdict)}:** "
                + " ".join(_build_text(reason) for reason in dict.fromkeys(reasons))
            )
    lines += [
        "",
        f"{len(reports)} combination(s) reported. A 🚫 ERROR row is a fault in the gate, not a performance "
        "result, and never blocks a pull request. A combination whose benchmark crashed shows both an ERROR "
        "row here and a failed job.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_json(report: Report, path: Path) -> None:
    """Write the comparison as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_text(value: object) -> str:
    """Keep artifact-provided labels inside their Markdown table cell or paragraph."""
    result = html.escape(str(value)).replace("\n", " ").replace("\r", " ")
    for character, entity in (("|", "&#124;"), ("`", "&#96;"), ("[", "&#91;"), ("]", "&#93;")):
        result = result.replace(character, entity)
    return result


def _build_commit(repository: str, commit: object) -> str:
    label = _build_text(str(commit or "unknown")[:12])
    if re.fullmatch(r"[\w.-]+/[\w.-]+", repository) and re.fullmatch(r"[0-9a-fA-F]{40}", str(commit)):
        return f"[{label}](https://github.com/{repository}/commit/{commit})"
    return label


def _build_identity(label: str, identity: dict | None) -> str:
    if not identity:
        return f"**{label}:** unavailable."
    run = _build_text(identity.get("run_id", "unknown"))
    attempt = _build_text(identity.get("run_attempt", "unknown"))
    # Rebuild links because URLs supplied by artifacts may be unsafe.
    repository = identity.get("repository", "")
    origin = f"https://github.com/{repository}" if re.fullmatch(r"[\w.-]+/[\w.-]+", repository) else None
    source = _build_commit(repository, identity.get("source_commit"))
    execution = f"run {run}, attempt {attempt}"
    run_url = None
    if origin and all(str(identity.get(field, "")).isdigit() for field in ("run_id", "run_attempt")):
        run_url = f"{origin}/actions/runs/{identity['run_id']}"
        execution = f"[{execution}]({run_url}/attempts/{identity['run_attempt']})"
    evidence = ""
    if run_url and str(identity.get("artifact_id", "")).isdigit():
        evidence = f" · [source results]({run_url}/artifacts/{identity['artifact_id']})"
    note = ""
    if identity.get("source_provenance") == "github_run_metadata":
        note = " · source SHA recovered from GitHub; tested checkout not independently recorded"
    head = str(identity.get("requested_head_commit") or "")
    if origin and head != identity.get("source_commit") and re.fullmatch(r"[0-9a-fA-F]{40}", str(head)):
        note += f" · requested PR head [{head[:12]}]({origin}/commit/{head})"
    return f"**{label}:** {source} · {execution}{evidence}{note}."


def _build_result(row: dict) -> str:
    if row["status"] != "compared" or row.get("absolute_change") is None:
        return "⚪ Not comparable"
    if row["absolute_change"] > 0:
        return "🟢 Improved"
    if row["absolute_change"] < 0:
        return "🔴 Regressed"
    return "⚪ Unchanged"


def _build_fps(value: float | None) -> str:
    """Make FPS readable at a glance while keeping small nonzero values visible."""
    if value is None:
        return "—"
    if 0 < abs(value) < 0.01:
        return f"{value:.6g}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _build_labels(rows: list[dict]) -> list[str]:
    """Use unambiguous CI configuration names, retaining full identities below."""
    labels = []
    for index, row in enumerate(rows, 1):
        legs = row.get("legs", {})
        current = legs.get("candidate", [])
        previous = legs.get("baseline", [])
        label = current[0] if len(current) == 1 else previous[0] if len(previous) == 1 else f"Workload {index}"
        labels.append(label)
    result, used = [], set(labels)
    for index, label in enumerate(labels, 1):
        if labels.count(label) > 1:
            while f"{label} ({index})" in used:
                index += 1
            label = f"{label} ({index})"
        used.add(label)
        result.append(label)
    return result


def _build_source_diagnostics(report: dict) -> list[str]:
    """Keep actual, intended and event revisions below the primary comparison."""
    selection = report.get("selection", {})
    lines = []
    source_context = report.get("source_context") or {}
    reference_commit = source_context.get("reference_commit") or selection.get("reference_commit")
    event_base_commit = source_context.get("event_base_commit") or selection.get("event_base_commit")
    if reference_commit and event_base_commit and reference_commit != event_base_commit:
        lines += [
            f"Resolved PR base: `{_build_text(reference_commit)}`. "
            f"The PR event reported `{_build_text(event_base_commit)}`; "
            "the resolved base identifies the tested merge's actual first parent.",
            "",
        ]
    failure = selection.get("paired_failure") or {}
    if failure:
        lines += ["**Source setup diagnostics:**", "", f"Stage: `{_build_text(failure.get('stage'))}`.", ""]
        failure_source = failure.get("source") or {}
        for field, label in (
            ("commit", "Actual checkout"),
            ("intended_commit", "Intended checkout"),
            ("reference_commit", "Resolved PR base"),
            ("event_base_commit", "PR event base"),
            ("requested_head_commit", "Requested PR head"),
        ):
            if failure_source.get(field):
                lines.append(f"- {label}: `{_build_text(failure_source[field])}`")
        if failure_source.get("commit_parents"):
            parents = ", ".join(f"`{_build_text(parent)}`" for parent in failure_source["commit_parents"])
            lines.append(f"- Checkout parents: {parents}")
        lines.append("")
    return lines


def render_build_comparison(report: dict) -> str:
    """Render exact-build FPS observations within the existing Performance smoke summary."""
    selection = report.get("selection", {})
    identities = {side: report.get(side) for side in ("baseline", "candidate")}
    unavailable_side = selection.get("unavailable_side")
    if unavailable_side in identities and not identities[unavailable_side]:
        identities[unavailable_side] = selection.get("unavailable_evidence")
    a_label = "A — PR base"
    b_label = "B — PR merge result" if report.get("candidate_kind") == "merge" else "B — PR benchmark"
    if (identities["candidate"] or {}).get("source_provenance") == "github_run_metadata":
        b_label = "B — PR revision (not verified)"
    a_label += " (evidence unavailable)" if unavailable_side == "baseline" else ""
    b_label += " (evidence unavailable)" if unavailable_side == "candidate" else ""
    rows = report.get("rows", [])
    labels = _build_labels(rows)
    results = Counter(_build_result(row) for row in rows)
    counts = [f"{status} {results[status]}" for status in ("🟢 Improved", "🔴 Regressed", "⚪ Not comparable")]
    if results["⚪ Unchanged"]:
        counts.append(f"⚪ Unchanged {results['⚪ Unchanged']}")
    lines = [
        "### PR performance comparison",
        "",
        "**" + (" · ".join(counts) if rows else "⚪ Not comparable: no workload results available") + "**",
        "",
    ]
    if selection.get("reason_code") or not report.get("baseline") or not rows:
        missing_reason = (
            "No baseline results were available." if rows else "No readable workload results were recorded."
        )
        lines += [
            "Comparison unavailable: " + _build_text(selection.get("reason", missing_reason)),
            "",
        ]
    if rows and (
        not selection.get("paired_failure")
        or any(row[side]["median"] is not None for row in rows for side in ("baseline", "candidate"))
    ):
        lines += [
            "| Status | Workload | Baseline FPS | PR FPS | Change % |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for label, row in zip(labels, rows):
            percent = _pct(row["change_pct"]) if row.get("change_pct") is not None else "—"
            if row["status"] == "compared" and row.get("change_pct") is None and row["baseline"]["median"] == 0:
                percent = "N/A (baseline is zero)"
            lines.append(
                f"| {_build_result(row)} | {_build_text(label)} | {_build_fps(row['baseline']['median'])} | "
                f"{_build_fps(row['candidate']['median'])} | {percent} |"
            )
        lines += [
            "",
            "FPS is the median of recorded samples. Improved and regressed describe observed FPS changes; "
            "positive change means higher FPS. These observations do not change the rolling-history CI gate below.",
            "",
        ]
    lines += [
        "<details>",
        "<summary>Sources and comparison details</summary>",
        "",
        _build_identity(a_label, identities["baseline"]),
        "",
        _build_identity(b_label, identities["candidate"]),
        "",
        "**Baseline selection:** " + _build_text(selection.get("reason", "Selection information is unavailable.")),
        "",
    ]
    if isinstance(selection.get("baseline_reused"), bool):
        measurement_status = "reused." if selection["baseline_reused"] else "run for this PR."
        lines += ["Baseline measurements: " + (measurement_status if report.get("baseline") else "unavailable."), ""]
    measurement_sources = report.get("measurement_sources", {})
    if measurement_sources:
        lines += ["Measurement sources (commit · SHA-256):", ""]
        for side, label in (("baseline", "A"), ("candidate", "B")):
            measurement = measurement_sources.get(side)
            if measurement:
                repository = (identities[side] or {}).get("repository", report.get("repository", ""))
                lines.append(
                    f"- {label}: {_build_commit(repository, measurement.get('commit'))} · "
                    f"`{_build_text(measurement.get('digest'))}`"
                )
        lines.append("")
    lines += _build_source_diagnostics(report)
    candidate = report.get("candidate") or {}
    if candidate.get("run_attempt") and report.get("report_attempt") != candidate["run_attempt"]:
        lines += [
            f"Measurements: attempt {_build_text(candidate['run_attempt'])}; "
            f"report: attempt {_build_text(report.get('report_attempt'))}.",
            "",
        ]
    for issue in selection.get("issues", []):
        lines.append("- Selection note: " + _build_text(issue))
    if rows:
        lines += ["", "**Workloads — samples A / B (valid/expected):**", ""]
    for label, row in zip(labels, rows):
        counts = []
        for side in ("baseline", "candidate"):
            item = row[side]
            expected = item.get("expected_count")
            counts.append(f"{item['count']}/{expected if expected is not None else '?'}")
        details = dict.fromkeys([*row.get("reasons", []), *row.get("notes", [])])
        lines.append(
            f"- **{_build_text(label)}:** {' / '.join(counts)} · {_build_text(row['label'])}. "
            + " ".join(_build_text(detail) for detail in details)
        )
    lines.append("")
    for note in dict.fromkeys(report.get("notes", [])):
        lines.append("- " + _build_text(note))
    lines += [
        "",
        "Full samples, environment differences and provenance are in `build-comparison.json`.",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines) + "\n"
