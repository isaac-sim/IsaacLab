# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Describe FPS changes between two selected evidence sets, without CI verdicts."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import baseline as baseline_mod
from .baseline import Evidence, workload_key


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _fps(bundle: dict) -> float | None:
    value = _object(_object(bundle.get("runtime")).get("total_fps")).get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _formula(evidence: Evidence | None) -> str | None:
    value = _object(evidence.context.get("metric_definition")).get("total_fps") if evidence else None
    return value if isinstance(value, str) and value.strip() and value.lower() != "unknown" else None


def _groups(evidence: Evidence | None, aliases: dict[str, set[str]]) -> dict:
    groups = {}
    if evidence is None:
        return groups
    for leg in sorted(evidence.samples.keys() | evidence.statuses.keys()):
        samples = evidence.samples.get(leg, [])
        for sample in samples or [None]:
            key = workload_key(sample["bundle"]) if sample else None
            unknown = key is None
            if unknown:
                possibilities = aliases.get(leg, set())
                key = next(iter(possibilities)) if len(possibilities) == 1 else "unknown:" + leg
            group = groups.setdefault(key, {"legs": set(), "samples": [], "unknown_identity": False})
            group["legs"].add(leg)
            group["unknown_identity"] |= unknown
            if sample:
                group["samples"].append(sample)
    return groups


def _summary(evidence: Evidence | None, group: dict | None) -> tuple[dict, list[str]]:
    items = group["samples"] if group else []
    measured = [(item["path"], value) for item in items if (value := _fps(item["bundle"])) is not None]
    values = [value for _, value in measured]
    expected = _object(evidence.context.get("execution")).get("expected_samples") if evidence else None
    known_expected = isinstance(expected, int) and not isinstance(expected, bool) and expected > 0
    reasons = []
    status = "complete"
    if not group or not items:
        status = "missing"
        reasons.append("No runtime samples identify this workload.")
    elif not known_expected:
        status = "unknown"
        reasons.append("The expected per-leg sample count is not recorded.")
    if group and evidence:
        if group["unknown_identity"]:
            reasons.append(
                "Workload identity is unavailable for some output; the leg name is only a display association."
            )
            if status == "complete":
                status = "unknown"
        for leg in sorted(group["legs"]):
            leg_status = evidence.statuses.get(leg)
            leg_items = evidence.samples.get(leg, [])
            if any(issue.startswith(leg + "/") for issue in evidence.issues):
                reasons.append(f"Leg {leg} has unresolved sample evidence issues; see the evidence notes.")
                if status == "complete":
                    status = "unknown"
            if leg_status != "ok":
                reasons.append(f"Leg {leg} status is {leg_status or 'not recorded'}.")
                if items:
                    status = "partial"
            if known_expected and len(leg_items) != expected:
                reasons.append(f"Leg {leg} has {len(leg_items)} runtime samples; expected {expected}.")
                if items:
                    status = "partial"
            identities = {workload_key(item["bundle"]) for item in leg_items}
            if len(identities) > 1:
                reasons.append(
                    f"Leg {leg} contains different workload identities; per-workload completeness is unknown."
                )
                if status == "complete":
                    status = "unknown"
        if any(_object(item["bundle"].get("run")).get("status") != "completed" for item in items):
            reasons.append("Some runtime samples did not complete.")
            status = "partial"
        if len(values) != len(items):
            reasons.append("Some FPS values are missing, non-numeric, non-finite or negative.")
            status = "partial"
    return {
        "count": len(values),
        "observed_count": len(items),
        "expected_count": expected * len(group["legs"]) if known_expected and group else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "samples": values,
        "sample_paths": [path for path, _ in measured],
        "status": status,
    }, reasons


def _protocol(bundle: dict) -> dict:
    runtime = _object(bundle.get("runtime"))
    timing = _object(runtime.get("environment_step_timing"))
    return {
        "seed": _object(bundle.get("run")).get("seed"),
        "measured_steps": runtime.get("iterations_completed"),
        "steps_per_iteration": runtime.get("steps_per_iteration"),
        "warmup_steps": timing.get("warmup_steps"),
        "measurement_mode": timing.get("measurement_mode"),
    }


def _flatten(value: Any, prefix: str = "") -> dict:
    if isinstance(value, dict) and value:
        result = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def _observations(group: dict | None, extractor) -> dict:
    observations = defaultdict(dict)
    for item in group["samples"] if group else []:
        for field, value in extractor(item["bundle"]).items():
            observations[field][json.dumps(value, sort_keys=True)] = value
    return {field: list(values.values()) for field, values in observations.items()}


def _display(values: list | None) -> Any:
    if not values:
        return None
    return values[0] if len(values) == 1 else {"sample_values": values}


def _differences(a: dict, b: dict, *, missing: bool = False) -> list[dict]:
    return [
        {"field": field, "baseline": _display(a.get(field)), "candidate": _display(b.get(field))}
        for field in sorted(a.keys() | b.keys())
        if a.get(field) != b.get(field)
        or len(a.get(field, [])) > 1
        or len(b.get(field, [])) > 1
        or (missing and (None in a.get(field, []) or None in b.get(field, [])))
    ]


def _context(evidence: Evidence | None, group: dict | None) -> dict:
    observed = _observations(
        group,
        lambda bundle: _flatten(
            {
                "hardware": bundle.get("hardware"),
                "versions": bundle.get("versions"),
                "config": _object(bundle.get("run")).get("config"),
            }
        ),
    )
    source = _object(evidence.context.get("source")) if evidence else {}
    for field in ("image_ref", "image_digest", "image_id"):
        observed["source." + field] = [source.get(field)]
    return observed


def _label(key: str) -> str:
    if key.startswith("unknown:"):
        return key.removeprefix("unknown:") + " (workload identity unavailable)"
    task, physics, renderer, envs, presets = json.loads(key)
    label = f"{task} · {physics} / {renderer} · {envs} envs"
    return label + (f" · presets: {', '.join(presets)}" if presets else "")


def _row(key: str, a: Evidence | None, b: Evidence, left: dict | None, right: dict | None) -> dict:
    before, left_reasons = _summary(a, left)
    after, right_reasons = _summary(b, right)
    reasons = ["Baseline: " + reason for reason in left_reasons] + ["Candidate: " + reason for reason in right_reasons]
    formulas = {"baseline": _formula(a), "candidate": _formula(b)}
    pa, pb = _observations(left, _protocol), _observations(right, _protocol)
    protocol_differences = _differences(pa, pb, missing=True)
    context_differences = _differences(_context(a, left), _context(b, right))
    status = "compared"
    if "missing" in (before["status"], after["status"]):
        status = "missing"
    elif "partial" in (before["status"], after["status"]):
        status = "partial"
    elif "unknown" in (before["status"], after["status"]):
        status = "unknown"
    if not all(formulas.values()):
        reasons.append("FPS formula identity is unknown for one or both selections.")
        if status == "compared":
            status = "unknown"
    elif formulas["baseline"] != formulas["candidate"]:
        reasons.append("FPS formula identities differ.")
        if status == "compared":
            status = "incompatible"
    if protocol_differences:
        reasons.append("Measurement protocol differs, varies within a selection, or has unrecorded fields.")
        if status == "compared":
            status = "unknown" if any(None in values for values in (*pa.values(), *pb.values())) else "incompatible"
    absolute_change = change_pct = None
    notes = []
    if status == "compared":
        absolute_change = after["median"] - before["median"]
        if before["median"] != 0:
            change_pct = 100 * absolute_change / before["median"]
        else:
            notes.append("Percentage change is undefined because the baseline median is zero.")
    if context_differences:
        notes.append(
            "Execution context differs or varies; this observed build comparison does not isolate a commit effect."
        )
    for side, evidence in (("Baseline", a), ("Candidate", b)):
        if evidence and not _object(evidence.context.get("source")).get("image_digest"):
            notes.append(f"{side} image digest is not recorded.")
    return {
        "workload_key": None if key.startswith("unknown:") else key,
        "label": _label(key),
        "legs": {"baseline": sorted(left["legs"]) if left else [], "candidate": sorted(right["legs"]) if right else []},
        "status": status,
        "reasons": reasons,
        "baseline": before,
        "candidate": after,
        "formula": formulas,
        "absolute_change": absolute_change,
        "change_pct": change_pct,
        "protocol_differences": protocol_differences,
        "context_differences": context_differences,
        "notes": notes,
    }


def compare_evidence(baseline: Evidence | None, candidate: Evidence) -> dict:
    """Return dynamically discovered workload rows and descriptive FPS deltas."""
    aliases = defaultdict(set)
    for evidence in (baseline, candidate):
        for leg, samples in evidence.samples.items() if evidence else []:
            for item in samples:
                key = workload_key(item["bundle"])
                if key is not None:
                    aliases[leg].add(key)
    left, right = _groups(baseline, aliases), _groups(candidate, aliases)
    notes = [
        "FPS deltas describe these samples; they are not statistical significance, causal attribution or a CI verdict."
    ]
    if baseline is None:
        notes.append("No baseline evidence was selected.")
    for side, evidence in (("Baseline", baseline), ("Candidate", candidate)):
        if evidence:
            notes.extend(f"{side}: {issue}" for issue in evidence.issues)
            definition = _object(evidence.context.get("metric_definition"))
            if definition.get("reason"):
                notes.append(f"{side} FPS definition: {definition['reason']}")
    if not left and not right:
        notes.append("No runtime workloads or leg statuses were present in the selected evidence.")
    return {
        "baseline": baseline.identity if baseline else None,
        "candidate": candidate.identity,
        "rows": [
            _row(key, baseline, candidate, left.get(key), right.get(key)) for key in sorted(left.keys() | right.keys())
        ],
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate the advisory automatic report for this workflow's benchmark evidence."""
    from .report import render_build_comparison

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--run_id", type=int, default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run_attempt", type=int, default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--output_dir", type=Path, default=Path("build-comparison"))
    args = parser.parse_args(argv)
    if not args.repository or not args.run_id or not args.run_attempt or min(args.run_id, args.run_attempt) < 1:
        parser.error("Repository, run ID and report attempt are needed to resolve current benchmark evidence")

    candidate = selected = pinned = None
    selection = {}
    try:
        client = baseline_mod.GitHubClient(args.repository)
        candidate = baseline_mod.resolve_candidate(client, args.run_id, args.run_attempt)
        if candidate.identity.get("event") == "pull_request":
            from .paired import select_pr_baseline

            result = select_pr_baseline(client, candidate)
        else:
            pinned = baseline_mod.load_previous_selection(client, candidate)
            result = baseline_mod.select_baseline(client, candidate, pinned)
        selected, selection = result.evidence, result.metadata
    except baseline_mod.EvidenceError as exc:
        selection = {
            "reason": str(exc),
            "reason_code": exc.code,
            "pinned": pinned is not None,
            "unavailable_evidence": pinned or exc.identity,
            "unavailable_side": "candidate" if candidate is None else "baseline",
        }

    payload = (
        compare_evidence(selected, candidate)
        if candidate is not None
        else {"candidate": None, "baseline": None, "rows": [], "notes": ["Candidate evidence is unavailable."]}
    )
    is_pr = (candidate is not None and candidate.identity.get("event") == "pull_request") or os.environ.get(
        "GITHUB_EVENT_NAME"
    ) == "pull_request"
    payload.update(
        schema_version=1,
        selector_version=1,
        repository=args.repository,
        run_id=args.run_id,
        report_attempt=args.run_attempt,
        selection=selection,
        comparison_mode="paired_pr" if is_pr else "historical",
        candidate_kind="merge" if is_pr else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build-comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "build-comparison.md").write_text(render_build_comparison(payload), encoding="utf-8")
    # The existing aggregate command remains the owner of the CI gate's exit status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
