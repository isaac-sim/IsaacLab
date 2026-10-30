# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for FPS comparisons, Summary rendering, and report failures."""

import copy
import json
from unittest.mock import patch

from . import build_compare, compare
from .baseline import Evidence
from .build_compare import compare_evidence
from .report import render_aggregate, render_build_comparison
from .test_baseline import HEAD, PARENT, FixtureClient, bundle, stamp


def evidence(samples=None, *, fps=(90, 100, 110), expected=3, formula="measurement-v1:" + "a" * 64, statuses=None):
    if samples is None:
        samples = {"leg": [bundle(value) for value in (fps if isinstance(fps, tuple) else (fps,) * 3)]}
    return Evidence(
        identity={"run_id": 1, "run_attempt": 1, "source_commit": "a" * 40},
        context={
            "source": {"image_digest": "sha256:fixture"},
            "execution": {"expected_samples": expected},
            "metric_definition": {"total_fps": formula},
        },
        samples={
            leg: [
                {"path": f"{leg}/sample-{index}/benchmark_runtime.json", "bundle": value}
                for index, value in enumerate(values, 1)
            ]
            for leg, values in samples.items()
        },
        statuses={leg: "ok" for leg in samples} if statuses is None else statuses,
        issues=[],
        files={},
    )


def set_sample_field(selection, path, value, *, index=None, remove=False):
    """Change or remove one field in all repeats or one selected repeat."""
    samples = selection.samples["leg"]
    for sample in samples if index is None else [samples[index]]:
        target = sample["bundle"]
        *parents, key = path.split(".")
        for parent in parents:
            target = target[parent]
        if remove:
            target.pop(key)
        else:
            target[key] = copy.deepcopy(value)


def assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected


def run_report(client, output, *, attempt=1, comparison_dir=None, gate_only=False):
    arguments = [
        "--repository",
        client.repository,
        "--run_id",
        "20",
        "--run_attempt",
        str(attempt),
        "--output_dir",
        str(output),
    ]
    if comparison_dir is not None:
        arguments += ["--comparison_dir", str(comparison_dir)]
    if gate_only:
        arguments += ["--gate-only"]
    with patch.object(build_compare.baseline_mod, "GitHubClient", return_value=client):
        status = build_compare.main(arguments)
    if gate_only:
        assert not output.exists()
        return (status, None, None)
    return (
        status,
        json.loads((output / "build-comparison.json").read_text()),
        (output / "build-comparison.md").read_text(),
    )


def check_comparisons(subtests):
    with subtests.test(msg="complete fps samples use medians and conventional delta"):
        a = evidence({"leg": [bundle(90), bundle(100), bundle(1100)]})
        b = evidence({"leg": [bundle(93), bundle(96), bundle(999)]})
        report = compare_evidence(a, b)
        row = report["rows"][0]
        assert json.loads(row["workload_key"]) == {
            "task": "Task",
            "physics_backend": "newton_mjwarp",
            "render_backend": None,
            "presets": [],
            "num_envs": 512,
        }
        assert_fields(row["baseline"], samples=[90, 100, 1100], median=100, min=90, max=1100, expected_count=3)
        assert row["candidate"]["median"] == 96
        assert_fields(row, absolute_change=-4, change_pct=-4, status="compared")
        assert "verdict" not in row
        assert any("not statistical significance" in note for note in report["notes"])
    with subtests.test(msg="dynamic workloads include every shared and missing identity"):
        a = evidence({"unfamiliar-leg": [bundle(100, "Shared")] * 3, "removed": [bundle(50, "Removed")] * 3})
        b = evidence({"renamed-leg": [bundle(110, "Shared")] * 3, "new": [bundle(60, "New")] * 3})
        rows = compare_evidence(a, b)["rows"]
        assert len(rows) == 3
        compared = [row for row in rows if row["status"] == "compared"]
        assert len(compared) == 1
        assert compared[0]["change_pct"] == 10
        assert sum(row["status"] == "missing" for row in rows) == 2
    with subtests.test(msg="status only leg is associated without claiming identity"):
        b = evidence({"leg": []}, statuses={"leg": "failed"})
        rows = compare_evidence(evidence(), b)["rows"]
        assert len(rows) == 1
        assert rows[0]["status"] == "missing"
        assert any("display association" in reason for reason in rows[0]["reasons"])
        unknown = compare_evidence(b, b)["rows"][0]
        assert unknown["workload_key"] is None
        assert "identity unavailable" in unknown["label"]
    with subtests.test(msg="incomplete or unknown sample counts and failed runs preserve values"):
        variants = [
            ("partial", evidence({"leg": [bundle(100), bundle(110)]})),
            ("partial", evidence(statuses={"leg": "failed"})),
            ("partial", evidence(statuses={})),
        ]
        runtime_failed = evidence()
        runtime_failed.samples["leg"][1]["bundle"]["run"]["status"] = "failed"
        variants.append(("partial", runtime_failed))
        variants.extend(("unknown", evidence(expected=value)) for value in (None, True, "3", 0))
        for status, b in variants:
            with subtests.test(statuses=b.statuses, execution=b.context["execution"], count=len(b.samples["leg"])):
                report = compare_evidence(evidence(), b)
                row = report["rows"][0]
                assert_fields(row, status=status, absolute_change=None, change_pct=None)
                assert row["candidate"]["samples"]
                if status == "unknown":
                    assert_fields(row["candidate"], count=3, expected_count=None)
                elif len(b.samples["leg"]) != 3:
                    assert "runtime samples; expected 3." in render_build_comparison(report)
    with subtests.test(msg="invalid fps is not zero or a valid repeat"):
        for value in (None, True, "100", float("nan"), float("inf"), -1):
            with subtests.test(value=value):
                b = evidence({"leg": [bundle(90), bundle(value), bundle(110)]})
                row = compare_evidence(evidence(), b)["rows"][0]
                assert_fields(row["candidate"], samples=[90, 110], observed_count=3, count=2)
                assert_fields(row, status="partial", change_pct=None)
    with subtests.test(msg="unknown different and legacy fps formulas are not equivalent"):
        known = "measurement-v1:" + "a" * 64
        legacy = "aggregate_frames_over_measured_seconds"
        for case, baseline_formula, candidate_formula, candidate_fps, expected in (
            ("missing", known, None, 100, "unknown"),
            ("unknown", known, "unknown", 100, "unknown"),
            ("padded_legacy", known, " AGGREGATE_FRAMES_OVER_MEASURED_SECONDS ", 100, "unknown"),
            ("different", known, "mean_instantaneous_fps", 100, "incompatible"),
            ("legacy_baseline", legacy, known, 110, "unknown"),
            ("legacy_candidate", known, legacy, 110, "unknown"),
            ("legacy_both", legacy, legacy, 110, "unknown"),
        ):
            with subtests.test(case=case):
                a = evidence(formula=baseline_formula)
                b = evidence(fps=(90, 100, 110) if candidate_fps == 100 else candidate_fps, formula=candidate_formula)
                report = compare_evidence(a, b)
                row = report["rows"][0]
                assert row["status"] == expected
                assert (row["baseline"]["median"], row["candidate"]["median"]) == (100, candidate_fps)
                if case == "different":
                    assert "FPS formula identities differ." in render_build_comparison(report)
                assert_fields(row, absolute_change=None, change_pct=None)
                if case.startswith("legacy_"):
                    assert "FPS formula identity is unknown for one or both selections." in row["reasons"]
    with subtests.test(msg="protocol changes and missing values are explicit"):
        for field, path, value in (
            ("seed", "run.seed", 7),
            ("warmup_steps", "runtime.environment_step_timing.warmup_steps", 20),
            ("measured_steps", "runtime.iterations_completed", 100),
            ("steps_per_iteration", "runtime.steps_per_iteration", 256),
            ("measurement_mode", "runtime.environment_step_timing.measurement_mode", "synchronized"),
            ("measurement_mode", "runtime.environment_step_timing.measurement_mode", None),
        ):
            with subtests.test(field=field, value=value):
                b = evidence()
                set_sample_field(b, path, value)
                row = compare_evidence(evidence(), b)["rows"][0]
                assert row["status"] == ("unknown" if value is None else "incompatible")
                assert field in [item["field"] for item in row["protocol_differences"]]
                assert row["change_pct"] is None
    with subtests.test(msg="identical protocol variation on both sides is not silently pooled"):
        a = evidence({"leg": [bundle(90, seed=1), bundle(100, seed=2), bundle(110, seed=3)]})
        row = compare_evidence(a, copy.deepcopy(a))["rows"][0]
        assert row["status"] == "incompatible"
        assert row["protocol_differences"][0]["baseline"] == {"sample_values": [1, 2, 3]}
        assert row["absolute_change"] is None
    with subtests.test(msg="hardware packages and image qualify without causal verdict"):
        for image, device_metadata in (
            ("sha256:other", {}),
            ("sha256:fixture", {"uuid": "different-device", "index": 7, "driver_version": "other-driver"}),
        ):
            with subtests.test(image=image, device_metadata=device_metadata):
                b = evidence(fps=110)
                b.context["source"]["image_digest"] = image
                for sample in b.samples["leg"]:
                    hardware = sample["bundle"]["hardware"]
                    hardware["hostname"] = "worker-b"
                    hardware["gpu_devices"][0].update(device_metadata)
                    sample["bundle"]["versions"]["warp"] = "2.0"
                report = compare_evidence(evidence(), b)
                row = report["rows"][0]
                assert_fields(row, status="compared", change_pct=10)
                assert row["context_differences"]
                fields = {item["field"] for item in row["context_differences"]}
                expected = {"hardware.hostname", "versions.warp"}
                if image != "sha256:fixture":
                    expected.add("source.image_digest")
                assert expected.issubset(fields)
                assert any("does not isolate a commit effect" in note for note in row["notes"])
                assert "does not isolate a commit effect" in render_build_comparison(report)
    with subtests.test(msg="changed or unavailable hardware preserves fps without comparing them"):
        cases = (
            ("incompatible", "candidate", "cpu_name", "AMD EPYC 9554", None),
            ("incompatible", "candidate", "cpu_count", 32, None),
            ("incompatible", "candidate", "gpu_devices", [{"name": "NVIDIA H100"}], None),
            (
                "incompatible",
                "candidate",
                "gpu_devices",
                [{"name": "NVIDIA RTX PRO 4500 Blackwell Server Edition"}] * 2,
                None,
            ),
            ("unknown", "both", None, None, None),
            ("unknown", "baseline", "cpu_name", None, None),
            ("unknown", "candidate", "cpu_count", None, None),
            ("unknown", "both", "gpu_devices", None, None),
            ("unknown", "candidate", "cpu_name", " unknown ", None),
            ("unknown", "baseline", "cpu_name", 7, None),
            ("unknown", "baseline", "cpu_count", True, None),
            ("unknown", "candidate", "cpu_count", 0, None),
            ("unknown", "candidate", "gpu_devices", [], None),
            ("unknown", "baseline", "gpu_devices", "GPU", None),
            ("unknown", "both", "gpu_devices", [{"name": "unknown"}], None),
            ("unknown", "candidate", "gpu_devices", [None], None),
            ("unknown", "baseline", "cpu_count", None, 1),
            ("unknown", "candidate", "gpu_devices", [], 1),
        )
        for status, side, field, value, index in cases:
            with subtests.test(status=status, side=side, field=field, value=value, index=index):
                a, b = (evidence(), evidence(fps=110))
                for selection in (a, b) if side == "both" else (a if side == "baseline" else b,):
                    set_sample_field(
                        selection, "hardware" + (f".{field}" if field else ""), value, index=index, remove=value is None
                    )
                report = compare_evidence(a, b)
                row = report["rows"][0]
                assert row["status"] == status
                assert (row["baseline"]["median"], row["candidate"]["median"]) == (100, 110)
                assert_fields(row, absolute_change=None, change_pct=None)
                primary, details = render_build_comparison(report).split("<details>", 1)
                assert "| ⚪ Not comparable | leg | 100 | 110 | — |" in primary
                if status == "incompatible":
                    assert "CPU/GPU hardware differs or varies within a selection." in row["reasons"]
                    assert "+10.00%" not in primary
                    assert "CPU/GPU hardware differs" in details
                else:
                    assert any("hardware" in reason for reason in row["reasons"])
    with subtests.test(msg="hardware variation and heterogeneous gpu order are not pooled"):
        a = evidence()
        set_sample_field(a, "hardware.cpu_count", 32, index=1)
        row = compare_evidence(a, copy.deepcopy(a))["rows"][0]
        assert_fields(row, status="incompatible", change_pct=None)
        a, b = (evidence(), evidence())
        for selection, models in ((a, ("A100", "H100")), (b, ("H100", "A100"))):
            set_sample_field(selection, "hardware.gpu_devices", [{"name": model} for model in models])
        row = compare_evidence(a, b)["rows"][0]
        assert_fields(row, status="incompatible", change_pct=None)
    with subtests.test(msg="unavailable hardware preserves missing and partial sample status"):
        for missing in (True, False):
            with subtests.test(missing=missing):
                b = evidence({"leg": [] if missing else [bundle(110)]})
                set_sample_field(b, "hardware", None, remove=True)
                row = compare_evidence(evidence(), b)["rows"][0]
                assert row["status"] == ("missing" if missing else "partial")
                assert row["candidate"]["median"] == (None if missing else 110)
                assert row["change_pct"] is None
    with subtests.test(msg="presets and environment count use validated workload identity"):
        for path, value in (("run.num_envs", 1024), ("run.config.presets", ["new_preset"])):
            b = evidence()
            set_sample_field(b, path, value)
            rows = compare_evidence(evidence(), b)["rows"]
            assert len(rows) == 2
            assert all(row["status"] == "missing" for row in rows)
    with subtests.test(msg="solver and coupling changes or variation preserve fps without comparing workloads"):
        config = {"physics_solvers": ["newton_mjwarp", "newton_vbd"], "physics_coupling": "proxy"}
        for field, value, index in (
            ("physics_solvers", ["newton_mjwarp"], None),
            ("physics_solvers", None, None),
            ("physics_solvers", ["newton_vbd"], None),
            ("physics_coupling", "admm", None),
            ("physics_coupling", None, None),
            ("physics_coupling", "admm", 1),
        ):
            with subtests.test(field=field, value=value, index=index):
                a, b = (evidence(), evidence(fps=110))
                for selection in (a, b):
                    for key, setting in config.items():
                        set_sample_field(selection, f"run.config.{key}", setting)
                set_sample_field(b, f"run.config.{field}", value, index=index, remove=value is None)
                report = compare_evidence(a, b)
                rows = report["rows"]
                assert all(row["change_pct"] is None for row in rows)
                for side in ("baseline", "candidate"):
                    assert sum(row[side]["count"] for row in rows) == 3
                primary, details = render_build_comparison(report).split("<details>", 1)
                assert "⚪ Not comparable" in primary
                assert "+10.00%" not in primary
                assert "solvers: newton_mjwarp, newton_vbd · coupling: proxy" in details
                if index is not None:
                    assert len(rows) == 2
                    shared = next(row for row in rows if row["baseline"]["count"])
                    assert shared["status"] == "unknown"
                    assert any("different workload identities" in reason for reason in shared["reasons"])
    check_rendering(subtests)


def check_rendering(subtests):
    with subtests.test(msg="headline counts and escaped artifact text preserve tables and sources"):
        a = evidence({name: [bundle(100, name)] * 3 for name in ("Higher", "Lower", "Same", "Tiny")})
        b = evidence(
            {
                name: [bundle(fps, name)] * 3
                for name, fps in (("Higher", 120), ("Lower", 80), ("Same", 100), ("Tiny", 100.0001), ("New", 130))
            }
        )
        report = compare_evidence(a, b)
        report["baseline"].update(repository="isaac-sim/IsaacLab", run_url="https://example.invalid/untrusted")
        row = report["rows"][0]
        unsafe_leg = "leg | </details><script>"
        row["legs"] = {"baseline": [unsafe_leg], "candidate": [unsafe_leg]}
        row["label"] = "Task | [click](https://example.invalid)\n<script>bad()</script>"
        row["notes"].append(unsafe_leg)
        before = copy.deepcopy(report)
        markdown = render_build_comparison(report)
        primary, diagnostics = markdown.split("<details>", 1)
        assert "**🟢 Improved 2 · 🔴 Regressed 1 · ⚪ Not comparable 1 · ⚪ Unchanged 1**" in primary
        assert "Status | Workload | Baseline FPS | PR FPS | Change %" in primary
        assert "PASS" not in markdown
        assert "FAIL" not in markdown
        rows = [line for line in primary.splitlines() if line.startswith("|")]
        assert len(rows) == 7
        assert all(len(line.split("|")) == 7 for line in rows)
        assert "Baseline: No runtime samples identify this workload." in diagnostics
        assert "samples A / B (valid/expected)" in diagnostics
        assert "3/3 / 3/3" in diagnostics
        assert "build-comparison.json" in diagnostics
        assert report == before
        assert "<script>" not in markdown
        assert "[click]" not in markdown
        assert "&#91;click&#93;" in diagnostics
        assert "&#124;" in markdown
        assert "&lt;/details&gt;&lt;script&gt;" in markdown
        assert "&lt;script&gt;" in markdown
        assert markdown.count("<details>") == 1
        assert markdown.count("</details>") == 1
        assert "https://example.invalid/untrusted" not in markdown
        assert "https://github.com/isaac-sim/IsaacLab/actions/runs/1/attempts/1" in markdown
    with subtests.test(msg="zero baseline and no rows are not misreported"):
        report = compare_evidence(evidence({"leg": [bundle(0)] * 3}), evidence({"leg": [bundle(10)] * 3}))
        row = report["rows"][0]
        assert_fields(row, absolute_change=10, change_pct=None)
        assert "zero" in row["notes"][0]
        primary = render_build_comparison(report).split("<details>", 1)[0]
        assert "🟢 Improved 1 · 🔴 Regressed 0 · ⚪ Not comparable 0" in primary
        assert "N/A (baseline is zero)" in primary
        empty = render_build_comparison({"rows": [], "selection": {"reason": "Current result artifact is missing."}})
        assert "⚪ Not comparable: no workload results available" in empty
        assert "Current result artifact is missing" in empty.split("<details>", 1)[0]
        assert "Not comparable 0" not in empty
    with subtests.test(msg="selection reason is visible when no baseline was found without an error code"):
        report = compare_evidence(None, evidence())
        report["selection"] = {"reason": "No verified baseline was available."}
        primary = render_build_comparison(report).split("<details>", 1)[0]
        assert "No verified baseline was available" in primary
        assert "| ⚪ Not comparable | leg | — | 100 | — |" in primary
    with subtests.test(msg="duplicate short labels do not collide with existing numbered label"):
        a = evidence({"leg": [bundle(100, "Before")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        b = evidence({"leg": [bundle(100, "Task")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        primary, diagnostics = render_build_comparison(compare_evidence(a, b)).split("<details>", 1)
        labels = [line.split("|")[2].strip() for line in primary.splitlines() if line.startswith("| ⚪")]
        assert len(labels) == 3
        assert len(set(labels)) == 3
        assert "leg (1)" in labels
        assert "leg (2)" in labels
        assert "newton_mjwarp" not in primary
        assert "Before · newton_mjwarp" in diagnostics
        assert "Task · newton_mjwarp" in diagnostics
        for side in ("Baseline", "Candidate"):
            assert f"{side}: No runtime samples identify this workload." in diagnostics
    with subtests.test(msg="paired pr labels exact sources and reused baseline inside details"):
        a, b = (evidence(), evidence(fps=110))
        a.context["measurement"] = {"commit": "d" * 40, "digest": "e" * 64}
        b.context["measurement"] = {"commit": HEAD, "digest": "e" * 64}
        report = compare_evidence(a, b)
        for side, commit, run in (("baseline", PARENT, 12), ("candidate", HEAD, 15)):
            report[side].update(
                repository="isaac-sim/IsaacLab", source_commit=commit, run_id=run, artifact_id=run + 100
            )
        report["candidate"].update(requested_head_commit="c" * 40, event="pull_request")
        report.update(comparison_mode="paired_pr", candidate_kind="merge", report_attempt=1)
        report["source_context"] = {"reference_commit": PARENT, "event_base_commit": HEAD}
        report["selection"] = {
            "reason": "Exact PR base and merge result.",
            "baseline_reused": True,
            "baseline_origin": report["baseline"],
        }
        primary, details = render_build_comparison(report).split("<details>", 1)
        assert "| 🟢 Improved | leg | 100 | 110 | +10.00% |" in primary
        assert "A — PR base" in details
        assert "B — PR merge result" in details
        for commit in (HEAD, PARENT, "c" * 40, "d" * 40):
            assert f"https://github.com/isaac-sim/IsaacLab/commit/{commit}" in details
            assert commit not in primary
        for run in (12, 15):
            assert details.count(f"https://github.com/isaac-sim/IsaacLab/actions/runs/{run}/attempts/1") == 1
            assert f"https://github.com/isaac-sim/IsaacLab/actions/runs/{run}/artifacts/{run + 100}" in details
        assert "Baseline measurements: reused." in details
        assert f"Resolved PR base: `{PARENT}`" in details
        assert f"The PR event reported `{HEAD}`" in details
        assert "Measurement sources (commit · SHA-256)" in details
        assert "e" * 64 in details
    with subtests.test(msg="paired baseline measurement status reflects available evidence"):
        for baseline, reused, expected in (
            (None, False, "unavailable."),
            (None, True, "unavailable."),
            (evidence(), False, "run for this PR."),
        ):
            with subtests.test(baseline_available=baseline is not None, reused=reused):
                report = compare_evidence(baseline, evidence())
                report.update(comparison_mode="paired_pr", candidate_kind="merge")
                report["selection"] = {"baseline_reused": reused}
                markdown = render_build_comparison(report)
                assert f"Baseline measurements: {expected}" in markdown
    with subtests.test(msg="rolling gate exposes recorded skip and error causes"):
        missing_credential = "No baseline store credential is available for this run"
        reports = [
            (
                "cartpole-newton",
                compare.Report(
                    verdict=compare.SKIP,
                    message=missing_credential,
                    metrics=(compare.MetricResult("total_fps", "Total FPS", 1801565.067428147),),
                ),
            ),
            (
                "partial",
                compare.Report(
                    verdict=compare.SKIP,
                    message="No gating metric could be compared",
                    metrics=(
                        compare.MetricResult(
                            "total_fps",
                            "Total FPS",
                            100,
                            note="insufficient independent runs for ASV significance testing",
                        ),
                    ),
                ),
            ),
            ("failed", compare.errored("comparison artifact could not be read: invalid JSON")),
        ]
        snapshot = [report.as_dict() for _, report in reports]
        markdown = render_aggregate(reports)
        assert "### Rolling-history CI gate: 🚫 ERROR" in markdown
        assert "<details>" not in markdown
        assert "| Combination | Total FPS | Baseline | FPS regression %" in markdown
        assert f"- **cartpole-newton — SKIP:** {missing_credential}" in markdown
        assert "Total FPS: insufficient independent runs for ASV significance testing" in markdown
        assert "- **failed — ERROR:** comparison artifact could not be read: invalid JSON" in markdown
        assert [report.as_dict() for _, report in reports] == snapshot
        mixed = reports[1:2] + [
            (name, compare.Report(verdict=compare.PASS, message="No performance regression detected"))
            for name in ("g1", "camera-newton", "camera-ovphysx")
        ]
        markdown = render_aggregate(mixed)
        introduction = markdown.split("| Combination |", 1)[0]
        assert "### Rolling-history CI gate: ✅ PASS" in introduction
        assert "Workload checks: 3 passed, 1 skipped." in introduction
        assert "No gating metric could be compared" not in introduction
        assert "No performance regression detected" not in introduction
        assert "- **partial — SKIP:** No gating metric could be compared" in markdown
    with subtests.test(msg="rolling gate does not invent an absent reason"):
        markdown = render_aggregate([("unknown", compare.Report(verdict=compare.SKIP))])
        assert "No reason was recorded in this comparison artifact" in markdown
        assert "credential" not in markdown
        assert "### Rolling-history CI gate: no results" in render_aggregate([])


def report_client():
    client = FixtureClient()
    samples = [bundle(80, start=stamp(12, 1), end=stamp(12, 5)) for _ in range(3)]
    client.add(20, HEAD, start=stamp(12), samples={"leg": samples}, event="pull_request")
    return client


def check_reports(tmp_path, subtests):
    with subtests.test(msg="capture failure explains missing results before pin identity error"):
        client = report_client()
        client.run_artifacts[20] = []
        reason = "The tested merge checkout has a different first parent than the requested base."
        failure = {
            "schema_version": 1,
            "stage": "capture_current",
            "reason": reason,
            "source": {
                "commit": "d" * 40,
                "intended_commit": "d" * 40,
                "reference_commit": PARENT,
                "event_base_commit": "c" * 40,
                "requested_head_commit": HEAD,
                "commit_parents": [PARENT, HEAD],
            },
            "execution": {"run_id": 20, "run_attempt": 1},
        }
        client.add_artifact(
            20,
            "performance-smoke-20-1",
            {
                "paired-failure.json": failure,
                "pr-comparison.json": {"reference_commit": PARENT, "tested_commit": "d" * 40},
                "leg/status": "error",
            },
        )
        status, result, markdown = run_report(
            client, tmp_path / "test_capture_failure_explains_missing_results_before_pin_identity_error"
        )
        primary, details = markdown.split("<details>", 1)
        assert status == 0
        assert_fields(result["selection"], reason=reason, paired_failure=failure)
        assert reason in primary
        assert "pinned comparison" not in primary
        assert "| Status |" not in primary
        assert "| ⚪ Not comparable |" not in primary
        assert primary.count(reason) == 1
        assert "⚪ Not comparable 1" in primary
        assert len(result["rows"]) == 1
        assert result["rows"][0]["change_pct"] is None
        assert result["candidate_kind"] is None
        assert "PR merge result" not in details
        assert "PR revision (not verified)" in details
        for commit in (HEAD, PARENT, "c" * 40, "d" * 40):
            assert commit in details
        assert "Actual checkout" in details
        assert "Intended checkout" in details
    with subtests.test(msg="baseline failure keeps available current fps and partial workloads"):
        client = report_client()
        candidate = build_compare.baseline_mod.resolve_candidate(client, 20, 1)
        files = candidate.files
        reason = "The base dependency image could not be pulled; baseline samples were not produced."
        files["paired-failure.json"] = json.dumps(
            {
                "schema_version": 1,
                "stage": "capture_baseline",
                "reason": reason,
                "source": {"commit": PARENT, "intended_commit": PARENT},
                "execution": {"run_id": 20, "run_attempt": 1},
            }
        ).encode()
        files["another-workload/status"] = b"error"
        client.run_artifacts[20] = []
        client.add_artifact(20, "performance-smoke-20-1", files)
        status, result, markdown = run_report(
            client, tmp_path / "test_baseline_failure_keeps_available_current_fps_and_partial_workloads"
        )
        primary = markdown.split("<details>", 1)[0]
        assert status == 0
        assert reason in primary
        assert "| Status | Workload | Baseline FPS | PR FPS | Change % |" in primary
        assert len(result["rows"]) == 2
        assert {row["candidate"]["median"] for row in result["rows"]} == {80, None}
        assert all(row["change_pct"] is None for row in result["rows"])
        assert "another-workload" in primary
    with subtests.test(msg="new benchmark attempt without artifact is not old measurements"):
        client = report_client()
        client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
        status, result, markdown = run_report(
            client, tmp_path / "test_new_benchmark_attempt_without_artifact_is_not_old_measurements", attempt=2
        )
        assert status == 0
        assert result["candidate"] is None
        assert result["selection"]["reason_code"] == "missing_candidate"
        assert "Candidate evidence is unavailable" in markdown
