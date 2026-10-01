# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure FPS comparison tests, independent of evidence retrieval and selection."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import build_compare, cli, compare
from .baseline import Evidence, workload_key
from .build_compare import compare_evidence
from .report import render_aggregate, render_build_comparison
from .test_baseline import HEAD, PARENT, FixtureClient, stamp


def bundle(fps, task="Task", *, seed=42):
    return {
        "run": {
            "task": task,
            "num_envs": 512,
            "status": "completed",
            "seed": seed,
            "config": {"physics_backend": "newton_mjwarp", "rendering_backend": "none", "presets": []},
        },
        "runtime": {
            "total_fps": {"mean": fps},
            "iterations_completed": 200,
            "steps_per_iteration": 512,
            "environment_step_timing": {"warmup_steps": 100, "measurement_mode": "host_return"},
        },
        "hardware": {"hostname": "worker-a", "gpu_devices": [{"name": "GPU"}]},
        "versions": {"warp": "1.0"},
    }


def evidence(samples=None, *, expected=3, formula="aggregate_frames_over_measured_seconds", statuses=None):
    samples = {"leg": [bundle(90), bundle(100), bundle(110)]} if samples is None else samples
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
        measurement_start=None,
        measurement_end=None,
        zip_bytes=b"",
        run={},
    )


class ReportRenderingTests(unittest.TestCase):
    def test_observed_direction_is_explicit_without_a_gate_verdict(self):
        for fps, expected in ((110, "🟢 Improved"), (90, "🔴 Regressed"), (100, "⚪ Unchanged")):
            with self.subTest(fps=fps):
                report = compare_evidence(evidence(), evidence({"leg": [bundle(fps)] * 3}))
                snapshot = copy.deepcopy(report)
                markdown = render_build_comparison(report)
                self.assertIn(expected, markdown)
                self.assertIn("Status | Workload | Baseline FPS | Current FPS | Change %", markdown)
                self.assertNotIn("PASS", markdown)
                self.assertNotIn("FAIL", markdown)
                self.assertEqual(report, snapshot)

    def test_missing_partial_and_incompatible_reasons_remain_visible(self):
        for a, b, expected, reason in (
            (None, evidence(), "No baseline samples", "Baseline:"),
            (evidence(), evidence({"leg": [bundle(100)]}), "Current results incomplete", "Candidate:"),
            (
                evidence(),
                evidence(formula="different"),
                "Cannot compare: measurement setup differs",
                "FPS formula identities differ.",
            ),
        ):
            with self.subTest(expected=expected):
                markdown = render_build_comparison(compare_evidence(a, b))
                self.assertIn(expected, markdown)
                self.assertIn(reason, markdown)
                self.assertIn("**Comparison details:**", markdown)
                self.assertIn("⚪ Not comparable", markdown.split("<details>", 1)[0])

    def test_headline_counts_every_workload_without_reclassifying_rounded_or_missing_deltas(self):
        a = evidence({name: [bundle(100, name)] * 3 for name in ("Higher", "Lower", "Same", "Tiny")})
        b = evidence(
            {
                name: [bundle(fps, name)] * 3
                for name, fps in (("Higher", 120), ("Lower", 80), ("Same", 100), ("Tiny", 100.0001), ("New", 130))
            }
        )
        report = compare_evidence(a, b)
        before = copy.deepcopy(report)
        primary, diagnostics = render_build_comparison(report).split("<details>", 1)
        self.assertIn("**🟢 Improved 2 · 🔴 Regressed 1 · ⚪ Not comparable 1 · ⚪ Unchanged 1**", primary)
        rows = [line for line in primary.splitlines() if line.startswith("|")]
        self.assertEqual(len(rows), 7)  # Header, separator and all five workloads.
        self.assertTrue(all(len(line.split("|")) == 7 for line in rows))
        self.assertIn("⚪ Not comparable", primary)
        self.assertIn("No baseline samples", diagnostics)
        self.assertNotIn("sample_paths", primary)
        self.assertNotIn("source.image_digest", primary)
        self.assertIn("Samples A / B", diagnostics)
        self.assertIn("**FPS samples:**", diagnostics)
        self.assertIn("Workload identities", diagnostics)
        self.assertEqual(report, before)

    def test_zero_baseline_and_no_rows_are_not_misreported(self):
        report = compare_evidence(evidence({"leg": [bundle(0)] * 3}), evidence({"leg": [bundle(10)] * 3}))
        primary = render_build_comparison(report).split("<details>", 1)[0]
        self.assertIn("🟢 Improved 1 · 🔴 Regressed 0 · ⚪ Not comparable 0", primary)
        self.assertIn("N/A (baseline is zero)", primary)
        empty = render_build_comparison({"rows": [], "selection": {"reason": "Current result artifact is missing."}})
        self.assertIn("⚪ Not comparable: no workload results available", empty)
        self.assertIn("Current result artifact is missing", empty.split("<details>", 1)[0])
        self.assertNotIn("Not comparable 0", empty)

    def test_selection_reason_is_visible_when_no_baseline_was_found_without_an_error_code(self):
        report = compare_evidence(None, evidence())
        report["selection"] = {"reason": "No earlier measured ancestor was found."}
        primary = render_build_comparison(report).split("<details>", 1)[0]
        self.assertIn("No earlier measured ancestor was found", primary)
        self.assertIn("| ⚪ Not comparable | leg | — | 100 | — |", primary)

    def test_configuration_labels_fall_back_when_the_same_name_covers_different_workloads(self):
        report = compare_evidence(evidence({"leg": [bundle(100, "Before")] * 3}), evidence())
        primary, diagnostics = render_build_comparison(report).split("<details>", 1)
        self.assertIn("leg (1)", primary)
        self.assertIn("leg (2)", primary)
        self.assertNotIn("newton_mjwarp", primary)
        self.assertIn("Before · newton_mjwarp", diagnostics)
        self.assertIn("Task · newton_mjwarp", diagnostics)
        self.assertIn("No baseline samples", diagnostics)
        self.assertIn("No current samples", diagnostics)

    def test_duplicate_short_labels_do_not_collide_with_existing_numbered_label(self):
        a = evidence({"leg": [bundle(100, "Before")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        b = evidence({"leg": [bundle(100, "After")] * 3, "leg (1)": [bundle(100, "Other")] * 3})
        primary = render_build_comparison(compare_evidence(a, b)).split("<details>", 1)[0]
        labels = [line.split("|")[2].strip() for line in primary.splitlines() if line.startswith("| ⚪")]
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 3)
        self.assertIn("leg (1)", labels)

    def test_paired_pr_labels_exact_sources_and_reused_baseline_inside_details(self):
        report = compare_evidence(evidence(), evidence({"leg": [bundle(110)] * 3}))
        for side, commit, run in (("baseline", PARENT, 12), ("candidate", HEAD, 15)):
            report[side].update(
                repository="isaac-sim/IsaacLab", source_commit=commit, run_id=run, artifact_id=run + 100
            )
        report["candidate"].update(requested_head_commit="c" * 40, event="pull_request")
        report.update(comparison_mode="paired_pr", candidate_kind="merge", report_attempt=1)
        report["selection"] = {
            "reason": "Exact PR base and merge result.",
            "baseline_reused": True,
            "baseline_origin": {**report["baseline"], "run_id": 11},
        }
        snapshot = copy.deepcopy(report)
        primary, details = render_build_comparison(report).split("<details>", 1)
        self.assertIn("### PR performance comparison", primary)
        self.assertIn("| Status | Workload | Baseline FPS | PR FPS | Change % |", primary)
        self.assertIn("| 🟢 Improved | leg | 100 | 110 | +10.00% |", primary)
        self.assertNotIn("historical", details)
        self.assertIn("A — PR base", details)
        self.assertIn("B — PR merge result", details)
        self.assertIn(f"https://github.com/isaac-sim/IsaacLab/commit/{HEAD}", details)
        self.assertIn(f"https://github.com/isaac-sim/IsaacLab/commit/{'c' * 40}", details)
        self.assertIn("https://github.com/isaac-sim/IsaacLab/actions/runs/15/attempts/1", details)
        self.assertIn("https://github.com/isaac-sim/IsaacLab/actions/runs/11/attempts/1", details)
        self.assertIn("Baseline measurements: reused.", details)
        self.assertEqual(report, snapshot)

    def test_paired_baseline_measurement_status_reflects_available_evidence(self):
        for baseline, reused, expected in (
            (None, False, "unavailable."),
            (None, True, "unavailable."),
            (evidence(), False, "run for this PR."),
            (evidence(), True, "reused."),
        ):
            with self.subTest(baseline_available=baseline is not None, reused=reused):
                report = compare_evidence(baseline, evidence())
                report.update(comparison_mode="paired_pr", candidate_kind="merge")
                report["selection"] = {"baseline_reused": reused}
                markdown = render_build_comparison(report)
                self.assertIn(f"Baseline measurements: {expected}", markdown)

    def test_historical_dispatch_is_not_labeled_as_a_pull_request(self):
        report = compare_evidence(evidence(), evidence())
        report["candidate"]["event"] = "workflow_dispatch"
        primary, details = render_build_comparison(report).split("<details>", 1)
        self.assertIn("### Automatic build comparison", primary)
        self.assertNotIn("### PR performance comparison", primary)
        self.assertIn("| Current FPS |", primary)
        self.assertNotIn("PR FPS", primary)
        self.assertIn("A — historical baseline", details)
        self.assertIn("B — current benchmark", details)

    def test_artifact_text_cannot_break_detail_blocks_or_table_columns(self):
        report = compare_evidence(evidence(), evidence())
        label = "leg | </details><script>"
        report["rows"][0]["legs"] = {"baseline": [label], "candidate": [label]}
        report["rows"][0]["notes"].append(label)
        markdown = render_build_comparison(report)
        self.assertNotIn("<script>", markdown)
        self.assertIn("&lt;/details&gt;&lt;script&gt;", markdown)
        self.assertEqual(markdown.count("<details>"), 2)
        self.assertEqual(markdown.count("</details>"), 2)
        primary = markdown.split("<details>", 1)[0]
        self.assertTrue(all(len(line.split("|")) == 7 for line in primary.splitlines() if line.startswith("|")))

    def test_context_details_are_outside_the_numeric_table_and_shared_notes_are_collapsed(self):
        a = evidence({"one": [bundle(100, "One")] * 3, "two": [bundle(200, "Two")] * 3})
        b = evidence({"one": [bundle(110, "One")] * 3, "two": [bundle(210, "Two")] * 3})
        for samples in b.samples.values():
            for sample in samples:
                sample["bundle"]["hardware"]["hostname"] = "worker-b"
        markdown = render_build_comparison(compare_evidence(a, b))
        table_lines = [line for line in markdown.splitlines() if line.startswith("|")]
        self.assertFalse(any("hardware.hostname" in line for line in table_lines))
        self.assertEqual(markdown.count("hardware.hostname"), 1)
        self.assertIn("**All workloads:** hardware.hostname", markdown)
        self.assertIn("does not isolate a commit effect", markdown)

    def test_rolling_gate_exposes_recorded_skip_and_error_causes(self):
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
        self.assertIn("### Rolling-history CI gate: 🚫 ERROR", markdown)
        self.assertIn(missing_credential, markdown)
        self.assertIn("Total FPS: insufficient independent runs for ASV significance testing", markdown)
        self.assertIn("comparison artifact could not be read: invalid JSON", markdown)
        self.assertEqual([report.as_dict() for _, report in reports], snapshot)

    def test_rolling_gate_does_not_invent_an_absent_reason(self):
        markdown = render_aggregate([("unknown", compare.Report(verdict=compare.SKIP))])
        self.assertIn("No reason was recorded in this comparison artifact", markdown)
        self.assertNotIn("credential", markdown)
        self.assertIn("### Rolling-history CI gate: no results", render_aggregate([]))


class BuildComparisonTests(unittest.TestCase):
    def test_complete_fps_samples_use_medians_and_conventional_delta(self):
        a = evidence()
        b = evidence({"leg": [bundle(93), bundle(96), bundle(99)]})
        report = compare_evidence(a, b)
        row = report["rows"][0]
        self.assertEqual(row["workload_key"], workload_key(bundle(100)))
        self.assertEqual(row["baseline"]["samples"], [90, 100, 110])
        self.assertEqual(row["baseline"]["median"], 100)
        self.assertEqual(row["baseline"]["min"], 90)
        self.assertEqual(row["baseline"]["max"], 110)
        self.assertEqual(row["baseline"]["expected_count"], 3)
        self.assertEqual(row["candidate"]["median"], 96)
        self.assertEqual(row["absolute_change"], -4)
        self.assertEqual(row["change_pct"], -4)
        self.assertEqual(row["status"], "compared")
        self.assertNotIn("verdict", row)
        self.assertTrue(any("not statistical significance" in note for note in report["notes"]))

    def test_zero_baseline_does_not_invent_a_percentage(self):
        a = evidence({"leg": [bundle(0)] * 3})
        row = compare_evidence(a, evidence())["rows"][0]
        self.assertEqual(row["absolute_change"], 100)
        self.assertIsNone(row["change_pct"])
        self.assertIn("zero", row["notes"][0])

    def test_dynamic_workloads_include_every_shared_and_missing_identity(self):
        a = evidence({"unfamiliar-leg": [bundle(100, "Shared")] * 3, "removed": [bundle(50, "Removed")] * 3})
        b = evidence({"renamed-leg": [bundle(110, "Shared")] * 3, "new": [bundle(60, "New")] * 3})
        rows = compare_evidence(a, b)["rows"]
        self.assertEqual(len(rows), 3)
        compared = [row for row in rows if row["status"] == "compared"]
        self.assertEqual(len(compared), 1)
        self.assertEqual(compared[0]["change_pct"], 10)
        self.assertEqual(sum(row["status"] == "missing" for row in rows), 2)

    def test_missing_baseline_keeps_candidate_measurements(self):
        report = compare_evidence(None, evidence())
        row = report["rows"][0]
        self.assertIsNone(report["baseline"])
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["candidate"]["median"], 100)
        self.assertEqual(row["baseline"]["samples"], [])
        self.assertIsNone(row["change_pct"])

    def test_status_only_leg_is_associated_without_claiming_identity(self):
        b = evidence({"leg": []}, statuses={"leg": "failed"})
        rows = compare_evidence(evidence(), b)["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "missing")
        self.assertTrue(any("display association" in reason for reason in rows[0]["reasons"]))
        unknown = compare_evidence(b, b)["rows"][0]
        self.assertIsNone(unknown["workload_key"])
        self.assertIn("identity unavailable", unknown["label"])

    def test_incomplete_counts_failed_status_and_runtime_failure_preserve_values(self):
        variants = [
            evidence({"leg": [bundle(100), bundle(110)]}),
            evidence(statuses={"leg": "failed"}),
            evidence(statuses={}),
        ]
        runtime_failed = evidence()
        runtime_failed.samples["leg"][1]["bundle"]["run"]["status"] = "failed"
        variants.append(runtime_failed)
        for b in variants:
            with self.subTest(statuses=b.statuses, count=len(b.samples["leg"])):
                row = compare_evidence(evidence(), b)["rows"][0]
                self.assertEqual(row["status"], "partial")
                self.assertTrue(row["candidate"]["samples"])
                self.assertIsNone(row["absolute_change"])

    def test_invalid_fps_is_not_zero_or_a_valid_repeat(self):
        for value in (None, True, "100", float("nan"), float("inf"), -1):
            with self.subTest(value=value):
                b = evidence({"leg": [bundle(90), bundle(value), bundle(110)]})
                row = compare_evidence(evidence(), b)["rows"][0]
                self.assertEqual(row["candidate"]["samples"], [90, 110])
                self.assertEqual(row["candidate"]["observed_count"], 3)
                self.assertEqual(row["candidate"]["count"], 2)
                self.assertEqual(row["status"], "partial")
                self.assertIsNone(row["change_pct"])

    def test_missing_expected_count_is_explicitly_unknown(self):
        for expected in (None, True, "3", 0):
            with self.subTest(expected=expected):
                row = compare_evidence(evidence(), evidence(expected=expected))["rows"][0]
                self.assertEqual(row["status"], "unknown")
                self.assertEqual(row["candidate"]["count"], 3)
                self.assertIsNone(row["candidate"]["expected_count"])
                self.assertIsNone(row["change_pct"])

    def test_unreadable_sample_issue_does_not_disappear_when_valid_count_matches(self):
        b = evidence()
        b.issues = ["leg/sample-bad/benchmark_runtime.json: runtime bundle is not valid JSON"]
        report = compare_evidence(evidence(), b)
        row = report["rows"][0]
        self.assertEqual(row["candidate"]["count"], 3)
        self.assertEqual(row["status"], "unknown")
        self.assertIsNone(row["change_pct"])
        self.assertIn("Candidate: " + b.issues[0], report["notes"])

    def test_unknown_or_different_fps_formula_is_not_equivalent(self):
        for formula, expected in (
            (None, "unknown"),
            ("unknown", "unknown"),
            ("mean_instantaneous_fps", "incompatible"),
        ):
            with self.subTest(formula=formula):
                row = compare_evidence(evidence(), evidence(formula=formula))["rows"][0]
                self.assertEqual(row["status"], expected)
                self.assertEqual(row["candidate"]["median"], 100)
                self.assertIsNone(row["absolute_change"])

    def test_protocol_changes_and_missing_values_are_explicit(self):
        for field, value in (
            ("seed", 7),
            ("warmup_steps", 20),
            ("measured_steps", 100),
            ("measurement_mode", "synchronized"),
            ("measurement_mode", None),
        ):
            with self.subTest(field=field, value=value):
                b = evidence()
                for sample in b.samples["leg"]:
                    raw = sample["bundle"]
                    if field == "seed":
                        raw["run"]["seed"] = value
                    elif field == "measured_steps":
                        raw["runtime"]["iterations_completed"] = value
                    else:
                        raw["runtime"]["environment_step_timing"][field] = value
                row = compare_evidence(evidence(), b)["rows"][0]
                self.assertEqual(row["status"], "unknown" if value is None else "incompatible")
                self.assertIn(field, [item["field"] for item in row["protocol_differences"]])
                self.assertIsNone(row["change_pct"])

    def test_identical_protocol_variation_on_both_sides_is_not_silently_pooled(self):
        a = evidence({"leg": [bundle(90, seed=1), bundle(100, seed=2), bundle(110, seed=3)]})
        row = compare_evidence(a, copy.deepcopy(a))["rows"][0]
        self.assertEqual(row["status"], "incompatible")
        self.assertEqual(row["protocol_differences"][0]["baseline"], {"sample_values": [1, 2, 3]})
        self.assertIsNone(row["absolute_change"])

    def test_hardware_packages_and_image_qualify_without_causal_verdict(self):
        b = evidence({"leg": [bundle(110)] * 3})
        b.context["source"]["image_digest"] = "sha256:other"
        for sample in b.samples["leg"]:
            sample["bundle"]["hardware"]["hostname"] = "worker-b"
            sample["bundle"]["versions"]["warp"] = "2.0"
        row = compare_evidence(evidence(), b)["rows"][0]
        self.assertEqual(row["status"], "compared")
        self.assertEqual(row["change_pct"], 10)
        fields = {item["field"] for item in row["context_differences"]}
        self.assertTrue({"hardware.hostname", "versions.warp", "source.image_digest"}.issubset(fields))
        self.assertTrue(any("does not isolate a commit effect" in note for note in row["notes"]))

    def test_presets_and_environment_count_use_validated_workload_identity(self):
        for field, value in (("num_envs", 1024), ("presets", ["new_preset"])):
            b = evidence()
            for sample in b.samples["leg"]:
                run = sample["bundle"]["run"]
                (run if field == "num_envs" else run["config"])[field] = value
            rows = compare_evidence(evidence(), b)["rows"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["status"] == "missing" for row in rows))

    def test_historical_missing_context_keeps_observations_without_delta(self):
        a, b = evidence(), evidence()
        a.context = b.context = {}
        a.issues = ["Historical formula unavailable"]
        report = compare_evidence(a, b)
        self.assertEqual(report["rows"][0]["status"], "unknown")
        self.assertEqual(report["rows"][0]["baseline"]["median"], 100)
        self.assertIsNone(report["rows"][0]["change_pct"])
        self.assertIn("Baseline: Historical formula unavailable", report["notes"])


class AutomaticReportTests(unittest.TestCase):
    def setUp(self):
        self.client = FixtureClient()
        for run_id, commit, hour, fps in ((10, PARENT, 8, 100), (20, HEAD, 12, 80)):
            samples = [bundle(fps) for _ in range(3)]
            for sample in samples:
                sample["run"].update(start_time_utc=stamp(hour, 1), end_time_utc=stamp(hour, 5))
            self.client.add(run_id, commit, start=stamp(hour), samples={"leg": samples})

    def run_report(self, output, attempt=1):
        with patch.object(build_compare.baseline_mod, "GitHubClient", return_value=self.client):
            result = build_compare.main(
                [
                    "--repository",
                    self.client.repository,
                    "--run_id",
                    "20",
                    "--run_attempt",
                    str(attempt),
                    "--output_dir",
                    str(output),
                ]
            )
        return (
            result,
            json.loads((output / "build-comparison.json").read_text()),
            (output / "build-comparison.md").read_text(),
        )

    def test_automatic_report_is_advisory_and_preserves_existing_gate_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "comparisons" / "leg"
            gate.mkdir(parents=True)
            (gate / "comparison.json").write_text(json.dumps({"verdict": "FAIL", "label": "leg"}))
            with contextlib.redirect_stdout(io.StringIO()):
                before = cli.main(["aggregate", "--comparison_dir", str(gate.parent)])
                status, result, markdown = self.run_report(root / "build-comparison")
                after = cli.main(["aggregate", "--comparison_dir", str(gate.parent)])
            self.assertEqual((before, after, status), (1, 1, 0))
            self.assertEqual(result["baseline"]["run_id"], 10)
            self.assertEqual(result["candidate"]["run_id"], 20)
            self.assertEqual(result["rows"][0]["change_pct"], -20)
            self.assertIn("-20.00%", markdown)
            self.assertIn("/actions/runs/10/attempts/1", markdown)
            self.assertIn("/actions/runs/20/attempts/1", markdown)
            self.assertIn("3/3 / 3/3", markdown)

    def test_unavailable_baseline_keeps_current_values_and_explicit_reason(self):
        self.client.run_artifacts[10][0]["expired"] = True
        with tempfile.TemporaryDirectory() as directory:
            status, result, markdown = self.run_report(Path(directory))
        self.assertEqual(status, 0)
        self.assertEqual(result["selection"]["reason_code"], "expired")
        self.assertFalse(result["selection"]["pinned"])
        self.assertEqual(result["rows"][0]["candidate"]["median"], 80)
        self.assertIsNone(result["rows"][0]["change_pct"])
        self.assertIn("Comparison unavailable", markdown)
        self.assertIn("expired", markdown)

    def test_new_benchmark_attempt_without_artifact_is_not_old_measurements(self):
        self.client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
        with tempfile.TemporaryDirectory() as directory:
            status, result, markdown = self.run_report(Path(directory), attempt=2)
        self.assertEqual(status, 0)
        self.assertIsNone(result["candidate"])
        self.assertEqual(result["selection"]["reason_code"], "missing_candidate")
        self.assertIn("Candidate evidence is unavailable", markdown)

    def test_report_only_rerun_reuses_producing_attempt_and_pinned_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _, first, _ = self.run_report(output)
            self.client.add_artifact(20, "performance-build-comparison-20-1", {"build-comparison.json": first})
            self.client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
            self.client.run_jobs[20, 2][0]["started_at"] = stamp(12)
            _, second, markdown = self.run_report(output, attempt=2)
        self.assertEqual(second["baseline"], first["baseline"])
        self.assertTrue(second["selection"]["pinned"])
        self.assertEqual(second["candidate"]["run_attempt"], 1)
        self.assertIn("Measurements: attempt 1; report: attempt 2", markdown)

    def test_expired_pinned_baseline_remains_identified_without_forcing_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _, first, _ = self.run_report(output)
            self.client.add_artifact(20, "performance-build-comparison-20-1", {"build-comparison.json": first})
            self.client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
            self.client.run_jobs[20, 2][0]["started_at"] = stamp(12)
            self.client.run_artifacts[10][0]["expired"] = True
            _, result, markdown = self.run_report(output, attempt=2)
            self.client.run_artifacts[20] = [
                item for item in self.client.run_artifacts[20] if item["name"] == "performance-smoke-20-1"
            ]
            self.client.add_artifact(20, "performance-build-comparison-20-2", {"build-comparison.json": result})
            self.client.add(20, HEAD, attempt=3, start=stamp(14), artifact=False)
            self.client.run_jobs[20, 3][0]["started_at"] = stamp(12)
            _, third, third_markdown = self.run_report(output, attempt=3)
        self.assertIsNone(result["baseline"])
        self.assertTrue(result["selection"]["pinned"])
        self.assertEqual(result["selection"]["unavailable_evidence"]["run_id"], 10)
        self.assertIsNone(result["rows"][0]["change_pct"])
        self.assertIn("historical baseline (evidence unavailable)", markdown)
        self.assertIn("/actions/runs/10/attempts/1", markdown)
        self.assertIn("expired", markdown)
        self.assertTrue(third["selection"]["pinned"])
        self.assertEqual(third["selection"]["unavailable_evidence"], result["selection"]["unavailable_evidence"])
        self.assertEqual(third["selection"]["reason_code"], "expired")
        self.assertIn("/actions/runs/10/attempts/1", third_markdown)

    def test_artifact_text_cannot_inject_markdown_rows_or_html(self):
        from .report import render_build_comparison

        report = compare_evidence(evidence(), evidence())
        report["baseline"].update(repository="isaac-sim/IsaacLab", run_url="https://example.invalid/untrusted")
        report["rows"][0]["label"] = "Task | [click](https://example.invalid)\n<script>bad()</script>"
        markdown = render_build_comparison(report)
        self.assertNotIn("<script>", markdown)
        self.assertNotIn("[click]", markdown)
        self.assertIn("&#124;", markdown)
        self.assertIn("&lt;script&gt;", markdown)
        self.assertNotIn("https://example.invalid/untrusted", markdown)
        self.assertIn("https://github.com/isaac-sim/IsaacLab/actions/runs/1/attempts/1", markdown)


if __name__ == "__main__":
    unittest.main()
