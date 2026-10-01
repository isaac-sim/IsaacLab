# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paired PR evidence reuse and selection using immutable local source fixtures."""

import hashlib
import json
import marshal
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

from . import baseline, paired, source_revision
from .test_baseline import HEAD, REPO, FixtureClient, bundle, stamp

RUNTIME = "source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py"


def encoded(value):
    return json.dumps(value).encode()


class PairedFixtureClient(FixtureClient):
    def __init__(self):
        super().__init__()
        self.pages_requested = []

    def paginate(self, path, key, **params):
        self.pages_requested.append((path, key, params))
        if key == "artifacts":
            return [item for items in self.run_artifacts.values() for item in items]
        if key == "workflow_runs":
            latest = {}
            for (run_id, attempt), run in self.attempts.items():
                if attempt >= latest.get(run_id, {}).get("run_attempt", 0):
                    latest[run_id] = run
            return list(latest.values())
        raise AssertionError(f"Unexpected fixture API request: {path}, {key}, {params}")

    def get(self, path):
        artifact_id = int(path.rsplit("/", 1)[1])
        matches = [item for items in self.run_artifacts.values() for item in items if item["id"] == artifact_id]
        if not matches:
            raise baseline.EvidenceError("inaccessible", "Fixture artifact is unavailable")
        return matches[0]


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.checkout = self.root / "base checkout"
        self.checkout.mkdir()
        self._git("init", "--quiet")
        source = self.checkout / RUNTIME
        source.parent.mkdir(parents=True)
        source.write_text('def run(argv):\n    """Selected base runtime."""\n    return argv\n')
        (self.checkout / "source/isaaclab/isaaclab/__init__.py").write_text("")
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Base"
        )
        self.commit = self._git("rev-parse", "HEAD")
        self.manifest = source_revision.prepare_manifest(self.checkout)
        self.client = PairedFixtureClient()
        self.output = self.root / "baseline output"
        self.selection_path = self.root / "pr-comparison.json"
        self.event = {
            "number": 42,
            "pull_request": {
                "number": 42,
                "base": {"sha": self.commit, "ref": "develop", "repo": {"full_name": REPO}},
                "head": {"sha": HEAD, "ref": "feature", "repo": {"full_name": REPO}},
            },
        }

    def _git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.checkout), *args], text=True).strip()

    def _files(self, run_id=10, attempt=1):
        files = {
            "source-manifest.json": encoded(self.manifest),
            "build-context.json": encoded(
                {
                    "source": {
                        "commit": self.commit,
                        "requested_head_commit": HEAD,
                        "reference_commit": self.commit,
                        "reference_branch": "develop",
                        "event_base_commit": self.commit,
                        "pull_request_number": 42,
                        "benchmark_role": "baseline",
                    },
                    "execution": {
                        "run_id": run_id,
                        "run_attempt": attempt,
                        "expected_samples": 3,
                        "expected_legs": ["first", "second"],
                        "measurement_not_before": stamp(8),
                        "job": "performance-smoke-benchmarks",
                        "runner_name": "fixture-gpu-runner",
                        "hostname": "fixture-gpu-host",
                    },
                    "metric_definition": {"total_fps": "aggregate_frames_over_measured_seconds"},
                }
            ),
        }
        runtime_code = next(
            value
            for value in compile((self.checkout / RUNTIME).read_bytes(), RUNTIME, "exec").co_consts
            if isinstance(value, types.CodeType) and value.co_name == "run"
        )
        modules = [
            {
                "name": "isaaclab.benchmark.entrypoints.runtime" if path == RUNTIME else "isaaclab",
                "path": "/workspace/isaaclab/" + path,
                "relative_path": path,
                "sha256": digest,
                "expected_sha256": digest,
                "status": "verified",
            }
            for path, digest in self.manifest["files"].items()
        ]
        for leg in ("first", "second"):
            files[f"{leg}/status"] = b"ok"
            for sample in range(1, 4):
                path = f"{leg}/sample-{sample}/benchmark_runtime_fixture.json"
                data = encoded(bundle(start=stamp(9, sample), end=stamp(9, sample + 1), task=leg))
                files[path] = data
                files[f"{leg}/sample-{sample}/source-revision.json"] = encoded(
                    {
                        "schema_version": 1,
                        "status": "verified",
                        "bytecode_policy": "fresh_process_cache",
                        "commit": self.commit,
                        "pid": 123,
                        "benchmark_exit_code": 0,
                        "exception": None,
                        "modules": modules,
                        "runtime_entrypoint": {
                            "module": "isaaclab.benchmark.entrypoints.runtime",
                            "function": "run",
                            "code_sha256": hashlib.sha256(marshal.dumps(runtime_code)).hexdigest(),
                            "doc_sha256": hashlib.sha256(b"Selected base runtime.").hexdigest(),
                            "source_code_matches": True,
                        },
                        "outputs": [
                            {
                                "path": "benchmark_runtime_fixture.json",
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "bytes": len(data),
                            }
                        ],
                        "mismatches": [],
                    }
                )
        return files

    def _evidence(self, files):
        run = self.client.add(10, HEAD, event="pull_request", branch="feature", artifact=False)
        artifact = self.client.add_artifact(10, "source-fixture", files)
        return baseline.read_evidence(self.client, run, artifact, 1, artifact_name=artifact["name"])

    def test_source_proof_matches_immutable_manifest_and_each_result(self):
        files = self._files()
        self.assertEqual(paired.source_issues(self._evidence(files), files, self.commit), [])

    def test_source_proof_detects_runtime_bytes_changed_after_verification(self):
        files = self._files()
        path = "first/sample-1/benchmark_runtime_fixture.json"
        changed = json.loads(files[path])
        changed["runtime"]["total_fps"]["mean"] = 900
        files[path] = encoded(changed)
        issues = paired.source_issues(self._evidence(files), files, self.commit)
        self.assertTrue(issues)
        self.assertIn("first/sample-1", str(issues))

    def test_source_proof_missing_for_one_sample_is_reported(self):
        files = self._files()
        del files["second/sample-3/source-revision.json"]
        issues = paired.source_issues(self._evidence(files), files, self.commit)
        self.assertTrue(issues)
        self.assertIn("second/sample-3", str(issues))

    def test_loaded_source_hash_cannot_disagree_with_selected_commit(self):
        files = self._files()
        path = "first/sample-1/source-revision.json"
        sidecar = json.loads(files[path])
        sidecar["modules"][0]["sha256"] = "0" * 64
        files[path] = encoded(sidecar)
        self.assertTrue(paired.source_issues(self._evidence(files), files, self.commit))

    def _add_baseline(self, files=None, *, run_id=10, attempt=1):
        self.client.add(run_id, HEAD, event="pull_request", branch="feature", attempt=attempt, artifact=False)
        return self.client.add_artifact(
            run_id,
            paired.baseline_name(42, self.commit, attempt),
            self._files(run_id, attempt) if files is None else files,
        )

    def _restore(self, *, run_id=20, attempt=1):
        return paired.restore_baseline(
            self.client, self.checkout, self.output, self.selection_path, self.event, run_id, attempt
        )

    def test_restore_unchanged_base_for_new_pr_head_copies_exact_results(self):
        files = self._files()
        artifact = self._add_baseline(files)
        self.event["pull_request"]["head"]["sha"] = "d" * 40
        selection = self._restore()
        self.assertTrue(selection["baseline_reused"])
        self.assertEqual(selection["requested_head_commit"], "d" * 40)
        self.assertEqual(selection["baseline_origin"]["artifact_id"], artifact["id"])
        self.assertEqual(selection["baseline_origin"]["source_commit"], self.commit)
        self.assertEqual(json.loads(self.selection_path.read_text()), selection)
        for name, data in files.items():
            self.assertEqual((self.output / name).read_bytes(), data)

    def test_changed_base_commit_requests_fresh_measurement(self):
        self._add_baseline()
        source = self.checkout / RUNTIME
        source.write_text(source.read_text().replace("Selected base", "Changed base"))
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Next"
        )
        new_commit = self._git("rev-parse", "HEAD")
        self.event["pull_request"]["base"]["sha"] = new_commit
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertEqual(selection["reference_commit"], new_commit)
        self.assertEqual(self.client.downloaded, [])

    def test_different_pr_does_not_reuse_another_pr_baseline(self):
        self._add_baseline()
        self.event["number"] = self.event["pull_request"]["number"] = 43
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertEqual(self.client.downloaded, [])

    def test_current_measurements_are_never_restored_as_baseline(self):
        files = self._files()
        context = json.loads(files["build-context.json"])
        context["source"]["benchmark_role"] = "current"
        files["build-context.json"] = encoded(context)
        self.client.add(10, HEAD, event="pull_request", artifact=False)
        self.client.add_artifact(10, "performance-smoke-10-1", files)
        self.assertFalse(self._restore()["baseline_reused"])
        self.assertEqual(self.client.downloaded, [])

    def test_expired_baseline_requests_fresh_measurement(self):
        artifact = self._add_baseline()
        artifact["expired"] = True
        self.assertFalse(self._restore()["baseline_reused"])
        self.assertFalse(self.output.exists())

    def test_missing_or_tampered_proof_requests_fresh_measurement(self):
        for mutation in ("missing", "hash"):
            with self.subTest(mutation=mutation):
                self.client = PairedFixtureClient()
                files = self._files()
                path = "first/sample-1/source-revision.json"
                if mutation == "missing":
                    del files[path]
                else:
                    proof = json.loads(files[path])
                    proof["outputs"][0]["sha256"] = "0" * 64
                    files[path] = encoded(proof)
                self._add_baseline(files)
                selection = self._restore()
                self.assertFalse(selection["baseline_reused"])
                self.assertTrue(selection["issues"])
                self.assertFalse(self.output.exists())

    def test_incomplete_or_failed_workload_requests_fresh_measurement(self):
        for mutation in ("missing-sample", "failed-leg"):
            with self.subTest(mutation=mutation):
                self.client = PairedFixtureClient()
                files = self._files()
                if mutation == "missing-sample":
                    del files["second/sample-3/benchmark_runtime_fixture.json"]
                else:
                    files["second/status"] = b"failed"
                self._add_baseline(files)
                self.assertFalse(self._restore()["baseline_reused"])

    def test_same_attempt_does_not_restore_its_own_results(self):
        self._add_baseline(run_id=20)
        self.assertFalse(self._restore()["baseline_reused"])

    def test_entire_missing_workload_cannot_appear_complete(self):
        files = {name: data for name, data in self._files().items() if not name.startswith("second/")}
        self._add_baseline(files)
        selection = self._restore()
        self.assertFalse(selection["baseline_reused"])
        self.assertTrue(selection["issues"])
        self.assertFalse(self.output.exists())

    def test_invalid_first_attempt_is_remeasured_then_second_attempt_is_reused(self):
        incomplete = self._files()
        del incomplete["second/sample-3/source-revision.json"]
        first = self._add_baseline(incomplete)
        first_bytes = self.client.contents[first["id"]]
        self.assertFalse(self._restore(run_id=10, attempt=2)["baseline_reused"])
        second = self._add_baseline(run_id=10, attempt=2)
        self.assertNotEqual(first["name"], second["name"])
        selection = self._restore(run_id=10, attempt=3)
        self.assertTrue(selection["baseline_reused"])
        self.assertEqual(selection["baseline_origin"]["artifact_id"], second["id"])
        self.assertEqual(selection["baseline_origin"]["run_attempt"], 2)
        self.assertEqual(self.client.contents[first["id"]], first_bytes)

    def _candidate(self, origin, *, reused=False, hostname="fixture-gpu-host", start=stamp(10)):
        files = self._files(run_id=20)
        manifest = json.loads(files["source-manifest.json"])
        manifest["commit"] = HEAD
        files["source-manifest.json"] = encoded(manifest)
        context = json.loads(files["build-context.json"])
        context["source"].update(commit=HEAD, benchmark_role="current")
        context["execution"].update(hostname=hostname, measurement_not_before=start)
        files["build-context.json"] = encoded(context)
        for name in list(files):
            if name.endswith("source-revision.json"):
                proof = json.loads(files[name])
                proof["commit"] = HEAD
                path = name.rsplit("/", 1)[0] + "/benchmark_runtime_fixture.json"
                data = json.loads(files[path])
                data["run"].update(start_time_utc=start, end_time_utc=stamp(11))
                files[path] = encoded(data)
                proof["outputs"][0].update(sha256=hashlib.sha256(files[path]).hexdigest(), bytes=len(files[path]))
                files[name] = encoded(proof)
        files["pr-comparison.json"] = encoded(
            {
                "schema_version": 1,
                "comparison_mode": "paired_pr",
                "pull_request_number": 42,
                "reference_commit": self.commit,
                "requested_head_commit": HEAD,
                "tested_commit": HEAD,
                "baseline_reused": reused,
                "baseline_origin": origin,
            }
        )
        run = self.client.add(20, HEAD, event="pull_request", branch="feature", artifact=False)
        artifact = self.client.add_artifact(20, "performance-smoke-20-1", files)
        return baseline.read_evidence(self.client, run, artifact, 1)

    def _origin(self, artifact, *, run_id=20):
        return {
            "repository": REPO,
            "artifact_id": artifact["id"],
            "run_id": run_id,
            "run_attempt": 1,
            "source_commit": self.commit,
        }

    def test_fresh_pair_uses_exact_pinned_base_from_same_runner_in_order(self):
        artifact = self._add_baseline(run_id=20)
        selected = paired.select_pr_baseline(self.client, self._candidate(self._origin(artifact)))
        self.assertEqual(selected.evidence.identity["artifact_id"], artifact["id"])
        self.assertEqual(selected.evidence.identity["source_commit"], self.commit)
        self.assertEqual(self.client.queried, [])
        self.assertEqual(self.client.pages_requested, [])

    def test_reused_base_remains_pinned_across_runner_change(self):
        artifact = self._add_baseline()
        candidate = self._candidate(self._origin(artifact, run_id=10), reused=True, hostname="different-gpu-host")
        selected = paired.select_pr_baseline(self.client, candidate)
        self.assertEqual(selected.evidence.identity["run_id"], 10)
        self.assertTrue(selected.metadata["baseline_reused"])

    def test_fresh_pair_reports_mismatched_runner(self):
        artifact = self._add_baseline(run_id=20)
        candidate = self._candidate(self._origin(artifact), hostname="different-gpu-host")
        selected = paired.select_pr_baseline(self.client, candidate)
        self.assertIsNone(selected.evidence)
        self.assertEqual(selected.metadata["reason_code"], "runner_mismatch")

    def test_fresh_pair_reports_base_not_finished_before_current(self):
        artifact = self._add_baseline(run_id=20)
        candidate = self._candidate(self._origin(artifact), start=stamp(9))
        selected = paired.select_pr_baseline(self.client, candidate)
        self.assertIsNone(selected.evidence)
        self.assertEqual(selected.metadata["reason_code"], "measurement_order")

    def test_expired_pinned_baseline_never_falls_back_to_history(self):
        artifact = self._add_baseline(run_id=20)
        artifact["expired"] = True
        selected = paired.select_pr_baseline(self.client, self._candidate(self._origin(artifact)))
        self.assertIsNone(selected.evidence)
        self.assertEqual(selected.metadata["reason_code"], "expired")
        self.assertEqual(selected.metadata["unavailable_evidence"]["artifact_id"], artifact["id"])
        self.assertEqual(self.client.queried, [])

    def test_missing_baseline_pointer_produces_explicit_absence(self):
        selected = paired.select_pr_baseline(self.client, self._candidate(None))
        self.assertIsNone(selected.evidence)
        self.assertEqual(self.client.queried, [])


if __name__ == "__main__":
    unittest.main()
