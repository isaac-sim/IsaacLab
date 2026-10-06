# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evidence selection tests using local GitHub API and artifact fixtures."""

import hashlib
import io
import json
import unittest
import urllib.error
import urllib.parse
import zipfile
from unittest.mock import Mock, patch

from . import baseline

REPO = "isaac-sim/IsaacLab"
HEAD, PARENT, GRANDPARENT = "a" * 40, "b" * 40, "c" * 40


def stamp(hour, minute=0):
    return f"2026-10-01T{hour:02d}:{minute:02d}:00+00:00"


def bundle(start=stamp(10), end=stamp(10, 5), task="task", fps=100, status="completed"):
    return {
        "run": {
            "task": task,
            "num_envs": 512,
            "status": status,
            "config": {"physics_backend": "newton_mjwarp", "rendering_backend": "none", "presets": []},
            "start_time_utc": start,
            "end_time_utc": end,
        },
        "runtime": {"total_fps": {"mean": fps}},
    }


def zipped(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, value in files.items():
            archive.writestr(path, value if isinstance(value, (str, bytes)) else json.dumps(value))
    return output.getvalue()


class FixtureClient:
    repository = REPO

    def __init__(self):
        self.attempts = {}
        self.run_artifacts = {}
        self.contents = {}
        self.run_jobs = {}
        self.parents = {HEAD: [PARENT], PARENT: [GRANDPARENT], GRANDPARENT: []}
        self.downloaded = []
        self.queried = []

    def add(
        self,
        run_id,
        commit,
        *,
        attempt=1,
        start=stamp(8),
        event="push",
        branch="develop",
        samples=None,
        context=True,
        artifact=True,
        statuses=None,
        conclusion="success",
        created=None,
    ):
        run = {
            "id": run_id,
            "run_attempt": attempt,
            "head_sha": commit,
            "head_branch": branch,
            "event": event,
            "workflow_id": 7,
            "run_started_at": start,
            "created_at": created or start,
            "status": "completed",
            "conclusion": conclusion,
        }
        self.attempts[run_id, attempt] = run
        self.run_jobs[run_id, attempt] = [
            {
                "name": "performance-smoke-benchmarks",
                "started_at": start,
                "run_attempt": attempt,
                "conclusion": conclusion,
            }
        ]
        self.run_artifacts.setdefault(run_id, [])
        if artifact:
            files = {
                f"{leg}/sample-{index}/benchmark_runtime_fixture.json": value
                for leg, values in (samples or {"leg": [bundle()]}).items()
                for index, value in enumerate(values, 1)
            }
            files.update({f"{leg}/status": value for leg, value in (statuses or {"leg": "ok"}).items()})
            if context:
                files["build-context.json"] = {
                    "source": {
                        "commit": commit,
                        "reference_branch": "develop",
                        "reference_commit": self.parents.get(commit, [None])[0] if self.parents.get(commit) else None,
                    },
                    "execution": {
                        "run_id": run_id,
                        "run_attempt": attempt,
                        "expected_samples": 3,
                        "measurement_not_before": start,
                    },
                    "metric_definition": {"total_fps": "source-fps-v2:" + "a" * 64},
                }
            self.add_artifact(run_id, f"performance-smoke-{run_id}-{attempt}", files)
        return run

    def add_artifact(self, run_id, name, files):
        data = zipped(files)
        artifact_id = len(self.contents) + 1
        item = {
            "id": artifact_id,
            "name": name,
            "workflow_run": {"id": run_id},
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "expired": False,
        }
        self.contents[artifact_id] = data
        self.run_artifacts.setdefault(run_id, []).append(item)
        return item

    def run_attempt(self, run_id, attempt):
        return self.attempts[run_id, attempt]

    def jobs(self, run_id, attempt):
        return self.run_jobs[run_id, attempt]

    def artifacts(self, run_id):
        return self.run_artifacts[run_id]

    def runs(self, commit, branch, event, workflow_id=None):
        self.queried.append((commit, branch, event))
        # Deliberately return extra rows: selector verifies API-filtered fields too.
        return [run for (run_id, attempt), run in self.attempts.items() if attempt == 1]

    def commit(self, commit):
        return {"sha": commit, "parents": [{"sha": parent} for parent in self.parents[commit]]}

    def download(self, artifact):
        self.downloaded.append(artifact["id"])
        if artifact.get("expired"):
            raise baseline.EvidenceError("expired", "Selected artifact has expired")
        return self.contents[artifact["id"]]


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.client = FixtureClient()
        self.client.add(20, HEAD, start=stamp(12), samples={"leg": [bundle(stamp(12, 1), stamp(12, 5))]})

    def candidate(self, attempt=1):
        return baseline.resolve_candidate(self.client, 20, attempt)

    def select(self):
        return baseline.select_baseline(self.client, self.candidate())

    def add_previous_report(self, report):
        self.client.run_artifacts[20] = self.client.run_artifacts[20][:1]
        self.client.add_artifact(20, "performance-build-comparison-20-1", {"build-comparison.json": report})

    def test_nearest_first_parent_branch_and_event_not_latest_branch_tip(self):
        self.client.add(10, GRANDPARENT)
        self.client.add(11, PARENT)
        self.client.add(12, PARENT, event="pull_request")
        self.client.add(13, PARENT, branch="feature")
        self.client.add(14, HEAD)
        evidence, metadata = self.select()
        self.assertEqual(evidence.identity["run_id"], 11)
        self.assertEqual(metadata["visited_commits"], [PARENT])

    def test_first_parent_walk_and_skips_no_shared_workload(self):
        self.client.add(10, GRANDPARENT)
        self.client.add(11, PARENT, samples={"other": [bundle(task="different")]})
        evidence, metadata = self.select()
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertEqual(metadata["visited_commits"], [PARENT, GRANDPARENT])
        self.assertIn("No shared workload", str(metadata["issues"]))

    def test_latest_measurement_not_artifact_upload_or_workflow_success(self):
        self.client.add(
            10, PARENT, context=False, samples={"leg": [bundle(end=stamp(12), fps=1)]}, conclusion="failure"
        )
        self.client.add(11, PARENT, samples={"leg": [bundle(end=stamp(10, 30))]})
        self.client.add(12, PARENT, samples={"leg": [bundle(end=stamp(12, 2))]})
        self.client.run_artifacts[11][0]["created_at"] = stamp(15)
        self.assertEqual(self.candidate().measurement_start, stamp(12, 1))
        evidence, _ = self.select()
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertEqual(evidence.context, {})
        self.assertIsNone(evidence.identity["tested_commit"])
        self.assertIn("provenance", evidence.issues[0])

    def test_latest_attempt_by_measurement_and_preserve_incomplete_legs(self):
        self.client.add(10, PARENT)
        self.client.add(
            10,
            PARENT,
            attempt=2,
            start=stamp(11),
            samples={"renamed-leg": [bundle(stamp(11), stamp(11, 5))]},
            statuses={"renamed-leg": "ok", "missing-leg": "failed"},
            conclusion="failure",
        )
        evidence, _ = self.select()
        self.assertEqual(evidence.identity["run_attempt"], 2)
        self.assertEqual(evidence.samples["missing-leg"], [])
        self.assertEqual(evidence.statuses["missing-leg"], "failed")

    def test_latest_run_creation_after_candidate_does_not_hide_original_attempt(self):
        self.client.add(10, PARENT, created=stamp(14))
        self.client.add(10, PARENT, attempt=2, start=stamp(14), samples={"leg": [bundle(stamp(14), stamp(14, 5))]})
        evidence, _ = self.select()
        self.assertEqual((evidence.identity["run_id"], evidence.identity["run_attempt"]), (10, 1))

    def test_missing_candidate_sample_start_uses_capture_or_unknown(self):
        self.client.add(30, HEAD, start=stamp(9), samples={"leg": [bundle(), bundle(start=None)]})
        candidate = baseline.resolve_candidate(self.client, 30, 1)
        self.assertEqual(candidate.measurement_start, stamp(9))
        self.assertIn("capture time", str(candidate.issues))
        self.client.add(31, HEAD, context=False, samples={"leg": [bundle(), bundle(start=None)]})
        self.assertIsNone(baseline.resolve_candidate(self.client, 31, 1).measurement_start)

    def test_failed_sample_without_end_does_not_prove_baseline_finished(self):
        self.client.add(10, PARENT, samples={"leg": [bundle(), bundle(end=None, status="failed")]})
        evidence, metadata = self.select()
        self.assertIsNone(evidence)
        self.assertIn("Measurement end time", str(metadata["issues"]))

    def test_corrupt_workload_preserved_while_valid_shared_leg_can_be_used(self):
        self.client.add(10, PARENT, artifact=False)
        self.client.run_jobs[10, 1][0]["completed_at"] = stamp(11)
        self.client.add_artifact(
            10,
            "performance-smoke-10-1",
            {
                "leg/sample-1/benchmark_runtime_valid.json": bundle(),
                "leg/status": "ok",
                "broken/sample-1/benchmark_runtime_broken.json": "{not json",
                "broken/status": "failed",
            },
        )
        evidence, _ = self.select()
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertEqual(evidence.samples["broken"], [])
        self.assertEqual(evidence.measurement_end, stamp(11))
        self.assertIn("broken/sample-1/benchmark_runtime_broken.json", str(evidence.issues))

    def test_all_failed_nearer_execution_does_not_replace_valid_older_execution(self):
        self.client.add(10, PARENT, statuses={"leg": "failed"})
        self.client.add(11, GRANDPARENT)
        evidence, metadata = self.select()
        self.assertEqual(evidence.identity["run_id"], 11)
        self.assertIn("status ok", str(metadata["issues"]))

    def test_malformed_fps_is_preserved_and_explicitly_unusable(self):
        malformed = bundle()
        malformed["runtime"]["total_fps"] = []
        self.client.add(10, PARENT, samples={"leg": [malformed, bundle(fps=10**1000)]})
        evidence = baseline.resolve_candidate(self.client, 10, 1)
        self.assertEqual(len(evidence.samples["leg"]), 2)
        self.assertFalse(evidence.has_completed_runtime)
        self.assertIn("FPS value", str(evidence.issues))
        selected, _ = self.select()
        self.assertIsNone(selected)

    def test_new_benchmark_attempt_uses_new_artifact(self):
        self.client.add(20, HEAD, attempt=2, start=stamp(13), samples={"leg": [bundle(stamp(13), stamp(13, 5))]})
        self.assertEqual(self.candidate(2).identity["run_attempt"], 2)

    def test_missing_job_time_is_explicit_not_old_evidence(self):
        self.client.add(20, HEAD, attempt=2, start=stamp(13), artifact=False)
        self.client.run_jobs[20, 2][0]["started_at"] = None
        with self.assertRaises(baseline.EvidenceError) as raised:
            self.candidate(2)
        self.assertEqual(raised.exception.code, "ambiguous_attempt")

    def test_expired_candidates_are_skipped_but_pinned_baseline_is_not_replaced(self):
        self.client.add(10, PARENT)
        self.client.add(11, GRANDPARENT)
        previous, _ = self.select()
        self.client.add(12, PARENT)
        for expired_run, expected_run in ((10, 12), (12, 11), (11, None)):
            with self.subTest(expired_run=expired_run):
                artifact = self.client.run_artifacts[expired_run][0]
                artifact["expired"] = True
                evidence, metadata = self.select()
                self.assertEqual(evidence.identity["run_id"] if evidence else None, expected_run)
                self.assertIn(
                    {"artifact_id": artifact["id"], "reason": "Selected artifact has expired"}, metadata["issues"]
                )
        self.client.run_artifacts[12][0]["expired"] = False
        with self.assertRaises(baseline.EvidenceError) as raised:
            baseline.select_baseline(self.client, self.candidate(), previous.identity)
        self.assertEqual(raised.exception.code, "expired")
        self.assertEqual(raised.exception.identity["artifact_id"], previous.identity["artifact_id"])
        self.client.run_artifacts[10][0].update(expired=False, digest="sha256:" + "0" * 64)
        with self.assertRaisesRegex(baseline.EvidenceError, "digest does not match") as raised:
            self.select()
        self.assertEqual(raised.exception.code, "corrupt")

    def test_corrupt_digest_and_zip_report_explicit_errors(self):
        artifact = self.client.run_artifacts[20][0]
        artifact["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(baseline.EvidenceError, "digest does not match") as raised:
            self.candidate()
        self.assertEqual(raised.exception.code, "corrupt")
        artifact.pop("digest")
        self.client.contents[artifact["id"]] = b"not a zip"
        with self.assertRaisesRegex(baseline.EvidenceError, "ZIP could not be read") as raised:
            self.candidate()
        self.assertEqual(raised.exception.code, "corrupt")

    def test_legacy_push_recovers_parent_but_legacy_pr_does_not_invent_merge_base(self):
        self.client.add(30, HEAD, context=False, event="push")
        self.client.add(10, PARENT)
        candidate = baseline.resolve_candidate(self.client, 30, 1)
        candidate.measurement_start = stamp(12)
        evidence, _ = baseline.select_baseline(self.client, candidate)
        self.assertEqual(evidence.identity["run_id"], 10)
        candidate.identity["event"] = "pull_request"
        evidence, _ = baseline.select_baseline(self.client, candidate)
        self.assertIsNone(evidence)

    def test_previous_selection_pins_identical_candidate_bytes(self):
        self.client.add(10, PARENT)
        candidate = self.candidate()
        previous, _ = baseline.select_baseline(self.client, candidate)
        self.add_previous_report({"candidate": candidate.identity, "baseline": previous.identity})
        candidate.identity["report_attempt"] = 2
        pinned = baseline.load_previous_selection(self.client, candidate)
        self.client.add(11, PARENT, samples={"leg": [bundle(end=stamp(11))]})
        evidence, metadata = baseline.select_baseline(self.client, candidate, pinned)
        self.assertEqual(evidence.identity["run_id"], 10)
        self.assertTrue(metadata["pinned"])
        candidate.identity["sha256"] = "different"
        self.assertIsNone(baseline.load_previous_selection(self.client, candidate))

    def test_previous_report_without_a_valid_pin_allows_fresh_selection(self):
        self.client.add(10, PARENT)
        candidate = self.candidate()
        candidate.identity["report_attempt"] = 2
        reports = [
            ("missing_candidate", {"candidate": None, "baseline": "irrelevant"}),
            (
                "different_candidate",
                {"candidate": {"artifact_id": 999, "sha256": "different"}, "baseline": "irrelevant"},
            ),
        ]
        for name, unavailable in (("unidentified_failure", {}), ("partial_failure", {"run_id": 10})):
            selection = {"pinned": False, "unavailable_side": "baseline", "unavailable_evidence": unavailable}
            reports.append((name, {"baseline": None, "selection": selection}))
        for name, report in reports:
            with self.subTest(case=name):
                self.add_previous_report({"candidate": candidate.identity, **report})
                pinned = baseline.load_previous_selection(self.client, candidate)
                self.assertIsNone(pinned)
                evidence, _ = baseline.select_baseline(self.client, candidate, pinned)
                self.assertEqual(evidence.identity["run_id"], 10)

    def test_saved_report_validates_identity_and_selection_and_reconstructs_links(self):
        self.client.add(10, PARENT)
        candidate = self.candidate()
        evidence, _ = baseline.select_baseline(self.client, candidate)
        original = evidence.identity
        candidate.identity["report_attempt"] = 2
        invalid = [("baseline_not_object", []), ("empty_baseline", {})]
        for field, value in (
            ("repository", "https://external.invalid"),
            ("run_id", 0),
            ("run_attempt", True),
            ("artifact_id", "3"),
            ("sha256", "not-a-digest"),
            ("source_commit", "short"),
        ):
            invalid.append((f"invalid_{field}", {**original, field: value}))
        for field in ("repository", "run_id", "run_attempt", "artifact_id", "sha256", "source_commit"):
            invalid.append((f"missing_{field}", {key: value for key, value in original.items() if key != field}))
        reports = [(name, {"baseline": saved}) for name, saved in invalid]
        selection = {"pinned": True, "unavailable_side": "baseline", "unavailable_evidence": None}
        reports += [
            ("candidate_not_object", {"candidate": [], "baseline": None}),
            ("selection_not_object", {"baseline": None, "selection": []}),
            ("missing_unavailable_identity", {"baseline": None, "selection": selection}),
        ]
        for name, report in reports:
            with self.subTest(case=name):
                self.add_previous_report({"candidate": candidate.identity, **report})
                with self.assertRaises(baseline.EvidenceError) as raised:
                    baseline.load_previous_selection(self.client, candidate)
                self.assertEqual(raised.exception.code, "corrupt")
        altered = {
            **original,
            "run_url": "https://external.invalid",
            "artifact_url": "https://external.invalid",
            "commit_url": "https://external.invalid",
        }
        self.add_previous_report({"candidate": candidate.identity, "baseline": altered})
        recovered = baseline.load_previous_selection(self.client, candidate)
        for field in ("run_url", "artifact_url", "commit_url"):
            self.assertEqual(recovered[field], original[field])

    def test_direct_malformed_pin_raises_evidence_error(self):
        with self.assertRaises(baseline.EvidenceError) as raised:
            baseline.select_baseline(self.client, self.candidate(), {"run_id": 10})
        self.assertEqual(raised.exception.code, "corrupt")

    def test_expired_evidence_retains_api_identity_without_claiming_tested_commit(self):
        self.client.run_artifacts[20][0]["expired"] = True
        with self.assertRaises(baseline.EvidenceError) as raised:
            self.candidate()
        identity = raised.exception.identity
        self.assertEqual((identity["repository"], identity["run_id"], identity["run_attempt"]), (REPO, 20, 1))
        self.assertEqual(identity["source_commit"], HEAD)
        self.assertEqual(identity["github_head_commit"], HEAD)
        self.assertEqual(identity["source_provenance"], "github_run_metadata")
        self.assertIsNone(identity["tested_commit"])
        self.assertEqual(identity["run_url"], f"https://github.com/{REPO}/actions/runs/20/attempts/1")

    def test_no_baseline_is_explicit_and_no_failed_samples_are_discarded(self):
        self.client.add(10, PARENT, samples={"leg": [bundle(status="failed")]}, statuses={"leg": "failed"})
        evidence, metadata = self.select()
        self.assertIsNone(evidence)
        self.assertIn("No usable completed runtime", str(metadata["issues"]))

    def test_candidate_without_readable_workloads_does_not_query_history(self):
        candidate = self.candidate()
        candidate.samples = {"failed-leg": []}
        candidate.context["source"].pop("reference_commit")
        self.client.commit = Mock(side_effect=AssertionError("Unexpected ancestry request"))
        self.client.runs = Mock(side_effect=AssertionError("Unexpected history request"))
        self.client.artifacts = Mock(side_effect=AssertionError("Unexpected artifact request"))
        evidence, metadata = baseline.select_baseline(self.client, candidate)
        self.assertIsNone(evidence)
        self.assertEqual(metadata["reason_code"], "no_readable_candidate_workload")
        self.client.commit.assert_not_called()
        self.client.runs.assert_not_called()
        self.client.artifacts.assert_not_called()

    def test_missing_or_invalid_renderer_is_unknown_workload_identity(self):
        sample = bundle()
        self.assertIsNotNone(baseline.workload_key(sample))
        sample["run"]["config"].pop("rendering_backend")
        self.assertIsNone(baseline.workload_key(sample))
        for renderer in (None, 42, "", " "):
            with self.subTest(renderer=renderer):
                sample["run"]["config"]["rendering_backend"] = renderer
                self.assertIsNone(baseline.workload_key(sample))


class TestGitHubTransport(unittest.TestCase):
    def test_pagination_at_client_boundary(self):
        client = baseline.GitHubClient(REPO, token="memory-only")
        paths = []

        def get(path):
            paths.append(path)
            page = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["page"][0]
            return {"total_count": 101, "artifacts": [{"id": i} for i in (range(100) if page == "1" else [100])]}

        client.get = get
        self.assertEqual(len(client.artifacts(10)), 101)
        self.assertEqual(len(paths), 2)

    def test_incomplete_history_is_not_treated_as_complete(self):
        client = baseline.GitHubClient(REPO)
        client.get = lambda path: {"total_count": 1001, "workflow_runs": []}
        with self.assertRaises(baseline.EvidenceError) as raised:
            client.runs(PARENT, "develop", "push")
        self.assertEqual(raised.exception.code, "incomplete_history")

    def test_redirect_does_not_forward_authorization(self):
        client = baseline.GitHubClient(REPO, token="memory-only")
        redirect = urllib.error.HTTPError(
            "https://api.github.com/artifact",
            302,
            "redirect",
            {
                "Location": "https://storage.example.invalid/artifact?sig=secret",
            },
            None,
        )
        client._opener = Mock()
        client._opener.open.side_effect = redirect
        response = io.BytesIO(b"original zip")
        response.headers = {}
        with patch("urllib.request.urlopen", return_value=response) as opened:
            self.assertEqual(client.download({"id": 7}), b"original zip")
        self.assertIsNone(opened.call_args.args[0].get_header("Authorization"))
        self.assertEqual(client._opener.open.call_args.args[0].get_header("Authorization"), "Bearer memory-only")

    def test_api_and_signed_download_read_timeouts_are_bounded_and_reported_safely(self):
        for download in (False, True):
            with self.subTest(download=download):
                client = baseline.GitHubClient(REPO, token="memory-only")
                client._opener = Mock()
                response = io.BytesIO(b"unfinished response")
                response.headers = {}
                client._opener.open.return_value = response
                if download:
                    client._opener.open.side_effect = urllib.error.HTTPError(
                        "https://api.github.com/artifact",
                        302,
                        "redirect",
                        {"Location": "https://storage.example.invalid/artifact?sig=secret"},
                        None,
                    )
                with (
                    patch("urllib.request.urlopen", return_value=response) as opened,
                    patch.object(response, "read", side_effect=TimeoutError("read stalled: ?sig=secret")),
                    self.assertRaises(baseline.EvidenceError) as raised,
                ):
                    if download:
                        client.download({"id": 7})
                    else:
                        client.run_attempt(10, 1)
                self.assertEqual(raised.exception.code, "timeout")
                self.assertNotIn("secret", str(raised.exception))
                self.assertTrue(response.closed)
                self.assertEqual(client._opener.open.call_args.kwargs["timeout"], 60)
                if download:
                    self.assertEqual(opened.call_args.kwargs["timeout"], 60)
                    self.assertIsNone(opened.call_args.args[0].get_header("Authorization"))

    def test_request_failures_have_safe_reason_codes_and_messages(self):
        for error in (
            urllib.error.URLError(TimeoutError("connect stalled: ?sig=secret")),
            urllib.error.HTTPError("https://storage.example.invalid/?sig=secret", 403, "secret", {}, None),
        ):
            with self.subTest(case=type(error).__name__):
                client = baseline.GitHubClient(REPO)
                client._opener = Mock()
                client._opener.open.side_effect = error
                with self.assertRaises(baseline.EvidenceError) as raised:
                    if isinstance(error, urllib.error.HTTPError):
                        client.download({"id": 7})
                    else:
                        client.run_attempt(10, 1)
                self.assertNotIn("secret", str(raised.exception))
                if isinstance(error, urllib.error.HTTPError):
                    self.assertIn("403", str(raised.exception))
                else:
                    self.assertEqual(raised.exception.code, "timeout")


if __name__ == "__main__":
    unittest.main()
