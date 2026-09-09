# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise publishing against an in-memory GitHub API and real artifact ZIP files."""

from __future__ import annotations

import copy
import os
import stat
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import docs_preview as publisher


class TestDocsPreview(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = {}
        self.runs = []
        self.prs = []
        self.artifacts = {}
        self.comments = {}
        self.statuses = []
        self.labels = [{"name": "docs-preview"}]
        self.permissions = {}
        self.writes = []
        self.fail_comments = False
        self.enterContext(
            patch.dict(
                os.environ,
                GITHUB_OUTPUT=str(self.root / "output"),
                GH_REPO="example/IsaacLab",
                REPO_NAME="example/IsaacLab",
            )
        )
        self.enterContext(patch.object(publisher, "_api", self.api))
        self.enterContext(patch.object(publisher, "_download", self.download))

    def api(self, endpoint, method="GET", **body):
        parsed = urlsplit(endpoint)
        query = parse_qs(parsed.query)
        route = parsed.path
        if method != "GET":
            self.writes.append((route, body))
        if route == "":
            return {"default_branch": "main", "id": 1}
        if route == "pages":
            return {"html_url": "https://example.github.io/IsaacLab/"}
        if route == "pulls":
            return self.prs
        if route.startswith("pulls/"):
            return next(pr for pr in self.prs if pr["number"] == int(route.split("/")[1]))
        if route == "labels":
            if method == "POST":
                self.labels.append(body)
            return self.labels
        if route.startswith("collaborators/"):
            return {"permission": self.permissions.get(route.split("/")[1], "read")}
        if route == "actions/workflows/docs-publish.yaml/dispatches":
            return None
        if route == "actions/workflows/docs.yaml/runs":
            runs = [run for run in self.runs if run["conclusion"] == "success"]
            if "branch" in query:
                runs = [run for run in runs if run["head_branch"] == query["branch"][0]]
            if "head_sha" in query:
                runs = [run for run in runs if run["head_sha"] == query["head_sha"][0]]
            return {"workflow_runs": sorted(runs, key=lambda run: run["id"], reverse=True)}
        if route.endswith("/artifacts"):
            run_id = int(route.split("/")[2])
            return {"artifacts": [self.artifacts[run_id]] if run_id in self.artifacts else []}
        if route.startswith("statuses/"):
            self.statuses.append(body)
            return {}
        if route.startswith("issues/"):
            if route.endswith("/labels"):
                pr = next(pr for pr in self.prs if pr["number"] == int(route.split("/")[1]))
                pr["labels"] = [{"name": label} for label in body["labels"]]
                return pr["labels"]
            if self.fail_comments and method != "GET":
                raise subprocess.CalledProcessError(1, "gh")
            if route.startswith("issues/comments/"):
                for comments in self.comments.values():
                    for comment in comments:
                        if comment["id"] == int(route.split("/")[-1]):
                            comment["body"] = body["body"]
                            return comment
            number = route.split("/")[1]
            comments = self.comments.setdefault(number, [])
            if method == "POST":
                comments.append({"id": int(number), "user": {"login": "github-actions[bot]"}, **body})
                return comments[-1]
            return comments
        raise AssertionError(f"Unexpected API request: {method} {endpoint}")

    def download(self, artifact, destination, prefix=""):
        publisher._extract(self.root / f"{artifact['id']}.zip", destination, prefix)

    def build(self, run_id, pr=None, files=None):
        run = {
            "id": run_id,
            "run_attempt": 1,
            "event": "pull_request" if pr else "schedule",
            "conclusion": "success",
            "head_sha": pr["head"]["sha"] if pr else "release-sha",
            "head_branch": pr["head"]["ref"] if pr else "main",
            "head_repository": pr["head"]["repo"] if pr else {"id": 1},
            # Exercise the fork case where GitHub omits this association.
            "pull_requests": [],
        }
        self.runs.append(run)
        self.artifacts[run_id] = {
            "id": run_id,
            "name": "docs-html" if pr else "docs-release-html",
            "expired": False,
        }
        with zipfile.ZipFile(self.root / f"{run_id}.zip", "w") as archive:
            for name, content in (files or {"index.html": str(run_id)}).items():
                archive.writestr(f"current/{name}" if pr else name, content)
        return run

    def pr(self, number, sha="initial-sha", requested=True):
        pr = {
            "number": number,
            "state": "open",
            "user": {"login": "contributor"},
            "labels": [{"name": "docs-preview"}] if requested else [],
            "head": {"sha": sha, "ref": f"feature-{number}", "repo": {"id": 2}},
        }
        self.prs.append(pr)
        return pr

    def request_event(self, author="contributor"):
        return {
            "issue": {"number": 10, "pull_request": {"url": "https://api.github.com/example/pulls/10"}},
            "comment": {"body": "publish-doc", "user": {"login": author}},
        }

    def test_command_opts_in_only_requested_pr_and_label_removal_cleans_up(self):
        self.build(1)
        first, second = self.pr(10, requested=False), self.pr(20, requested=False)
        self.build(2, first)
        self.build(3, second)
        publisher._prepare(self.root, self.state)
        self.assertFalse((self.root / "public/pr-preview").exists())
        self.labels.clear()
        publisher._request(self.request_event("Contributor"))
        self.assertEqual(self.writes[-1], ("actions/workflows/docs-publish.yaml/dispatches", {"ref": "main"}))
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.assertTrue((self.root / "public/pr-preview/10/index.html").is_file())
        self.assertFalse((self.root / "public/pr-preview/20").exists())
        first["labels"].clear()
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.assertFalse((self.root / "public/pr-preview/10").exists())
        self.assertIn("label was removed", self.comments["10"][0]["body"])

    def test_command_authorization_matches_run_ci_and_rejects_closed_prs(self):
        pr = self.pr(10, requested=False)
        with self.assertRaises(PermissionError):
            publisher._request(self.request_event("reader"))
        self.assertEqual(self.writes, [])
        for permission in ("write", "admin"):
            self.permissions["maintainer"] = permission
            publisher._request(self.request_event("maintainer"))
            self.assertEqual(self.writes[-1][0], "actions/workflows/docs-publish.yaml/dispatches")
        self.writes.clear()
        pr["state"] = "closed"
        with self.assertRaisesRegex(ValueError, "not open"):
            publisher._request(self.request_event())
        self.assertEqual(self.writes, [])
        with patch.dict(os.environ, REPO_NAME="other/repository"), self.assertRaises(PermissionError):
            publisher._request(self.request_event())
        self.assertEqual(self.writes, [])

    def test_request_survives_dispatch_failure_and_waits_for_successful_build(self):
        self.build(1)
        pr = self.pr(10, requested=False)

        def dispatch_fails(endpoint, method="GET", **body):
            if endpoint.endswith("/dispatches"):
                raise subprocess.CalledProcessError(1, "gh")
            return self.api(endpoint, method, **body)

        with patch.object(publisher, "_api", dispatch_fails), self.assertRaises(subprocess.CalledProcessError):
            publisher._request(self.request_event())
        publisher._prepare(self.root, self.state)
        self.assertFalse((self.root / "public/pr-preview/10").exists())
        self.build(2, pr)
        publisher._prepare(self.root, self.state)
        self.assertTrue((self.root / "public/pr-preview/10/index.html").is_file())

    def test_reconciles_multiple_prs_and_preserves_them_during_nightly_build(self):
        self.build(1)
        first, second = self.pr(10), self.pr(20)
        self.build(2, first, {"index.html": "old", "removed.html": "old"})
        self.build(3, second)
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.assertEqual(len(self.comments["10"]), 1)
        self.assertIn("/pr-preview/10/", self.comments["10"][0]["body"])

        self.build(4, files={"index.html": "new release"})
        first["head"]["sha"] = "new-sha"
        self.build(5, first)
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        public = self.root / "public"
        self.assertEqual((public / "index.html").read_text(), "new release")
        self.assertEqual((public / "pr-preview/10/index.html").read_text(), "5")
        self.assertFalse((public / "pr-preview/10/removed.html").exists())
        self.assertEqual((public / "pr-preview/20/index.html").read_text(), "3")
        self.assertEqual(len(self.comments["10"]), 1)
        self.assertIn("new-sha", self.comments["10"][0]["body"])

        count = len(self.statuses)
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.assertEqual(len(self.statuses), count)
        self.assertTrue((self.root / "output").read_text().endswith("changed=false\n"))

    def test_close_and_reopen_updates_existing_comment(self):
        self.build(1)
        pr = self.pr(10)
        self.build(2, pr)
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.prs.clear()
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.assertFalse((self.root / "public/pr-preview/10").exists())
        self.assertIn("expired", self.comments["10"][0]["body"])
        self.assertEqual(self.statuses[-1]["target_url"], "https://example.github.io/IsaacLab")

        self.prs.append(pr)
        publisher._prepare(self.root, self.state)
        publisher._notify(self.root, self.state)
        self.assertTrue((self.root / "public/pr-preview/10/index.html").is_file())
        self.assertEqual(len(self.comments["10"]), 1)
        self.assertIn("View documentation preview", self.comments["10"][0]["body"])

    def test_stale_failed_unrelated_and_expired_builds_do_not_replace_preview(self):
        self.build(1)
        pr = self.pr(10)
        self.build(2, pr)
        publisher._prepare(self.root, self.state)
        pr["head"]["sha"] = "latest-sha"
        self.build(3, pr)["conclusion"] = "failure"
        self.build(4, pr)["head_repository"] = {"id": 999}
        self.build(5, pr)["pull_requests"] = [{"number": 99}]
        self.build(6, pr)
        self.artifacts[6]["expired"] = True
        publisher._prepare(self.root, self.state)
        self.assertEqual((self.root / "public/pr-preview/10/index.html").read_text(), "2")
        self.assertEqual(self.state["previews"]["10"]["sha"], "initial-sha")

    def test_rejected_artifact_keeps_last_good_preview_and_allows_cleanup(self):
        self.build(1)
        first, second = self.pr(10), self.pr(20)
        self.build(2, first)
        self.build(3, second)
        publisher._prepare(self.root, self.state)
        self.prs.remove(second)
        first["head"]["sha"] = "malformed-sha"
        self.build(4, first, {"../../escaped": "bad"})
        publisher._prepare(self.root, self.state)
        self.assertEqual((self.root / "public/pr-preview/10/index.html").read_text(), "2")
        self.assertFalse((self.root / "public/pr-preview/20").exists())

    def test_force_push_back_to_older_commit_and_rerun_refresh_the_preview(self):
        self.build(1)
        pr = self.pr(10)
        old = self.build(2, pr)
        pr["head"]["sha"] = "new-sha"
        self.build(3, pr)
        publisher._prepare(self.root, self.state)
        pr["head"]["sha"] = "initial-sha"
        publisher._prepare(self.root, self.state)
        self.assertEqual((self.root / "public/pr-preview/10/index.html").read_text(), "2")
        old["run_attempt"] = 2
        with zipfile.ZipFile(self.root / "2.zip", "w") as archive:
            archive.writestr("current/index.html", "rerun")
        publisher._prepare(self.root, self.state)
        self.assertEqual((self.root / "public/pr-preview/10/index.html").read_text(), "rerun")

    def test_notifications_retry_without_editing_user_comments(self):
        self.build(1)
        self.build(2, self.pr(10))
        publisher._prepare(self.root, self.state)
        spoof = {"id": 100, "user": {"login": "contributor"}, "body": publisher._MARKER}
        self.comments["10"] = [copy.deepcopy(spoof)]
        self.fail_comments = True
        with self.assertRaisesRegex(RuntimeError, "retry"):
            publisher._notify(self.root, self.state)
        self.assertNotIn("notified", self.state["previews"]["10"])
        self.fail_comments = False
        publisher._notify(self.root, self.state)
        self.assertEqual(self.comments["10"][0], spoof)
        self.assertEqual(len(self.comments["10"]), 2)

    def test_requires_release_seed_and_enforces_combined_size_limit(self):
        with self.assertRaisesRegex(RuntimeError, "seed release"):
            publisher._prepare(self.root, self.state)
        self.build(1, files={"index.html": "1234"})
        self.build(2, self.pr(10), {"index.html": "5678"})
        with patch.object(publisher, "_MAX_SITE_BYTES", 6), self.assertRaisesRegex(ValueError, "Combined docs"):
            publisher._prepare(self.root, self.state)
        self.assertFalse((self.root / "state.json").exists())

    def test_artifact_paths_links_and_missing_index_are_rejected(self):
        archive = self.root / "unsafe.zip"
        symlink = zipfile.ZipInfo("current/link")
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = ["../escape", "/absolute", "current//absolute", "current/../../escape", "current/.git/config", symlink]
        for entry in cases:
            with self.subTest(entry=entry):
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(entry, "bad")
                with self.assertRaises(ValueError):
                    publisher._extract(archive, self.root / "extracted", "current/")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("other/index.html", "not the preview")
        with self.assertRaisesRegex(ValueError, "no index.html"):
            publisher._extract(archive, self.root / "extracted", "current/")


if __name__ == "__main__":
    unittest.main()
