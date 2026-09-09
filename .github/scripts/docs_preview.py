# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconcile static docs artifacts using trusted GitHub API metadata, then notify PRs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

_MARKER = "<!-- isaaclab-docs-preview -->"
_LABEL = "docs-preview"
_MAX_SITE_BYTES = 950 * 1024 * 1024


def _api(endpoint: str, method: str = "GET", **body):
    command = ["gh", "api", f"repos/{os.environ['GH_REPO']}/{endpoint}", "--method", method]
    if body:
        command += ["--input", "-"]
    result = subprocess.run(
        command, input=json.dumps(body) if body else None, text=True, capture_output=True, check=True
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def _pages(endpoint: str, key: str | None = None, **query):
    page = 1
    while True:
        result = _api(f"{endpoint}?{urlencode(dict(query, per_page=100, page=page))}")
        items = result[key] if key else result
        yield from items
        if len(items) < 100:
            return
        page += 1


def _extract(archive: Path, destination: Path, prefix: str = "") -> None:
    """Extract regular files within one artifact subtree, rejecting unsafe archive entries."""
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if sum(entry.file_size for entry in entries) > _MAX_SITE_BYTES:
            raise ValueError("Docs artifact exceeds the site size budget")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or any(part.lower().startswith(".git") for part in path.parts)
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            ):
                raise ValueError(f"Unsafe docs artifact entry: {entry.filename!r}")
        for entry in entries:
            if entry.is_dir() or not entry.filename.startswith(prefix):
                continue
            relative = entry.filename[len(prefix) :]
            if not relative:
                continue
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Docs artifact entry escapes its destination: {entry.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if not (destination / "index.html").is_file():
        raise ValueError("Docs artifact has no index.html")


def _download(artifact: dict, destination: Path, prefix: str = "") -> None:
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "docs.zip"
        with archive.open("wb") as output:
            subprocess.run(
                ["gh", "api", f"repos/{os.environ['GH_REPO']}/actions/artifacts/{artifact['id']}/zip"],
                stdout=output,
                check=True,
            )
        _extract(archive, destination, prefix)


def _artifact(run: dict, name: str) -> dict | None:
    return next(
        (
            item
            for item in _pages(f"actions/runs/{run['id']}/artifacts", "artifacts")
            if item["name"] == name and not item["expired"]
        ),
        None,
    )


def _revision(run: dict) -> list[int]:
    return [run["id"], run["run_attempt"]]


def _matches_pr(run: dict, pr: dict) -> bool:
    # Fork workflow runs can have an empty pull_requests array. In that case,
    # authenticate the source through the API's repository, branch, and commit.
    associations = run.get("pull_requests", [])
    return (
        run["event"] == "pull_request"
        and run["conclusion"] == "success"
        and run["head_sha"] == pr["head"]["sha"]
        and run["head_branch"] == pr["head"]["ref"]
        and run["head_repository"]["id"] == pr["head"]["repo"]["id"]
        and (not associations or any(item["number"] == pr["number"] for item in associations))
    )


def _save(root: Path, state: dict) -> None:
    (root / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _request(event: dict) -> None:
    """Authorize the same requesters as run-ci, persist opt-in, and wake the publisher."""
    if os.environ.get("REPO_NAME") != os.environ["GH_REPO"]:
        raise PermissionError("Preview publishing is not enabled for this repository")
    if not event["issue"].get("pull_request") or event["comment"]["body"] != "publish-doc":
        raise ValueError("Expected a publish-doc comment on a pull request")
    number = event["issue"]["number"]
    pr = _api(f"pulls/{number}")
    if pr["state"] != "open":
        raise ValueError(f"PR #{number} is not open")
    author = event["comment"]["user"]["login"]
    if author.casefold() != pr["user"]["login"].casefold():
        permission = _api(f"collaborators/{author}/permission")["permission"]
        if permission not in ("admin", "write"):
            raise PermissionError("Only the PR author or a user with write access can request a preview")

    if not any(label["name"] == _LABEL for label in _pages("labels")):
        try:
            _api("labels", "POST", name=_LABEL, color="0e8a16", description="Host docs while this PR is open")
        except subprocess.CalledProcessError:
            # Another PR command may have created the shared label concurrently.
            _api(f"labels/{_LABEL}")
    _api(f"issues/{number}/labels", "POST", labels=[_LABEL])
    # GITHUB_TOKEN label changes do not trigger workflows. Explicit dispatch is
    # supported and the persistent label also survives a replaced pending run.
    _api("actions/workflows/docs-publish.yaml/dispatches", "POST", ref=_api("")["default_branch"])
    print(f"Requested docs preview for PR #{number}; awaiting a successful current-head Docs artifact and publication")


def _prepare(root: Path, state: dict) -> None:
    before = json.dumps(state, sort_keys=True)
    public = root / "public"
    public.mkdir(exist_ok=True)
    repository = _api("")
    # Only successful runs of the known workflow on the default branch can
    # replace release docs. PR artifacts never control the site's root.
    for run in _pages(
        "actions/workflows/docs.yaml/runs", "workflow_runs", branch=repository["default_branch"], status="success"
    ):
        if run["event"] not in ("schedule", "workflow_dispatch") or run["head_repository"]["id"] != repository["id"]:
            continue
        if _revision(run) <= state.get("release", [0, 0]):
            break
        artifact = _artifact(run, "docs-release-html")
        if artifact is None:
            continue
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / "release"
            _download(artifact, release)
            if (release / "pr-preview").exists():
                raise ValueError("Release artifact uses the reserved pr-preview directory")
            for child in public.iterdir():
                if child.name != "pr-preview":
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            shutil.copytree(release, public, dirs_exist_ok=True)
        state["release"] = _revision(run)
        break
    if not (public / "index.html").is_file():
        raise RuntimeError("Run the Docs workflow on the default branch once to seed release docs before publishing")

    previews = state.setdefault("previews", {})
    open_prs = {
        str(pr["number"]): pr
        for pr in _pages("pulls", state="open")
        if any(label["name"] == _LABEL for label in pr["labels"])
    }
    for number, preview in previews.items():
        if number not in open_prs and preview["active"]:
            shutil.rmtree(public / "pr-preview" / number)
            preview["active"] = False

    for number, pr in open_prs.items():
        if not pr["head"]["repo"]:
            continue
        previous = previews.get(number, {})
        for run in _pages(
            "actions/workflows/docs.yaml/runs",
            "workflow_runs",
            event="pull_request",
            status="success",
            head_sha=pr["head"]["sha"],
        ):
            if not _matches_pr(run, pr):
                continue
            if previous.get("active") and previous["sha"] == pr["head"]["sha"] and _revision(run) <= previous["run"]:
                break
            artifact = _artifact(run, "docs-html")
            if artifact is None:
                continue
            with tempfile.TemporaryDirectory() as temporary:
                preview = Path(temporary) / "preview"
                try:
                    _download(artifact, preview, "current/")
                except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
                    print(f"Rejected docs artifact for PR #{number}: {error}")
                    break
                target = public / "pr-preview" / number
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(preview, target)
            previews[number] = {"active": True, "sha": pr["head"]["sha"], "run": _revision(run)}
            break

    if sum(path.stat().st_size for path in public.rglob("*") if path.is_file()) > _MAX_SITE_BYTES:
        raise ValueError("Combined docs exceed the GitHub Pages size budget; the existing deployment is unchanged")
    _save(root, state)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"changed={str(before != json.dumps(state, sort_keys=True)).lower()}\n")


def _notify(root: Path, state: dict) -> None:
    url = _api("pages")["html_url"].rstrip("/")
    failed = []
    for number, preview in state.get("previews", {}).items():
        signature = [preview["active"], *preview["run"], url]
        if preview.get("notified") == signature:
            continue
        active = preview["active"]
        link = f"{url}/pr-preview/{number}/"
        body = (
            f"{_MARKER}\n[View documentation preview]({link})\n\n"
            f"Built from `{preview['sha']}`. Updated after successful docs builds; removed when this PR closes."
            if active
            else f"{_MARKER}\nDocumentation preview expired: the PR was closed or its `{_LABEL}` label was removed."
        )
        try:
            comment = next(
                (
                    item
                    for item in _pages(f"issues/{number}/comments")
                    if item["user"]["login"] == "github-actions[bot]" and item["body"].startswith(_MARKER)
                ),
                None,
            )
            if comment:
                if comment["body"] != body:
                    _api(f"issues/comments/{comment['id']}", "PATCH", body=body)
            elif active:
                _api(f"issues/{number}/comments", "POST", body=body)
            _api(
                f"statuses/{preview['sha']}",
                "POST",
                state="success",
                context=f"Docs preview / PR #{number}",
                description="Documentation preview ready" if active else "Preview expired (closed or opted out)",
                target_url=link if active else url,
            )
            preview["notified"] = signature
        except subprocess.CalledProcessError:
            failed.append(number)
    _save(root, state)
    if failed:
        raise RuntimeError(f"Could not notify PRs {', '.join(failed)}; will retry on the next publisher run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("request", "prepare", "notify"))
    parser.add_argument("path", type=Path, help="Event JSON for request; publication state directory otherwise")
    args = parser.parse_args()
    if args.operation == "request":
        _request(json.loads(args.path.read_text()))
    else:
        state_file = args.path / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        {"prepare": _prepare, "notify": _notify}[args.operation](args.path, state)
