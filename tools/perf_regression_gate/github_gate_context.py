# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve GitHub event context for the performance regression gate"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MIRRORED_PR_BRANCH_RE = re.compile(r"^pull-request/(\d+)$")
_BASELINE_PUBLISH_BRANCHES = frozenset({"main", "develop", "angehu/perf-gate-poc"})
_BASELINE_PUBLISH_PREFIXES = ("release/",)


@dataclass(frozen=True)
class GateContext:
    base_sha: str
    target_branch: str
    source_branch: str
    allow_update: bool
    trusted_source: str
    event_kind: str

    def outputs(self) -> dict[str, str]:
        return {
            "base_sha": self.base_sha,
            "target_branch": self.target_branch,
            "source_branch": self.source_branch,
            "allow_update": "true" if self.allow_update else "false",
            "trusted_source": self.trusted_source,
            "event_kind": self.event_kind,
        }


def _strip_heads_ref(value: str) -> str:
    prefix = "refs/heads/"
    return value[len(prefix) :] if value.startswith(prefix) else value


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _event_value(env: Mapping[str, str], key: str) -> str:
    try:
        return env[key]
    except KeyError as exc:
        raise RuntimeError(f"Missing required GitHub environment variable: {key}") from exc


def _load_event(event_path: str | None) -> dict[str, Any]:
    if not event_path:
        return {}
    with Path(event_path).open(encoding="utf-8") as fh:
        event = json.load(fh)
    return event if isinstance(event, dict) else {}


def _baseline_publish_branch(ref: str) -> bool:
    branch = _strip_heads_ref(ref)
    return branch in _BASELINE_PUBLISH_BRANCHES or branch.startswith(_BASELINE_PUBLISH_PREFIXES)


def fetch_pr_from_github(repo: str, pr_number: str, env: Mapping[str, str]) -> dict[str, Any]:
    token = env.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to resolve mirrored PR context")
    api_url = env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    request = urllib.request.Request(
        f"{api_url}/repos/{repo}/pulls/{pr_number}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        pr = json.load(response)
    if not isinstance(pr, dict):
        raise RuntimeError(f"GitHub API returned non-object PR payload for PR {pr_number}")
    return pr


def _pr_context(pr: Mapping[str, Any], *, allow_update: bool, trusted_source: str, event_kind: str) -> GateContext:
    base = pr["base"]
    head = pr["head"]
    return GateContext(
        base_sha=str(base["sha"]),
        target_branch=str(base["ref"]),
        source_branch=str(head["ref"]),
        allow_update=allow_update,
        trusted_source=trusted_source,
        event_kind=event_kind,
    )


def resolve_gate_context(
    env: Mapping[str, str] | None = None,
    event: Mapping[str, Any] | None = None,
    fetch_pr: Callable[[str, str, Mapping[str, str]], Mapping[str, Any]] | None = None,
) -> GateContext:
    env = env or os.environ
    event = event if event is not None else _load_event(env.get("GITHUB_EVENT_PATH"))
    fetch_pr = fetch_pr or fetch_pr_from_github

    event_name = _event_value(env, "GITHUB_EVENT_NAME")
    github_ref = _event_value(env, "GITHUB_REF")
    ref_name = _event_value(env, "GITHUB_REF_NAME")
    github_sha = _event_value(env, "GITHUB_SHA")

    if event_name == "pull_request":
        return _pr_context(
            event["pull_request"],
            allow_update=False,
            trusted_source="read_only",
            event_kind="pull_request",
        )

    if event_name == "merge_group":
        merge_group = event.get("merge_group") or {}
        return GateContext(
            base_sha=str(merge_group.get("base_sha") or github_sha),
            target_branch=_strip_heads_ref(str(merge_group.get("base_ref") or ref_name)),
            source_branch=_strip_heads_ref(str(merge_group.get("head_ref") or ref_name)),
            allow_update=False,
            trusted_source="read_only",
            event_kind="merge_group",
        )

    if event_name == "push" and github_ref.startswith("refs/heads/pull-request/"):
        match = _MIRRORED_PR_BRANCH_RE.fullmatch(ref_name)
        if not match:
            raise RuntimeError(f"Cannot parse mirrored PR number from GITHUB_REF_NAME={ref_name!r}")
        repo = _event_value(env, "GITHUB_REPOSITORY")
        pr = fetch_pr(repo, match.group(1), env)
        allow_update = _truthy(env.get("ALLOW_MIRROR_UPDATE") or env.get("PERF_GATE_ALLOW_MIRROR_BASELINE_UPDATE"))
        return _pr_context(
            pr,
            allow_update=allow_update,
            trusted_source="mirrored_pr_push" if allow_update else "read_only",
            event_kind="mirrored_pr_push",
        )

    if event_name == "push":
        allow_update = _baseline_publish_branch(github_ref)
        return GateContext(
            base_sha=github_sha,
            target_branch=ref_name,
            source_branch=ref_name,
            allow_update=allow_update,
            trusted_source="protected_branch" if allow_update else "read_only",
            event_kind="protected_push" if allow_update else "push",
        )

    return GateContext(
        base_sha=github_sha,
        target_branch=ref_name,
        source_branch=ref_name,
        allow_update=False,
        trusted_source="read_only",
        event_kind=event_name,
    )


def write_github_outputs(outputs: Mapping[str, str], output_path: str | None) -> None:
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={value}\n")


def main() -> int:
    context = resolve_gate_context()
    outputs = context.outputs()
    write_github_outputs(outputs, os.environ.get("GITHUB_OUTPUT"))
    print(
        "gate_context "
        f"event={outputs['event_kind']} base={outputs['base_sha'][:12]} "
        f"target={outputs['target_branch']} source={outputs['source_branch']} "
        f"allow_update={outputs['allow_update']} trusted_source={outputs['trusted_source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
