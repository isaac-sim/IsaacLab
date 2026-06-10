# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for local action references in GitHub workflows."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_LOCAL_ACTION_RE = re.compile(r"^\s*uses:\s*[\"']?(?P<path>\./\.github/actions/[^\"'\s#]+)", re.MULTILINE)


def test_workflow_local_action_references_have_action_metadata():
    missing_actions = []

    for workflow_path in sorted(_WORKFLOWS_DIR.glob("*.y*ml")):
        workflow_text = workflow_path.read_text(encoding="utf-8")
        for match in _LOCAL_ACTION_RE.finditer(workflow_text):
            action_path = _REPO_ROOT / match.group("path")
            if not (action_path / "action.yml").is_file() and not (action_path / "action.yaml").is_file():
                missing_actions.append(f"{workflow_path.relative_to(_REPO_ROOT)} -> {match.group('path')}")

    assert not missing_actions, "Missing local GitHub action metadata:\n" + "\n".join(missing_actions)
