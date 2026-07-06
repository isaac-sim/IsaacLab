# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for changed-file Testmon mode selection."""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("select_testmon_mode.py")
_SPEC = importlib.util.spec_from_file_location("select_testmon_mode", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
select_testmon_mode = _MODULE.select_testmon_mode


def test_python_only_relevant_changes_select_affected_tests() -> None:
    assert select_testmon_mode(["source/isaaclab/isaaclab/app.py", "tools/helper.py"]) == "select"


def test_irrelevant_changes_select_no_affected_tests() -> None:
    assert select_testmon_mode(["docs/guide.md", "source/isaaclab/changelog.d/change.skip"]) == "select"


def test_static_relevant_change_collects_full_suite() -> None:
    assert select_testmon_mode(["source/isaaclab/config/extension.toml"]) == "collect"


def test_mixed_change_collects_full_suite() -> None:
    assert select_testmon_mode(["source/isaaclab/code.py", "docker/Dockerfile.base"]) == "collect"


def test_workflow_yaml_change_collects_full_suite() -> None:
    # A workflow change that testmon cannot reason about must run the full suite,
    # otherwise a workflow-only PR would trigger a job that deselects every test.
    assert select_testmon_mode([".github/workflows/install-ci.yml"]) == "collect"


def test_action_python_change_collects_full_suite() -> None:
    # Python helpers under .github/actions/ run outside pytest, so testmon has no
    # dependency data for them; a change to one must run the full suite rather than
    # taking the Python-only fast path and deselecting every test.
    assert select_testmon_mode([".github/actions/run-package-tests/select_testmon_mode.py"]) == "collect"
