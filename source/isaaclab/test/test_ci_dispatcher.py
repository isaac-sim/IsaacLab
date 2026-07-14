# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the outer per-file CI dispatcher."""

import importlib.util
import sys
from pathlib import Path


def _load_dispatcher_module():
    tools_dir = Path(__file__).resolve().parents[3] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        spec = importlib.util.spec_from_file_location("isaaclab_ci_dispatcher", tools_dir / "conftest.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(tools_dir))


def test_dispatcher_timeout_is_a_failing_exit_status():
    """A timeout path recorded in failed_tests must fail the composite action."""

    dispatcher = _load_dispatcher_module()

    assert dispatcher.dispatcher_return_code([]) == 0
    assert dispatcher.dispatcher_return_code(["timed_out_test.py"]) == 1
