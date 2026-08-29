# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-file pytest result handling in the test orchestrator."""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[4] / "tools"

posix_only = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"),
    reason="the orchestrator's process handling and the stack-dump signal are both POSIX-only",
)


def _load_orchestrator_module() -> ModuleType:
    module_path = TOOLS_DIR / "conftest.py"
    module_name = "isaaclab_test_orchestrator"
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@posix_only
def test_hung_process_report_names_where_it_is_stuck(monkeypatch, tmp_path: Path) -> None:
    orchestrator = _load_orchestrator_module()
    monkeypatch.setattr(orchestrator, "_capture_system_diagnostics", lambda: "=== SYSTEM DIAGNOSTICS BODY ===")
    monkeypatch.setattr(orchestrator, "HANG_DUMP_GRACE", 1, raising=False)

    test_file = tmp_path / "test_wedges.py"
    test_file.write_text(
        "import threading\n\n\ndef test_wedges():\n    wedged_call()\n\n\ndef wedged_call():\n"
        "    threading.Event().wait()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["ISAACLAB_HANG_DUMP"] = str(tmp_path / "hangdump.log")
    cmd = [sys.executable, "-m", "pytest", "-p", "hang_dump", "-p", "no:cacheprovider", str(test_file)]

    _returncode, _stdout, _stderr, kill_reason, _wall_time, pre_kill_diag = (
        orchestrator.capture_test_output_with_timeout(cmd, timeout=8, env=env)
    )

    assert kill_reason == "timeout"
    assert "HANG STACK DUMP" in pre_kill_diag
    assert "wedged_call" in pre_kill_diag
    assert pre_kill_diag.count("----- dump ") > 1
    assert pre_kill_diag.index("HANG STACK DUMP") < pre_kill_diag.index("SYSTEM DIAGNOSTICS BODY")


def test_hang_dump_plugin_is_inert_without_signal_support(monkeypatch) -> None:
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    import hang_dump

    monkeypatch.setattr(hang_dump, "DUMP_SIGNAL", None)

    assert hang_dump.is_supported() is False
    assert hang_dump.register() is False
    hang_dump.pytest_configure(config=None)
