# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for packaged programs that import sibling helpers."""

import sys

import pytest

from isaaclab.programs import _run_script


def test_program_can_import_sibling_helper_without_leaking_search_path(tmp_path, monkeypatch):
    """The CLI must provide normal script imports and restore the caller on failure."""
    helper = "_isaaclab_program_sibling_regression"
    (tmp_path / f"{helper}.py").write_text("VALUE = 42\n")
    script = tmp_path / "demo.py"
    script.write_text(f"from {helper} import VALUE\nassert VALUE == 42\nraise RuntimeError('script completed')\n")
    monkeypatch.delitem(sys.modules, helper, raising=False)
    original_path = sys.path
    original_main = sys.modules["__main__"]
    try:
        with pytest.raises(RuntimeError, match="script completed"):
            _run_script(script)
        assert sys.path is original_path
        assert sys.modules["__main__"] is original_main
    finally:
        sys.modules.pop(helper, None)
