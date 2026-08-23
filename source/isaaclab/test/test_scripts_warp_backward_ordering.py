# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static scanner ensuring entry scripts disable Warp adjoint codegen early enough.

``warp.config.enable_backward`` is captured when a Warp module is created, which happens at
import time. Setting it after an Isaac Lab import has no effect on modules already created,
and nothing fails loudly when that happens -- the kernels are simply built with adjoints
again. These tests pin the ordering so a future import reshuffle cannot silently undo it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ENTRY_SCRIPTS = [
    "scripts/reinforcement_learning/train.py",
    "scripts/reinforcement_learning/play.py",
    "scripts/environments/zero_agent.py",
    "scripts/environments/random_agent.py",
]

_SETTING = "wp.config.enable_backward = False"


@pytest.mark.unit
@pytest.mark.parametrize("script", _ENTRY_SCRIPTS)
def test_entry_script_disables_warp_backward_before_isaaclab_imports(script: str):
    lines = (_REPO_ROOT / script).read_text().splitlines()

    setting_at = next((i for i, line in enumerate(lines) if line.strip() == _SETTING), None)
    assert setting_at is not None, f"{script} does not set '{_SETTING}'"

    first_isaaclab_import = next(
        (i for i, line in enumerate(lines) if line.startswith(("import isaaclab", "from isaaclab"))),
        None,
    )
    assert first_isaaclab_import is not None, f"{script} imports no Isaac Lab module"
    assert setting_at < first_isaaclab_import, (
        f"{script} sets enable_backward on line {setting_at + 1}, after the Isaac Lab import on line"
        f" {first_isaaclab_import + 1}; Warp modules created by that import keep adjoint codegen."
    )
