# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep CLI unit tests independent of the developer's local Kit installation."""

from pathlib import Path

import pytest

import isaaclab.cli as cli


@pytest.fixture(autouse=True)
def isolate_local_kit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "DEFAULT_ISAAC_SIM_PATH", tmp_path / "_isaac_sim")
