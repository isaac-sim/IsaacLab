# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import isaaclab.app.app_launcher as app_launcher_module
from isaaclab.app import AppLauncher


def test_constructor_reports_missing_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_launcher_module, "SimulationApp", None)

    with pytest.raises(ImportError, match="requires the full Isaac Sim runtime"):
        AppLauncher()
