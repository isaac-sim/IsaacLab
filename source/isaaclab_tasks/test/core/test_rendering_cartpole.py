# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering correctness tests for Cartpole environment backend combinations."""

# Launch Isaac Sim Simulator first for kit-based combinations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    PHYSICS_RENDERER_AOV_COMBINATIONS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    rendering_test_cartpole,
)

pytestmark = pytest.mark.isaacsim_ci

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)


# ----------------------------------------------------------------------------------------------------
# TEMPORARY CI PROBE -- DO NOT MERGE
#
# Wedges this file partway through the run so CI exercises the hang stack dump against a real Isaac Sim
# process: Kit is fully up, has rendered actual frames, and all of its own threads are running when
# SIGUSR1 arrives. The unit tests only prove the mechanism against a toy subprocess; this proves it
# reports a live Kit process, whose main thread is parked in a native call.
#
# Expect: this file is killed for "timeout", and its job log and JUnit report carry a
# "=== HANG STACK DUMP (all threads) ===" section naming _probe_wedge, twice.
#
# Revert by deleting this block and the _probe_wedge() call in test_rendering_cartpole.
# ----------------------------------------------------------------------------------------------------
import sys  # noqa: E402
import threading  # noqa: E402

_PROBE_CASES_BEFORE_WEDGE = 1
"""Cases allowed to render normally before the process is wedged."""

_probe_cases_done = 0


def _probe_wedge():
    """Block forever, once enough cases have actually rendered."""
    global _probe_cases_done
    _probe_cases_done += 1
    if _probe_cases_done <= _PROBE_CASES_BEFORE_WEDGE:
        return
    print(
        f"[CI PROBE] wedging after {_PROBE_CASES_BEFORE_WEDGE} rendered case(s);"
        " expect a hang stack dump naming _probe_wedge",
        file=sys.__stderr__,
        flush=True,
    )
    threading.Event().wait()


@pytest.mark.parametrize("physics_backend,renderer,data_type", PHYSICS_RENDERER_AOV_COMBINATIONS)
def test_rendering_cartpole(physics_backend, renderer, data_type):
    """Test cartpole environment rendering correctness."""
    rendering_test_cartpole(physics_backend, renderer, data_type, _COMPARISON_SCORES, compare_golden=True)
    _probe_wedge()  # TEMPORARY CI PROBE -- DO NOT MERGE
