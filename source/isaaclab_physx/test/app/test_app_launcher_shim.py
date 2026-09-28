# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that a script written against the deprecated ``AppLauncher`` still runs."""

import subprocess
import sys
import textwrap

import pytest

_READY_MARKER = "OLD_STYLE_SCRIPT_RAN"

_OLD_STYLE_SCRIPT = textwrap.dedent(f"""
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(["--device", "cpu"])
    simulation_app = AppLauncher(args_cli).app

    import isaaclab.sim as sim_utils

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.reset()
    for _ in range(3):
        sim.step()
    assert simulation_app.is_running() and not simulation_app.is_exiting()
    print("{_READY_MARKER}", flush=True)
    simulation_app.close()
""")


# ``AppLauncher(headless=True).app`` in a local that goes out of scope: Kit must keep running until exit
_DROPPED_LAUNCHER_SCRIPT = textwrap.dedent(f"""
    import gc

    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, device="cpu")
    gc.collect()

    import isaaclab.sim as sim_utils

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu"))
    sim.reset()
    sim.step()
    print("{_READY_MARKER}", flush=True)
""")


@pytest.mark.integration
@pytest.mark.parametrize("script", [_OLD_STYLE_SCRIPT, _DROPPED_LAUNCHER_SCRIPT], ids=["kept-app", "dropped-launcher"])
def test_old_style_script_starts_kit_steps_and_closes(script):
    """The shim starts Kit, the script steps a simulation, and ``close()`` ends the process with 0."""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    output = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    assert _READY_MARKER in result.stdout, output
    assert result.returncode == 0, output
