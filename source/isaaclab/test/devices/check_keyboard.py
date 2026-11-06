# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check Capture keyboard commands and visualizer hotkeys with --viz kit or --viz newton_gl."""

import argparse
from contextlib import ExitStack

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Check the keyboard teleoperation device.")
add_launcher_args(parser)
args_cli = parser.parse_args()

from isaaclab_teleop import CLOUDXR_STANDALONE_ENV, create_isaac_teleop_device, se3_keyboard_teleop_cfg

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.visualizers import add_key_callbacks


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    teleop_cfg = se3_keyboard_teleop_cfg(pos_sensitivity=0.1, rot_sensitivity=0.1, sim_device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli), ExitStack() as cleanup:
        sim = SimulationContext(sim_cfg)
        sim.reset()
        teleop = cleanup.enter_context(
            create_isaac_teleop_device(
                teleop_cfg,
                visualizers=sim.visualizers,
                use_kit_xr_bridge=False,
                cloudxr_env_file=CLOUDXR_STANDALONE_ENV,
            )
        )
        teleop.request_start()
        cleanup.callback(add_key_callbacks(sim.visualizers, {"L": lambda: print("Print callback"), "Escape": sim.stop}))
        print("Press 'L' to print a message, 'Escape' to quit, and 'B'/'P' to resume/pause teleoperation.")
        while sim.is_running() and not sim.is_stopped():
            command = teleop.advance()
            if command is not None:
                print(command)
            sim.step()


if __name__ == "__main__":
    main()
