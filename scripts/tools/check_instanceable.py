# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script uses the cloner API to check if asset has been instanced properly.

Usage with different inputs (replace `<Asset-Path>` and `<Asset-Path-Instanced>` with the path to the
original asset and the instanced asset respectively):

```bash
./isaaclab.sh  -p source/tools/check_instanceable.py <Asset-Path> -n 4096 --headless --physics
./isaaclab.sh  -p source/tools/check_instanceable.py <Asset-Path-Instanced> -n 4096 --headless --physics
./isaaclab.sh  -p source/tools/check_instanceable.py <Asset-Path> -n 4096 --headless
./isaaclab.sh  -p source/tools/check_instanceable.py <Asset-Path-Instanced> -n 4096 --headless
```

Output from the above commands:

```bash
>>> Cloning time (cloner.clone): 0.648198 seconds
>>> Setup time (sim.reset): : 5.843589 seconds
[#clones: 4096, physics: True] Asset: <Asset-Path-Instanced> : 6.491870 seconds

>>> Cloning time (cloner.clone): 0.693133 seconds
>>> Setup time (sim.reset): 50.860526 seconds
[#clones: 4096, physics: True] Asset: <Asset-Path> : 51.553743 seconds

>>> Cloning time (cloner.clone) : 0.687201 seconds
>>> Setup time (sim.reset) : 6.302215 seconds
[#clones: 4096, physics: False] Asset: <Asset-Path-Instanced> : 6.989500 seconds

>>> Cloning time (cloner.clone) : 0.678150 seconds
>>> Setup time (sim.reset) : 52.854054 seconds
[#clones: 4096, physics: False] Asset: <Asset-Path> : 53.532287 seconds
```

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Utility to empirically check if asset in instanced properly.")
parser.add_argument("input", type=str, help="The path to the USD file.")
parser.add_argument("-n", "--num_clones", type=int, default=128, help="Number of clones to spawn.")
parser.add_argument("-s", "--spacing", type=float, default=1.5, help="Spacing between instances in a grid.")
parser.add_argument("-p", "--physics", action="store_true", default=False, help="Clone assets using physics cloner.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.carb import set_carb_setting

from isaaclab.utils import Timer
from isaaclab.utils.assets import check_file_path


def main():
    """Spawns the USD asset robot and clones it using Isaac Gym Cloner API."""
    # check valid file path
    if not check_file_path(args_cli.input):
        raise ValueError(f"Invalid file path: {args_cli.input}")
    # Load kit helper
    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda:0"
    )
    # enable fabric which avoids passing data over to USD structure
    # this speeds up the read-write operation of GPU buffers
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_fabric(True)
    # increase GPU buffer dimensions
    sim.get_physics_context().set_gpu_found_lost_aggregate_pairs_capacity(2**25)
    sim.get_physics_context().set_gpu_total_aggregate_pairs_capacity(2**21)
    # enable hydra scene-graph instancing
    # this is needed to visualize the scene when fabric is enabled
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=args_cli.spacing)
    cloner.define_base_env("/World/envs")
    prim_utils.define_prim("/World/envs/env_0")
    # Spawn things into stage
    prim_utils.create_prim("/World/Light", "DistantLight")

    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.create_prim("/World/envs/env_0/Asset", "Xform", usd_path=os.path.abspath(args_cli.input))
    # Clone the scene
    num_clones = args_cli.num_clones

    # Create a timer to measure the cloning time
    with Timer(f"[#clones: {num_clones}, physics: {args_cli.physics}] Asset: {args_cli.input}"):
        # Clone the scene
        with Timer(">>> Cloning time (cloner.clone)"):
            cloner.define_base_env("/World/envs")
            envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_clones)
            _ = cloner.clone(
                source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=args_cli.physics
            )
        # Play the simulator
        with Timer(">>> Setup time (sim.reset)"):
            sim.reset()

    # Simulate scene (if not headless)
    if not args_cli.headless:
        with contextlib.suppress(KeyboardInterrupt):
            while sim.is_playing():
                # perform step
                sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
