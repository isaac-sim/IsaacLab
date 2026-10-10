# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script uses the interactive scene cloner to check if asset has been instanced properly.

An asset path may be a local file or a Nucleus/HTTPS URL; remote assets are downloaded before use.

Usage with different inputs (replace `<Asset-Path>` and `<Asset-Path-Instanced>` with the path to the
original asset and the instanced asset respectively):

```bash
uv run python scripts/tools/check_instanceable.py <Asset-Path> -n 4096 --physics
uv run python scripts/tools/check_instanceable.py <Asset-Path-Instanced> -n 4096 --physics
uv run python scripts/tools/check_instanceable.py <Asset-Path> -n 4096
uv run python scripts/tools/check_instanceable.py <Asset-Path-Instanced> -n 4096
```

Output from the above commands:

```bash
>>> Cloning time (scene creation): 0.648198 seconds
>>> Setup time (sim.reset): : 5.843589 seconds
[#clones: 4096, physics: True] Asset: <Asset-Path-Instanced> : 6.491870 seconds

>>> Cloning time (scene creation): 0.693133 seconds
>>> Setup time (sim.reset): 50.860526 seconds
[#clones: 4096, physics: True] Asset: <Asset-Path> : 51.553743 seconds

>>> Cloning time (scene creation) : 0.687201 seconds
>>> Setup time (sim.reset) : 6.302215 seconds
[#clones: 4096, physics: False] Asset: <Asset-Path-Instanced> : 6.989500 seconds

>>> Cloning time (scene creation) : 0.678150 seconds
>>> Setup time (sim.reset) : 52.854054 seconds
[#clones: 4096, physics: False] Asset: <Asset-Path> : 53.532287 seconds
```

"""

"""Parse the command line first."""

import argparse
import contextlib

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser("Utility to empirically check if asset in instanced properly.")
parser.add_argument("input", type=str, help="The path to the USD file.")
parser.add_argument("-n", "--num_clones", type=int, default=128, help="Number of clones to spawn.")
parser.add_argument("-s", "--spacing", type=float, default=1.5, help="Spacing between instances in a grid.")
parser.add_argument("-p", "--physics", action="store_true", default=False, help="Clone assets using physics cloner.")
# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import Timer, instantiate
from isaaclab.utils.assets import check_file_path, retrieve_file_path


def main():
    """Spawns the USD asset and clones it through the interactive scene."""
    # check valid file path
    if not check_file_path(args_cli.input):
        raise ValueError(f"Invalid file path: {args_cli.input}")
    # Load kit helper
    sim_cfg = SimulationCfg(dt=0.01)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)

        # Fabric and PhysX GPU buffers are configured through SimulationCfg/PhysxCfg defaults.
        # enable hydra scene-graph instancing
        # this is needed to visualize the scene when fabric is enabled
        sim.set_setting("/persistent/omnihydra/useSceneGraphInstancing", True)

        num_clones = args_cli.num_clones
        scene_cfg = InteractiveSceneCfg(
            num_envs=num_clones, env_spacing=args_cli.spacing, replicate_physics=args_cli.physics
        )
        scene_cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg())
        # Resolve through retrieve_file_path so Nucleus/HTTPS inputs are downloaded first; applying
        # os.path.abspath() to a URL would prepend the working directory and corrupt it.
        scene_cfg.asset = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Asset", spawn=sim_utils.UsdFileCfg(usd_path=retrieve_file_path(args_cli.input))
        )

        # Create a timer to measure the cloning time
        with Timer(f"[#clones: {num_clones}, physics: {args_cli.physics}] Asset: {args_cli.input}"):
            # Clone the scene
            with Timer(">>> Cloning time (scene creation)"):
                instantiate(scene_cfg)
            # Play the simulator
            with Timer(">>> Setup time (sim.reset)"):
                sim.reset()

        # Simulate scene (if a GUI is open)
        if sim.has_gui:
            with contextlib.suppress(KeyboardInterrupt):
                while sim.is_playing():
                    # perform step
                    sim.step()


if __name__ == "__main__":
    # run the main function
    main()
