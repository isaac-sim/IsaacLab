# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates procedural terrains with flat patches.

Example usage:

.. code-block:: bash

    # Generate terrain with height color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme height

    # Generate terrain with random color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme random

    # Generate terrain with no color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme none

    # Generate terrain with curriculum
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --use_curriculum

    # Generate terrain with curriculum along with flat patches
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --use_curriculum --show_flat_patches

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates procedural terrain generation.")
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme to use for the terrain generation.",
)
parser.add_argument(
    "--use_curriculum",
    action="store_true",
    default=False,
    help="Whether to use the curriculum for the terrain generation.",
)
parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Whether to show the flat patches computed during the terrain generation.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


def design_scene() -> tuple[dict, torch.Tensor]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Parse terrain generation
    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme)

    # Add flat patch configuration
    # Note: To have separate colors for each sub-terrain type, we set the flat patch sampling configuration name
    #   to the sub-terrain name. However, this is not how it should be used in practice. The key name should be
    #   the intention of the flat patch. For instance, "source" or "target" for spawn and command related flat patches.
    if args_cli.show_flat_patches:
        for sub_terrain_name, sub_terrain_cfg in terrain_gen_cfg.sub_terrains.items():
            sub_terrain_cfg.flat_patch_sampling = {
                sub_terrain_name: FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)
            }

    # Handler for terrains importing
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=2048,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
    )
    # Remove visual material for height and random color schemes to use the default material
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None
    # Create terrain importer
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Show the flat patches computed
    if args_cli.show_flat_patches:
        # Configure the flat patches
        vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
        for name in terrain_importer.flat_patches:
            vis_cfg.markers[name] = sim_utils.CylinderCfg(
                radius=0.5,  # note: manually set to the patch radius for visualization
                height=0.1,
                visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
            )
        flat_patches_visualizer = VisualizationMarkers(vis_cfg)

        # Visualize the flat patches
        all_patch_locations = []
        all_patch_indices = []
        for i, patch_locations in enumerate(terrain_importer.flat_patches.values()):
            num_patch_locations = patch_locations.view(-1, 3).shape[0]
            # store the patch locations and indices
            all_patch_locations.append(patch_locations.view(-1, 3))
            all_patch_indices += [i] * num_patch_locations
        # combine the patch locations and indices
        flat_patches_visualizer.visualize(torch.cat(all_patch_locations), marker_indices=all_patch_indices)

    # return the scene information
    scene_entities = {"terrain": terrain_importer}
    return scene_entities, terrain_importer.env_origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, AssetBase], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
