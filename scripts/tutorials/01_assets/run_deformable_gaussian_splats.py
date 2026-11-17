# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates a Gaussian splat visual that follows a deformable object.

A raspberry is a Gaussian splat asset embedded in a volume deformable. The script holds the top of each berry
and shakes it side to side, and saves the camera images in which the splats follow the simulated body.

.. code-block:: bash

    # Usage with Newton VBD physics and the Newton Warp renderer.
    uv run python scripts/tutorials/01_assets/run_deformable_gaussian_splats.py

    # Usage with Newton VBD physics and the OVRTX renderer.
    uv run --extra ovrtx python scripts/tutorials/01_assets/run_deformable_gaussian_splats.py --renderer ovrtx

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on Gaussian splats that follow a deformable object.")
parser.add_argument("--backend", type=str, default="newton_vbd", choices=["newton_vbd"], help="Physics backend.")
parser.add_argument("--renderer", type=str, default="newton_warp", choices=["newton_warp", "ovrtx"])
parser.add_argument("--asset", type=str, default=None, help="USD of a Gaussian splat embedded in a volume deformable.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of cloned environments.")
parser.add_argument("--output_dir", type=str, default="output/deformable_gaussian_splats", help="Image directory.")
# append simulation launcher CLI arguments
add_launcher_args(parser)
# Splats are drawn by the camera renderer, so no visualizer is used by default
parser.set_defaults(visualizer=None)
# parse the arguments
args_cli = parser.parse_args()
args_cli.physics = args_cli.backend

"""Rest everything follows."""

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.scene import InteractiveScene

# The raspberry's VBD asset sits next to its static Gaussian splat base.
ASSET = args_cli.asset or f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Objects/Berries/deformable_raspberry_omniphysics_vbd.usd"


def renderer_cfg():
    """Select the renderer that draws the splats."""
    if args_cli.renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    return NewtonWarpRendererCfg()


@configclass
class GaussianSplatsSceneCfg(InteractiveSceneCfg):
    """Raspberries seen by one camera each."""

    filter_collisions = False

    # The splat asset is spawned as a deformable object; its splats follow the simulation nodes.
    berry = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Berry", spawn=sim_utils.UsdFileCfg(usd_path=ASSET, func=sim_utils.spawn_from_usd)
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        width=256,
        height=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05196, 0.0, 0.04), rot=(0.35355, 0.35355, 0.61237, 0.61237), convention="opengl"
        ),
        renderer_cfg=renderer_cfg(),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: "InteractiveScene"):
    """Shake each berry by its top nodes, saving the rendered images."""
    berry: DeformableObject = scene["berry"]
    camera = scene["camera"]
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dt = sim.get_physics_dt()

    # Constrain the top quarter of the nodes to a target that moves them side to side; the rest hangs and sways.
    rest = berry.data.default_nodal_state_w.torch[..., :3].clone()
    top = rest[..., 2] > rest[..., 2].amin(dim=1, keepdim=True) * 0.25 + rest[..., 2].amax(dim=1, keepdim=True) * 0.75
    target = berry.data.nodal_kinematic_target.torch.clone()
    target[..., :3] = rest
    period = 2.0
    for count in range(int(2 * period / sim_dt)):
        shake = 0.003 * math.sin(2.0 * math.pi * count * sim_dt / period)
        target[..., 0] = rest[..., 0] + shake * top
        target[..., 3] = torch.where(top, 0.0, 1.0)
        berry.write_nodal_kinematic_target_to_sim_index(target)
        berry.write_data_to_sim()
        sim.step()
        berry.update(sim_dt)
        camera.update(sim_dt)
        if count % int(0.25 / sim_dt) == 0:
            frames = camera.data.output["rgb"].torch.cpu().numpy()
            for env, frame in enumerate(frames):
                Image.fromarray(frame[..., :3]).save(output_dir / f"env{env}_{count:04d}.png")
            print(f"Time {count * sim_dt:.2f}s: saved {len(frames)} images to {output_dir}")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli) as physics_cfg:
        physics_cfg.solver_cfg.iterations = 10
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(GaussianSplatsSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0))
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)
        print("[INFO]: Simulation complete...")


if __name__ == "__main__":
    main()
