# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a small scene with a foveated camera.

Run with ``--backend newton`` or ``--backend rtx``. The retina looks through
the spawned camera, an ideal pinhole or, with ``--lens fisheye``, an ideal
equidistant fisheye, and its gaze moves inside that camera's field of view.
``--pool`` renders a denser photoreceptor grid and pools it to the output grid.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("newton", "rtx"), required=True)
parser.add_argument("--lens", choices=("pinhole", "fisheye"), default="pinhole", help="The camera the retina uses.")
parser.add_argument("--pool", action="store_true", help="Pool a denser photoreceptor grid to the output grid.")
parser.add_argument("--resolution", type=int, default=64, help="Side length of the output grid.")
parser.add_argument("--cmf_a_deg", type=float, default=None, help="CMF constant [deg]; a large value is non-foveated.")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--output", type=Path, help="Save the RGB frames and warm frame timings.")
add_launcher_args(parser)
# Camera output is required, but the example does not need an interactive visualizer.
parser.set_defaults(enable_cameras=True)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.resolution <= 0:
    parser.error("--resolution must be positive")

import torch
import warp as wp
from isaaclab_fovi.sensors import FoveatedCameraCfg

import isaaclab.sim as sim_utils


def make_camera_cfg(renderer) -> FoveatedCameraCfg:
    """The foveated camera, looking at the cube through the selected lens."""

    if args.lens == "pinhole":
        # An ideal pinhole with a 60 degree field of view.
        spawn = sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=27.713)
    else:
        # An ideal equidistant fisheye: a 640 px image circle spanning 160 degrees.
        focal = 320.0 / math.radians(80.0)
        spawn = sim_utils.PinholeCameraCfg(
            distortion=sim_utils.OpenCvFisheyeDistortionCfg(
                fx=focal, fy=focal, cx=319.5, cy=319.5, image_size=(640, 640)
            )
        )
    camera_cfg = FoveatedCameraCfg(
        prim_path="{ENV_REGEX_NS}/Eye",
        data_types=["rgb"],
        renderer_cfg=renderer,
        foveated_res=args.resolution,
        pool_enabled=args.pool,
        spawn=spawn,
        # The retina sees only what the spawned camera can see at the current gaze.
        mask_to_sensor_fov=True,
        # OpenGL camera half a metre above the ground, looking along +Y with +Z up.
        offset=FoveatedCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.5), rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="opengl"
        ),
    )
    if args.cmf_a_deg is not None:
        camera_cfg.cmf_a_deg = args.cmf_a_deg
    return camera_cfg


def main() -> None:
    """Render RGB of a static cube while the gaze moves."""
    if args.backend == "newton":
        from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)
        renderer = NewtonWarpRendererCfg(render_order="pixel_priority")
    else:
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        physics = PhysxCfg()
        renderer = IsaacRtxRendererCfg()

    sim_cfg = sim_utils.SimulationCfg(device=args.device, physics=physics)
    # The launcher starts the runtime that the physics and the camera's renderer need.
    with launch_simulation(SimpleNamespace(sim=sim_cfg, renderer=renderer), args):
        # Scene types load USD support, which Kit must provide first.
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils.configclass import configclass

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            """A red cube on the ground in front of the camera."""

            ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
            light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=500.0))
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    mass_props=sim_utils.MassCfg(mass=1.0),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 0.25)),
            )
            eye = make_camera_cfg(renderer)

        sim_utils.create_new_stage()
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.0))
        camera = scene["eye"]
        sim.reset()

        frame_seconds: list[float] = []
        frames: list[torch.Tensor] = []
        for step in range(args.steps):
            # Gaze (tilt, pan) [deg] from the optical axis: look ahead, then 8 degrees to the right.
            pan = (0.0, -8.0, -8.0, 0.0)[step % 4]
            camera.set_gaze_angles(torch.tensor([[0.0, pan]], device=sim.device))
            wp.synchronize_device(sim.device)
            start = time.perf_counter()
            sim.step()
            camera.update(sim.get_physics_dt(), force_recompute=True)
            wp.synchronize_device(sim.device)
            if step > 0:
                frame_seconds.append(time.perf_counter() - start)
            rgb = camera.data.output["rgb"].torch
            if args.output is not None:
                frames.append(rgb.clone())
        print(
            f"{args.backend}: RGB shape={tuple(rgb.shape)}, dtype={rgb.dtype}, warm frame seconds={frame_seconds}",
            flush=True,
        )
        if args.output is not None:
            torch.save(
                {
                    "frames": torch.stack(frames).cpu(),
                    "sampling_coordinates": camera.sampling_coords.cartesian.cpu(),
                    "frame_seconds": frame_seconds,
                },
                args.output,
            )
        sim.stop()


if __name__ == "__main__":
    main()
