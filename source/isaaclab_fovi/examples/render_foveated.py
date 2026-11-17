# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a small scene with a foveated camera.

Run with ``--backend newton`` or ``--backend rtx``. By default the fixation
moves inside the fixed camera's calibrated field of view; ``--planar`` samples
the image plane of the spawn optics instead, with no gaze. ``--pool`` renders a
denser photoreceptor grid and pools it to the output grid.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("newton", "rtx"), required=True)
parser.add_argument("--planar", action="store_true", help="Sample the image plane of the spawn optics.")
parser.add_argument("--pool", action="store_true", help="Pool a denser photoreceptor grid to the output grid.")
parser.add_argument("--resolution", type=int, default=64, help="Side length of the output grid.")
parser.add_argument("--cmf-a-deg", type=float, default=None, help="CMF constant [deg]; a large value is non-foveated.")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--output", type=Path, help="Save the RGB frames and warm frame timings.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.resolution <= 0:
    parser.error("--resolution must be positive")
args.enable_cameras = True
launcher = AppLauncher(args)


def main() -> None:
    """Render RGB of a static cube while the fixation moves."""
    # Isaac Sim plugins must be available before importing simulator-bound types.
    import torch
    import warp as wp
    from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg

    import isaaclab.sim as sim_utils

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

    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, physics=physics))
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    cube = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
    )
    cube.func("/World/envs/env_0/Cube", cube, translation=(0.0, 2.0, 0.0))
    light = sim_utils.DistantLightCfg(intensity=2000.0)
    light.func("/World/Light", light)

    cfg = FoveatedCameraCfg(
        prim_path="/World/envs/env_.*/Eye",
        data_types=["rgb"],
        renderer_cfg=renderer,
        foveated_res=args.resolution,
        pool_enabled=args.pool,
        # A 60 degree horizontal field of view, used by planar sampling.
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=27.713),
        # OpenGL camera looking along +Y with +Z up.
        offset=FoveatedCameraCfg.OffsetCfg(rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="opengl"),
    )
    if args.planar:
        cfg.field_geometry = "planar"
    else:
        # The physical camera the retina looks through: a 640x480 pinhole.
        cfg.camera_calibration = {
            "model": "pinhole",
            "image_size": (480, 640),
            "intrinsics": (550.0, 550.0, 319.5, 239.5),
        }
        cfg.mask_to_sensor_fov = True
    if args.cmf_a_deg is not None:
        cfg.cmf_a_deg = args.cmf_a_deg
    camera = FoveatedCamera(cfg)
    sim.reset()

    frame_seconds: list[float] = []
    frames: list[torch.Tensor] = []
    for step in range(args.steps):
        if not args.planar:
            # Fixation in normalized image coordinates (row, column) of the calibrated camera.
            fixation = torch.tensor([[0.5, (0.5, 0.58, 0.58, 0.5)[step % 4]]], device=sim.device)
            camera.set_calibrated_fixation(fixation)
        wp.synchronize_device(sim.device)
        start = time.perf_counter()
        sim.step()
        camera.update(sim.get_physics_dt(), force_recompute=True)
        wp.synchronize_device(sim.device)
        if step > 0:
            frame_seconds.append(time.perf_counter() - start)
        frames.append(camera.data.output["rgb"].torch.clone())
    print(
        f"{args.backend}: RGB shape={tuple(frames[-1].shape)}, dtype={frames[-1].dtype}, "
        f"warm frame seconds={frame_seconds}",
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
    try:
        main()
    except Exception:
        # Kit shutdown can terminate before Python prints an uncaught exception.
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
