# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a small scene with a foveated camera.

Run with ``--backend newton`` or ``--backend rtx``; add ``--pool`` for
photoreceptor-to-output pooling. ``--steps`` bounds the simulation run.
With ``--virtual-gaze`` the camera stays fixed and the fixation moves inside
its calibrated field of view.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("newton", "rtx"), required=True)
parser.add_argument("--pool", action="store_true")
parser.add_argument("--field-geometry", choices=("planar", "spherical"), default="planar")
parser.add_argument(
    "--style",
    choices=("isotropic", "warped_cartesian_as_grid"),
    default="isotropic",
)
parser.add_argument("--fov-type", choices=("circular", "square", "wang"), default="circular")
parser.add_argument("--radius-norm", type=float, choices=(2.0, float("inf")), default=2.0)
parser.add_argument(
    "--virtual-gaze",
    action="store_true",
    help="Move the fixation inside the fixed camera's calibrated field of view.",
)
parser.add_argument("--lens", choices=("pinhole", "fisheye"), default="pinhole")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--resolution", type=int, default=16)
parser.add_argument("--output", type=Path, help="Save final RGB tensor and warm frame timings.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.resolution <= 0:
    parser.error("--resolution must be positive")
if args.virtual_gaze and args.field_geometry != "spherical":
    parser.error("--virtual-gaze requires --field-geometry spherical")
args.enable_cameras = True
launcher = AppLauncher(args)


def main() -> None:
    """Render RGB of a static cube, optionally moving the fixation."""
    # Isaac Sim plugins must be available before importing simulator-bound types.
    import torch
    import warp as wp
    from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg

    import isaaclab.sim as sim_utils
    from isaaclab.utils.warp import ProxyArray

    if args.backend == "newton":
        from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)
        renderer = NewtonWarpRendererCfg(render_order="pixel_priority")
    elif args.backend == "rtx":
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        physics = PhysxCfg()
        renderer = IsaacRtxRendererCfg()
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

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
    calibration = None
    if args.field_geometry == "spherical":
        calibration = {
            "model": args.lens,
            "image_size": (480, 640),
            "intrinsics": (550.0, 550.0, 319.5, 239.5),
            "distortion": (0.02, -0.003, 0.0, 0.0),
        }
    cfg = FoveatedCameraCfg(
        prim_path="/World/envs/env_.*/Eye",
        data_types=["rgb"],
        renderer_cfg=renderer,
        foveated_res=args.resolution,
        foveated_style=args.style,
        fov_type=args.fov_type,
        radius_norm=args.radius_norm,
        ndc_layout="native_grid" if args.style == "warped_cartesian_as_grid" else "cortical_disc",
        field_geometry=args.field_geometry,
        camera_calibration=calibration,
        fixed_software_fov=args.virtual_gaze,
        software_fov_image_height=480,
        software_fov_image_width=640,
        retinal_window_reference_side="long",
        pool_enabled=args.pool,
        pool_res_mult=2.0,
        pool_mode="avg",
        camera_fov_deg=60.0,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=27.713),
        # OpenGL camera looking along +Y with +Z up.
        offset=FoveatedCameraCfg.OffsetCfg(rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="opengl"),
    )
    camera = FoveatedCamera(cfg)
    sim.reset()
    frame_seconds: list[float] = []
    frames: list[torch.Tensor] = []
    for step in range(args.steps):
        if args.virtual_gaze:
            fixation = torch.tensor([[0.5, (0.5, 0.58, 0.58, 0.5)[step % 4]]], device=sim.device)
            camera.set_calibrated_fixation(fixation)
        wp.synchronize_device(sim.device)
        start = time.perf_counter()
        sim.step()
        camera.update(sim.get_physics_dt(), force_recompute=True)
        wp.synchronize_device(sim.device)
        if step > 0:
            frame_seconds.append(time.perf_counter() - start)
        frame = camera.data.output["rgb"]
        frames.append((frame.torch if isinstance(frame, ProxyArray) else frame).clone())
    rgb = camera.data.output["rgb"]
    rgb_tensor = rgb.torch if isinstance(rgb, ProxyArray) else rgb
    print(
        f"{args.backend}: pooled={args.pool}, RGB shape={tuple(rgb_tensor.shape)}, "
        f"dtype={rgb_tensor.dtype}, warm frame seconds={frame_seconds}",
        flush=True,
    )
    if args.output is not None:
        torch.save(
            {
                "rgb": rgb_tensor.cpu(),
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
