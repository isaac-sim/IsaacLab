# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a pile of cables that collide and settle on each other.

.. code-block:: bash

    # Usage with default Newton VBD physics and Kit visualizer.
    uv run python scripts/demos/cables.py

    # Usage with explicit Newton VBD physics and Newton visualizer.
    uv run python scripts/demos/cables.py --physics newton_vbd --visualizer newton

    # Usage without a visualizer and with a larger cable pile.
    uv run python scripts/demos/cables.py --visualizer none --num_cables 40 --num_segments 15

    # Usage with a rendering camera, to observe cable rendering on a given renderer.
    uv run python scripts/demos/cables.py --visualizer none --camera ovrtx
    uv run python scripts/demos/cables.py --visualizer none --camera newton_warp

"""

from __future__ import annotations

import argparse
import math
import os
import random

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Spawn a pile of cables with Newton VBD.", conflict_handler="resolve")
parser.add_argument("--num_cables", type=int, default=25, help="Number of cables to spawn.")
parser.add_argument("--num_segments", type=int, default=20, help="Number of segments per cable.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed the cable spawn RNG so two runs are comparable frame-for-frame.",
)
parser.add_argument(
    "--camera",
    nargs="?",
    const="ovrtx",
    choices=["ovrtx", "newton_warp", "isaac_rtx"],
    default=None,
    help="Attach a camera that renders the cable pile with the given renderer.",
)
parser.add_argument(
    "--frames_dir",
    default=None,
    help="Write each rendered frame as a PNG into this directory, for assembly into an animation.",
)
parser.add_argument("--physics", default="newton_vbd", choices=["newton_vbd"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

if args_cli.num_cables < 1:
    parser.error("--num_cables must be at least 1.")
if args_cli.num_segments < 2:
    parser.error("--num_segments must be at least 2.")

if args_cli.camera == "isaac_rtx":
    # ``launch_simulation`` decides Kit-vs-kitless and camera support by scanning the cfg it is
    # handed, but this demo builds its camera after that call, so the scan sees neither. Both flags
    # are needed and both are additive.
    args_cli.require_kit = True
    args_cli.enable_cameras = True

from isaaclab.physics import PhysicsCfg


def _load_scene_modules() -> None:
    """Import the USD-backed modules, deferred until after :func:`launch_simulation` has opened.

    ``--camera isaac_rtx`` brings up Kit, which loads its own USD; importing the environment's copy
    first makes the two collide and the process dies inside ``libusd_tf`` during startup. Safe
    because ``from __future__ import annotations`` leaves the annotations below unevaluated.
    """
    global torch, sim_utils, CableObject, CableObjectCfg, RendererCfg, Camera, CameraCfg  # noqa: PLW0603

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import CableObject, CableObjectCfg
    from isaaclab.renderers import RendererCfg
    from isaaclab.sensors import Camera, CameraCfg


def _renderer_cfg(renderer: str) -> RendererCfg:
    """Import the requested renderer's config lazily, so the demo does not require all of them."""
    if renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if renderer == "newton_warp":
        from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg()
    if renderer == "isaac_rtx":
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg()
    raise ValueError(f"Unknown renderer: {renderer}")


def design_scene(num_cables: int, num_segments: int, colorize: bool) -> dict[str, CableObject]:
    """Spawn a ground plane, light, and randomly oriented cable pile.

    Args:
        num_cables: Number of cables to spawn.
        num_segments: Number of segments per cable.
        colorize: Whether to give each cable a random visual material.
    """
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    cable_length = 0.5
    segment_length = cable_length / num_segments
    thickness = 0.01
    radius = 0.5 * thickness
    target_stretch_stiffness = 5.0e5  # [N/m] per joint
    target_bend_stiffness = 20.0  # [N.m/rad] per joint
    stretch_modulus = target_stretch_stiffness * segment_length / (math.pi * radius**2)
    bend_modulus = target_bend_stiffness * segment_length / (0.25 * math.pi * radius**4)
    xy_jitter = 0.3
    z_spacing = 1.5 * thickness
    z_base = 0.8
    positions = [(index * segment_length, 0.0, 0.0) for index in range(num_segments + 1)]

    if args_cli.seed is not None:
        random.seed(args_cli.seed)

    print(f"[INFO]: Spawning {num_cables} cables...")
    entities: dict[str, CableObject] = {}
    for index in range(num_cables):
        angle = random.uniform(0.0, 2.0 * math.pi)
        position = (
            random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.cos(angle),
            random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.sin(angle),
            z_base + index * z_spacing,
        )
        orientation = (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))
        visual_material = None
        if colorize:
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random())
            )
        cfg = CableObjectCfg(
            prim_path=f"/World/envs/env_0/Cable{index:03d}",
            spawn=sim_utils.CableCfg(
                positions=positions,
                visual_material=visual_material,
                physics_material=sim_utils.CableMaterialCfg(
                    thickness=thickness,
                    density=100.0,
                    stretch_stiffness=stretch_modulus,
                    bend_stiffness=bend_modulus,
                ),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=position, rot=orientation),
        )
        entities[f"cable_{index:03d}"] = CableObject(cfg=cfg)

    return entities


def reset_cables(entities: dict[str, CableObject]) -> None:
    """Restore every cable to its initial segment state."""
    for cable in entities.values():
        cable.write_segment_pose_to_sim_index(segment_pose=cable.data.default_segment_pose_w)
        cable.write_segment_velocity_to_sim_index(segment_velocity=cable.data.default_segment_velocity_w)


# Framed to keep the whole fall in view: cables spawn near z=0.8 and settle at z~0.
_CAM_EYE = (3.0, 3.0, 1.6)
_CAM_TARGET = (0.0, 0.0, 0.35)


def _look_at_quat(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Camera orientation looking from ``eye`` to ``target``, as a world-convention ``(w, x, y, z)``."""
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    matrix = create_rotation_matrix_from_view(
        torch.tensor([eye], dtype=torch.float32),
        torch.tensor([target], dtype=torch.float32),
        "Z",
        device="cpu",
    )
    return tuple(float(value) for value in quat_from_matrix(matrix)[0].tolist())


def make_camera(renderer: str) -> Camera:
    """Spawn a camera aimed at the cable pile, using the requested renderer.

    The prim must sit under the standard env namespace: scene-partition primvars are authored by
    walking that subtree, so a prim outside it inherits no partition and renders nothing. The pose
    is baked into the cfg because :meth:`~isaaclab.sensors.Camera.set_world_poses_from_view` does
    not reach the render — the camera stays at the origin and the pile falls out of frame.
    """
    return Camera(
        CameraCfg(
            prim_path="/World/envs/env_0/CableCam",
            width=640,
            height=480,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(),
            offset=CameraCfg.OffsetCfg(pos=_CAM_EYE, rot=_look_at_quat(_CAM_EYE, _CAM_TARGET), convention="opengl"),
            renderer_cfg=_renderer_cfg(renderer),
        )
    )


def save_frame(camera: Camera, frames_dir: str, index: int) -> None:
    """Write one RGB frame as a zero-padded PNG, so a run can be assembled into an animation."""
    from PIL import Image

    rgb = camera.data.output["rgb"]
    tensor = rgb.torch if hasattr(rgb, "torch") else rgb
    image = tensor[0, ..., :3].detach().to("cpu", torch.uint8).numpy()
    Image.fromarray(image).save(os.path.join(frames_dir, f"frame_{index:05d}.png"))


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, CableObject],
    max_steps: int = -1,
    camera: Camera | None = None,
    frames_dir: str | None = None,
) -> None:
    """Run the simulation and periodically restore the cable pile."""
    sim_dt = sim.get_physics_dt()
    reset_steps = max(1, int(2.0 / sim_dt))
    count = 0
    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)

    while (max_steps < 0 or count < max_steps) and sim.is_headless_or_exist_active_visualizer():
        if count > 0 and count % reset_steps == 0:
            reset_cables(entities)
            print("[INFO]: Resetting cable state...")

        sim.step(render=False)
        for cable in entities.values():
            cable.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        if camera is not None:
            camera.update(sim_dt)
            if frames_dir is not None:
                save_frame(camera, frames_dir, count)
        count += 1


def main() -> None:
    """Launch and run the cable pile demo."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        _load_scene_modules()
        physics_cfg.solver_cfg.iterations = 20
        physics_cfg.num_substeps = 8
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(2.0, 2.0, 1.0), target=(0.0, 0.0, 0.25))
        colorize = bool(args_cli.visualizer and "kit" in args_cli.visualizer)
        entities = design_scene(args_cli.num_cables, args_cli.num_segments, colorize)
        camera = make_camera(args_cli.camera) if args_cli.camera else None
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, entities, args_cli.max_steps, camera, args_cli.frames_dir)


if __name__ == "__main__":
    main()
