# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a foveated camera move its gaze between objects.

The viewer shows the camera's retina: dense where it looks, sparse in the periphery. The gaze steps from one object
to the next, so the fovea lands on each in turn.

.. code-block:: bash

    uv run --extra fovi isaaclab demo foveated-camera
    uv run --extra fovi --extra isaacsim isaaclab demo foveated-camera renderer=isaacsim_rtx
    uv run --extra fovi --extra ovrtx isaaclab demo foveated-camera renderer=ovrtx
    # A large CMF constant gives the matched non-foveated retina, for comparison:
    uv run --extra fovi isaaclab demo foveated-camera env.scene.eye.cmf_a_deg=1000
"""

from __future__ import annotations

import argparse
import math

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Number of render frames; -1 runs until the viewer closes."
)
parser.add_argument("--dwell", type=int, default=60, help="Frames the gaze rests on each object.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli, overrides = setup_preset_cli(parser)
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")
if args_cli.dwell <= 0:
    parser.error("--dwell must be positive.")

import torch
from isaaclab_fovi.sensors import FoveatedCameraCfg
from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_isaacsim.renderers import IsaacRtxRendererCfg
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.visualizers import SceneCameraCfg, VisualizerCfg

from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

EYE_HEIGHT = 0.5
"""Height of the camera above the ground [m]; it looks along +Y with +Z up, so +X is to its right."""

OBJECTS = {
    "red_cube": ((-0.9, 2.5, 0.2), (0.8, 0.15, 0.1)),
    "green_sphere": ((0.8, 3.0, 0.25), (0.15, 0.7, 0.2)),
    "blue_cube": ((0.0, 4.0, 0.6), (0.15, 0.3, 0.85)),
    "yellow_sphere": ((-0.3, 1.8, 0.15), (0.9, 0.8, 0.1)),
}
"""Each object's centre [m] and colour, in the order the gaze visits them."""


def _object(prim: str, centre: tuple[float, float, float], colour: tuple[float, float, float]) -> AssetBaseCfg:
    """A static, coloured cube or sphere sized to its centre's height."""
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=colour)
    if "sphere" in prim:
        spawn = sim_utils.SphereCfg(radius=centre[2], visual_material=material)
    else:
        spawn = sim_utils.CuboidCfg(size=(2 * centre[2],) * 3, visual_material=material)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}", spawn=spawn, init_state=AssetBaseCfg.InitialStateCfg(pos=centre)
    )


@configclass
class FoveatedSceneCfg(InteractiveSceneCfg):
    """Coloured objects on a ground plane, in front of a foveated camera."""

    ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=800.0))
    red_cube = _object("red_cube", *OBJECTS["red_cube"])
    green_sphere = _object("green_sphere", *OBJECTS["green_sphere"])
    blue_cube = _object("blue_cube", *OBJECTS["blue_cube"])
    yellow_sphere = _object("yellow_sphere", *OBJECTS["yellow_sphere"])
    eye = FoveatedCameraCfg(
        prim_path="{ENV_REGEX_NS}/Eye",
        data_types=["rgb"],
        renderer_cfg=MultiBackendRendererCfg(
            default=NewtonWarpRendererCfg(render_order="pixel_priority"),
            newton_warp=NewtonWarpRendererCfg(render_order="pixel_priority"),
            ovrtx=OVRTXRendererCfg(),
            isaacsim_rtx=IsaacRtxRendererCfg(),
        ),
        # An ideal pinhole with a 90 degree field of view; the retina spans its short side.
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=24.0),
        foveated_res=128,
        mask_to_sensor_fov=True,
        offset=FoveatedCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, EYE_HEIGHT), rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="opengl"
        ),
    )


@configclass
class FoveatedDemoCfg(ManagerBasedEnvCfg):
    """A static scene; physics initializes it but never advances."""

    decimation = 1
    actions = {}
    observations = {}
    events = None
    ui_window_class_type = None
    scene: FoveatedSceneCfg = FoveatedSceneCfg(num_envs=1, env_spacing=0.0)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        physics=preset(
            default=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False, load_visual_shapes=True),
            isaacsim_rtx=PhysxCfg(),
        ),
        visualizer_cfgs=VisualizerCfg(cameras=[SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Eye")]),
    )


def gaze_toward(centre: tuple[float, float, float]) -> tuple[float, float]:
    """(tilt, pan) [deg] that look from the camera at ``centre``: positive tilt looks up, positive pan looks left."""
    x, y, z = centre
    return math.degrees(math.atan2(z - EYE_HEIGHT, math.hypot(x, y))), -math.degrees(math.atan2(x, y))


def main() -> None:
    """Render the scene while the gaze steps between the objects."""
    # Config-only registration reuses the standard preset and Hydra override resolver.
    gym.register(
        "FoveatedCameraDemo",
        entry_point="isaaclab.envs:ManagerBasedEnv",
        kwargs={"env_cfg_entry_point": FoveatedDemoCfg},
    )
    cfg, _ = resolve_task_config("FoveatedCameraDemo", None, overrides=overrides)
    with launch_simulation(cfg, args_cli):
        sim_utils.create_new_stage()
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.register_interactive_scene(scene)
        sim.reset()
        camera = scene["eye"]
        gazes = torch.tensor([gaze_toward(centre) for centre, _ in OBJECTS.values()], device=sim.device)
        frame = 0
        while sim.is_running() and (args_cli.max_steps < 0 or frame < args_cli.max_steps):
            if frame % args_cli.dwell == 0:
                camera.set_gaze_angles(gazes[(frame // args_cli.dwell) % len(gazes)].unsqueeze(0))
            scene.update(sim.get_physics_dt())
            sim.render()
            frame += 1


if __name__ == "__main__":
    main()
