# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch two experimental foveated eyes on a pan-tilt head fixate near and far objects together.

A binocular teacher moves the eyes in hybrid gaze: the retinas jump to each object while the head turns after them,
and the vergence turns the eyes toward each other for near objects and back toward parallel for far ones. The viewer
shows the left eye; switch to the right eye or the overview in its camera menu.

.. code-block:: bash

    uv run --extra fovi isaaclab example foveated-stereo
    uv run --extra fovi --extra ovrtx isaaclab example foveated-stereo renderer=ovrtx
"""

from __future__ import annotations

import argparse

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Number of environment steps; -1 runs until the viewer closes."
)
parser.add_argument("--dwell", type=int, default=45, help="Steps the eyes stay on each object.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli, overrides = setup_preset_cli(parser)
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")
if args_cli.dwell <= 0:
    parser.error("--dwell must be positive.")

import torch
import warp as wp
from isaaclab_fovi.assets import PAN_TILT_CFG, PAN_TILT_STEREO_OFFSETS
from isaaclab_fovi.mdp import BinocularGazeActionCfg, GazeRigCfg, binocular_gaze_state
from isaaclab_fovi.sensors import FoveatedCameraCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg

from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.hydra import resolve_task_config

EYE_HEIGHT = 0.5
"""Height of the head's base above the ground [m]; the head looks along +Y with +Z up, so +X is to its right."""

OBJECTS = {
    "far_cube": ((0.6, 3.5, 0.3), 0.3, (0.15, 0.3, 0.85)),
    "close_sphere": ((0.12, 0.35, 0.56), 0.025, (0.15, 0.7, 0.2)),
    "near_sphere": ((-0.1, 0.45, 0.47), 0.03, (0.9, 0.8, 0.1)),
    "middle_cube": ((-0.8, 1.8, 0.2), 0.2, (0.8, 0.15, 0.1)),
}
"""Each object's centre [m], half size [m] and colour, in the order the eyes visit them; each lies within both eyes'
view from the one before."""


def _object(prim: str, centre, half_size: float, colour) -> AssetBaseCfg:
    """A static, coloured cube or sphere."""
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=colour)
    if "sphere" in prim:
        spawn = sim_utils.SphereCfg(radius=half_size, visual_material=material)
    else:
        spawn = sim_utils.CuboidCfg(size=(2 * half_size,) * 3, visual_material=material)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}", spawn=spawn, init_state=AssetBaseCfg.InitialStateCfg(pos=centre)
    )


EYE_PATHS = tuple(
    f"{{ENV_REGEX_NS}}/PanTilt/Geometry/world/pan_link/tilt_link/{eye}" for eye in ("LeftEye", "RightEye")
)
"""The eyes' prim paths (left, right), on the head's tilting link."""


def _eye(prim_path: str, offset) -> FoveatedCameraCfg:
    """A foveated eye on the head's tilting link."""
    return FoveatedCameraCfg(
        prim_path=prim_path,
        data_types=["rgb"],
        renderer_cfg=preset(
            default=NewtonWarpRendererCfg(render_order="pixel_priority"),
            newton_warp=NewtonWarpRendererCfg(render_order="pixel_priority"),
            ovrtx=OVRTXRendererCfg(),
        ),
        # An ideal pinhole with a 90 degree field of view; the retina spans its short side.
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=24.0, clipping_range=(0.05, 100.0)),
        foveated_res=128,
        mask_to_sensor_fov=True,
        offset=offset,
    )


@configclass
class StereoSceneCfg(InteractiveSceneCfg):
    """Objects near and far in front of a pan-tilt head with two foveated eyes."""

    ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=800.0))
    far_cube = _object("far_cube", *OBJECTS["far_cube"])
    near_sphere = _object("near_sphere", *OBJECTS["near_sphere"])
    middle_cube = _object("middle_cube", *OBJECTS["middle_cube"])
    close_sphere = _object("close_sphere", *OBJECTS["close_sphere"])
    head = PAN_TILT_CFG.replace(init_state=PAN_TILT_CFG.init_state.replace(pos=(0.0, 0.0, EYE_HEIGHT)))
    left_eye = _eye(EYE_PATHS[0], PAN_TILT_STEREO_OFFSETS[0])
    right_eye = _eye(EYE_PATHS[1], PAN_TILT_STEREO_OFFSETS[1])


@configclass
class ActionsCfg:
    gaze = BinocularGazeActionCfg(
        asset_name="left_eye", right_asset_name="right_eye", mode="hybrid", rig=GazeRigCfg(asset_name="head")
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        gaze = ObservationTermCfg(func=binocular_gaze_state)

    policy = PolicyCfg()


@configclass
class StereoExampleCfg(ManagerBasedEnvCfg):
    """The head turns on Newton's MJWarp solver at Isaac Lab's usual 120 Hz physics and 30 Hz control."""

    decimation = 4
    actions = ActionsCfg()
    observations = ObservationsCfg()
    ui_window_class_type = None
    scene: StereoSceneCfg = StereoSceneCfg(num_envs=1, env_spacing=0.0)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()),
        visualizer_cfgs=VisualizerCfg(
            cameras=[
                SceneCameraCfg(prim_path=EYE_PATHS[0]),
                SceneCameraCfg(prim_path=EYE_PATHS[1]),
                PerspectiveCameraCfg(eye=(1.4, -1.2, 1.4), lookat=(0.0, 1.2, 0.4)),
            ]
        ),
    )


def main() -> None:
    """Step the eyes toward each object in turn, with the binocular teacher's actions."""
    # Config-only registration reuses the standard preset and Hydra override resolver.
    gym.register(
        "FoveatedStereoExample",
        entry_point="isaaclab.envs:ManagerBasedEnv",
        kwargs={"env_cfg_entry_point": StereoExampleCfg},
    )
    cfg, _ = resolve_task_config("FoveatedStereoExample", None, overrides=overrides)
    with launch_simulation(cfg, args_cli):
        env = ManagerBasedEnv(cfg)
        env.reset()
        gaze = env.action_manager.get_term("gaze")
        centres = torch.tensor([centre for centre, _, _ in OBJECTS.values()], device=env.device)
        point = centres[:1].clone()
        labels = torch.zeros(1, 4, device=env.device)
        point_wp, labels_wp = wp.from_torch(point, dtype=wp.vec3f), wp.from_torch(labels)
        step = 0
        while env.sim.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
            point.copy_(centres[(step // args_cli.dwell) % len(centres)] + env.scene.env_origins[:1])
            # The teacher's labels step both eyes toward the object; an invalid target holds them still.
            gaze.teacher_labels(point_wp, labels_wp)
            env.step(labels[:, :3])
            step += 1
        env.close()


if __name__ == "__main__":
    main()
