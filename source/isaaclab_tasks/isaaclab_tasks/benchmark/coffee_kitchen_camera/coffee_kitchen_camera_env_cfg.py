# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Rizon-Sharpa coffee demo's kitchen scene with one camera per environment.

The scene comes from the demo's own code (:mod:`isaaclab_tasks.contrib.rizon_sharpa_coffee`): its
``configure_scene`` builds the robot, table, bowl, cup and spoon, and its kinematic mount moves the spoon
with the palm. Every environment replays the demo's recorded arm motion (``coffee_arm_motion.json``) in a
loop from its own time offset.

Differences from the demo:

* no coffee powder or water, so no MPM solver;
* a kitchen in every environment instead of one around the first, without its ceiling, lights and cameras;
* the scene's dome and key lights light it instead of the demo's kitchen lighting, which turns them off.
  RTX Minimal mode shades with the first distant light only, which is the key light;
* the recorded motion replaces the live IK controller, at 100 Hz physics instead of 800 Hz;
* a square camera with the demo's kitchen view.

The camera data type and resolution are ``presets=`` selectable with the same names as the Kuka-Allegro
camera presets, e.g. ``presets=simple_shading_full_mdl64``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from isaaclab_newton.physics import NewtonCfg

if TYPE_CHECKING:
    from pxr import Usd

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import COFFEE_ASSET_DIR
from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env_cfg import CoffeeDemoEnvCfg, CoffeeSceneCfg, configure_scene
from isaaclab_tasks.utils import PresetCfg, preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


def _demo_cfg() -> CoffeeDemoEnvCfg:
    """The demo's configuration after its own ``configure_scene``, which builds configs only."""
    demo = CoffeeDemoEnvCfg()
    # configure_scene reads the visualizer list that the demo's launcher normally resolves.
    demo.sim.visualizer_cfgs = []
    configure_scene(demo)
    return demo


_DEMO = _demo_cfg()

##
# Kitchen
##

KITCHEN_LIGHT_COUNT = 15
KITCHEN_CAMERA_COUNT = 6


def spawn_coffee_kitchen(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn the demo's kitchen without its ceiling, lights and cameras, and hide the table's studio floor.

    Like the demo's presentation, the kitchen keeps its authored placement around the table, so the scene's
    default pose is ignored, and the studio floor is hidden. The ceiling is removed so that the key light reaches
    the table. Cloned into every environment, the kitchen's lights and cameras would multiply with the
    environment count, and RTX Minimal mode does not use them.
    """
    from pxr import Usd, UsdGeom, UsdLux  # noqa: PLC0415

    kitchen = sim_utils.spawn_from_usd(prim_path, cfg, translation=None, orientation=None, **kwargs)
    stage = kitchen.GetStage()
    for root in sim_utils.find_matching_prims(prim_path, stage=stage):
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        lights = [prim for prim in prims if prim.HasAPI(UsdLux.LightAPI)]
        cameras = [prim for prim in prims if prim.IsA(UsdGeom.Camera)]
        # A changed bundle must not silently add lights to the scene.
        if len(lights) != KITCHEN_LIGHT_COUNT or len(cameras) != KITCHEN_CAMERA_COUNT:
            raise RuntimeError(
                f"{root.GetPath()} has {len(lights)} lights and {len(cameras)} cameras; expected "
                f"{KITCHEN_LIGHT_COUNT} and {KITCHEN_CAMERA_COUNT}"
            )
        ceiling = root.GetPrimAtPath("KitchenRoom/Ceiling")
        if not ceiling.IsValid():
            raise RuntimeError(f"{root.GetPath()} has no KitchenRoom/Ceiling")
        for prim in [*lights, *cameras, ceiling]:
            if prim.IsInstanceProxy():
                raise RuntimeError(f"{prim.GetPath()} is inside an instance and cannot be deactivated")
            prim.SetActive(False)
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix("/World/envs") and prim.GetName() == "StudioFloor":
            UsdGeom.Imageable(prim).MakeInvisible()
    return kitchen


KITCHEN_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Kitchen",
    # Keep Newton's visual geometry when using its renderer.
    # RTX renderers get the kitchen from the USD stage or their own clone context.
    cloning_contexts=preset(default=None, ovrtx=(), isaacsim_rtx=(), rtx=()),
    spawn=UsdFileCfg(func=spawn_coffee_kitchen, usd_path=f"{COFFEE_ASSET_DIR}/kitchen.usda"),
)

##
# Camera presets
##


def _view_rotation(eye: tuple[float, float, float], lookat: tuple[float, float, float]) -> tuple[float, ...]:
    """OpenGL-convention camera rotation looking from eye at lookat, as Isaac Lab builds viewer cameras."""
    matrix = create_rotation_matrix_from_view(torch.tensor([eye]), torch.tensor([lookat]), "Z", device="cpu")
    return tuple(quat_from_matrix(matrix)[0].tolist())


# The demo's kitchen view (examples/demos/rizon_sharpa_coffee.py), relative to the environment origin.
_CAMERA_EYE = (0.86, -0.61, 0.70)
_CAMERA_LOOKAT = (0.53, -0.03, 0.10)
# The demo's viewers turn its 24 mm focal length into a vertical field of view with USD's default 15.2908 mm
# aperture (isaaclab/visualizers/base_visualizer.py). A square sensor of that size keeps the demo's
# vertical view and crops the sides of its 4:3 window.
_CAMERA_APERTURE_MM = 15.2908

KITCHEN_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(pos=_CAMERA_EYE, rot=_view_rotation(_CAMERA_EYE, _CAMERA_LOOKAT), convention="opengl"),
    data_types=MISSING,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, horizontal_aperture=_CAMERA_APERTURE_MM, vertical_aperture=_CAMERA_APERTURE_MM
    ),
    width=MISSING,
    height=MISSING,
    renderer_cfg=MultiBackendRendererCfg(),
)


_IMAGE_OBSERVATION_FUNCS = {
    "rgb": mdp.image_rgb,
    "depth": mdp.image_depth,
    "albedo": mdp.image_rgb,
    "simple_shading_constant_diffuse": mdp.image_rgb,
    "simple_shading_diffuse_mdl": mdp.image_rgb,
    "simple_shading_full_mdl": mdp.image_rgb,
    "semantic_segmentation": mdp.image_segmentation,
}
"""Image observation function per camera data type."""

_IMAGE_RESOLUTIONS = (64, 128, 256)


def _camera_presets(make_option: Callable[[str, int], Any]) -> PresetCfg:
    """One option per data type and resolution, named like the Kuka-Allegro camera presets (e.g. ``rgb64``).

    The camera and its image observation share these names, so one ``presets=`` name selects both.
    """
    options = {
        f"{data_type}{resolution}": make_option(data_type, resolution)
        for data_type in _IMAGE_OBSERVATION_FUNCS
        for resolution in _IMAGE_RESOLUTIONS
    }
    return preset(default=options["rgb64"], **options)


def _kitchen_camera(data_type: str, resolution: int) -> CameraCfg:
    return replace(KITCHEN_CAMERA_CFG, data_types=[data_type], width=resolution, height=resolution)


def _camera_image_term(data_type: str, resolution: int) -> ObsTerm:
    return ObsTerm(
        func=_IMAGE_OBSERVATION_FUNCS[data_type],
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": data_type,
            "normalize": False,
            "channel_first": True,
        },
    )


##
# Environment configuration
##


@configclass
class CoffeeKitchenSceneCfg(CoffeeSceneCfg):
    """The demo's scene without its powder, a kitchen in every environment, and one camera per environment."""

    robot = _DEMO.scene.robot
    table = _DEMO.scene.table
    bowl = _DEMO.scene.bowl
    cup = _DEMO.scene.cup
    spoon = _DEMO.scene.spoon
    kitchen: AssetBaseCfg = KITCHEN_CFG
    camera: CameraCfg = _camera_presets(_kitchen_camera)


@configclass
class CoffeeMotionPlaybackCfg(ActionTermCfg):
    """Replay of the demo's recorded arm motion; see :class:`CoffeeMotionPlayback`."""

    class_type: type[ActionTerm] | str = "{DIR}.coffee_kitchen_camera_env:CoffeeMotionPlayback"
    asset_name: str = "robot"
    motion_file: str = str(Path(__file__).with_name("coffee_arm_motion.json"))
    """JSON recording with ``rate_hz``, arm ``joint_names`` and per-step ``positions`` [rad]."""


@configclass
class CoffeeKitchenActionsCfg:
    """No actions: the recorded motion drives the arm."""

    arm_motion = CoffeeMotionPlaybackCfg()


@configclass
class CoffeeKitchenObservationsCfg:
    """The camera image only.

    Reading the image once per step is what triggers the single lazy render per ``env.step``.
    """

    @configclass
    class CameraImageCfg(ObsGroup):
        """Camera observations."""

        image = _camera_presets(_camera_image_term)

    camera_image: CameraImageCfg = CameraImageCfg()


@configclass
class CoffeeKitchenPhysicsCfg(PresetCfg):
    """The demo's robot solver without its MPM coupling."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=_DEMO.sim.physics.solver_cfg.entries[0].solver_cfg,
        collision_cfg=_DEMO.sim.physics.collision_cfg,
        num_substeps=1,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class CoffeeKitchenCameraEnvCfg(ManagerBasedRLEnvCfg):
    """The coffee demo's kitchen scene with one camera per environment."""

    # Each kitchen is 5.9 x 5.4 m, so 8 m spacing keeps the neighbouring kitchens apart.
    scene: CoffeeKitchenSceneCfg = CoffeeKitchenSceneCfg(num_envs=256, env_spacing=8.0)
    observations: CoffeeKitchenObservationsCfg = CoffeeKitchenObservationsCfg()
    actions: CoffeeKitchenActionsCfg = CoffeeKitchenActionsCfg()
    commands = None
    rewards = None
    terminations = None
    curriculum = None
    # Without terminations nothing resets after the first reset, so episodes never end.
    episode_length_s = 3600.0

    def __post_init__(self):
        super().__post_init__()
        # One physics step and one render per step, at the recording's 100 Hz.
        self.decimation = 1
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.use_newton_actuators = True
        self.sim.physics = CoffeeKitchenPhysicsCfg()
