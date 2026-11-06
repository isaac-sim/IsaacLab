# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""View the static MicroDuck Warehouse with selectable camera-renderer presets.

.. code-block:: bash

    uv run isaaclab demo microduck-warehouse
    uv run isaaclab demo microduck-warehouse \\
        renderer=isaacsim_rtx presets=minimal
    uv run isaaclab demo microduck-warehouse \\
        renderer=isaacsim_rtx presets=path_traced
    uv run isaaclab demo microduck-warehouse \\
        renderer=newton_renderer presets=depth

Use ``presets=aisle`` for the other saved camera. Resolution and renderer settings
use normal overrides, e.g. ``env.scene.camera.width=1920 env.scene.camera.height=1080``.
The existing viewer's camera/channel controls display the selected sensor output.
"""

from __future__ import annotations

import argparse

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Number of render frames; -1 runs until the viewer closes."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli, overrides = setup_preset_cli(parser)
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.visualizers import SceneCameraCfg, VisualizerCfg

from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

WAREHOUSE_USD = f"{ISAACLAB_NUCLEUS_DIR}/Environments/MicroDuckWarehouse/microduck_warehouse_static.usdc"


@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """Borrow the static geometry and one camera from the loaded stage."""

    warehouse = AssetBaseCfg(prim_path="/World", spawn=None)
    lighting = AssetBaseCfg(prim_path="/RoofReview", spawn=None)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        spawn=None,
        width=1280,
        height=720,
        data_types=preset(
            default=["rgb"],
            rgb=["rgb"],
            minimal=["simple_shading_full_mdl"],
            depth=["depth"],
            normals=["normals"],
            semantic_segmentation=["semantic_segmentation"],
            instance_segmentation=["instance_segmentation"],
        ),
        renderer_cfg=MultiBackendRendererCfg(
            default=OVRTXRendererCfg(enable_shadows=True),
            ovrtx=OVRTXRendererCfg(enable_shadows=True),
            newton_renderer=NewtonWarpRendererCfg(enable_textures=True, enable_shadows=True),
            isaacsim_rtx=IsaacRtxRendererCfg(
                global_settings=preset(
                    default=IsaacRtxRendererGlobalSettingsCfg(carb_settings={"/rtx/rendermode": "RealTimePathTracing"}),
                    path_traced=IsaacRtxRendererGlobalSettingsCfg(
                        carb_settings={"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 128}
                    ),
                )
            ),
        ),
    )


@configclass
class WarehouseDemoCfg(ManagerBasedEnvCfg):
    """Rendering presets; physics initializes the scene but never advances."""

    decimation = 1
    actions = {}
    observations = {}
    events = None
    ui_window_class_type = None
    scene: WarehouseSceneCfg = WarehouseSceneCfg(num_envs=1, env_spacing=0.0)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        physics=preset(
            default=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False, load_visual_shapes=True),
            isaacsim_rtx=PhysxCfg(),
        ),
        visualizer_cfgs=VisualizerCfg(cameras=[SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera")]),
    )
    camera_exposure_iso = preset(default=3200.0, isaacsim_rtx=None, newton_renderer=None)
    camera_path = preset(
        default="/RoofReview/Cameras/duck_showcase_animation",
        poster="/RoofReview/Cameras/duck_showcase_animation",
        aisle="/RoofReview/Cameras/sunlight_lower",
    )


def main() -> None:
    """Resolve normal Isaac Lab presets and render the scene without stepping physics."""
    # Config-only registration reuses the standard preset and Hydra override resolver.
    gym.register(
        "MicroDuckWarehouseDemo",
        entry_point="isaaclab.envs:ManagerBasedEnv",
        kwargs={"env_cfg_entry_point": WarehouseDemoCfg},
    )
    cfg, _ = resolve_task_config("MicroDuckWarehouseDemo", None, overrides=overrides)
    output = cfg.scene.camera.data_types[0]
    channel = {
        "simple_shading_full_mdl": "rgb",
        "semantic_segmentation": "segmentation",
        "instance_segmentation": "segmentation",
    }.get(output, output)
    for viewer in cfg.sim.visualizer_cfgs:
        viewer.streaming_gt_types = (channel,)
    with launch_simulation(cfg, args_cli):
        from pxr import UsdGeom

        stage = sim_utils.create_new_stage()
        stage.GetRootLayer().subLayerPaths = [retrieve_file_path(WAREHOUSE_USD)]
        source_camera = UsdGeom.Camera(stage.GetPrimAtPath(cfg.camera_path))
        if not source_camera:
            raise ValueError(f"No USD camera at {cfg.camera_path!r} in {WAREHOUSE_USD!r}.")
        camera = UsdGeom.Camera.Define(stage, "/World/envs/env_0/Camera")
        camera.SetFromCamera(source_camera.GetCamera())
        if output == "rgb" and cfg.camera_exposure_iso is not None:
            # Use a fixed physical exposure for the warehouse's dim interior in OVRTX.
            camera.GetPrim().AddAppliedSchema("OmniRtxCameraExposureAPI_1")
            camera.CreateExposureFStopAttr(5.0)
            camera.CreateExposureTimeAttr(0.02)
            camera.CreateExposureIsoAttr(cfg.camera_exposure_iso)
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.register_interactive_scene(scene)
        sim.reset()
        frame = 0
        while sim.is_running() and (args_cli.max_steps < 0 or frame < args_cli.max_steps):
            scene.update(sim.get_physics_dt())
            sim.render()
            frame += 1


if __name__ == "__main__":
    main()
