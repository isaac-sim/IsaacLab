# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Example on using the Multi-Mesh Raycaster Camera sensor.

.. code-block:: bash

    # with allegro hand
    python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type allegro_hand

    # with anymal-D bodies
    python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type anymal_d

    # with random multiple objects
    python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type objects

"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the multi-mesh raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument(
    "--asset_type",
    type=str,
    default="allegro_hand",
    help="Asset type to use.",
    choices=["allegro_hand", "anymal_d", "objects"],
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
import warp as wp

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

if args_cli.asset_type == "allegro_hand":
    asset_cfg = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ray_caster_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(-0.70, -0.7, -0.25), rot=(0.268976, 0.268976, 0.653951, 0.653951)
        ),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/thumb_link_.*/visuals_xform"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/index_link.*/visuals_xform"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/middle_link_.*/visuals_xform"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/ring_link_.*/visuals_xform"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/palm_link/visuals_xform"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/allegro_mount/visuals_xform"),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=120,
            width=240,
        ),
        debug_vis=not args_cli.headless,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

elif args_cli.asset_type == "anymal_d":
    asset_cfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ray_caster_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0, -0.1, 1.5), rot=(0.0, 1.0, 0.0, 0.0)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LF_.*/visuals"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RF_.*/visuals"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LH_.*/visuals"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RH_.*/visuals"),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/base/visuals"),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=120,
            width=240,
        ),
        debug_vis=not args_cli.headless,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

elif args_cli.asset_type == "objects":
    asset_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
                sim_utils.CylinderCfg(
                    radius=0.2,
                    height=0.5,
                    axis="Y",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CapsuleCfg(
                    radius=0.15,
                    height=0.5,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.ConeCfg(
                    radius=0.2,
                    height=0.5,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    ray_caster_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0, 0.0, 1.5), rot=(0.0, 1.0, 0.0, 0.0)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Object"),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=120,
            width=240,
        ),
        debug_vis=not args_cli.headless,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )
else:
    raise ValueError(f"Unknown asset type: {args_cli.asset_type}")


@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the asset."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            scale=(1, 1, 1),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # asset
    asset = asset_cfg
    # ray caster
    ray_caster = ray_caster_cfg


def randomize_shape_color(prim_path_expr: str):
    """Randomize the color of the geometry."""

    stage = sim_utils.get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression

    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            print("Applying prim scale to:", prim_path)
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())

            # randomize scale
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            scale_spec.default = Gf.Vec3f(random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    triggered = True
    countdown = 42

    # Simulate physics
    while simulation_app.is_running():
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state = wp.to_torch(scene["asset"].data.default_root_state).clone()
            root_state[:, :3] += scene.env_origins
            scene["asset"].write_root_pose_to_sim(root_state[:, :7])
            scene["asset"].write_root_velocity_to_sim(root_state[:, 7:])

            if isinstance(scene["asset"], Articulation):
                # set joint positions with some noise
                joint_pos, joint_vel = (
                    wp.to_torch(scene["asset"].data.default_joint_pos).clone(),
                    wp.to_torch(scene["asset"].data.default_joint_vel).clone(),
                )
                joint_pos += torch.rand_like(joint_pos) * 0.1
                scene["asset"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Asset state...")

        if isinstance(scene["asset"], Articulation):
            # -- generate actions/commands
            default_joint_pos = wp.to_torch(scene["asset"].data.default_joint_pos)
            targets = default_joint_pos + 5 * (torch.rand_like(default_joint_pos) - 0.5)
            # -- apply action to the asset
            scene["asset"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        if not triggered:
            if countdown > 0:
                countdown -= 1
                continue

            data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()  # noqa: F841
            triggered = True
        else:
            continue


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    if args_cli.asset_type == "objects":
        randomize_shape_color(scene_cfg.asset.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*"))

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
