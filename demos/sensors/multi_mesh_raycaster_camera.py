# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Example on using the Multi-Mesh Raycaster Camera sensor.

.. code-block:: bash

    # with allegro hand
    uvx --from 'isaaclab[isaacsim]' isaaclab demo multi-mesh-ray-caster-camera --num_envs 16 --asset_type allegro_hand

    # with anymal-D bodies
    uvx --from 'isaaclab[isaacsim]' isaaclab demo multi-mesh-ray-caster-camera --num_envs 16 --asset_type anymal_d

    # with random multiple objects
    uvx --from 'isaaclab[isaacsim]' isaaclab demo multi-mesh-ray-caster-camera --num_envs 16 --asset_type objects

"""

import argparse
import random

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Example on using the multi-mesh raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument(
    "--asset_type",
    type=str,
    default="allegro_hand",
    help="Asset type to use.",
    choices=["allegro_hand", "anymal_d", "objects"],
)
parser.add_argument("--log_interval", type=int, default=100, help="Steps between compact sensor summaries.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx"],
    help="Physics backend.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.log_interval < 1:
    parser.error("--log_interval must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Simulator-dependent imports must follow AppLauncher initialization.
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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
            rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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


def randomize_shape_color(prim_path_expr: str) -> None:
    """Randomize the color of the geometry."""

    stage = sim_utils.get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression

    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            scale_spec.default = Gf.Vec3f(random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Run the simulator."""
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running() and (args_cli.max_steps < 0 or count < args_cli.max_steps):
        if count % 500 == 0:
            root_pose = scene["asset"].data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            scene["asset"].write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = scene["asset"].data.default_root_vel.torch.clone()
            scene["asset"].write_root_velocity_to_sim_index(root_velocity=root_vel)

            if isinstance(scene["asset"], Articulation):
                joint_pos, joint_vel = (
                    scene["asset"].data.default_joint_pos.torch.clone(),
                    scene["asset"].data.default_joint_vel.torch.clone(),
                )
                joint_pos += torch.rand_like(joint_pos) * 0.1
                scene["asset"].write_joint_position_to_sim_index(position=joint_pos)
                scene["asset"].write_joint_velocity_to_sim_index(velocity=joint_vel)
            scene.reset()
            print("[INFO]: Resetting Asset state...")

        if isinstance(scene["asset"], Articulation):
            default_joint_pos = scene["asset"].data.default_joint_pos.torch
            targets = default_joint_pos + 5 * (torch.rand_like(default_joint_pos) - 0.5)
            scene["asset"].set_joint_position_target_index(target=targets)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        if count % args_cli.log_interval == 0:
            hits = scene["ray_caster"].data.ray_hits_w.torch
            valid = torch.isfinite(hits).all(dim=-1)
            print(f"[INFO] step={count} ray hit rate={valid.float().mean().item():.1%}")


def main() -> None:
    """Run the multi-mesh ray-caster camera demo."""

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
    try:
        main()
    finally:
        simulation_app.close()
