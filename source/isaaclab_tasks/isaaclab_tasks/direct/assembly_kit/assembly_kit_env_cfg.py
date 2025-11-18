# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import json
import random
import torch
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, PinholeCameraCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def parse_json_episode_data(asset_dir) -> tuple[
    list[dict],
    list[tuple[float, float, float]],
    float,
    bool,
]:
    """Parse the JSON episode data from the asset directory."""
    with open(asset_dir.joinpath("episodes.json")) as json_data:
        episode_json = json.load(json_data)
    episodes = episode_json["episodes"]

    color = episode_json["config"]["color"]
    object_scale = episode_json["config"]["object_scale"]

    symmetry = episode_json["config"]["symmetry"]

    return episodes, color, object_scale, symmetry


def compute_asset_paths(kit_dir: Path, model_dir: Path) -> tuple[
    list[str],
    list[list[int]],
    list[list[str]],
    list[list[tuple[float, float, float]]],
    list[list[tuple[float, float, float, float]]],
    list[tuple[float, float, float]],
]:
    """Compute the asset paths for the environment."""

    # Collecting all the kit USD paths
    kit_usd_paths = sorted(
        [str(p / f"{p.stem}.usda") for p in kit_dir.iterdir() if p.is_dir()],
        key=lambda p: int(Path(p).stem.split("_")[-1]),
    )

    # Collecting all the model paths from the kit JSON files
    kit_json_paths = sorted(
        [str(p) for p in kit_dir.iterdir() if p.match("*.json")],
        key=lambda p: int(Path(p).stem.split("_")[-1]),
    )

    # Parsing the kit JSON files to get the object IDs and the goal poses
    kit_model_ids = []
    kit_model_positions = []
    kit_model_rots = []
    kit_target_starting_pos = []
    for kit_json_path in kit_json_paths:
        with open(kit_json_path) as json_data:
            kit_json = json.load(json_data)
        kit_model_ids.append([obj["object_id"] for obj in kit_json["objects"]])
        poses = [o["pos"] for o in kit_json["objects"]]
        rots = [o["rot"] for o in kit_json["objects"]]
        kit_model_positions.append(poses)
        kit_model_rots.append(rots)
        kit_target_starting_pos.append(kit_json["start_pos_proposal"])

    # Grouping the model paths for MultiUsdFileCfg
    kit_models_paths = [
        [str(model_dir.joinpath(f"model_{model_id:02d}", f"model_{model_id:02d}.usda")) for model_id in group]
        for group in zip(*kit_model_ids)
    ]

    return kit_usd_paths, kit_model_ids, kit_models_paths, kit_model_positions, kit_model_rots, kit_target_starting_pos


def get_kit_cfg(kit_usd_paths: list[str]) -> RigidObjectCfg:
    """Builds the RigidObjectCfg for the kit assembly platform.

    Args:
        kit_usd_paths: List of USD paths for kit variants.

    Returns:
        Config object for spawning the kit in all envs.
    """
    kit_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Kit",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=kit_usd_paths,
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.27807487, 0.20855615, 0.16934046),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.5,
                metallic=0.0,
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.007), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    return kit_cfg


def get_model_cfg(model_paths: list[str], model_idx: int, color) -> RigidObjectCfg:
    """Generates the RigidObjectCfg for a single model with randomized color.

    Args:
        model_paths: List of USD paths for the model variants.
        model_idx:   Index of this model in the kit.

    Returns:
        Config object for spawning the model in the scene.
    """
    r, g, b, a = color[random.randint(0, len(color) - 1)]

    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Model_{model_idx}",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=model_paths,
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(r, g, b),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.5,
                metallic=0.0,
                opacity=a,
            ),
            scale=(0.98, 0.98, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0, 0.1), rot=(0.0, 0.0, 0.0, 1.0)),
    )


@configclass
class AssemblyKitEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 4  # 4 / (decimation * dt) = 200 steps
    # - spaces definition
    action_space = 9
    observation_space = 20
    state_space = 0

    obs_mode: str = "state"
    robot_controller: str = "joint_space"

    TABLE_OFFSET = 0.55

    asset_dir = Path("/home/johann/Downloads/assembly_kit_noremesh")
    kit_dir = asset_dir.joinpath("kits")
    model_dir = asset_dir.joinpath("models")

    # simulation
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 100,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
        replicate_physics=False,
    )

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    sensors = (
        CameraCfg(
            prim_path="/World/envs/env_.*/Camera",  # Fixed camera
            # prim_path="/World/envs/env_.*/Robot/panda_hand/Camera",  # attach to the wrist link
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=1.0,
                horizontal_aperture=36.0,  # gives ~90° HFOV with f=24
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(TABLE_OFFSET + 0.3, 0.0, 0.4),  # offset from center view
                rot=(
                    tuple(
                        quat_from_euler_xyz(
                            torch.tensor(0.6),
                            torch.tensor(torch.pi),
                            torch.tensor(-torch.pi / 2.0),
                        ).tolist()
                    )
                ),  # identity quaternion
                convention="ros",  # right-handed, X-forward, Z-up
            ),
        ),
    )

    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    tcp_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link7",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                ),
            ),
        ],
    )

    episode, color, object_scale, symmetry = parse_json_episode_data(asset_dir)

    kit_usd_paths, kit_model_ids, kit_models_paths, kit_model_positions, kit_model_rots, kit_target_starting_pos = (
        compute_asset_paths(kit_dir, model_dir)
    )


@configclass
class AssemblyKitCameraEnvCfg(AssemblyKitEnvCfg):

    obs_mode: str = "camera"


@configclass
class AssemblyKitTeleopCfg(AssemblyKitEnvCfg):

    robot_controller = "task_space"

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True

    viewer = ViewerCfg()
    viewer.eye = (1.6, 0.0, 1.2)
    viewer.lookat = (0.0, 0.0, 0.2)
    viewer.origin_type = "env"
    viewer.env_index = 0
