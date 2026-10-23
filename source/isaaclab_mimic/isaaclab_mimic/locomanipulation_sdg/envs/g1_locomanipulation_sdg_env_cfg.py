# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.locomanip_pick_place import mdp as locomanip_mdp
from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
    ObservationsCfg,
    manip_mdp,
)

from .locomanipulation_sdg_env_cfg import LocomanipulationSDGEnvCfg, LocomanipulationSDGRecorderManagerCfg

ISAAC_RTX_GAUSSIAN_CAMERA_RENDERER_CARB_SETTINGS = {
    "/rtx/rtpt/gaussian/accumulatedDepth/allHits/enabled": True,
    "/rtx/rtpt/gaussian/accumulatedAlbedo/enabled": True,
    "/rtx/rtpt/gaussian/maxGaussiansToAccumulate": 360,
}


@configclass
class G1LocomanipulationSDGSceneCfg(LocomanipulationG1SceneCfg):
    packing_table_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2, -3.55, -0.3],
            rot=[0, 0, -0.3826834, 0.9238795],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
    )

    def add_robot_pov_cam(self, height, width, use_nurec_renderer_settings: bool = False):
        camera_kwargs = {}
        if use_nurec_renderer_settings:
            camera_kwargs["renderer_cfg"] = IsaacRtxRendererCfg(
                global_settings=IsaacRtxRendererGlobalSettingsCfg(
                    enable_dl_denoiser=True,
                    antialiasing_mode="DLSS",
                    carb_settings=ISAAC_RTX_GAUSSIAN_CAMERA_RENDERER_CARB_SETTINGS,
                )
            )

        self.robot_pov_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, -0.1736482, 0.0, 0.9848078), convention="world"),
            **camera_kwargs,
        )

    def add_background_asset(self, background_usd_path: str):
        self.background = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Background",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0, 0, 0],
                rot=[0.0, 0.0, 0.0, 1.0],
            ),
            spawn=UsdFileCfg(
                usd_path=background_usd_path,
            ),
        )

    def add_forklifts(self, num_forklifts: int):
        for i in range(num_forklifts):
            forklift = AssetBaseCfg(
                prim_path=f"/World/envs/env_[^/]+/Forklift{i}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                ),
            )
            setattr(self, f"forklift_{i}", forklift)

    def add_boxes(self, num_boxes: int):
        for i in range(num_boxes):
            box = AssetBaseCfg(
                prim_path=f"/World/envs/env_[^/]+/Box{i}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd",
                ),
            )
            setattr(self, f"box_{i}", box)


@configclass
class G1LocomanipulationSDGObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        robot_pov_cam = ObsTerm(
            func=manip_mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "normalize": False},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1LocomanipulationSDGEnvCfg(LocomanipulationG1EnvCfg, LocomanipulationSDGEnvCfg):
    """Configuration for the G1 29DoF environment."""

    # Scene settings
    scene: G1LocomanipulationSDGSceneCfg = G1LocomanipulationSDGSceneCfg(num_envs=1, env_spacing=2.5)
    recorders: LocomanipulationSDGRecorderManagerCfg = LocomanipulationSDGRecorderManagerCfg()
    observations: G1LocomanipulationSDGObservationsCfg = G1LocomanipulationSDGObservationsCfg()

    background_usd_path: str | None = None
    background_occupancy_yaml_file: str | None = None
    high_res_video: bool = False

    def __post_init__(self):
        """Post initialization."""
        # This class overrides LocomanipulationG1EnvCfg.__post_init__, so preserve the
        # contact reporting required by the inherited per-hand contact sensors.
        self.scene.robot.spawn.activate_contact_sensors = True

        # general settings
        self.decimation = 4
        self.episode_length_s = 50.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 6
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.0, 3.0, 1.25), lookat=(0.0, 0.0, 0.5))

        # The IK controller retrieves the URDF when it initializes.
        self.actions.upper_body_ik.controller.urdf_path = (
            f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/"
            "g1_29dof_with_hand_only_kinematics.urdf"
        )

        # Override success condition: check placement relative to the drop-off table (packing_table_2),
        # since this env has two tables: packing_table (pick-up) and packing_table_2 (drop-off).
        self.terminations.success = DoneTerm(
            func=locomanip_mdp.task_done_pick_place_table_frame,
            params={
                "task_link_name": "right_wrist_yaw_link",
                "table_cfg": SceneEntityCfg("packing_table_2"),
            },
        )
