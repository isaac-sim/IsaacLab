# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGInputData
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import OccupancyMap
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose, SceneBody, SceneFixture

from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
    ObservationsCfg,
    manip_mdp,
)

from .locomanipulation_sdg_env import LocomanipulationSDGEnv
from .locomanipulation_sdg_env_cfg import LocomanipulationSDGEnvCfg, LocomanipulationSDGRecorderManagerCfg

NUM_FORKLIFTS = 6
NUM_BOXES = 12


@configclass
class G1LocomanipulationSDGSceneCfg(LocomanipulationG1SceneCfg):
    packing_table_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2, -3.55, -0.3],
            # rot=[0, 0, 0, 1]),
            rot=[0, 0, -0.3826834, 0.9238795],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    def add_robot_pov_cam(self, height, width):
        robot_pov_cam = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/torso_link/d435_link/camera",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, -0.1736482, 0.0, 0.9848078), convention="world"),
        )
        setattr(self, "robot_pov_cam", robot_pov_cam)

    def add_background_asset(self, background_usd_path: str):
        background = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Background",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0, 0, 0],
                rot=[0.0, 0.0, 0.0, 1.0],
            ),
            spawn=UsdFileCfg(
                usd_path=background_usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )

        setattr(self, "background", background)

    def add_forklifts(self, num_forklifts: int):
        for i in range(num_forklifts):
            forklift = AssetBaseCfg(
                prim_path=f"/World/envs/env_.*/Forklift{i}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            )
            setattr(self, f"forklift_{i}", forklift)

    def add_boxes(self, num_boxes: int):
        for i in range(num_boxes):
            box = AssetBaseCfg(
                prim_path=f"/World/envs/env_.*/Box{i}",
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
            func=manip_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": False},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1LocomanipulationSDGEnvCfg(LocomanipulationG1EnvCfg, LocomanipulationSDGEnvCfg):
    """Configuration for the G1 29DoF environment."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 3.0, 1.25), lookat=(0.0, 0.0, 0.5), origin_type="asset_body", asset_name="robot", body_name="pelvis"
    )

    # Scene settings
    scene: G1LocomanipulationSDGSceneCfg = G1LocomanipulationSDGSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    recorders: LocomanipulationSDGRecorderManagerCfg = LocomanipulationSDGRecorderManagerCfg()
    observations: G1LocomanipulationSDGObservationsCfg = G1LocomanipulationSDGObservationsCfg()

    background_usd_path: str | None = None
    background_occupancy_yaml_file: str | None = None
    high_res_video: bool = False

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 50.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 6

        # Set the URDF and mesh paths for the IK controller
        urdf_omniverse_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"  # noqa: E501

        # Retrieve local paths for the URDF and mesh files. Will be cached for call after the first time.
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)


class G1LocomanipulationSDGEnv(LocomanipulationSDGEnv):
    def __init__(self, cfg: G1LocomanipulationSDGEnvCfg, **kwargs):
        if cfg.background_usd_path is not None:
            self._num_forklifts = 0
            self._num_boxes = 0
            remove_virtual_ground = True
            cfg.scene.add_background_asset(cfg.background_usd_path)
        else:
            self._num_forklifts = NUM_FORKLIFTS
            self._num_boxes = NUM_BOXES
            remove_virtual_ground = False

        if cfg.high_res_video:
            cfg.scene.add_robot_pov_cam(540, 960)
        else:
            cfg.scene.add_robot_pov_cam(160, 256)

        cfg.scene.add_forklifts(self._num_forklifts)
        cfg.scene.add_boxes(self._num_boxes)

        if remove_virtual_ground:
            delattr(cfg.scene, "ground")

        super().__init__(cfg)
        self.sim.set_camera_view([10.5, 10.5, 10.5], [0.0, 0.0, 0.5])
        self._upper_body_dim = self.action_manager.get_term("upper_body_ik").action_dim
        self._waist_dim = 0  # self._env.action_manager.get_term("waist_joint_pos").action_dim
        self._lower_body_dim = self.action_manager.get_term("lower_body_joint_pos").action_dim
        self._frame_pose_dim = 7
        self._number_of_finger_joints = 7

    def load_input_data(self, episode_data: EpisodeData, step: int) -> LocomanipulationSDGInputData | None:
        dataset_action = episode_data.get_action(step)
        dataset_state = episode_data.get_state(step)

        if dataset_action is None:
            return None

        if dataset_state is None:
            return None

        object_pose = dataset_state["rigid_object"]["object"]["root_pose"]

        base_pose = episode_data.get_initial_state()["articulation"]["robot"]["root_pose"]
        data = LocomanipulationSDGInputData(
            left_hand_pose_target=dataset_action[0:7],
            right_hand_pose_target=dataset_action[7:14],
            left_hand_joint_positions_target=dataset_action[14:21],
            right_hand_joint_positions_target=dataset_action[21:28],
            base_pose=base_pose,
            object_pose=object_pose,
            fixture_pose=torch.tensor(
                [0.0, 0.55, -0.3, 0.0, 0.0, 0.0, 1.0], device=base_pose.device
            ),  # Table pose is not recorded for this env. Quaternion in XYZW format (identity).
        )

        return data

    def build_action_vector(
        self,
        left_hand_pose_target: torch.Tensor,
        right_hand_pose_target: torch.Tensor,
        left_hand_joint_positions_target: torch.Tensor,
        right_hand_joint_positions_target: torch.Tensor,
        base_velocity_target: torch.Tensor,
    ):
        """Build the action vector for the G1 locomanipulation environment.

        Action layout (upper_body_dim = left_pose + right_pose + left_fingers + right_fingers = 28):

            Indices                          Content
            [0:7]                            left hand pose (pos[0:3] + quat[3:7])
            [7:14]                           right hand pose (pos[7:10] + quat[10:14])
            [14:21]                          left hand finger joints (7)
            [21:28]                          right hand finger joints (7)
            [upper_body_dim:upper_body_dim+3]  base velocity (vx, vy, yaw_rate)
            [upper_body_dim+3]               base height
        """
        action = torch.zeros(self.action_space.shape)

        # Set base height
        lower_body_index_offset = self._upper_body_dim + self._waist_dim
        action[0, lower_body_index_offset + 3 : lower_body_index_offset + 4] = torch.tensor([0.8])

        # Left hand pose
        assert left_hand_pose_target.shape == (self._frame_pose_dim,), (
            f"Expected pose shape ({self._frame_pose_dim},), got {left_hand_pose_target.shape}"
        )
        action[0, : self._frame_pose_dim] = left_hand_pose_target

        # Right hand pose
        assert right_hand_pose_target.shape == (self._frame_pose_dim,), (
            f"Expected pose shape ({self._frame_pose_dim},), got {right_hand_pose_target.shape}"
        )
        action[0, self._frame_pose_dim : 2 * self._frame_pose_dim] = right_hand_pose_target

        # Left hand joint positions
        assert left_hand_joint_positions_target.shape == (self._number_of_finger_joints,), (
            f"Expected joint_positions shape ({self._number_of_finger_joints},), got"
            f" {left_hand_joint_positions_target.shape}"
        )
        action[0, 2 * self._frame_pose_dim : 2 * self._frame_pose_dim + self._number_of_finger_joints] = (
            left_hand_joint_positions_target
        )

        # Right hand joint positions
        assert right_hand_joint_positions_target.shape == (self._number_of_finger_joints,), (
            f"Expected joint_positions shape ({self._number_of_finger_joints},), got"
            f" {right_hand_joint_positions_target.shape}"
        )
        action[
            0,
            2 * self._frame_pose_dim + self._number_of_finger_joints : 2 * self._frame_pose_dim
            + 2 * self._number_of_finger_joints,
        ] = right_hand_joint_positions_target

        # Base velocity
        assert base_velocity_target.shape == (3,), f"Expected velocity shape (3,), got {base_velocity_target.shape}"
        lower_body_index_offset = self._upper_body_dim + self._waist_dim
        action[0, lower_body_index_offset : lower_body_index_offset + 3] = base_velocity_target

        return action

    def get_base(self) -> HasPose:
        return SceneBody(self.scene, "robot", "pelvis")

    def get_left_hand(self) -> HasPose:
        return SceneBody(self.scene, "robot", "left_wrist_yaw_link")

    def get_right_hand(self) -> HasPose:
        return SceneBody(self.scene, "robot", "right_wrist_yaw_link")

    def get_object(self) -> HasPose:
        return SceneBody(self.scene, "object", "sm_steeringwheel_a01_01")

    def get_start_fixture(self) -> SceneFixture:
        return SceneFixture.from_boundary(
            self.scene,
            "packing_table",
            np.array([[-1.45, -0.45], [1.45, -0.45], [1.45, 0.45], [-1.45, 0.45]]),
            0.05,
        )

    def get_end_fixture(self) -> SceneFixture:
        return SceneFixture.from_boundary(
            self.scene,
            "packing_table_2",
            np.array([[-1.45, -0.45], [1.45, -0.45], [1.45, 0.45], [-1.45, 0.45]]),
            0.05,
        )

    def get_obstacle_fixtures(self):
        obstacles = [
            SceneFixture.from_boundary(
                self.scene,
                f"forklift_{i}",
                np.array([[-1.0, -1.9], [1.0, -1.9], [1.0, 2.1], [-1.0, 2.1]]),
                0.05,
            )
            for i in range(self._num_forklifts)
        ]

        obstacles += [
            SceneFixture.from_boundary(
                self.scene,
                f"box_{i}",
                np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
                0.05,
            )
            for i in range(self._num_boxes)
        ]

        return obstacles

    def get_background_fixture(self) -> SceneFixture | None:
        if "background" not in self.scene.keys():
            return None

        background_map = OccupancyMap.from_ros_yaml(
            ros_yaml_path=self.cfg.background_occupancy_yaml_file,
        )
        background_fixture = SceneFixture(self.scene, "background", background_map)
        return background_fixture
