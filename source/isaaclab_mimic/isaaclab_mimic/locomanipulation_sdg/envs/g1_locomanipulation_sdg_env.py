# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGInputData
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import OccupancyMap
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose, SceneBody, SceneFixture

from .g1_locomanipulation_sdg_env_cfg import G1LocomanipulationSDGEnvCfg as G1LocomanipulationSDGEnvCfg
from .g1_locomanipulation_sdg_env_cfg import (
    G1LocomanipulationSDGObservationsCfg as G1LocomanipulationSDGObservationsCfg,
)
from .g1_locomanipulation_sdg_env_cfg import G1LocomanipulationSDGSceneCfg as G1LocomanipulationSDGSceneCfg
from .locomanipulation_sdg_env import LocomanipulationSDGEnv

NUM_FORKLIFTS = 0
NUM_BOXES = 0


class G1LocomanipulationSDGEnv(LocomanipulationSDGEnv):
    def __init__(self, cfg: G1LocomanipulationSDGEnvCfg, **kwargs):
        if cfg.background_usd_path is not None:
            self._num_forklifts = 0
            self._num_boxes = 0
            set_ground_invisible = True
            cfg.scene.add_background_asset(cfg.background_usd_path)
        else:
            self._num_forklifts = NUM_FORKLIFTS
            self._num_boxes = NUM_BOXES
            set_ground_invisible = False

        camera_height, camera_width = (320, 512) if cfg.high_res_video else (160, 256)
        cfg.scene.add_robot_pov_cam(
            camera_height, camera_width, use_nurec_renderer_settings=cfg.background_usd_path is not None
        )

        cfg.scene.add_forklifts(self._num_forklifts)
        cfg.scene.add_boxes(self._num_boxes)

        if set_ground_invisible:
            cfg.scene.ground.spawn.visible = False

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
