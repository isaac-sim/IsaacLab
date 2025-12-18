# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose, SceneBody, SceneFixture

from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_fii_env_cfg import (
    FiibotEnvCfg,
    FiibotObservationsCfg,
    FiibotSceneCfg,
    manip_mdp,
)

from .locomanipulation_sdg_env import LocomanipulationSDGEnv
from .locomanipulation_sdg_env_cfg import LocomanipulationSDGEnvCfg, LocomanipulationSDGRecorderManagerCfg


@configclass
class FiibotLocomanipSceneCfg(FiibotSceneCfg):

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.85, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    packing_table_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-17.5, 10.8, 0.0),
            # rot=[0, 0, 0, 1]),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    robot_pov_cam = CameraCfg(
        prim_path="/World/envs/env_.*/robot/head_pitch_Link2",
        update_period=0.0,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.9848078, 0.0, -0.1736482, 0.0), convention="world"),
    )


@configclass
class FiibotLocomanipObservationsCfg(FiibotObservationsCfg):
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    @configclass
    class PolicyCfg(FiibotObservationsCfg.PolicyCfg):

        robot_pov_cam = ObsTerm(
            func=manip_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": False},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class FiibotLocomanipEnvCfg(FiibotEnvCfg, LocomanipulationSDGEnvCfg):
    """Configuration for the G1 29DoF environment."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 5.0, 2.0), lookat=(0.0, 0.0, 0.7), origin_type="asset_body", asset_name="robot", body_name="base_link"
    )

    # Scene settings
    scene: FiibotLocomanipSceneCfg = FiibotLocomanipSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    recorders: LocomanipulationSDGRecorderManagerCfg = LocomanipulationSDGRecorderManagerCfg()
    observations: FiibotLocomanipObservationsCfg = FiibotLocomanipObservationsCfg()

    def __post_init__(self):
        FiibotEnvCfg.__post_init__(self)


class FiibotLocomanipEnv(LocomanipulationSDGEnv):

    def __init__(self, cfg: FiibotLocomanipEnvCfg, **kwargs):
        super().__init__(cfg)
        self.sim.set_camera_view([10.5, 10.5, 10.5], [0.0, 0.0, 0.5])
        self._upper_body_dim = self.action_manager.get_term("upper_body_ik").action_dim
        self._lower_body_dim = self.action_manager.get_term("lower_body_ik").action_dim

    def load_input_data(self, episode_data: EpisodeData, step: int) -> LocomanipulationSDGInputData | None:
        dataset_action = episode_data.get_action(step)
        dataset_state = episode_data.get_state(step)

        if dataset_action is None:
            return None

        if dataset_state is None:
            return None

        object_pose = dataset_state["rigid_object"]["object"]["root_pose"]

        data = LocomanipulationSDGInputData(
            left_hand_pose_target=dataset_action[0:7],
            right_hand_pose_target=dataset_action[7:14],
            left_hand_joint_positions_target=dataset_action[14:16],
            right_hand_joint_positions_target=dataset_action[16:18],
            base_pose=episode_data.get_initial_state()["articulation"]["robot"]["root_pose"],
            object_pose=object_pose,
            fixture_pose=torch.tensor([0.0, 0.85, 0.0, 1.0, 0.0, 0.0, 0.0]),  # Table pose is not recorded for this env.
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

        action = torch.zeros(self.action_space.shape)
        action[0, 0:7] = left_hand_pose_target
        action[0, 7:14] = right_hand_pose_target
        action[0, 14:16] = left_hand_joint_positions_target
        action[0, 16:18] = right_hand_joint_positions_target
        action[0, 18:21] = base_velocity_target
        action[0, 21:22] = 0.7

        return action

    def get_base(self) -> HasPose:
        return SceneBody(self.scene, "robot", "base_link")

    def get_left_hand(self) -> HasPose:
        return SceneBody(self.scene, "robot", "left_7_Link")

    def get_right_hand(self) -> HasPose:
        return SceneBody(self.scene, "robot", "right_7_Link")

    def get_object(self) -> HasPose:
        return SceneBody(self.scene, "object", "Cube")

    def get_start_fixture(self) -> SceneFixture:
        return SceneFixture(
            self.scene,
            "packing_table",
            occupancy_map_boundary=np.array([[-1.55, -0.45], [1.55, -0.45], [1.55, 0.45], [-1.55, 0.45]]),
            occupancy_map_resolution=0.05,
        )

    def get_end_fixture(self) -> SceneFixture:
        return SceneFixture(
            self.scene,
            "packing_table_2",
            occupancy_map_boundary=np.array([[-1.55, -0.45], [1.55, -0.45], [1.55, 0.45], [-1.55, 0.45]]),
            occupancy_map_resolution=0.05,
        )

    def get_obstacle_fixtures(self):
        return []
