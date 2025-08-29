# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from pathlib import Path

from common import DisjointNavRecording, DisjointNavRecordingItem, DisjointNavScenario, HasPose, SceneBody, SceneFixture
from mdp.actions import G1_UPPER_BODY_IK_ACTION_CFG, LowerBodyActionCfg
from occupancy_map import OccupancyMap

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.manager_based_rl_mimic_env import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import ObservationsCfg
from dataclasses import asdict
from common import DisjointNavReplayState

NUM_FORKLIFTS = 6
NUM_BOXES = 12

G1_LOCOMANIPULATION_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/agile/Robots/Collected_g1/g1_collision_geom_simplified_bigger_offset_with_hand_collision.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,  # Configurable - can be set to True for fixed base
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 3000.0,
                "waist_roll_joint": 3000.0,
                "waist_pitch_joint": 3000.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=3000.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*_joint": 0.01,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_.*",
                ".*_middle_.*",
                ".*_thumb_.*",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=4000,
            damping=50,
            armature=0.001,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    packing_table_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2, -3.55, -0.3],
            # rot=[0, 0, 0, 1]),
            rot=[0.9238795, 0, 0, -0.3826834],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.9996 - 0.3], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = G1_LOCOMANIPULATION_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# Add forklifts
for i in range(NUM_FORKLIFTS):
    setattr(
        ObjectTableSceneCfg, 
        f"forklift_{i}", 
        AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/Forklift{i}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )        
    )

# Add boxes
for i in range(NUM_BOXES):
    setattr(
        ObjectTableSceneCfg, 
        f"box_{i}", 
        AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/Box{i}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
                
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )        
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

    # This term can be removed on the ik supports the waist joints.
    waist_joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_.*_joint"],
        use_default_offset=True,
        clip={".*": (-0.0, 0.0)},
    )

    lower_body_joint_pos = LowerBodyActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
        ],
        scale=0.25,
        obs_group_name="lower_body_policy",  # need to be the same name as the on in ObservationCfg
        policy_path=Path(__file__).parent / "policy/g1/agile_locomotion.pt",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


class PreStepLowerBodyPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs_lower_body", self._env.obs_buf["lower_body_policy"]


@configclass
class PreStepLowerBodyPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = PreStepLowerBodyPolicyObservationsRecorder


class DisjointNavReplayStateRecorder(RecorderTerm):

    def record_pre_step(self):
        replay_state: DisjointNavReplayState = self._env._replay_state

        replay_state_dict = {
            "left_hand_pose_target": replay_state.left_hand_pose_target[None, :],
            "right_hand_pose_target": replay_state.right_hand_pose_target[None, :],
            "left_hand_joint_positions_target": replay_state.left_hand_joint_positions_target[None, :],
            "right_hand_joint_positions_target": replay_state.right_hand_joint_positions_target[None, :],
            "base_velocity_target": replay_state.base_velocity_target[None, :],
            "start_fixture_pose": replay_state.start_fixture_pose,
            "end_fixture_pose": replay_state.end_fixture_pose,
            "object_pose": replay_state.object_pose,
            "base_pose": replay_state.base_pose,
            "task": torch.tensor([[replay_state.task]]),
            "base_goal_pose": replay_state.base_goal_pose,
            "base_goal_approach_pose": replay_state.base_goal_approach_pose,
            "base_path": replay_state.base_path[None, :],
            "recording_step": torch.tensor([[replay_state.recording_step]]),
            "obstacle_fixture_poses": replay_state.obstacle_fixture_poses
        }
        
        return "replay_state", replay_state_dict


@configclass
class DisjointNavReplayStateRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = DisjointNavReplayStateRecorder


class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_lower_body_policy_observations = PreStepLowerBodyPolicyObservationsRecorderCfg()
    record_pre_step_disjoint_nav_replay_state = DisjointNavReplayStateRecorderCfg()


@configclass
class G129DoFDisjointNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 29DoF environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 1 / 200  # 100Hz
        self.sim.render_interval = 2

        # Set the URDF and mesh paths for the IK controller
        urdf_omniverse_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/agile/Robots/urdf/g1/g1_minimal_with_leg_hand_collision_corrected.urdf"
        mesh_omniverse_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/agile/Robots/urdf/g1/meshes"

        # Retrieve local paths for the URDF and mesh files. Will be cached for call after the first time.
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)
        self.actions.upper_body_ik.controller.mesh_path = retrieve_file_path(mesh_omniverse_path)


class PackingTable(SceneFixture):

    def get_occupancy_map(self):

        local_occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=np.array([[-1.45, -0.45], [1.45, -0.45], [1.45, 0.45], [-1.45, 0.45]]), resolution=0.05
        )

        transform = self.get_transform_2d().detach().cpu().numpy()

        occupancy_map = local_occupancy_map.transformed(transform)

        return occupancy_map


class Forklift(SceneFixture):

    def get_occupancy_map(self):

        local_occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=np.array([[-1.0, -1.9], [1.0, -1.9], [1.0, 2.1], [-1.0, 2.1]]), resolution=0.05
        )

        transform = self.get_transform_2d().detach().cpu().numpy()

        occupancy_map = local_occupancy_map.transformed(transform)

        return occupancy_map


class CardboardBox(SceneFixture):

    def get_occupancy_map(self):

        local_occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]), resolution=0.05
        )

        transform = self.get_transform_2d().detach().cpu().numpy()

        occupancy_map = local_occupancy_map.transformed(transform)

        return occupancy_map


class G1DisjointNavRecording(DisjointNavRecording):

    def __init__(self, path: str, demo: str = "demo_0", device: str = "cpu"):
        self.dataset_file_handler = HDF5DatasetFileHandler()
        self.dataset_file_handler.open(path)
        self.episode_data = self.dataset_file_handler.load_episode(demo, device)
        self._robot_base_pose = self.episode_data.get_initial_state()["articulation"]["robot"]["root_pose"]

    def get_initial_state(self):

        initial_state = self.episode_data.get_initial_state()
        return initial_state

    def get_item(self, step: int) -> DisjointNavRecordingItem | None:

        dataset_action = self.episode_data.get_action(step)
        dataset_state = self.episode_data.get_state(step)

        if dataset_action is None:
            return None

        if dataset_state is None:
            return None

        base_pose = dataset_state["articulation"]["robot"]["root_pose"]
        object_pose = dataset_state["rigid_object"]["object"]["root_pose"]

        target = DisjointNavRecordingItem(
            left_hand_pose_target=dataset_action[0:7],
            right_hand_pose_target=dataset_action[7:14],
            left_hand_joint_positions_target=dataset_action[14:21],
            right_hand_joint_positions_target=dataset_action[21:28],
            base_pose=self._robot_base_pose,
            object_pose=object_pose,
            fixture_pose=torch.tensor(
                [0.0, 0.55, -0.3, 1.0, 0.0, 0.0, 0.0]
            ),  # Table pose is not recorded for this env.
        )

        return target


class G1DisjointNavScenario(DisjointNavScenario):

    def __init__(self, output_dir: str, output_file_name: str):
        self._env_cfg = G129DoFDisjointNavEnvCfg()
        self._env_cfg.sim.device = "cpu"
        # self._env_cfg.sim.render.rendering_mode = "performance"

        self._env_cfg.scene.num_envs = 1

        self._env_cfg.recorders = RecorderManagerCfg()
        self._env_cfg.recorders.dataset_export_dir_path = output_dir
        self._env_cfg.recorders.dataset_filename = output_file_name

        self._env = ManagerBasedRLMimicEnv(cfg=self._env_cfg)

        self._env.sim.set_camera_view([10.5, 10.5, 10.5], [0.0, 0.0, 0.5])
        self._upper_body_dim = self._env.action_manager.get_term("upper_body_ik").action_dim
        self._waist_dim = self._env.action_manager.get_term("waist_joint_pos").action_dim
        self._lower_body_dim = self._env.action_manager.get_term("lower_body_joint_pos").action_dim
        self._frame_pose_dim = 7
        self._number_of_finger_joints = 7
        self._env_action = torch.zeros(self._env.action_space.shape)
        self.set_base_height_target()

    def set_left_hand_pose_target(self, pose: torch.Tensor):
        assert pose.shape == (self._frame_pose_dim,), f"Expected pose shape ({self._frame_pose_dim},), got {pose.shape}"
        self._env_action[0, : self._frame_pose_dim] = pose

    def set_right_hand_pose_target(self, pose: torch.Tensor):
        assert pose.shape == (self._frame_pose_dim,), f"Expected pose shape ({self._frame_pose_dim},), got {pose.shape}"
        self._env_action[0, self._frame_pose_dim : 2 * self._frame_pose_dim] = pose

    def set_left_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        assert joint_positions.shape == (
            self._number_of_finger_joints,
        ), f"Expected joint_positions shape ({self._number_of_finger_joints},), got {joint_positions.shape}"
        self._env_action[0, 2 * self._frame_pose_dim : 2 * self._frame_pose_dim + self._number_of_finger_joints] = (
            joint_positions
        )

    def set_right_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        assert joint_positions.shape == (
            self._number_of_finger_joints,
        ), f"Expected joint_positions shape ({self._number_of_finger_joints},), got {joint_positions.shape}"
        self._env_action[
            0,
            2 * self._frame_pose_dim
            + self._number_of_finger_joints : 2 * self._frame_pose_dim
            + 2 * self._number_of_finger_joints,
        ] = joint_positions

    def set_base_velocity_target(self, velocity: torch.Tensor):
        assert velocity.shape == (3,), f"Expected velocity shape (3,), got {velocity.shape}"
        lower_body_index_offset = self._upper_body_dim + self._waist_dim
        self._env_action[0, lower_body_index_offset : lower_body_index_offset + 3] = velocity

    def set_base_height_target(self, height: torch.Tensor = torch.tensor([0.72])):
        assert height.shape == (1,), f"Expected height shape (1,), got {height.shape}"
        lower_body_index_offset = self._upper_body_dim + self._waist_dim
        self._env_action[0, lower_body_index_offset + 3 : lower_body_index_offset + 4] = height

    def get_base(self) -> HasPose:
        return SceneBody(self._env.scene, "robot", "pelvis")

    def get_left_hand(self) -> HasPose:
        return SceneBody(self._env.scene, "robot", "left_wrist_yaw_link")

    def get_right_hand(self) -> HasPose:
        return SceneBody(self._env.scene, "robot", "right_wrist_yaw_link")

    def get_object(self) -> HasPose:
        return SceneBody(self._env.scene, "object", "sm_steeringwheel_a01_01")

    def get_start_fixture(self) -> SceneFixture:
        return PackingTable(self._env.scene, "packing_table")

    def get_end_fixture(self) -> SceneFixture:
        return PackingTable(self._env.scene, "packing_table_2")

    def get_obstacle_fixtures(self):
        obstacles = [Forklift(self._env.scene, f"forklift_{i}") for i in range(NUM_FORKLIFTS)]
        obstacles += [CardboardBox(self._env.scene, f"box_{i}") for i in range(NUM_BOXES)]
        return obstacles

    def reset(self, initial_state=None):
        if initial_state is not None:
            self._env.reset_to(initial_state, env_ids=torch.tensor([0]))
        else:
            self._env.reset()

    def get_env(self):
        return self._env
    
    def step(self):
        self._env.step(self._env_action)

    def close(self):
        self._env.close()
