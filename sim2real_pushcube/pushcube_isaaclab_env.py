"""IsaacLab PushCube environment mirroring ManiSkill PushCube-v1 (Panda, RGB+state).

Standalone (does NOT use the manager-based task framework). Built directly on
``InteractiveScene`` so the observation/action contract can be matched to
ManiSkill exactly, which is required for the ManiSkill-trained checkpoint to be
loaded and evaluated here.

Contract (must match ManiSkill PushCube-v1 + ppo_rgb.py):
- obs: dict ``{"rgb": uint8[N,128,128,3] (HWC), "state": float32[N,35]}``
  state order = [qpos(9), qvel(9), tcp_pose(7, wxyz), goal_pos(3), obj_pose(7, wxyz)]
    qpos/qvel in ManiSkill joint order: panda_joint1..7, panda_finger_joint1,2
    tcp_pose = panda_hand link pose + [0,0,0.1034] offset  (wxyz quaternion)
    goal_pos = cube_xy + [0.2, 0, 0]  (world frame)
    obj_pose = cube root pose (wxyz quaternion)
- action: float32[N,8] = [arm_delta(7) in [-0.1,0.1], gripper_abs_target(1) in [-0.01,0.04]]
    arm:  target = current_arm_qpos + clip(action[:7], -0.1, 0.1)   (pd_joint_delta_pos)
    grip: target = clip(action[7], -0.01, 0.04) applied to BOTH fingers (mimic)
- dynamics: sim_dt=0.01, 5 sim substeps per action (control 20Hz), 50 actions/episode
- PD gains (arm + fingers): stiffness=1e3, damping=1e2, effort_limit=100
- success: cube xy within 0.1 of goal_xy (Euclidean) AND cube z < 0.025  (sticky per episode)

IsaacLab 3.x notes baked in:
- ``_index``-suffixed methods; data tensors accessed via ``.torch``
- quaternion convention is (x,y,z,w); ManiSkill is (w,x,y,z) -> convert in state obs
- write_root_pose_to_sim_index expects WORLD frame -> add scene.env_origins
"""
from __future__ import annotations

import copy

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

# ManiSkill PushCube-v1 constants
SIM_DT = 0.01            # sim_freq = 100 Hz
DECIMATION = 5           # control_freq = 20 Hz  (sim_freq // control_freq)
MAX_STEPS = 50           # max_episode_steps
CUBE_HALF = 0.02         # 4 cm cube
GOAL_RADIUS = 0.1
GOAL_DX = 0.2            # goal = cube_xy + [0.1 + goal_radius, 0, 0] = cube_xy + [0.2, 0, 0]
ROBOT_BASE = (-0.615, 0.0, 0.0)
TCP_OFFSET = (0.0, 0.0, 0.1034)
ARM_REST_QPOS = torch.tensor(
    [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4]
)
ARM_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
# ManiSkill cube color (12, 42, 160) / 255
CUBE_COLOR = (12 / 255, 42 / 255, 160 / 255)


@configclass
class PushCubeSceneCfg(InteractiveSceneCfg):
    """Scene: ground (= table surface at z=0), Franka Panda, 4cm cube, one 128x128 camera."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # --- robot (Franka Panda, base shifted to match ManiSkill, stiff PD gains) ---
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = ROBOT_BASE
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": np.pi / 8,
        "panda_joint3": 0.0,
        "panda_joint4": -np.pi * 5 / 8,
        "panda_joint5": 0.0,
        "panda_joint6": np.pi * 3 / 4,
        "panda_joint7": np.pi / 4,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    # deepcopy actuators so we don't mutate the global FRANKA_PANDA_CFG, then match ManiSkill gains
    robot.actuators = copy.deepcopy(robot.actuators)
    for _grp in ("panda_shoulder", "panda_forearm", "panda_hand"):
        robot.actuators[_grp].stiffness = 1e3
        robot.actuators[_grp].damping = 1e2
        robot.actuators[_grp].effort_limit_sim = 100.0

    # --- object (4 cm rigid cube, blue, resting on ground at z=0.02) ---
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CUBE_HALF)),
        spawn=sim_utils.CuboidCfg(
            size=(2 * CUBE_HALF, 2 * CUBE_HALF, 2 * CUBE_HALF),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # TODO: match ManiSkill cube mass
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_COLOR),
        ),
    )

    # --- camera (128x128, 90 deg fov -> fx=fy=64, cx=cy=64; pose set at runtime via look_at) ---
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,  # render every sim step (set to SIM_DT*DECIMATION to render once per action)
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[64.0, 0.0, 64.0, 0.0, 64.0, 64.0, 0.0, 0.0, 1.0],
            width=128,
            height=128,
            clipping_range=(0.01, 100.0),  # ManiSkill near=0.01, far=100
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.6), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    )


class PushCubeIsaacLabEnv:
    """Standalone IsaacLab PushCube env matching the ManiSkill PushCube-v1 contract."""

    def __init__(self, num_envs: int, device: str, sim_dt: float = SIM_DT,
                 decimation: int = DECIMATION, max_steps: int = MAX_STEPS):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.max_steps = max_steps

        # simulation
        sim_cfg = sim_utils.SimulationCfg(device=device, dt=sim_dt)
        self.sim = SimulationContext(sim_cfg)

        # scene
        scene_cfg = PushCubeSceneCfg(num_envs=num_envs, env_spacing=2.5)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # entities
        self.robot = self.scene["robot"]
        self.cube = self.scene["object"]
        self.camera = self.scene["camera"]

        # camera pose = sapien look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1]), per env (world frame)
        eye = self.scene.env_origins + torch.tensor([0.3, 0.0, 0.6], device=self.device)
        target = self.scene.env_origins + torch.tensor([-0.1, 0.0, 0.1], device=self.device)
        self.camera.set_world_poses_from_view(eye, target)

        # joint / body indices (resolve via names - never assume USD order)
        arm_idx_list, _ = self.robot.find_joints(ARM_NAMES, preserve_order=True)
        finger_idx_list, _ = self.robot.find_joints(FINGER_NAMES, preserve_order=True)
        self.arm_idx = torch.tensor(arm_idx_list, device=self.device, dtype=torch.long)      # (7,)
        self.finger_idx = torch.tensor(finger_idx_list, device=self.device, dtype=torch.long)  # (2,)
        self.joint_idx = torch.cat([self.arm_idx, self.finger_idx])  # (9,) ManiSkill order
        self.num_joints = self.robot.num_joints
        self.hand_body_idx = self.robot.find_bodies("panda_hand")[0][0]  # int
        self.tcp_offset = torch.tensor(TCP_OFFSET, device=self.device)
        self.arm_rest_qpos = ARM_REST_QPOS.to(self.device)

        # buffers
        self.step_count = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.ever_success = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.goal_pos = torch.zeros(num_envs, 3, device=self.device)

        # prime one step so the camera has rendered before the first obs
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim_dt)

    # ------------------------------------------------------------------ obs
    def get_obs(self) -> dict:
        rgb = self.camera.data.output["rgb"].torch[..., :3].contiguous()  # (N,128,128,3) uint8

        qpos = self.robot.data.joint_pos.torch[:, self.joint_idx]   # (N,9)
        qvel = self.robot.data.joint_vel.torch[:, self.joint_idx]   # (N,9)

        # tcp pose = panda_hand link pose + [0,0,0.1034] offset (IsaacLab quat is xyzw)
        hand_pose = self.robot.data.body_link_pose_w.torch[:, self.hand_body_idx]  # (N,7) [xyz, xyzw]
        hand_pos = hand_pose[:, :3]
        hand_quat_xyzw = hand_pose[:, 3:7]
        tcp_pos = hand_pos + math_utils.quat_apply(hand_quat_xyzw, self.tcp_offset.expand(self.num_envs, 3))
        tcp_quat_wxyz = hand_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        tcp_pose = torch.cat([tcp_pos, tcp_quat_wxyz], dim=1)  # (N,7)

        # object pose (world); convert xyzw -> wxyz
        obj_pos = self.cube.data.root_pos_w.torch                  # (N,3)
        obj_quat_xyzw = self.cube.data.root_quat_w.torch           # (N,4)
        obj_quat_wxyz = obj_quat_xyzw[:, [3, 0, 1, 2]]
        obj_pose = torch.cat([obj_pos, obj_quat_wxyz], dim=1)      # (N,7)

        # goal_pos is stored in world frame
        state = torch.cat([qpos, qvel, tcp_pose, self.goal_pos, obj_pose], dim=1)  # (N,35)
        return {"rgb": rgb, "state": state}

    # --------------------------------------------------------------- success
    def _compute_success(self) -> torch.Tensor:
        cube_pos = self.cube.data.root_pos_w.torch  # (N,3) world
        dist = torch.linalg.norm(cube_pos[:, :2] - self.goal_pos[:, :2], dim=1)
        on_table = cube_pos[:, 2] < (CUBE_HALF + 5e-3)  # 0.025
        return (dist < GOAL_RADIUS) & on_table

    # ----------------------------------------------------------------- step
    def step(self, action: torch.Tensor):
        """Apply one control step (5 sim substeps). action: (N,8)."""
        # --- action -> joint targets (ManiSkill pd_joint_delta_pos semantics) ---
        arm_delta = torch.clamp(action[:, :7], -0.1, 0.1)            # (N,7)
        current_arm = self.robot.data.joint_pos.torch[:, self.arm_idx]  # (N,7)
        arm_target = current_arm + arm_delta
        grip = torch.clamp(action[:, 7:8], -0.01, 0.04)             # (N,1) absolute finger target
        # build full (N, num_joints) target in joint_names order
        full_target = self.robot.data.joint_pos.torch.clone()
        full_target[:, self.arm_idx] = arm_target
        full_target[:, self.finger_idx] = grip.expand(-1, 2)
        self.robot.set_joint_position_target_index(target=full_target)

        # --- step simulation (decimation substeps) ---
        self.scene.write_data_to_sim()
        for _ in range(self.decimation):
            self.sim.step()
            self.scene.update(self.sim_dt)

        # --- bookkeeping ---
        self.step_count += 1
        success = self._compute_success()
        self.ever_success |= success
        done = self.step_count >= self.max_steps  # timeout only (ManiSkill PushCube has no early term)
        return self.get_obs(), done, {"success": success}

    # ---------------------------------------------------------------- reset
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)

        # robot joints: rest qpos + arm noise 0.02 ; fingers at 0.04
        arm_noise = (torch.rand((n, 7), device=self.device) * 2 - 1) * 0.02
        arm_rest = self.arm_rest_qpos.unsqueeze(0).expand(n, 7) + arm_noise  # (n,7)
        finger_rest = torch.full((n, 2), 0.04, device=self.device)
        pos = torch.zeros((n, self.num_joints), device=self.device)
        pos[:, self.arm_idx] = arm_rest
        pos[:, self.finger_idx] = finger_rest
        self.robot.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros((n, self.num_joints), device=self.device), env_ids=env_ids
        )
        self.robot.set_joint_position_target_index(target=pos, env_ids=env_ids)  # hold rest during settle

        # robot root pose (world frame = env_origin + base offset), identity quat (xyzw)
        base = torch.tensor(ROBOT_BASE, device=self.device, dtype=torch.float32)
        root_pos = self.scene.env_origins[env_ids] + base
        ident_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, 4)
        self.robot.write_root_pose_to_sim_index(
            root_pose=torch.cat([root_pos, ident_xyzw], dim=1), env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )

        # cube: random xy in [-0.1, 0.1], z = CUBE_HALF, identity quat (world frame)
        cube_xy = torch.rand((n, 2), device=self.device) * 0.2 - 0.1
        cube_pos_world = self.scene.env_origins[env_ids] + torch.cat(
            [cube_xy, torch.full((n, 1), CUBE_HALF, device=self.device)], dim=1
        )
        self.cube.write_root_pose_to_sim_index(
            root_pose=torch.cat([cube_pos_world, ident_xyzw], dim=1), env_ids=env_ids
        )
        self.cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )

        # goal (world): cube_xy + [GOAL_DX, 0, 0]
        self.goal_pos[env_ids] = self.scene.env_origins[env_ids] + torch.cat(
            [cube_xy + GOAL_DX, torch.zeros((n, 1), device=self.device)], dim=1
        )

        # counters
        self.step_count[env_ids] = 0
        self.ever_success[env_ids] = False

        # flush + one settle step so the camera renders the reset state
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim_dt)
        return self.get_obs()

    def close(self):
        # teardown is handled by simulation_app.close() in the eval script
        pass
