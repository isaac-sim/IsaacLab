from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
import ipdb

from isaaclab_assets.robots.anubis_fix import ANUBIS_CFG



@configclass
class AnubisCabinetEnvCfg(DirectRLEnvCfg):
    # env parameters
    episode_length_s: float = 8.3333  # 500 timesteps
    decimation: int = 2

    # new action & observation sizes for bimanual Anubis
    action_scale: float = 7.5
    dof_velocity_scale: float = 0.1

    action_space: int = 19  # 7 joints per arm
    observation_space: int = 40  # 14 pos + 14 vel + 6 to targets + 2 drawer states
    state_space: int = 0

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene config
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=3.0, replicate_physics=True)

    # robot uses URDF loader for Anubis
    robot: ArticulationCfg = ANUBIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class AnubisCabinetEnv(DirectRLEnv):
    cfg: AnubisCabinetEnvCfg

    def __init__(self, cfg: AnubisCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # limits and speed scales
        self.joint_lower = self._robot.data.soft_joint_pos_limits[0,:,0].to(self.device)
        self.joint_upper = self._robot.data.soft_joint_pos_limits[0,:,1].to(self.device)
        self.speed_scale = torch.ones_like(self.joint_lower)
        # reduce speed for gripper joints
        for name in ("gripper1_joint", "gripper2_joint"):
            idx = self._robot.find_joints(name)[0]
            self.speed_scale[idx] = 0.1
        self.targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # compute local grasp poses for both arms
        stage = get_current_stage()
        env_origin = self.scene.env_origins[0]
        # arm1
        hand1 = get_env_local_pose(env_origin, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/ee_link1")), self.device)
        lf1 = get_env_local_pose(env_origin, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper1L")), self.device)
        rf1 = get_env_local_pose(env_origin, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper1R")), self.device)
        finger1 = torch.zeros(7, device=self.device)
        finger1[:3] = (lf1[:3] + rf1[:3]) / 2
        finger1[3:] = lf1[3:]
        inv_r1, inv_p1 = tf_inverse(hand1[3:], hand1[:3])
        r1, p1 = tf_combine(inv_r1, inv_p1, finger1[3:], finger1[:3])
        p1 += torch.tensor([0,0.04,0], device=self.device)
        self.local_grasp_pos1 = p1.repeat((self.num_envs,1))
        self.local_grasp_rot1 = r1.repeat((self.num_envs,1))
        # arm2
        hand2 = get_env_local_pose(env_origin, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/ee_link2")), self.device)
        lf2 = get_env_local_pose(env_origin, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper2L")), self.device)
        rf2 = get_env_local_pose(env_origin, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper2R")), self.device)
        finger2 = torch.zeros(7, device=self.device)
        finger2[:3] = (lf2[:3] + rf2[:3]) / 2
        finger2[3:] = lf2[3:]
        inv_r2, inv_p2 = tf_inverse(hand2[3:], hand2[:3])
        r2, p2 = tf_combine(inv_r2, inv_p2, finger2[3:], finger2[:3])
        p2 += torch.tensor([0,0.04,0], device=self.device)
        self.local_grasp_pos2 = p2.repeat((self.num_envs,1))
        self.local_grasp_rot2 = r2.repeat((self.num_envs,1))

        # drawer grasp
        dlocal = torch.tensor([0.3,0.01,0.0,1.0,0.0,0.0,0.0], device=self.device)
        self.drawer_pos_local = dlocal[:3].repeat((self.num_envs,1))
        self.drawer_rot_local = dlocal[3:].repeat((self.num_envs,1))

        # axes
        axes = dict(
            forward=torch.tensor([0,0,1], device=self.device),
            inward=torch.tensor([-1,0,0], device=self.device),
            up=torch.tensor([0,1,0], device=self.device)
        )
        self.g_fwd = axes['forward'].repeat((self.num_envs,1))
        self.d_inward = axes['inward'].repeat((self.num_envs,1))
        self.g_up = axes['up'].repeat((self.num_envs,1))
        self.d_up = axes['up'].repeat((self.num_envs,1))

        # placeholders for transforms
        self.g1_pos = torch.zeros((self.num_envs,3), device=self.device)
        self.g1_rot = torch.zeros((self.num_envs,4), device=self.device)
        self.g2_pos = torch.zeros((self.num_envs,3), device=self.device)
        self.g2_rot = torch.zeros((self.num_envs,4), device=self.device)
        self.drawer_pos = torch.zeros((self.num_envs,3), device=self.device)
        self.drawer_rot = torch.zeros((self.num_envs,4), device=self.device)

        # link indices
        self.hand1_idx = self._robot.find_bodies("ee_link1")[0][0]
        self.hand2_idx = self._robot.find_bodies("ee_link2")[0][0]
        self.lf1_idx = self._robot.find_bodies("gripper1L")[0][0]
        self.rf1_idx = self._robot.find_bodies("gripper1R")[0][0]
        self.lf2_idx = self._robot.find_bodies("gripper2L")[0][0]
        self.rf2_idx = self._robot.find_bodies("gripper2R")[0][0]
        self.drawer_idx = self._cabinet.find_bodies("drawer_top")[0][0]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clone().clamp(-1,1)
        delta = self.speed_scale * self.dt * a * self.cfg.action_scale
        self.targets[:] = torch.clamp(self.targets + delta, self.joint_lower, self.joint_upper)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.targets)

    def _get_dones(self):
        open_pos = self._cabinet.data.joint_pos[:,3]
        done = open_pos > 0.39
        trig = self.episode_length_buf >= self.max_episode_length-1
        return done, trig

    def _get_rewards(self):
        self._compute_intermediate_values()
        # end-effector positions
        p1 = self.g1_pos; p2 = self.g2_pos; dp = self.drawer_pos
        # distances
        d1 = torch.norm(p1-dp, dim=-1); d2 = torch.norm(p2-dp, dim=-1)
        # distance rewards
        rd1 = 1/(1+d1**2); rd1 = rd1**2; rd1 = torch.where(d1<=0.02, rd1*2, rd1)
        rd2 = 1/(1+d2**2); rd2 = rd2**2; rd2 = torch.where(d2<=0.02, rd2*2, rd2)
        dist_reward = rd1 + rd2
        # orientation
        a1 = tf_vector(self.g1_rot.to(torch.float32), self.g_fwd.to(torch.float32))
        a2 = tf_vector(self.drawer_rot.to(torch.float32), self.d_inward.to(torch.float32))
        a3 = tf_vector(self.g1_rot.to(torch.float32), self.g_up.to(torch.float32))
        a4 = tf_vector(self.drawer_rot.to(torch.float32), self.d_up.to(torch.float32))
        dot1_1 = torch.bmm(a1.view(-1,1,3), a2.view(-1,3,1)).squeeze()
        dot2_1 = torch.bmm(a3.view(-1,1,3), a4.view(-1,3,1)).squeeze()
        rot1 = 0.5*(torch.sign(dot1_1)*dot1_1**2 + torch.sign(dot2_1)*dot2_1**2)
        # for arm2
        b1 = tf_vector(self.g2_rot.to(torch.float32), self.g_fwd.to(torch.float32))
        dot1_2 = torch.bmm(b1.view(-1,1,3), a2.view(-1,3,1)).squeeze()
        dot2_2 = torch.bmm(tf_vector(self.g2_rot.to(torch.float32), self.g_up.to(torch.float32)).view(-1,1,3), a4.view(-1,3,1)).squeeze()
        rot2 = 0.5*(torch.sign(dot1_2)*dot1_2**2 + torch.sign(dot2_2)*dot2_2**2)
        rot_reward = rot1 + rot2
        # action penalty
        act_pen = torch.sum(self.actions**2, dim=-1)
        # open reward
        open_r = self._cabinet.data.joint_pos[:,3]
        # combine
        r = (self.cfg.dist_reward_scale*dist_reward
             + self.cfg.rot_reward_scale*rot_reward
             + self.cfg.open_reward_scale*open_r
             - self.cfg.action_penalty_scale*act_pen)
        # bonuses
        r = torch.where(open_r>0.01, r+0.25, r)
        r = torch.where(open_r>0.2, r+0.25, r)
        r = torch.where(open_r>0.35, r+0.25, r)
        return r

    def _reset_idx(self, env_ids=None):
        super()._reset_idx(env_ids)
        # robot reset
        default = self._robot.data.default_joint_pos[env_ids]
        jitter = sample_uniform(-0.125,0.125,(len(env_ids), self._robot.num_joints), self.device)
        pos = torch.clamp(default + jitter, self.joint_lower, self.joint_upper)
        vel = torch.zeros_like(pos)
        self._robot.set_joint_position_target(pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
        # cabinet reset
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self._compute_intermediate_values(env_ids)

    def _get_observations(self):
        q = self._robot.data.joint_pos; v = self._robot.data.joint_vel
        pos_s = 2*(q-self.joint_lower)/(self.joint_upper-self.joint_lower)-1
        vel_s = v*self.cfg.dof_velocity_scale
        # vector to targets for both arms
        t1 = self.drawer_pos - self.g1_pos
        t2 = self.drawer_pos - self.g2_pos
        cz = self._cabinet.data.joint_pos[:,3].unsqueeze(-1)
        cv = self._cabinet.data.joint_vel[:,3].unsqueeze(-1)
        obs = torch.cat((pos_s, vel_s, t1, t2, cz, cv), dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None: env_ids = self._robot._ALL_INDICES
        # end-effector world
        h1_p = self._robot.data.body_pos_w[env_ids,self.hand1_idx]
        h1_r = self._robot.data.body_quat_w[env_ids,self.hand1_idx]
        h2_p = self._robot.data.body_pos_w[env_ids,self.hand2_idx]
        h2_r = self._robot.data.body_quat_w[env_ids,self.hand2_idx]
        # drawer world
        d_p = self._cabinet.data.body_pos_w[env_ids,self.drawer_idx]
        d_r = self._cabinet.data.body_quat_w[env_ids,self.drawer_idx]
        # compute grasp transforms
        g1_r,g1_p = tf_combine(h1_r,h1_p,self.local_grasp_rot1[env_ids],self.local_grasp_pos1[env_ids])
        g2_r,g2_p = tf_combine(h2_r,h2_p,self.local_grasp_rot2[env_ids],self.local_grasp_pos2[env_ids])
        # compute drawer handle transform
        dr_r,dr_p = tf_combine(d_r,d_p,self.drawer_rot_local[env_ids],self.drawer_pos_local[env_ids])
        # store back
        self.g1_rot[env_ids],self.g1_pos[env_ids] = g1_r,g1_p
        self.g2_rot[env_ids],self.g2_pos[env_ids] = g2_r,g2_p
        self.drawer_rot[env_ids],self.drawer_pos[env_ids] = dr_r,dr_p
