
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np 
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import wrap_to_pi
from parkour_isaaclab.managers import ParkourTerm
from parkour_isaaclab.terrains import ParkourTerrainGeneratorCfg, ParkourTerrainImporter, ParkourTerrainGenerator

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from .parkour_events_cfg import ParkourEventsCfg


class ParkourEvent(ParkourTerm):
    cfg: ParkourEventsCfg

    def __init__(
        self, 
        cfg: ParkourEventsCfg, 
        env: ParkourManagerBasedRLEnv
        ):
        super().__init__(cfg, env)

        self.episode_length_s = env.cfg.episode_length_s
        self.reach_goal_delay = cfg.reach_goal_delay
        self.num_future_goal_obs = cfg.num_future_goal_obs
        self.next_goal_threshold = cfg.next_goal_threshold
        self.simulation_time = env.step_dt
        self.arrow_num = cfg.arrow_num
        self.env = env 
        self.debug_vis = cfg.debug_vis
               
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- metrics
        self.metrics["far_from_current_goal"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["how_far_from_start_point"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["terrain_levels"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["current_goal_idx"] = torch.zeros(self.num_envs, device='cpu')
        self.dis_to_start_pos = torch.zeros(self.num_envs, device=self.device)
        self.terrain: ParkourTerrainImporter = self.env.scene.terrain
        terrain_generator: ParkourTerrainGenerator = self.terrain.terrain_generator_class
        parkour_terrain_cfg :ParkourTerrainGeneratorCfg = self.terrain.cfg.terrain_generator
        self.num_goals = parkour_terrain_cfg.num_goals
        self.env_class = torch.zeros(self.num_envs, device=self.device)
        self.env_origins = self.terrain.env_origins
        self.terrain_type = terrain_generator.terrain_type
        self.terrain_class = torch.from_numpy(self.terrain_type).to(self.device).to(torch.float)
        self.env_class[:] = self.terrain_class[self.terrain.terrain_levels, self.terrain.terrain_types]
        
        terrain_goals = terrain_generator.goals
        self.terrain_goals = torch.from_numpy(terrain_goals).to(self.device).to(torch.float)
        self.env_goals = torch.zeros(self.num_envs,  self.terrain_goals.shape[2] + self.num_future_goal_obs, 3, device=self.device, requires_grad=False)
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float).to(device = self.device)
        
        if self.debug_vis:
            self.total_heights = torch.from_numpy(terrain_generator.goal_heights).to(device = self.device)
            self.future_goal_idx = torch.ones(self.num_goals, device=self.device, dtype=torch.bool).repeat(self.num_envs, 1)
            self.future_goal_idx[:, 0] = False
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]
       
        self.total_terrain_names = terrain_generator.terrain_names
        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy() ## string type can't convert to torch
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]
        self._reset_offset = self.env.event_manager.get_term_cfg('reset_root_state').params['offset']

        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])


    def __call__(self):
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def __str__(self) -> str:
        msg = "ParkourCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg
    
    def _update_command(self):
        """Re-target the current goal position to the current root state."""
        next_flag = self.reach_goal_timer > self.reach_goal_delay / self.simulation_time
        if self.debug_vis:
            tmp_mask = torch.nonzero(self.cur_goal_idx>0).squeeze(-1)
            if tmp_mask.numel() > 0:
                self.future_goal_idx[tmp_mask, self.cur_goal_idx[tmp_mask]] = False
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.reached_goal_ids = torch.norm(robot_root_pos_w - self.cur_goals[:, :2], dim=1) < self.next_goal_threshold
        reached_goal_idx = self.reached_goal_ids.nonzero(as_tuple=False).squeeze(-1)
        if reached_goal_idx.numel() > 0:
            self.reach_goal_timer[reached_goal_idx] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        start_pos = self.env_origins[:,:2] - \
                    torch.tensor((self.terrain.cfg.terrain_generator.size[1] + \
                                  self._reset_offset, 0)).to(self.device)

        self.dis_to_start_pos = torch.norm(start_pos - self.robot.data.root_pos_w[:, :2], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        ## we are use reset_root_state events for initalize robot position in a subterrain
        ## original robot root init position is (0,0) in the subterrain axis, so we subtracted off from current robot position 

        start_pos = self.env_origins[env_ids,:2] - \
                    torch.tensor((self.terrain.cfg.terrain_generator.size[1] + \
                                  self._reset_offset, 0)).to(self.device)

        self.dis_to_start_pos = torch.norm(start_pos - self.robot.data.root_pos_w[env_ids, :2], dim=1)
        threshold = self.env.command_manager.get_command("base_velocity")[env_ids, 0] * self.episode_length_s
        move_up = self.dis_to_start_pos > 0.8*threshold
        move_down = self.dis_to_start_pos < 0.4*threshold

        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain.terrain_levels[env_ids] = torch.where(self.terrain.terrain_levels[env_ids]>=self.terrain.max_terrain_level,
                                                   torch.randint_like(self.terrain.terrain_levels[env_ids], self.terrain.max_terrain_level),
                                                   torch.clip(self.terrain.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain.terrain_origins[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy()
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]

        self.reach_goal_timer[env_ids] = 0
        self.cur_goal_idx[env_ids] = 0

        if self.debug_vis:
            self.future_goal_idx[env_ids, 0] = False
            self.future_goal_idx[env_ids, 1:] = True
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]

    def _update_metrics(self):
        # logs data
        self.metrics["terrain_levels"] = (self.terrain.terrain_levels.float()).to(device = 'cpu')
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.metrics["far_from_current_goal"] = (torch.norm(self.cur_goals[:, :2] - robot_root_pos_w,dim =-1) - self.next_goal_threshold).to(device = 'cpu')
        self.metrics["current_goal_idx"] = self.cur_goal_idx.to(device='cpu', dtype=float)
        self.metrics["how_far_from_start_point"] = self.dis_to_start_pos.to(device = 'cpu')
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer = VisualizationMarkers(self.cfg.current_goal_pose_visualizer_cfg)
            # set their visibility to true
            self.current_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer = VisualizationMarkers(self.cfg.future_goal_poses_visualizer_cfg)
            self.future_goal_poses_visualizer.set_visibility(True)


            if not hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer = VisualizationMarkers(self.cfg.current_arrow_visualizer_cfg)
            # set their visibility to true
            self.current_arrow_visualizer.set_visibility(True)
            if not hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer = VisualizationMarkers(self.cfg.future_arrow_visualizer_cfg)
            self.future_arrow_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer.set_visibility(False)

            if hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer.set_visibility(False)
            if hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        env_per_goals = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types] 
        env_per_xy_goals = env_per_goals[:,:,:2].reshape(self.num_envs, -1,2) ## (env_num, 8, 2 )
        env_per_xy_goals = env_per_xy_goals + self.env_origins[:, :2].unsqueeze(1)
        goal_height = self.env_per_heights.unsqueeze(-1)*self.terrain.cfg.terrain_generator.vertical_scale
        env_per_goal_pos = torch.concat([env_per_xy_goals, goal_height],dim=-1)
        env_per_current_goal_pos = env_per_goal_pos[~self.future_goal_idx, :]
        env_per_future_goal_pos = env_per_goal_pos[self.future_goal_idx, :] .reshape(-1,3)
        self.current_goal_pose_visualizer.visualize(
            translations=env_per_current_goal_pos,
        )
        if len(env_per_future_goal_pos) > 0:
            self.future_goal_poses_visualizer.visualize(
                translations=env_per_future_goal_pos ,
            )
        current_arrow_list = []
        future_arrow_list = []
        for i in range(self.arrow_num):
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            current_pose_arrow = self.robot.data.root_pos_w[:, :2] + 0.1*(i+3) * target_vec_norm[:, :2]
            current_arrow_list.append(torch.concat([
                current_pose_arrow[:,0][:,None], 
                current_pose_arrow[:,1][:,None], 
                self.robot.data.root_pos_w[:, 2][:,None]
                ], dim = 1))
            if len(env_per_future_goal_pos) > 0:
                    
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                future_pose_arrow = self.robot.data.root_pos_w[:, :2] + 0.2*(i+3) * target_vec_norm[:, :2]
                future_arrow_list.append(torch.concat([
                    future_pose_arrow[:,0][:,None], 
                    future_pose_arrow[:,1][:,None], 
                    self.robot.data.root_pos_w[:, 2][:,None]
                    ], dim = 1))
            else:
                future_arrow_list.append(torch.concat([
                    current_pose_arrow[:,0][:,None], 
                    current_pose_arrow[:,1][:,None], 
                    self.robot.data.root_pos_w[:, 2][:,None]
                    ], dim = 1))

        current_arrow_positions = torch.cat(current_arrow_list, dim=0)
        future_arrow_positions = torch.cat(future_arrow_list, dim=0)
        self.current_arrow_visualizer.visualize(
            translations=current_arrow_positions,
        )

        self.future_arrow_visualizer.visualize(
            translations=future_arrow_positions,
        )

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")
