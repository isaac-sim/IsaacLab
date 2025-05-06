from __future__ import annotations
from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
import torch
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedEnv

class UniformHeightCommand(CommandTerm):
    """Command generator for height control."""
    
    cfg: UniformHeightCommandCfg
    
    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.robot = env.scene[cfg.asset_name]
        self.height_scan = None
        if cfg.sensor_name is not None:
            self.height_scan = env.scene[cfg.sensor_name]

        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.height_command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        return self.height_command
    
    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        if self.height_scan is not None:
            ray_hits = self.height_scan.data.ray_hits_w[..., 2]
            if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
                adjusted_target_height = self.robot.data.root_link_pos_w[:, 2]
            else:
                adjusted_target_height = self.height_command + torch.mean(ray_hits, dim=1)
        else:
            adjusted_target_height = self.height_command
            
        self.metrics["height_error"] = torch.abs(self.robot.data.root_pos_w[:, 2] - adjusted_target_height) / max_command_step
    
    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.z_range)
    
    def _update_command(self):
        pass
    
    
@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for height command generator."""
    
    asset_name: str = MISSING
    sensor_name: str = MISSING
    class_type: type = UniformHeightCommand
    
    @configclass
    class Ranges:
        z_range: tuple[float, float] = MISSING
        
    ranges: Ranges = MISSING
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    
    goal_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_height"
    )