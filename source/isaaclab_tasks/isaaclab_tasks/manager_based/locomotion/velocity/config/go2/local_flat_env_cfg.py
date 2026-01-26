"""Local offline configuration for Unitree Go2 flat terrain training."""

import os
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    """Configuration for Unitree Go2 locomotion on flat terrain using local assets."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Get IsaacLab root path for local assets
        isaaclab_path = os.environ.get('ISAACLAB_PATH', os.getcwd())
        
        # Reward tuning for flat terrain
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        
        # Create flat terrain with grid pattern
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": MeshPlaneTerrainCfg(size=(8.0, 8.0))
        }
        self.scene.terrain.terrain_generator.proportion = [1.0]
        self.scene.terrain.terrain_generator.curriculum = False
        
        # Grid-like visual material (light gray with darker grid lines)
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.8, 0.8),  # Light gray base
            metallic=0.0,
            roughness=0.5,
        )
        
        # Apply grid texture overlay
        # Note: For true grid lines, you'd need a grid texture file
        # For now, this creates a clean, flat appearance
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        
        # Local asset paths
        self.scene.sky_light.spawn.texture_file = f"{isaaclab_path}/local_assets/Textures/kloofendal_43d_clear_puresky_4k.hdr"
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].usd_path = f"{isaaclab_path}/local_assets/Props/arrow_x.usd"
        self.commands.base_velocity.current_vel_visualizer_cfg.markers["arrow"].usd_path = f"{isaaclab_path}/local_assets/Props/arrow_x.usd"
        
        # Disable height scanning and terrain curriculum
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    """Play configuration with smaller scene and no randomization."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        
        # Smaller scene for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None