from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, DeformableObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG

@configclass
class DeformableCubeEnvCfg(DirectRLEnvCfg):
	# required params
	num_envs = 256
	env_spacing = 4.0
	dt = 1 / 120
	observation_space = 15
	action_space = 3
	state_space = 0

	# env params
	decimation = 1
	episode_length_s = 15.0

	# simulation
	sim: SimulationCfg = SimulationCfg(
		dt=dt, 
		render_interval=decimation
	)

	# robot
	robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

	# objects
	object_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
				rest_offset=0.0,
				contact_offset=0.001
			),
            visual_material=sim_utils.PreviewSurfaceCfg(
				diffuse_color=(0.5, 0.1, 0.0)
			),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
				poissons_ratio=0.4,
				youngs_modulus=1e5
			),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(),
        debug_vis=True,
    )

	container_cfg: AssetBaseCfg = MISSING #TODO: fill this out

	# scene
	scene: InteractiveSceneCfg = InteractiveSceneCfg(
		num_envs=num_envs, 
		env_spacing=env_spacing, 
		replicate_physics=True,
	)