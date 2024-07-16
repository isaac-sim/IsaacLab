# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import DeformableObjectCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka import joint_pos_env_cfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaDeformableCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.replicate_physics = False

        FRANKA_PANDA_CFG.actuators["panda_hand"].effort_limit = 2.0

        # Set Deformable Cube as object
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.05], rot=[1, 0, 0, 0]),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.06, 0.06, 0.06),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                    self_collision_filter_distance=0.005,
                    settling_threshold=0.1,
                    sleep_damping=1.0,
                    sleep_threshold=0.05,
                    solver_position_iteration_count=20,
                    vertex_velocity_damping=0.5,
                    simulation_hexahedral_resolution=4,
                    rest_offset=0.0001,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    dynamic_friction=0.95,
                    youngs_modulus=500000,
                ),
            ),
        )

        self.events.reset_object_position = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
        )


@configclass
class FrankaDeformableCubeLiftEnvCfg_PLAY(FrankaDeformableCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
