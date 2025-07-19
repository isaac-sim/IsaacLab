from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab_tasks.manager_based.manipulation.bernie_proj.kitchen.config.rby1.joint_pos_env_cfg import RBY1CubeLiftEnvCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg


@configclass
class RBY1FreezerEnvCfg(RBY1CubeLiftEnvCfg):
    """
    Class Description: This is class that all of our enviornments will extend
                         - it enables a user to specify the start and goal object
                           that a user wants to use for a task
                       This class itself extends FrankaYCBEnvCfg
                         - this class sets up the enviornment
    """
    def __init__(self, scale=(1.0, 1.0, 1.0)):
        """
        parameters:
        - object_name: string for the YCB object name
        - scale: scaling of the object
        """
        self.scale = scale
        super().__init__()

    def __post_init__(self):
        super().__post_init__()

        # Set Cube as object
        self.scene.object = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/object",
            init_state=ArticulationCfg.InitialStateCfg(pos=[0.75, 0, 0.055], rot=(0.0, 0.0, -0.70710678, 0.70710678)),
            spawn=UsdFileCfg(
                usd_path=str("/home/davin123/IsaacLab/assets/bernie_proj/kitchen/") + "freezer" + ".usd",
                scale=(0.01, 0.01, 0.01),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
            actuators={},
        )
