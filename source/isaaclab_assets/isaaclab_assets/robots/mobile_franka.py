import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

MOBILE_FRANKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/xuezhi/Downloads/ridgeback_franka6_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # franka_panda
            "panda_joint1": 0.0,
            "panda_joint2": -1.0,
            "panda_joint3": 0.0,
            "panda_joint4": -2.2,
            "panda_joint5": 0.0,
            "panda_joint6": 2.4,
            "panda_joint7": 0.8,
            "panda_finger_joint1": 0.035,
            "panda_finger_joint2": 0.035,
            
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "panda_joint[1-7]"
            ],
            effort_limit=87.0,
            
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper_actuators": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1", "panda_finger_joint2"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=100000.0,
            damping=1000.0,
        ),
        "base_actuators": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint"],
            effort_limit=1000.0,
            velocity_limit=100.0,  # Assuming position control
            stiffness=999999986991104.0,
            damping=100000.0,
        ),
        "base_rot_actuators": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_revolute_z_joint"],
            effort_limit=1000.0,
            velocity_limit=100.0,  # Assuming position control
            stiffness=17453292716032.0,
            damping=1745.32922,
        ),
    },
)