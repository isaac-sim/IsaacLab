import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import math as math_utils
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.scene import InteractiveScene

# 注意：这里路径是相对于 IsaacLab 根目录
A10_USD_PATH = "A10_Single/assets/a10_single.usd"


# ============================
# A10 单臂机器人配置
# ===========================

A10_SINGLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=A10_USD_PATH,
        activate_contact_sensors=False,
        # For IK teleoperation, disabling gravity avoids immediate sagging when PD/controller is not yet settled.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=True,  # 固定根链接
        ),
    ),

            # "joint1": 0.0,
            # "joint2": 0.0,
            # "joint3": 2.618,
            # "joint4": -2.618,
            # "joint5": -1.5708,
            # "joint6": 0.0,
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # "joint1": 0.0,
            # "joint2": 0.0,
            # "joint3": 2.618,
            # "joint4": -2.618,
            # "joint5": -1.5708,
            # "joint6": 0.0,
            # 完全没效果 -p=0
            # "joint1": -14 / 180 * math.pi,
            # "joint2": -24 / 180 * math.pi,
            # "joint3": 139 / 180 * math.pi,
            # "joint4": -77 / 180 * math.pi,
            # "joint5": -76 / 180 * math.pi,
            # "joint6": -8 / 180 * math.pi,
            # 完全无效果 -p=1
            # "joint1": -19.5 / 180 * math.pi,
            # "joint2": -19.3 / 180 * math.pi,
            # "joint3": 126.5 / 180 * math.pi,
            # "joint4": -61 / 180 * math.pi,
            # "joint5": -70 / 180 * math.pi,
            # "joint6": -9 / 180 * math.pi,
            # "gripper_left_joint": 0.0,
            # "gripper_right_joint": 0.0,
            #有一点效果 -p=2
            # "joint1": -31 / 180 * math.pi,
            # "joint2": -23.5 / 180 * math.pi,
            # "joint3": 123.5 / 180 * math.pi,
            # "joint4": -51.6 / 180 * math.pi,
            # "joint5": -62.4 / 180 * math.pi,
            # "joint6": -22 / 180 * math.pi,
            #效果很好 -p=3
            "joint1": -40 / 180 * math.pi,
            "joint2": -8.8 / 180 * math.pi,
            "joint3": 106.5 / 180 * math.pi,
            "joint4": -32.8 / 180 * math.pi,
            "joint5": -54.3 / 180 * math.pi,
            "joint6": -33 / 180 * math.pi,
            #
            # "joint1": -16.6 / 180 * math.pi,
            # "joint2": -11.2 / 180 * math.pi,
            # "joint3": 103 / 180 * math.pi,
            # "joint4": -28.9 / 180 * math.pi,
            # "joint5": -83.5 / 180 * math.pi,
            # "joint6": -17 / 180 * math.pi,
            #一般
            # "joint1": 12.8 / 180 * math.pi,
            # "joint2": -7.6 / 180 * math.pi,
            # "joint3": 110 / 180 * math.pi,
            # "joint4": -48 / 180 * math.pi,
            # "joint5": -111 / 180 * math.pi,
            # "joint6": 10.1 / 180 * math.pi,

        },
    ),

    # 采用正则匹配为所有关节设置隐式关节驱动；后续可根据具体命名拆分左右臂
    # actuators={
    #     "A1": ImplicitActuatorCfg(joint_names_expr=["joint1"], stiffness=3000.0, damping=80.0),
    #     "A2": ImplicitActuatorCfg(joint_names_expr=["joint2"], stiffness=3000.0, damping=80.0),
    #     "A3": ImplicitActuatorCfg(joint_names_expr=["joint3"], stiffness=3000.0, damping=80.0),
    #     "A4": ImplicitActuatorCfg(joint_names_expr=["joint4"], stiffness=3000.0, damping=80.0),
    #     "A5": ImplicitActuatorCfg(joint_names_expr=["joint5"], stiffness=3000.0, damping=80.0),
    #     "A6": ImplicitActuatorCfg(joint_names_expr=["joint6"], stiffness=3000.0, damping=80.0),
    #     #"A7": ImplicitActuatorCfg(joint_names_expr=["Arm1_ee"], stiffness=3000.0, damping=80.0),
    # },
actuators={
    "arm": ImplicitActuatorCfg(
        # Use a permissive regex to ensure all arm joints are captured even if naming slightly differs.
        joint_names_expr=["joint.*"],
        # Higher gains/effort help the arm hold posture after reset.
        effort_limit_sim=1500.0,
        stiffness=3000.0,
        damping=80.0,
    ),
    "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_left_joint", "gripper_right_joint"],
            effort_limit_sim=12.0,
            velocity_limit_sim=0.2,
            stiffness=200.0,
            damping=20.0,
        ),
},
)

