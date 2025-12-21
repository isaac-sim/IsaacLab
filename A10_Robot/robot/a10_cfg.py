import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import math as math_utils
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.scene import InteractiveScene

# 注意：这里路径是相对于 IsaacLab 根目录
A10_USD_PATH = "A10_Robot/assets/dualarm.usd"


# ============================
# A10 双臂机器人配置（假定总 14 DOF：左7+右7）
# ============================

A10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=A10_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),

    # init_state=ArticulationCfg.InitialStateCfg(
    #     joint_pos={f"Joint{i}": 0.0 for i in range(1, 15)},
    # ),

    # 采用正则匹配为所有关节设置隐式关节驱动；后续可根据具体命名拆分左右臂
    actuators={
        "A1": ImplicitActuatorCfg(joint_names_expr=["Arm1_joint1"], stiffness=3000.0, damping=80.0),
        "A2": ImplicitActuatorCfg(joint_names_expr=["Arm1_joint2"], stiffness=3000.0, damping=80.0),
        "A3": ImplicitActuatorCfg(joint_names_expr=["Arm1_joint3"], stiffness=3000.0, damping=80.0),
        "A4": ImplicitActuatorCfg(joint_names_expr=["Arm1_joint4"], stiffness=3000.0, damping=80.0),
        "A5": ImplicitActuatorCfg(joint_names_expr=["Arm1_joint5"], stiffness=3000.0, damping=80.0),
        "A6": ImplicitActuatorCfg(joint_names_expr=["Arm1_joint6"], stiffness=3000.0, damping=80.0),
        #"A7": ImplicitActuatorCfg(joint_names_expr=["Arm1_ee"], stiffness=3000.0, damping=80.0),

        # 右臂 Joint8~Joint14
        "B1": ImplicitActuatorCfg(joint_names_expr=["Arm2_joint1"], stiffness=3000.0, damping=80.0),
        "B2": ImplicitActuatorCfg(joint_names_expr=["Arm2_joint2"], stiffness=3000.0, damping=80.0),
        "B3": ImplicitActuatorCfg(joint_names_expr=["Arm2_joint3"], stiffness=3000.0, damping=80.0),
        "B4": ImplicitActuatorCfg(joint_names_expr=["Arm2_joint4"], stiffness=3000.0, damping=80.0),
        "B5": ImplicitActuatorCfg(joint_names_expr=["Arm2_joint5"], stiffness=3000.0, damping=80.0),
        "B6": ImplicitActuatorCfg(joint_names_expr=["Arm2_joint6"], stiffness=3000.0, damping=80.0),
       # "B7": ImplicitActuatorCfg(joint_names_expr=["Arm2_ee"], stiffness=3000.0, damping=80.0),
    },
)

