import math
import torch
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import math as math_utils
from isaacsim.sensors.camera import Camera

# 注意：这里路径是相对于 IsaacLab 根目录
X7_USD_PATH = "X7_Robot/assets/X7/x7_duo.usd"


# ============================
# 机器人配置（14 DOF）
# ============================

X7_DUO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=X7_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={f"Joint{i}": 0.0 for i in range(1, 15)},
        pos=(0.0, 0.0, 1.8),
        # 绕 Y 轴 -90° 旋转
        rot=tuple(
            math_utils.quat_from_euler_xyz(
                torch.tensor([0.0]),               # roll
                torch.tensor([-math.pi / 2]),      # pitch
                torch.tensor([0.0]),               # yaw
            )[0].tolist()
        ),
    ),

    actuators={
        # 左臂 Joint1~Joint7
        "A1": ImplicitActuatorCfg(joint_names_expr=["Joint1"], stiffness=3000.0, damping=80.0),
        "A2": ImplicitActuatorCfg(joint_names_expr=["Joint2"], stiffness=3000.0, damping=80.0),
        "A3": ImplicitActuatorCfg(joint_names_expr=["Joint3"], stiffness=3000.0, damping=80.0),
        "A4": ImplicitActuatorCfg(joint_names_expr=["Joint4"], stiffness=3000.0, damping=80.0),
        "A5": ImplicitActuatorCfg(joint_names_expr=["Joint5"], stiffness=3000.0, damping=80.0),
        "A6": ImplicitActuatorCfg(joint_names_expr=["Joint6"], stiffness=3000.0, damping=80.0),
        "A7": ImplicitActuatorCfg(joint_names_expr=["Joint7"], stiffness=3000.0, damping=80.0),

        # 右臂 Joint8~Joint14
        "B1": ImplicitActuatorCfg(joint_names_expr=["Joint8"], stiffness=3000.0, damping=80.0),
        "B2": ImplicitActuatorCfg(joint_names_expr=["Joint9"], stiffness=3000.0, damping=80.0),
        "B3": ImplicitActuatorCfg(joint_names_expr=["Joint10"], stiffness=3000.0, damping=80.0),
        "B4": ImplicitActuatorCfg(joint_names_expr=["Joint11"], stiffness=3000.0, damping=80.0),
        "B5": ImplicitActuatorCfg(joint_names_expr=["Joint12"], stiffness=3000.0, damping=80.0),
        "B6": ImplicitActuatorCfg(joint_names_expr=["Joint13"], stiffness=3000.0, damping=80.0),
        "B7": ImplicitActuatorCfg(joint_names_expr=["Joint14"], stiffness=3000.0, damping=80.0),
    },
)

# class X7DuoSceneCfg(InteractiveSceneCfg):
#     # 地面
#     ground = AssetBaseCfg(
#         prim_path="/World/defaultGroundPlane",
#         spawn=sim_utils.GroundPlaneCfg(),
#     )

#     # 机器人
#     robot = X7_DUO_CFG.replace(
#         prim_path="{ENV_REGEX_NS}/X7_duo"
#     )

