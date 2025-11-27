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

    # # 不预设具体关节名，依 URDF 加载；初始位姿设定为抬高并旋转对齐
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 1.0),
    #     rot=tuple(
    #         math_utils.quat_from_euler_xyz(
    #             torch.tensor([0.0]),               # roll
    #             torch.tensor([-math.pi / 2]),      # pitch
    #             torch.tensor([0.0]),               # yaw
    #         )[0].tolist()
    #     ),
    # ),

    # 采用正则匹配为所有关节设置隐式关节驱动；后续可根据具体命名拆分左右臂
    actuators={
        "all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=3000.0, damping=80.0),
    },
)

class A10SceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg()
    robot = A10_CFG.replace(prim_path="/World/A10")


def build_scene(sim):
    """Create and return an InteractiveScene for A10 duo."""
    scene = InteractiveScene(sim=sim, scene_cfg=A10SceneCfg())
    return scene


