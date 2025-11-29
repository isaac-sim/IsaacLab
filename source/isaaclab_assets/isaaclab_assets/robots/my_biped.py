import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

MY_BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # ⚠️ 必须是你刚才确认无误的那个 USD 的绝对路径
        usd_path="/home/fan/workspace/IsaacLab/source/isaaclab_assets/data/Robots/MyBiped/robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, # 初期建议关闭自碰撞，训练快且稳
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False, # ⚠️ 绝对不能为 True，否则钉在空中
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # ⚠️ 高度调整：双足机器人建议给 0.6~0.8m，保证脚底刚好接触或略高于地面
        pos=(0.0, 0.0, 0.05), 
        # ⚠️ 关节名称：必须和你右侧 Stage 面板里看到的一字不差！
        # 这里设置为 0.0 (直立姿态)
        joint_pos={
            "L_hip_roll_joint": 0.0, "L_hip_pitch_joint": 0.0, "L_knee_joint": 0.0, "L_foot_joint": 0.0,
            "R_hip_roll_joint": 0.0, "R_hip_pitch_joint": 0.0, "R_knee_joint": 0.0, "R_foot_joint": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"], # 控制所有关节
            effort_limit=20.0,       # 最大扭矩 (根据真实电机修改)
            velocity_limit=10.0,     # 最大速度 (rad/s)
            stiffness=25.0,          # P (刚度)：双足需要硬一点，建议 20-40
            damping=0.5,             # D (阻尼)：防止震荡，建议 0.5-2.0
        ),
    },
)