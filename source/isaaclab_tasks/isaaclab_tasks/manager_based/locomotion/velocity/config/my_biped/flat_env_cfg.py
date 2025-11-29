from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg
from isaaclab_assets.robots.my_biped import MY_BIPED_CFG
from isaaclab.managers import RewardTermCfg as RewTerm  
from isaaclab.envs import mdp                           

@configclass
class MyBipedFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # ========== 优化训练指令  ==========
        
        # 1. 锁定指令：只能向前走 (Lin Vel X > 0)
        # 原来是 (-1.0, 1.0)，现在改成 (0.0, 1.0)，甚至 (0.3, 0.5) 让它先学慢走
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        
        # 2. 禁用横向移动 (Lin Vel Y = 0)
        # 双足机器人很难侧向走，初期必须关掉
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        
        # 3. 禁用原地转圈 (Ang Vel Z = 0)
        # 先学会走直线，再学转弯
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


        # 1. 替换机器人
        self.scene.robot = MY_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 2. 地形设为平地
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None 
        self.scene.height_scanner = None 
        
        # 3. 动作缩放
        self.actions.joint_pos.scale = 0.25 
        
        # 4. 奖励函数调整
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.is_alive = RewTerm(func=mdp.is_alive, weight=0.5)

        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -2.0

        # 惩罚关节偏离 (Joint Deviation)
        # 强迫它走完一步后，关节要尝试回到 0 度 (直立状态)
        # 这会逼着它把拖在后面的腿收回来
        #self.rewards.joint_pos_limits = None # 先关掉旧的限位奖励
        #self.rewards.joint_deviation_l2 = RewTerm(
        #    func=mdp.joint_pos_l2,
        #    weight=-0.3,  # 权重给大一点，比如 -0.2 到 -1.0
        #)

        # 疯涨腾空奖励 (Feet Air Time)
        # 原来可能是 0.125，现在加倍！
        # 只有当指令速度大于 0.1 时，脚离地才给分
        #if hasattr(self.rewards, "feet_air_time"):
        #    self.rewards.feet_air_time.weight = 2.0  # 大幅提高！
        #    self.rewards.feet_air_time.params["threshold"] = 0.6 # 稍微提高一点判定阈值


        # ========== 修复所有正则匹配错误 (终极版) ==========
        
        # A. 接触传感器
        self.scene.contact_forces.body_filter = [".*foot.*", "body_bottom"] 

        # B. 摔倒判定
        if hasattr(self.terminations, "base_contact"):
            self.terminations.base_contact.params["sensor_cfg"].body_names = ["body_bottom"]
        
        # C. 脚部腾空奖励
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*foot.*"]
            
        # D. 【新增修复】禁用“不当碰撞”奖励
        # 因为你的机器人没有叫 "THIGH" (大腿) 的部件，所以直接关掉这个惩罚
        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts = None

        # E. 移除高度扫描观测
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        # F. 【新增修复】禁用质量随机化 (先跑通再说)
        # 因为父类的配置在新版 API 下报错了，我们先把它关掉
        if hasattr(self.events, "add_base_mass"):
            self.events.add_base_mass = None
            
        # G. 顺便把推力干扰也关了 (初期训练不需要人踹它)
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        # H. 【新增修复】禁用地形课程学习
        # 因为我们是无限平地，不需要根据步数升级地形难度
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None