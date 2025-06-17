from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG  # isort: skip
from ...dextrah_env_cfg import DextrahEnvCfg
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroPCAActionCfg:
    actions = mdp.PCAHandActionCfg(asset_name="robot")


@configclass
class KukaAllegroFabricActionCfg:
    actions = mdp.FabricActionCfg(asset_name="robot")



@configclass
class DextrahKukaAllegroEnvCfg(DextrahEnvCfg):

    def __post_init__(self: DextrahEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    'iiwa7_joint_1': -0.85,
                    'iiwa7_joint_2': 0.0,
                    'iiwa7_joint_3': 0.76,
                    'iiwa7_joint_4': 1.25,
                    'iiwa7_joint_5': -1.76,
                    'iiwa7_joint_6': 0.90,
                    'iiwa7_joint_7': 0.64,
                    '(index|middle|ring)_joint_0': 0.0,
                    '(index|middle|ring)_joint_1': 0.3,
                    '(index|middle|ring)_joint_2': 0.3,
                    '(index|middle|ring)_joint_3': 0.3,
                    'thumb_joint_0': 1.5,
                    'thumb_joint_1': 0.60147215,
                    'thumb_joint_2': 0.33795027,
                    'thumb_joint_3': 0.60845138
                },
            ),
            # Default dof friction coefficients
            # NOTE: these are set based on how far out they will scale multiplicatively
            # with the above joint_friction EventTerm above.
            actuators={
                "kuka_allegro_actuators": KUKA_ALLEGRO_CFG.actuators["kuka_allegro_actuators"].replace(
                    friction={
                        "iiwa7_joint_(1|2|3|4|5|6|7)": 1.,
                        "index_joint_(0|1|2|3)": 0.01,
                        "middle_joint_(0|1|2|3)": 0.01,
                        "ring_joint_(0|1|2|3)": 0.01,
                        "thumb_joint_(0|1|2|3)": 0.01,
                    }
                )
            },
        )

        self.observations.policy.hand_tips_pos.params["asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.observations.critic.measured_body_forces.params["asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.finger_curl_reg.params["asset_cfg"].joint_names = "(thumb|index|middle|ring).*"


@configclass
class DextrahKukaAllegroEnvRelJointPosCfg(DextrahKukaAllegroEnvCfg):
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()


@configclass
class DextrahKukaAllegroEnvPCACfg(DextrahKukaAllegroEnvCfg):
    actions: KukaAllegroPCAActionCfg = KukaAllegroPCAActionCfg()


@configclass
class DextrahKukaAllegroEnvFabricCfg(DextrahKukaAllegroEnvCfg):
    actions: KukaAllegroFabricActionCfg = KukaAllegroFabricActionCfg()
