from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg
from ...dextrah_reorient_env_cfg import DexSuiteReorientEnvCfg, DexSuiteLiftEnvCfg, RewardsCfg
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
class KukaAllegroReorientRewardCfg(RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb. 
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    
    def __post_init__(self: DexSuiteReorientEnvCfg):
        super().__post_init__()
        self.observations.policy.contact = ObsTerm(func=mdp.fingers_contact_force_w)
        self.observations.critic.contact = ObsTerm(func=mdp.fingers_contact_force_w)
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

        self.observations.policy.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.observations.critic.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.scene.robot.spawn.activate_contact_sensors = True
        for finger in ["index", "middle", "ring", "thumb"]:
            link_name = f"{finger}_link_3"
            link_contact_senor = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                update_period=0.0,
                history_length=6,
                debug_vis=False,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]
            )
            setattr(self.scene, f"{link_name}_contact_sensor", link_contact_senor)


@configclass
class DexsuiteKukaAllegroReorientRelJointPosActionEnvCfg(KukaAllegroMixinCfg, DexSuiteReorientEnvCfg):
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()


@configclass
class DexsuiteKukaAllegroReorientFabricActionEnvCfg(KukaAllegroMixinCfg, DexSuiteReorientEnvCfg):
    actions: KukaAllegroFabricActionCfg = KukaAllegroFabricActionCfg()


@configclass
class DexsuiteKukaAllegroLiftRelJointPosEnvCfg(KukaAllegroMixinCfg, DexSuiteLiftEnvCfg):
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()


@configclass
class DexsuiteKukaAllegroLiftPCAEnvCfg(KukaAllegroMixinCfg, DexSuiteLiftEnvCfg):
    actions: KukaAllegroPCAActionCfg = KukaAllegroPCAActionCfg()


@configclass
class DexsuiteKukaAllegroLiftFabricEnvCfg(KukaAllegroMixinCfg, DexSuiteLiftEnvCfg):
    actions: KukaAllegroFabricActionCfg = KukaAllegroFabricActionCfg()
