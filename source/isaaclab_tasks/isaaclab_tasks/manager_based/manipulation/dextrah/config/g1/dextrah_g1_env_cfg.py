from isaaclab.utils import configclass
from ...dextrah_env_cfg import DextrahEnvCfg
from ... import mdp

from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

HAND_BODY_REGEX = ".*_(palm|zero|one|two|three|four|five|six)_link"
HAND_JOINT_REGEX = ".*_(zero|one|two|three|four|five|six)_joint"

@configclass
class G1RelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class G1PCAActionCfg:
    actions = mdp.PCAHandActionCfg(asset_name="robot")


@configclass
class G1FabricActionCfg:
    actions = mdp.FabricActionCfg(asset_name="robot")



@configclass
class DextrahG1EnvCfg(DextrahEnvCfg):

    def __post_init__(self: DextrahEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "left_palm_link"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.34)

        self.observations.policy.hand_tips_pos.params["asset_cfg"].body_names = HAND_BODY_REGEX
        self.observations.critic.measured_body_forces.params["asset_cfg"].body_names = HAND_BODY_REGEX
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = HAND_BODY_REGEX
        self.rewards.finger_curl_reg.params["asset_cfg"].joint_names = HAND_JOINT_REGEX
        self.events.reset_robot.params['pose_range']['yaw'] = [3.1415, 3.1415]
        


@configclass
class DextrahG1EnvRelJointPosCfg(DextrahG1EnvCfg):
    actions: G1RelJointPosActionCfg = G1RelJointPosActionCfg()


@configclass
class DextrahG1EnvPCACfg(DextrahG1EnvCfg):
    actions: G1PCAActionCfg = G1PCAActionCfg()


@configclass
class DextrahG1EnvFabricCfg(DextrahG1EnvCfg):
    actions: G1FabricActionCfg = G1FabricActionCfg()
