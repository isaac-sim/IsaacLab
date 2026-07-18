# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp
from .camera_cfg import StateObservationCfg

FINGERTIP_LIST = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
THUMB_SENSOR = "thumb_link_3_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "thumb_link_3"]


@configclass
class KukaAllegroSceneCfg(dexsuite.SceneCfg):
    """KukaAllegro scene for the dexsuite lift/reorient tasks.

    The ``base_camera`` / ``wrist_camera`` slots are left unset (``None``) for the state task; the
    camera env config populates them (see ``dexsuite_kuka_allegro_camera_env_cfg``).
    """

    robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    base_camera: CameraCfg | None = None
    wrist_camera: CameraCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        for link_name in FINGERTIP_LIST:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={"threshold": 0.01, "sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
    )

    def __post_init__(self):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class KukaAllegroMixinCfg:
    @configclass
    class PhysicsCfg(dexsuite.PhysicsCfg):
        """KUKA-Allegro solver presets owned by this task composition."""

        feather_pgs = NewtonCfg(
            solver_cfg=FeatherPGSSolverCfg(
                pgs_mode="matrix_free",
                update_mass_matrix_interval=2,
                enable_joint_limits=True,
                joint_limit_activation_gap=0.1,
                pgs_iterations=8,
                pgs_velocity_iterations=0,
                dense_max_constraints=384,
                mf_max_constraints=64,
                hinv_jt_kernel="par_row",
                pgs_warmstart=False,
                pgs_omega=1.0,
                pgs_beta=0.05,
                pgs_cfm=1.0e-6,
                serial_kernel_block_dim=64,
                row_watermark=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        )

    scene: KukaAllegroSceneCfg = KukaAllegroSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    observations: StateObservationCfg = StateObservationCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.sim.physics = self.PhysicsCfg()
        self.commands.object_pose.body_name = "palm_link"
        events = self.events.conditional_reset.params["terms"]
        events["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="iiwa7_joint_7")
        events["reset_object_to_target"].params["target_cfg"] = SceneEntityCfg("robot", body_names="palm_link")
        events["reset_object_to_target"].params["pose_range"] = {
            "x": [0.03, 0.07],
            "y": [-0.04, 0.04],
            "z": [0.02, 0.08],
        }
        # table/ground clearance: everything but the ground-mounted arm base (allegro finger links
        # are also named *_link_N, so exclude exactly iiwa7_link_0)
        self.events.conditional_reset.params["valid_criteria"][
            "robot_table_clearance"
        ].body_names = "(?!iiwa7_link_0$).*"
        # finger closing-speed DR: armature sets tau/M.
        self.events.finger_closing_speed = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="(index|middle|ring|thumb)_joint_(0|1|2|3)"),
                "armature_distribution_params": (0.1, 1.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
