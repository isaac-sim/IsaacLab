# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 bindings for the shared state-based lift task."""

from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MassCfg, UsdPhysicsCollisionCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.so101 import SO101_CFG

from ... import lift_env_cfg as lift
from ... import mdp

FINGERTIP_LIST = ["gripper", "moving_jaw_so101_v1"]
"""Static and moving finger bodies of the SO-101 gripper."""

THUMB_SENSOR = "gripper_object_s"
"""Object contact with the static finger."""

FINGER_SENSORS = ["moving_jaw_so101_v1_object_s"]
"""Object contact with the moving finger."""


@configclass
class SO101SceneCfg(lift.SceneCfg):
    """Table-mounted SO-101 and a 3 cm, 50 g cube within its grasp envelope."""

    robot: ArticulationCfg = replace(
        SO101_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=replace(
            SO101_CFG.spawn,
            make_uninstanceable=True,
            collision_props={
                "/(gripper|moving_jaw_so101_v1)/collisions/.*": [NewtonCollisionCfg(contact_margin=0.0015)],
            },
            variants={
                "Robot": "robot",
                "Sensor": "sensors",
                "Physics": preset(default="physics", isaacsim_physx="physx", physx="physx", ovphysx="physx"),
            },
        ),
        init_state=replace(
            SO101_CFG.init_state,
            # The asset root sits 3.008 cm above the clamp foot; the tabletop is at z=0.255 m.
            pos=(-0.16, 0.2, 0.22492),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),
        ),
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=replace(
            lift.ObjectCfg().cube,
            size=(0.03, 0.03, 0.03),
            mass_props=MassCfg(mass=0.05),
            activate_contact_sensors=True,
            collision_props=[
                UsdPhysicsCollisionCfg(),
                MujocoCollisionCfg(priority=1, solref=(0.01, 1.0)),
            ],
        ),
        # Two millimeters of table clearance before settling.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.43, 0.2, 0.272)),
    )

    def __post_init__(self):
        super().__post_init__()
        for link_name in FINGERTIP_LIST:
            # Object-side filtering supports the static and moving jaws on both PhysX and Newton.
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{link_name}"],
                ),
            )


@configclass
class StateObservationCfg(lift.ObservationsCfg):
    """Shared root-frame state, contact forces and object point clouds with history."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [THUMB_SENSOR, *FINGER_SENSORS]},
            # Sensors measure force on the object; report the opposing force on each finger.
            scale=-1.0,
            clip=(-20.0, 20.0),
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = FINGERTIP_LIST


@configclass
class SO101RelJointPosActionCfg:
    """Relative position targets for the five arm joints and the moving jaw."""

    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class SO101RewardCfg(lift.RewardsCfg):
    """Shared lift rewards gated by contact with both SO-101 fingers."""

    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.75,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )
    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=0.1,
        params={"threshold": 0.01, "sensor_names": [THUMB_SENSOR, *FINGER_SENSORS]},
    )

    def __post_init__(self):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=FINGERTIP_LIST)
        for term in (self.fingers_to_object, self.position_tracking, self.orientation_tracking, self.success):
            if term is not None:
                term.params["thumb_name"] = THUMB_SENSOR
                term.params["finger_names"] = FINGER_SENSORS


@configclass
class SO101EventCfg(lift.EventCfg):
    """Shared randomization and reset bank adapted to the small arm and cube."""

    def __post_init__(self):
        super().__post_init__()
        # Scale the small cube's own inertia instead of adding an absolute inertia for larger objects.
        self.object_physics_inertia.params["inertia_distribution_params"] = (0.8, 1.2)
        self.object_physics_inertia.params["operation"] = "scale"
        reset = self.conditional_reset.params
        terms = reset["terms"]
        terms["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["shoulder_.*", "elbow_flex", "wrist_flex", "gripper"]
        )
        terms["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="wrist_roll")
        terms["reset_robot_wrist_joint"].params["position_range"] = (-0.5, 0.5)
        terms["reset_object"].params["pose_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (0.0, 0.10),
            "yaw": (-0.785398, 0.785398),
        }
        # Retain the shared near-hand proposal; clearance rejection removes intersecting draws.
        near_hand = terms["reset_object_to_target"].params
        near_hand["target_cfg"] = SceneEntityCfg("robot", body_names="gripper")
        near_hand["pose_range"] = {"x": (-0.004, 0.016), "y": (-0.01, 0.01), "z": (-0.105, -0.085)}
        reset["diversity_feature"].body_names = FINGERTIP_LIST
        table_clearance = reset["valid_criteria"]["robot_table_clearance"]
        # The clamp/base is mounted on the table and is excluded from moving-link clearance.
        table_clearance.body_names = ["shoulder", "upper_arm", "lower_arm", "wrist", *FINGERTIP_LIST]
        table_clearance.min_clearance = 0.001


@configclass
class SO101CurriculumCfg(lift.CurriculumCfg):
    """Shared ADR with height bounds that admit the SO-101's lower tabletop resets."""

    oob_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.object_out_of_bound.params.in_bound_range.z",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": (0.26, 0.75), "final_value": (0.20, 0.75), "difficulty_term_str": "adr"},
        },
    )


@configclass
class SO101MixinCfg:
    """Robot-specific bindings for the shared lift environment."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=2.0)
    observations: StateObservationCfg = StateObservationCfg()
    actions: SO101RelJointPosActionCfg = SO101RelJointPosActionCfg()
    rewards: SO101RewardCfg = SO101RewardCfg()
    events: SO101EventCfg = SO101EventCfg()
    curriculum: SO101CurriculumCfg | None = SO101CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        physics = lift.PhysicsCfg()
        self.sim.physics = preset(
            default=physics.newton_mjwarp,
            newton_mjwarp=physics.newton_mjwarp,
            isaacsim_physx=physics.isaacsim_physx,
            ovphysx=physics.ovphysx,
            physx=physics.physx,
        )
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(-0.8, -0.25, 0.6), lookat=(-0.29, 0.2, 0.33))
        self.commands.object_pose.ranges.pos_x = (-0.025, 0.025)
        self.commands.object_pose.ranges.pos_y = (-0.295, -0.245)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.20)
        self.terminations.object_out_of_bound.params["in_bound_range"] = {
            "x": (-0.65, 0.25),
            "y": (-0.25, 0.65),
            "z": (0.26, 0.75),
        }
        self.terminations.abnormal_robot.func = mdp.abnormal_robot_state


@configclass
class SO101LiftEnvCfg(SO101MixinCfg, lift.LiftEnvCfg):
    """Lift a small cube using the shared state observations, reset bank and gravity ADR.

    The reset bank rejects intersecting robot/object/table proposals. Before rejection and bank
    sampling, 25% of object proposals are near the gripper; the remainder occupy a 5 cm tabletop
    square with up to 10 cm of vertical offset. Evaluation uses this mixed bank, rather than
    measuring tabletop-only picks. Set
    ``events.conditional_reset.params.terms.reset_object_to_target.params.probability=0`` and
    ``events.conditional_reset.params.terms.reset_object.params.pose_range.z=[0,0]`` before
    initialization for tabletop-only evaluation proposals.

    Both finger contacts gate the shared position-progress and success rewards. The sticky
    success flag records reaching at least one command during an episode; it does not establish
    success at every resampled command. Old compact-observation SO-101 lift checkpoints require
    retraining for this task's observation history and relative joint action contract.
    """
