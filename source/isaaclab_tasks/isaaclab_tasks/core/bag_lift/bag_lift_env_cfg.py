# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CONFIG for the bag-lift task (Direct workflow): numbers and descriptions only, no logic.

The logic lives in bag_lift_env.py; scene objects (arm, bag, tray) live in bag_sand_scene.py.
Any value here can be overridden from the CLI with Hydra, e.g.  env.rew_scale_collision=-10.0

TASK
    1. grasp   : gripper commanded closed and the strap held geometrically (rest on the rigid jaw, or pinched
                 between the jaws), debounced over a few steps. No force threshold.
    2. lift    : while grasped, the bag rises ``lift_height`` (5 cm)
    3. final   : gripper commanded closed, a jaw within ``success_strap_gap`` of the strap, bag still
                 ``lift_height`` up, and arm joints 1-6 within ``final_pose_tol`` (10 deg) of return_home.
    Success = (3). The episode ends on first success with a flat bonus. The episode is the demo length plus
    ``episode_extra_s``. Contact on the lower 95% of the bag body is penalised; the top rim and the handle are not.
"""

import math
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from .bag_sand_scene import (
    AMBIENT_LIGHT_CFG,
    ARM_CFG,
    ARM_CONTACT_CFG,
    ARM_GRIPPER_JOINT,
    BAG_CFG,
    EE_BODY_NAME,
    EE_TIP_OFFSET,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    HANDLE_CFG,
    HANDLE_INFO_JSON,
    JAW_INSERT_AXIS,
    JAW_POINTS_JSON,
    MOVING_JAW_BODY_NAME,
    SANDBOX_CFG,
    USE_DEFORMABLE_HANDLE,
    USE_SAND_PARTICLES,
)

if USE_SAND_PARTICLES:
    _PHYSICS_CFG = PhysxCfg(gpu_max_particle_contacts=2**24)
elif USE_DEFORMABLE_HANDLE:
    _PHYSICS_CFG = PhysxCfg(gpu_found_lost_pairs_capacity=2**22)  # deformable contacts need a larger pair buffer
else:
    _PHYSICS_CFG = PhysxCfg()


@configclass
class BagLiftSceneCfg(InteractiveSceneCfg):
    sandbox = SANDBOX_CFG  # plain rigid tray
    arm = ARM_CFG  # SO-101, physics layer switched on
    bag = BAG_CFG  # bag merged into one rigid body
    contact: ContactSensorCfg = ARM_CONTACT_CFG  # every arm link vs the bag
    ambient = AMBIENT_LIGHT_CFG  # one dim dome light

    def __post_init__(self):
        super().__post_init__()
        if USE_DEFORMABLE_HANDLE:
            self.handle = HANDLE_CFG  # bendable strap clamped to the bag body (bag_sand_scene.py)


@configclass
class EventCfg:
    # Bag grip and servo stiffness are re-drawn every reset. Set a range to a single value to freeze it.
    bag_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bag", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("arm", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.25),
            "damping_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class BagLiftEnvCfg(DirectRLEnvCfg):
    # --- timing: physics 120 Hz, policy 30 Hz --------------------------------
    # 30 Hz is plenty for a residual on a slow teacher, and halves the steps PPO must look ahead over.
    decimation = 4
    # Placeholder: with use_demo_residual the env sets episode_length_s = demo length + episode_extra_s.
    episode_length_s = 30.0
    episode_extra_s = 4.0  # [s] time after the teacher ends to settle into the return_home box

    # --- sizes (MUST match bag_lift_env.py) ----------------------------------
    action_space = 7  # 6 arm joints + gripper
    observation_space = 47  # see BagLiftEnv._get_observations
    state_space = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=_PHYSICS_CFG,
    )
    # --num_envs overrides num_envs. Keep it a multiple of 4 (see agents/rl_games_direct_ppo_cfg.yaml).
    scene: BagLiftSceneCfg = BagLiftSceneCfg(num_envs=128, env_spacing=2.5)

    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0005, operation="abs"),
    )

    # --- arm / hand ------------------------------------------------------------
    ee_body_name = EE_BODY_NAME  # link holding the rigid jaw
    ee_tip_offset = EE_TIP_OFFSET  # rigid-jaw tip in that link's frame [m]
    jaw_insert_axis = JAW_INSERT_AXIS  # direction the rigid jaw points (link -Y)
    moving_jaw_body_name = MOVING_JAW_BODY_NAME
    gripper_joint_name = ARM_GRIPPER_JOINT
    gripper_open_pos = GRIPPER_OPEN_POS
    gripper_closed_pos = GRIPPER_CLOSED_POS
    # Training: joint target = teacher(t) + action_scale * action. Zero residual replays grasp_poses.json.
    # 0.35 rad (~20 deg) is enough to correct for the bag randomization below, not enough to throw the demo away.
    # The pose tool sets use_demo_residual False and widens action_scale so it can jump to any edited waypoint.
    use_demo_residual = True
    action_scale = 0.35
    # Over the last residual_fade_frac of the demo the residual shrinks from action_scale to carry_action_scale.
    # It never reaches zero, so the policy can still keep the strap in the jaws on the way home and after the demo.
    residual_fade_frac = 0.25
    carry_action_scale = 0.07  # [rad] ~4 deg, under final_pose_tol so it cannot push the arm out of the box
    demo_speed = 1.0  # teacher playback: 1 = grasp_poses.json timing; >1 finishes the demo earlier (speed ceiling)
    arm_joint_noise = 0.05  # [rad] start-pose jitter

    # --- handle geometry (bag frame) -------------------------------------------
    # The handle is a thin strap (13 mm wide along bag +Y, about 1 mm thick) bent into a U that sticks 38 mm out
    # of the bag face (bag +Z, so the apex is at z = 0.114). The rigid jaw is a wedge (9 mm at the tip, 25 mm
    # 5 cm up, then a bulky housing), so it cannot be threaded through the loop. The jaws pinch the strap instead.
    # Pinch target: strap apex, 12 mm below its lower edge, so the thin tip ends up beside the strap.
    pinch_target = (0.0, -0.012, 0.1135)  # [m] bag frame
    # Written by `grasp_pose_tool.py --record`. When it exists, pinch_target above is replaced by the recorded tip
    # position and the align reward matches the recorded jaw orientation (both in the bag frame).
    goal_file = str(Path(__file__).parent / "grasp_goal.json")
    reach_radius = 0.015  # [m] tip this close to the pinch target counts as reached
    carry_radius = 0.02  # [m] hand-rise reward fades out as the tip moves this far off the strap

    # Rigid-handle fallback only (USE_DEFORMABLE_HANDLE=False): a contact counts as "on the handle strap"
    # from its location in the bag frame (x along the strap, y its 13 mm width, z outward from the body).
    # The body's face is at z = 0.076 and the strap spans z 0.076..0.114.
    handle_zone_half_x = 0.040  # [m]
    handle_zone_half_y = 0.015  # [m]
    handle_zone_min_z = 0.078  # [m]
    # Bag-frame +Z: body bottom -0.076 .. opening +0.076, handle above that (bag_sand_scene.py). Collision
    # penalises only the lower collision_lower_frac of the body; the top rim and the handle are free.
    bag_body_z_min = -0.076  # [m]
    bag_body_z_max = 0.076  # [m]
    collision_lower_frac = 0.95  # lower this fraction of the body is a collision; top (1-frac) + handle is not

    # --- deformable handle (only when USE_DEFORMABLE_HANDLE in bag_sand_scene.py) -------------------------------
    # PhysX cannot report contacts on deformables, so the grasp is inferred from geometry (no forces):
    #   1. the gripper is commanded closed (target within grip_closed_margin of its closed limit)
    #   2. EITHER (preferred) >= min_strap_nodes strap nodes REST on the rigid jaw (within pinch_dist of its proxy
    #      points in assets/jaw_points.json, on the mouth side) at depth < mouth_mid_frac + mouth_tol, i.e. midway or
    #      further back; OR >= min_strap_nodes nodes lie BETWEEN the jaws (within pinch_dist of both, on opposite
    #      sides) at depth mouth_mid_frac +- mouth_tol.
    #   Depth: 0 = housing end (mouth_base_y), 1 = jaw tip (mouth_tip_y), in the rigid-jaw link frame (-Y is the
    #   jaw's long axis).
    handle_info_file = HANDLE_INFO_JSON
    jaw_points_file = JAW_POINTS_JSON
    touch_dist = 0.006  # [m] a jaw 'touches' the strap within this (status display only)
    pinch_dist = 0.008  # [m] each jaw within this of a strap node, on opposite sides = node is between the jaws
    min_strap_nodes = 3  # strap nodes that must be between the jaws
    grip_closed_margin = 0.05  # [rad] gripper target this close to the closed limit counts as 'commanded closed'
    # Debounce: the geometric test flickers while the strap shifts in the jaws. grasped turns on after
    # grasp_on_steps consecutive raw hits and off after grasp_off_steps consecutive misses (policy steps).
    grasp_on_steps = 3
    grasp_off_steps = 15  # 0.5 s at 30 Hz
    mouth_base_y = -0.040  # [m] rigid-jaw frame y of the housing end of the mouth (depth 0)
    mouth_tip_y = -0.095  # [m] rigid-jaw frame y of the jaw tip (depth 1)
    mouth_mid_frac = 0.5  # target depth: midway into the mouth
    # depth may be off by this much (+-16 mm on the 55 mm mouth). The demo carries the bag with the strap at
    # depth 0.62-0.75, so 0.2 (up to 0.7) flipped grasped on and off from run to run on the same demo.
    mouth_tol = 0.3

    # --- final pose (the success condition) -----------------------------------------------------------------------
    # Success = gripper commanded closed, a jaw within success_strap_gap of the strap, the bag lift_height up
    # (it only stays up if something holds it), and joints 1-6 within final_pose_tol of return_home.
    final_pose_file = str(Path(__file__).parent / "grasp_poses.json")
    approach_pose_waypoint = "grasp_closed"  # joint-space target until the strap is held
    final_pose_waypoint = "return_home"  # waypoint in that file (joint angles inherit like in grasp_pose_tool.py)
    final_pose_tol = math.radians(10.0)  # [rad] every one of joints 1-6 must be this close
    return_scale = 0.5  # [rad] RMS joint error where the return reward has dropped to 24%
    track_scale = 0.25  # [rad] RMS tracking error where the demo-follow reward has dropped to 24%
    demo_hold_s = 0.3  # [s] pause after each waypoint, also the grasp_pose_tool.py --hold_s default
    demo_move_s = 2.0  # [s] fallback move duration when a waypoint has no speed_deg_per_s
    demo_close_s = 1.5  # [s] fallback duration of the grasp_closed move
    demo_min_move_s = 0.3  # [s] shortest move, same as grasp_pose_tool.MIN_MOVE_S

    # --- success ----------------------------------------------------------------
    terminate_on_success = True  # stop the clock on first success (time-to-success)
    success_strap_gap = 0.010  # [m] closest jaw within this of a strap node counts as still on the strap
    lift_height = 0.05  # [m] bag must rise this far above its resting height while grasped
    bag_out_of_bounds = 0.35  # [m] bag pushed this far from the tray centre = episode over

    # --- collision penalty ---------------------------------------------------------
    # Per step: collision = tanh(F / collision_force_scale), F = contact force [N] on the LOWER
    # collision_lower_frac of the bag body. The top rim and the handle are not penalised.
    collision_force_scale = 5.0  # [N] the open-loop demo brushes the body at ~1 N; don't make that a cliff
    # The penalty weight ramps linearly 0 -> rew_scale_collision over this many env steps (1 PPO iteration =
    # horizon_length = 64 steps, so 16000 steps ~ 250 iterations). At full weight from step 0 the policy learned
    # to stay away from the bag instead of approaching it.
    collision_ramp_steps = 16000

    # --- bag reset randomization (relative to each env origin) ------------------------
    # Full spawn jitter around the demo bag pose. +-5 cm and +-20 deg move the strap up to ~9 cm, the edge of what
    # the 0.35 rad residual can correct.
    bag_xy_range = (-0.05, 0.05)  # [m] tray-plane spawn
    bag_yaw_range = (-math.radians(20.0), math.radians(20.0))  # heading +-20 deg
    bag_random_flip = False  # keep the handle on the same face
    bag_tilt_range = math.radians(5.0)  # small roll/pitch [rad]
    # Curriculum: the ranges above are multiplied by a scale that starts at rand_scale_init and steps up by
    # rand_scale_step each time a window of ~num_envs episodes reaches rand_curriculum_success final-pose rate.
    # rand_curriculum = False uses the full ranges from the start.
    rand_curriculum = True
    rand_scale_init = 0.0
    rand_scale_step = 0.1
    rand_curriculum_success = 0.8

    # --- reward weights (per-step terms are multiplied by step_dt; success is a one-off) ----------
    # reach / align / reached only pay before the grasp, so they do not pull the hand back down during the carry
    rew_scale_reach = 0.5  # rigid tip toward the pinch target on the strap
    rew_scale_align = 0.5  # rigid jaw matching the recorded goal orientation
    rew_scale_reached = 0.5  # rigid tip at the pinch target
    rew_scale_close = 1.0  # gripper closing while a jaw is at the strap (leads to the grasp)
    rew_scale_track = 2.0  # all 7 joints toward the teacher pose this step (zero residual = the demo)
    rew_scale_grasp = 5.0  # geometric grasp (must beat hovering at the pinch without holding)
    rew_scale_squeeze = 2.0  # gripper closed fraction while grasped
    rew_scale_ee_up = 3.0  # rigid jaw pointing world +Z while grasped (carry the bag with the EE up)
    rew_scale_hand_up = 5.0  # hand rising 0..lift_height while grasped (dense: pays at the first cm)
    rew_scale_lift = 5.0  # bag rising 0..lift_height while grasped
    rew_scale_return = 8.0  # arm toward return_home while still holding the strap
    rew_scale_collision = -0.5  # contact on the lower 95% of the bag body (full weight after the ramp)
    rew_scale_action_rate = -0.01  # jerky motion
    rew_scale_residual = -0.08  # ||action||^2: stay on the teacher unless the bag draw requires a correction
    rew_scale_time = -2.0  # per second of wall time
    # One-off on first success, NOT scaled by step_dt. Must beat what staying grasped would still earn: the
    # holding-gated terms total ~28/s and at most ~episode_extra_s remain after the demo, so ~110.
    rew_scale_success = 200.0
