# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm pick-up task: grasp and lift the red cube (cube_2) with the LEFT arm only.

Action space (14D flat):
  [0:6]   left arm IK delta pose  (dx dy dz drx dry drz)
  [6]     left gripper command    (±1.0)
  [7:13]  right arm IK delta pose (kept zero during single-arm pick-up)
  [13]    right gripper command   (±1.0)

Scene: only cube_2 (red) is present — cube_1 and cube_3 are removed.

Cube randomisation: left-arm workspace only.
  Looking at the robot from the front camera (1,0,0.5):
    x = forward from robot base (depth in front of robot)
    y = left  (positive y = robot's left arm side)
  cube_2 is randomized within x:[0.17, 0.27]  y:[0.03, 0.17]
  — a compact band centred on the left arm's comfortable sweet spot.

Success condition: cube_2 centre rises 2.5 cm above its resting height (see `cube_rest_z` in
OpenarmPickUpRedCubeEnvCfg.__post_init__ -- resting height itself depends on the pad/cube
geometry in stack_joint_pos_env_cfg.py, so this is computed at cfg-build time, not a literal;
the +2.5 cm offset is calibrated against real recorded teleop demos' peak lift height, not
picked by analogy to an older design -- see the comment above `cube_rest_z` for why).
"""

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp_core
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import openarm_domain_randomization

from . import stack_ik_abs_visuomotor_env_cfg


# ── Subtask / termination helpers ─────────────────────────────────────────────

def cube_is_lifted(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.30,
) -> torch.Tensor:
    """Return 1.0 when cube centre is above *min_height* world-Z."""
    obj = env.scene[asset_cfg.name]
    return (obj.data.root_pos_w[:, 2] > min_height).unsqueeze(-1).float()


def cube_pickup_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.20,
) -> torch.Tensor:
    """Episode terminates with success when cube centre rises above *min_height* world-Z."""
    obj = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] > min_height


# ── Events ────────────────────────────────────────────────────────────────────

@configclass
class PickUpEventCfg:
    """Reset robot to default pose; randomize cube_2 in the left-arm workspace only.

    cube_2 default world position: [0.55, 0.05, CUBE_Z]  (CUBE_Z ≈ 0.15 m)
    Target randomisation range:    x=[0.17, 0.27]  y=[0.03, 0.17]
    Required offsets:              x=[-0.38, -0.28]  y=[-0.02, 0.12]

    Range is intentionally compact — centred on the left arm's comfortable sweet spot.
    To widen: increase x max offset toward -0.20, or widen the y band symmetrically.
    """

    init_robot_pose = EventTerm(
        func=mdp_core.reset_scene_to_default,
        mode="reset",
    )

    randomize_cube_2 = EventTerm(
        func=mdp_core.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.38, -0.28),   # cube_2 default x=0.55 → actual x:[0.17, 0.27]
                "y": (-0.02, 0.12),    # cube_2 default y=0.05 → actual y:[0.03, 0.17]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )


@configclass
class PickUpDomainRandomizationEventCfg(PickUpEventCfg):
    """Extends `PickUpEventCfg` with optional appearance/pose domain-randomization terms.

    Not used by default -- only attached to the env cfg when domain randomization is
    explicitly requested (see generate_dataset.py's --enable_domain_randomization flag).
    Visual randomization requires scene.replicate_physics=False, which is already the
    default for this task family (see stack_env_cfg.py's ObjectTableSceneCfg).
    """

    # NOTE: no texture (image-swap) randomization for cube_2/workspace_pad. An earlier version
    # routed both color AND texture through the Replicator API (`rep.get.prims` +
    # `rep.randomizer.color`/`.texture`), which creates a *new* OmniGraph node + OmniPBR
    # material every single call. Invoked once per "reset" across many episodes, this produced
    # a growing pile of orphaned nodes and surfaced as repeated `UsdExpiredPrimAccessError: Used
    # null prim` errors from `OgnSampleOmniPBR` during an actual generate_dataset.py run (visible
    # once enough resets had happened for an older node's cached prim to be replaced/invalidated
    # by a newer call -- non-fatal to that particular run, but not something to keep doing across
    # a full 50-trial generation). Color below now writes directly to the material's Shader prim
    # (`randomize_visual_color`'s docstring), which has none of that risk -- but true image
    # texture swapping needs an actual UV-mapped texture shader graph, which isn't safe to
    # hand-build without a live Isaac Sim session to verify against. Color + roughness jitter is
    # the appearance-randomization budget for these two assets until that's built properly.
    randomize_cube_2_color = EventTerm(
        func=openarm_domain_randomization.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_2"),
            "colors": {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)},
            "roughness_range": (0.0, 1.0),
        },
    )

    randomize_pad_color = EventTerm(
        func=openarm_domain_randomization.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("workspace_pad"),
            "colors": {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)},
            "roughness_range": (0.0, 1.0),
        },
    )

    randomize_pad_position = EventTerm(
        func=openarm_domain_randomization.randomize_static_asset_pose,
        mode="reset",
        params={
            # Small jitter only -- height (z) and orientation stay fixed so the deliberately
            # tuned pad height / robot clearance (see stack_joint_pos_env_cfg.py) still holds.
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            # Must match workspace_pad's spawn position in stack_joint_pos_env_cfg.py
            # (PAD_CENTER_X, 0.0, PAD_HEIGHT / 2) -- update this if that changes.
            "base_pos": (0.3925, 0.0, 0.095),
            "asset_cfg": SceneEntityCfg("workspace_pad"),
        },
    )

    randomize_light = EventTerm(
        func=openarm_domain_randomization.randomize_scene_lighting,
        mode="reset",
        params={
            "intensity_range": (500.0, 12000.0),
            "color_range": (0.3, 1.0),
            # Every path below is one already referenced by other shipped Isaac Lab task
            # configs/tests (grepped for "Skies/"+".hdr" across the repo) -- kept to
            # confirmed-existing Nucleus assets rather than guessed filenames, since a wrong
            # path fails silently (no texture applied) rather than erroring.
            "skybox_textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ],
            "asset_cfg": SceneEntityCfg("light"),
        },
    )

    # ── Cameras ────────────────────────────────────────────────────────────
    # front_cam is env-anchored (fixed relative to the env origin, not the robot) -- jitter
    # its world position and where it's aimed. wrist/right_wrist/body cams are mounted on
    # moving robot links, so they use randomize_mounted_camera_pose (mount-offset jitter
    # relative to their parent link) instead -- see that function's docstring for why a
    # world-frame jitter like front_cam's would be wrong for them.
    randomize_camera_pose = EventTerm(
        func=openarm_domain_randomization.randomize_fixed_camera_pose,
        mode="reset",
        params={
            # Eye and look-at target jitter independently, so their worst-case combination (both
            # pushed to opposite extremes) matters more than either range alone -- e.g. the old
            # (0.15, 0.08) pair could swing the aim ~20 deg off nominal, enough to lose the
            # robot/workspace out of frame (front_cam's half-FOV is ~24 deg). Halved below to
            # keep the worst case comfortably inside the frame while still varying the viewpoint.
            "pos_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (-0.05, 0.05)},
            "look_at_offset": (0.45, 0.0, 0.15),
            "look_at_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.03, 0.03)},
            "asset_cfg": SceneEntityCfg("front_cam"),
        },
    )

    # Kept moderate (not maxed out like color/lighting above): these mounts sit close to the
    # gripper and are the views the grasp itself depends on, so over-jittering risks losing the
    # cube/gripper out of frame the same way front_cam did before it got dialed back.
    randomize_wrist_cam_pose = EventTerm(
        func=openarm_domain_randomization.randomize_mounted_camera_pose,
        mode="reset",
        params={
            "pos_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
            "rot_range": {"roll": (-0.14, 0.14), "pitch": (-0.14, 0.14), "yaw": (-0.14, 0.14)},
            "parent_body_name": "openarm_left_ee_tcp",
            "asset_cfg": SceneEntityCfg("wrist_cam"),
        },
    )

    randomize_right_wrist_cam_pose = EventTerm(
        func=openarm_domain_randomization.randomize_mounted_camera_pose,
        mode="reset",
        params={
            "pos_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
            "rot_range": {"roll": (-0.14, 0.14), "pitch": (-0.14, 0.14), "yaw": (-0.14, 0.14)},
            "parent_body_name": "openarm_right_ee_tcp",
            "asset_cfg": SceneEntityCfg("right_wrist_cam"),
        },
    )

    randomize_body_cam_pose = EventTerm(
        func=openarm_domain_randomization.randomize_mounted_camera_pose,
        mode="reset",
        params={
            "pos_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
            "rot_range": {"roll": (-0.14, 0.14), "pitch": (-0.14, 0.14), "yaw": (-0.14, 0.14)},
            "parent_body_name": "openarm_body_link",
            "asset_cfg": SceneEntityCfg("body_cam"),
        },
    )

    # NOTE: robot/table texture randomization (via franka_stack_events.randomize_visual_texture_material)
    # was removed here -- it goes through the same repeated-Replicator-call pattern that produced
    # the "cube_2"/"workspace_pad" texture bug above (see the NOTE near randomize_cube_2_color).
    # That function is pre-existing, shared code used elsewhere in this task family, so it wasn't
    # rewritten; it's just not worth attaching two more copies of a known-fragile call pattern to
    # this event cfg on top of the two (cube/pad) that already caused a real, observed failure.


# ── Observations ──────────────────────────────────────────────────────────────

@configclass
class PickUpObservationsCfg:
    """Camera observations + EEF pose (needed by Mimic) + subtask completion signals."""

    @configclass
    class PolicyCfg(ObsGroup):
        front_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        body_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False},
        )
        # EEF pose — required by OpenArmPickUpIKAbsMimicEnv.get_robot_eef_pose()
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask termination signals for Mimic auto-annotation.

        Both signals use cube height so they are robust and require no EEF tracking:
          grasp (min_height=cube_rest_z + 5 mm):  cube_2 has just left the pad.
                                                   Marks end of the reach-and-grasp segment.
          lift  (min_height=cube_rest_z + 2.5 cm): cube_2 is clearly picked up.
                                                    Stored for debugging; not used as a Mimic
                                                    term signal (it is the last subtask). Must
                                                    still fire at least once during a demo or
                                                    annotate_demos.py rejects the episode --
                                                    calibrated against real recorded peak
                                                    heights, see __post_init__.

        The literal `min_height` values below are placeholders overwritten in
        OpenarmPickUpRedCubeEnvCfg.__post_init__ once cube_2's actual resting height
        (`cube_rest_z`) is known -- see that method for why a hardcoded literal here is
        wrong whenever the pad height / cube size changes.
        """

        grasp = ObsTerm(
            func=cube_is_lifted,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.155},
        )
        lift = ObsTerm(
            func=cube_is_lifted,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.30},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ── Environment config ────────────────────────────────────────────────────────

@configclass
class OpenarmPickUpRedCubeEnvCfg(stack_ik_abs_visuomotor_env_cfg.OpenarmCubeStackVisuomotorEnvCfg):
    """OpenArm pick-up task: grasp and lift the red cube (cube_2) with the left arm.

    Key differences from the visuomotor stack env:
      - Only cube_2 (red) is present in the scene (cube_1 and cube_3 removed).
      - cube_2 is randomized within the left arm's reachable workspace only.
      - ee_frame targets openarm_left_ee_tcp (required by eef_pos/eef_quat obs).
      - subtask_terms observation group added for Mimic auto-annotation.
      - Success terminates when cube_2 rises above 0.20 m.
      - Right arm IK action kept so TAB-switching works during teleoperation.

    Mimic task: Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0
    """

    gripper_joint_names: list = ["openarm_left_finger_joint.*"]
    gripper_open_val: float = 0.044
    gripper_threshold: float = 0.018

    def __post_init__(self):
        super().__post_init__()  # cameras, IK-Abs action, pad, cubes

        # ── Remove blue and green cubes — only red cube participates ──────────
        # The scene builder skips attributes set to None.
        self.scene.cube_1 = None
        self.scene.cube_3 = None

        # ── Observations ─────────────────────────────────────────────────────
        self.observations.policy = PickUpObservationsCfg.PolicyCfg()
        self.observations.subtask_terms = PickUpObservationsCfg.SubtaskCfg()

        # cube_2's resting height (its spawn Z, set in stack_joint_pos_env_cfg.py from
        # PAD_HEIGHT/CUBE_SIZE) drives every height threshold below. Reading it back here
        # instead of hardcoding a literal keeps "grasp"/"lift"/"success" correct if pad height
        # or cube size ever change again -- a stale literal here previously broke the "grasp"
        # subtask signal outright (it sat *below* the resting height, so cube_is_lifted was
        # always True and the 0->1 transition annotate_demos.py/generate_dataset.py look for
        # never happened).
        #
        # The +0.025/+0.025 offsets below (lift/success) are calibrated against the actual
        # peak cube_2 height reached across 10 recorded teleop demos under this same geometry
        # (rest=0.1975, observed max_z ranged 0.2006-0.2444, 9/10 demos cleared ~0.232+) -- a
        # first attempt used a much larger +0.15/+0.05 offset by analogy to an older design
        # tuned for a lower pad, and it made annotate_demos.py's success_term/"lift" signal
        # unreachable by *every* recorded demo (all fell short of 0.245+). Keep any future
        # offset choice grounded in the demos' actual achieved lift height, not a guessed value.
        cube_rest_z = float(self.scene.cube_2.init_state.pos[2])
        self.observations.subtask_terms.grasp.params["min_height"] = cube_rest_z + 0.005
        self.observations.subtask_terms.lift.params["min_height"] = cube_rest_z + 0.025

        # ── ee_frame: track left EEF (required by eef_pos / eef_quat obs) ────
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link1",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        )

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.time_out = None
        self.terminations.success = TerminationTermCfg(
            func=cube_pickup_success,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": cube_rest_z + 0.025},
            time_out=False,
        )

        # ── Events: robot reset + cube_2 randomised in left-arm workspace ────
        self.events = PickUpEventCfg()

        # ── Right arm action (for TAB-switching during keyboard recording) ────
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
        )
        self.actions.right_gripper_action = mdp_core.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        self.image_obs_list = ["front_cam", "wrist_cam", "right_wrist_cam", "body_cam"]
