# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration shared by the VLA benchmark tasks."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import InclusionSet
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import HeterogeneousTaskCfg, HeterogeneousTaskManager, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from . import VLA_BENCHMARKS_ASSETS_DIR, mdp
from .config.franka.robot_cfg import FRANKA_ROBOTIQ_CFG

GRIPPER_BODY = "robot/Robotiq_2F_85_edit/Robotiq_2F_85/left_inner_finger"
"""Gripper link, relative to the environment prim, whose contacts decide whether an object is held."""


def _camera(prim_path: str, focal_length: float, pos: tuple, rot: tuple, follow_link: bool = False) -> CameraCfg:
    """Return a 448 x 252 RGB camera with the RoboLab lens (the 1280 x 720 view, downsampled)."""
    return CameraCfg(
        prim_path=prim_path,
        width=448,
        height=252,
        data_types=["rgb"],
        update_period=0.0,
        update_latest_camera_pose=follow_link,
        # The renderer that draws the objects selected as mesh variants.
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
            clipping_range=(0.01, 1.0e6),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


@configclass
class VLABenchmarkSceneCfg(InteractiveSceneCfg):
    """Robot, robot table, ground, lights, cameras and contact sensor of the VLA benchmark tasks.

    Task inclusions select rigid objects from the flat catalog below, the furniture first. The contact sensor
    senses every rigid object and the gripper link against each other.
    """

    robot = FRANKA_ROBOTIQ_CFG
    table_fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/scene/franka_table/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.087, 0.0, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
    )
    # One ground below the furniture of all tasks; the RoboLab scenes hide theirs.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.697)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            texture_file=f"{VLA_BENCHMARKS_ASSETS_DIR}/scene/home_office.exr",
            texture_format="latlong",
            visible_in_primary_ray=True,
        ),
    )
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )
    over_shoulder_left_camera = _camera(
        "{ENV_REGEX_NS}/over_shoulder_left_camera", 2.1, (0.05, 0.57, 0.66), (-0.195, 0.399, 0.805, -0.393)
    )
    # The wrist camera rides on the gripper, so its pose follows the simulated link.
    wrist_cam = _camera(
        "{ENV_REGEX_NS}/robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/wrist_cam",
        2.8,
        (-0.074, 0.031, 0.011),
        (0.704, -0.692, -0.11, 0.114),
        follow_link=True,
    )
    contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/([^/]+|{GRIPPER_BODY})",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/([^/]+|{GRIPPER_BODY})"],
        update_period=0.0,
    )


@configclass
class ActionsCfg:
    """Arm joint position targets and a binary gripper command (negative closes)."""

    arm = isaaclab_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], preserve_order=True, use_default_offset=False
    )
    gripper = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": math.pi / 4},
    )


@configclass
class ObservationsCfg:
    """Camera images and arm and gripper positions for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        over_shoulder_left_camera = ObsTerm(
            func=isaaclab_mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("over_shoulder_left_camera"), "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=isaaclab_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "normalize": False}
        )
        arm_joint_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"], preserve_order=True)},
        )
        # Gripper opening from 0 (open) to 1 (closed).
        gripper_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
            clip=(0.0, math.pi / 4),
            scale=4.0 / math.pi,
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy = PolicyCfg()


@configclass
class EventsCfg:
    """Reset the robot and the objects to their start states."""

    reset = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class TerminationsCfg:
    """Time-out; tasks add their success term."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)


# Asset definitions are shared by all task records; poses belong to the records, not this catalog.
ASSET_CATALOG = {
    name: RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/objects/{name}/{name}.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True) if kinematic else None,
        ),
    )
    for name, kinematic in {
        "appleseed_endtable": True,
        "appleseed_sofatable": True,
        "av_aftonconsole": True,
        "av_apple": False,
        "av_book_01": False,
        "av_book_11": False,
        "av_book_21": False,
        "av_book_33": False,
        "av_book_51": False,
        "av_book_66": False,
        "av_book_77": False,
        "av_book_99": False,
        "av_bowl_plate": False,
        "av_chess_bishop": False,
        "av_chess_king": False,
        "av_chess_pawn": False,
        "av_chess_queen": False,
        "av_chess_rook": False,
        "av_coaster_hexagon": False,
        "av_decorativebowl": False,
        "av_desperes_table": True,
        "av_eastrural_table": True,
        "av_ellendale": True,
        "av_ellisville": True,
        "av_eraser": False,
        "av_eurekaconsole": True,
        "av_hexagon": True,
        "av_hippoorganizer": False,
        "av_jennings_table": True,
        "av_kiwi01": False,
        "av_knife": False,
        "av_mustard_jar": False,
        "av_p_glassware_short": False,
        "av_p_glassware_tall": False,
        "av_pan_b": False,
        "av_pan_d": False,
        "av_photoalbum_closed": False,
        "av_plant_succulent_01": False,
        "av_plant_succulent_02": False,
        "av_salt_bowl": False,
        "av_sketchbook": False,
        "av_stawberry": False,
        "av_tincan": False,
        "av_white_tray": False,
        "av_whittershins": True,
        "av_willow": True,
        "av_woodenchmedium": False,
        "av_woodenchsmall": False,
        "av_woodentray_medium": False,
        "avocado01": False,
        "bagel": False,
        "bagel_00": False,
        "bbq_sauce_bottle": False,
        "bbq_sauce_bottle_hot3d": False,
        "bin_a02": False,
        "bin_a03": False,
        "bin_a05": False,
        "bin_b04": False,
        "birdhouse": False,
        "blackandbrassbowl_large": False,
        "blackandbrassbowl_small": False,
        "blue_block": False,
        "bowl": False,
        "bowl_ycb": False,
        "brick": False,
        "butter": False,
        "canned_mushrooms": False,
        "canned_peaches": False,
        "canned_tuna": False,
        "ceramic_mug": False,
        "cheez_it": False,
        "chocolate_pudding": False,
        "chocolate_pudding_mix": False,
        "clamp": False,
        "clay_plates": False,
        "cline": True,
        "coffee_can": False,
        "coffee_pot": False,
        "contemporary_laterialfile": True,
        "contemporary_lockedstorage": True,
        "contemporary_lowfulldesk": True,
        "contemporary_lowhalfdesk": True,
        "contemporary_openstorage": True,
        "cordless_drill": False,
        "corn_can": False,
        "crabbypenholder": False,
        "cream_cheese": False,
        "cubebox_a02": False,
        "desk_01": True,
        "dry_erase_marker": False,
        "dumbbell": False,
        "foam_roller": False,
        "fork_big": False,
        "frozen_vegetable_block": False,
        "frozen_waffles": False,
        "fstylejug_a01": False,
        "fstylejug_a02": False,
        "gilbert": True,
        "glasses": False,
        "granola_bars": False,
        "green_beans_can": False,
        "green_block": False,
        "jello": False,
        "ketchup_bottle": False,
        "lime01": False,
        "macaroni_and_cheese": False,
        "mayonnaise_bottle": False,
        "measuring_cups": False,
        "measuring_cups_1": False,
        "measuring_spoon": False,
        "milk_carton": False,
        "milkjug_a01": False,
        "milkjug_a02": False,
        "milkjug_a03": False,
        "mug": False,
        "mug_ycb": False,
        "mustard": False,
        "mustard_bottle": False,
        "naturalbostonroundbottle_a01": False,
        "naturalbostonroundbottle_a02": False,
        "naturalbostonroundbottle_a03": False,
        "oatmeal_raisin_cookies": False,
        "orange_02": False,
        "orange_juice_carton": False,
        "parmesan_cheese_canister": False,
        "peas_and_carrots": False,
        "pineapple_slices_can": False,
        "pitcher": False,
        "pitcher_ycb": False,
        "pitted_cherries": False,
        "plate_small": False,
        "popcorn_box": False,
        "potato_masher": False,
        "pumpkinlarge": False,
        "pumpkinsmall": False,
        "raisin_box": False,
        "ranch_dressing": False,
        "red_block": False,
        "rubiks_cube": False,
        "scissors": False,
        "serving_bowl": False,
        "smartphone": False,
        "soft_scrub": False,
        "soup_can": False,
        "spaghetti": False,
        "spam_can": False,
        "spatula_01": False,
        "spatula_03": False,
        "spatula_04": False,
        "spatula_05": False,
        "spatula_06": False,
        "spatula_07": False,
        "spatula_08": False,
        "spatula_09": False,
        "spatula_10": False,
        "spatula_11": False,
        "spatula_12": False,
        "spatula_13": False,
        "spatula_16": False,
        "spatula_17": False,
        "spatula_18": False,
        "spatula_19": False,
        "spoon": False,
        "spoon_1": False,
        "spoon_2": False,
        "spoon_big": False,
        "standard_halfunit": True,
        "standard_smallunit": True,
        "steam": True,
        "storage_box": False,
        "sugar_box": False,
        "table_bamboo": True,
        "table_black": True,
        "table_maple": True,
        "table_oak": True,
        "tomato_sauce_can": False,
        "tomato_soup_can": False,
        "tuna_can": False,
        "whitepackerbottle_a01": False,
        "whitepackerbottle_a02": False,
        "whitepackerbottle_a03": False,
        "whitepackerbottle_a04": False,
        "wood_block": False,
        "wooden_bowl": False,
        "yellow_block": False,
        "yogurt_cup": False,
    }.items()
}
ASSET_CATALOG.update(
    {
        "empty_1": HeterogeneousTaskManager.placeholder(1),
        "empty_2": HeterogeneousTaskManager.placeholder(2),
        "empty_3": HeterogeneousTaskManager.placeholder(3),
        "empty_4": HeterogeneousTaskManager.placeholder(4),
        "empty_5": HeterogeneousTaskManager.placeholder(5),
        "empty_6": HeterogeneousTaskManager.placeholder(6),
        "empty_7": HeterogeneousTaskManager.placeholder(7),
        "empty_8": HeterogeneousTaskManager.placeholder(8),
    }
)


# Fixed benchmark records. Compatibility is declared with InclusionSet; no task environment classes.
TASKS = {
    "PickPlace000": HeterogeneousTaskCfg(
        instruction="put the chess pawn upright on the white tray",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "empty_1",
                "av_white_tray",
                "spatula_04",
                "av_chess_pawn",
                "popcorn_box",
                "av_chess_king",
                "naturalbostonroundbottle_a01",
                "coffee_can",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.65799, 0.19674, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_white_tray": RigidObjectCfg.InitialStateCfg(
                pos=(0.37227, 0.06462, 0.07106), rot=(0.0, 0.0, -0.522537, 0.852617)
            ),
            "spatula_04": RigidObjectCfg.InitialStateCfg(
                pos=(0.52317, 0.12638, 0.07773), rot=(0.0, 0.0, 0.861227, 0.50822)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.35978, -0.11828, 0.071), rot=(0.0, 0.0, 0.692404, 0.72151)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.55136, -0.0821, 0.11349), rot=(0.0, 0.0, 0.360769, 0.932655)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.30816, -0.04352, 0.071), rot=(0.0, 0.0, -0.596253, 0.802796)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.35389, -0.20447, 0.071), rot=(0.0, 0.0, -0.977237, 0.212152)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.50815, -0.18363, 0.14107), rot=(0.0, 0.0, 0.278894, 0.960322)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_chess_pawn")],
                    "reference": SceneEntityCfg("av_white_tray"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace001": HeterogeneousTaskCfg(
        instruction="put a chess piece in the red bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "bowl_ycb",
                "av_coaster_hexagon",
                "empty_3",
                "av_chess_pawn",
                "milkjug_a01",
                "av_chess_king",
                "av_chess_rook",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.44445, -0.06131, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.42061, 0.13864, -0.00352), rot=(0.0, 0.0, 0.763881, 0.645357)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.33901, -0.19627, -0.031), rot=(0.0, 0.0, -0.919043, 0.394156)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.55422, 0.06806, -0.031), rot=(0.0, 0.0, -0.213145, 0.977021)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29936, 0.04439, -0.031), rot=(0.0, 0.0, 0.788763, 0.614697)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.42411, -0.04901, -0.031), rot=(0.0, 0.0, -0.32347, 0.946238)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.54345, -0.11546, -0.031), rot=(0.0, 0.0, -0.606052, 0.795425)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("av_chess_rook"),
                        SceneEntityCfg("av_chess_pawn"),
                        SceneEntityCfg("av_chess_king"),
                    ],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace002": HeterogeneousTaskCfg(
        instruction="put the mayonnaise bottle in the wooden bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "wooden_bowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "empty_5",
                "storage_box",
                "mayonnaise_bottle",
                "empty_8",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.437, 0.21564, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.4415, -0.04025, 0.10892), rot=(0.0, 0.0, 0.990727, 0.135868)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.32239, 0.2552, 0.10887), rot=(0.0, 0.0, -0.710787, 0.703408)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.43489, 0.19939, 0.18295), rot=(0.0, 0.0, -0.725997, 0.687698)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("mayonnaise_bottle")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace003": HeterogeneousTaskCfg(
        instruction="place a can into the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "canned_peaches",
                "av_mustard_jar",
                "tomato_sauce_can",
                "pitted_cherries",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.56543, 0.02839, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.45577, -0.01553, 0.07696), rot=(0.0, 0.0, -0.978571, 0.20591)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.47724, 0.23338, 0.077), rot=(0.0, 0.0, 0.999972, 0.00751)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.41202, -0.2461, 0.10589), rot=(0.0, 0.0, 0.101339, 0.994852)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.39327, 0.25957, 0.07708), rot=(0.0, 0.0, -0.993944, 0.109888)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.29392, -0.04795, 0.11842), rot=(0.0, 0.0, -0.797466, 0.603364)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.36515, 0.158, 0.10611), rot=(0.0, 0.0, -0.928736, 0.370742)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("pitted_cherries"),
                        SceneEntityCfg("tomato_sauce_can"),
                        SceneEntityCfg("canned_peaches"),
                    ],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace004": HeterogeneousTaskCfg(
        instruction="put a box in the black bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "blackandbrassbowl_small",
                "empty_2",
                "spatula_07",
                "parmesan_cheese_canister",
                "raisin_box",
                "av_chess_bishop",
                "empty_7",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.59233, -0.40532, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.48008, -0.14738, 0.07096), rot=(0.0, 0.0, -0.711185, 0.703005)
            ),
            "spatula_07": RigidObjectCfg.InitialStateCfg(
                pos=(0.31522, -0.15653, 0.07414), rot=(0.0, 0.0, 0.698943, 0.715178)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.45045, 0.22182, 0.12243), rot=(0.0, 0.0, 0.860687, 0.509135)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.32572, 0.21769, 0.13259), rot=(0.0, 0.0, -0.43192, 0.901912)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.30827, 0.07762, 0.071), rot=(0.0, 0.0, 0.451788, 0.892125)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.48348, 0.03854, 0.15413), rot=(0.0, 0.0, -0.000918, 1.0)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("raisin_box"), SceneEntityCfg("macaroni_and_cheese")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace005": HeterogeneousTaskCfg(
        instruction="pick up the lime and put it in the pan",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "av_pan_d",
                "empty_2",
                "crabbypenholder",
                "empty_4",
                "empty_5",
                "empty_6",
                "dry_erase_marker",
                "lime01",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.50524, 0.60417, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.42515, -0.05703, 0.09026), rot=(0.0, 0.0, 0.99929, 0.037675)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.27894, 0.25622, 0.08997), rot=(0.0, 0.0, -0.035785, 0.99936)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.34612, -0.24289, 0.09944), rot=(0.0, 0.0, 0.573546, 0.819173)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.48219, -0.25128, 0.09022), rot=(0.0, 0.0, 0.488966, 0.872303)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("lime01")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace006": HeterogeneousTaskCfg(
        instruction="place the eraser on top of the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "empty_1",
                "av_book_77",
                "empty_3",
                "ranch_dressing",
                "smartphone",
                "av_chess_king",
                "av_eraser",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.40516, 0.07129, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_77": RigidObjectCfg.InitialStateCfg(
                pos=(0.41784, -0.11617, 0.198), rot=(0.0, 0.0, -0.087316, 0.996181)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.5155, 0.0789, 0.27172), rot=(0.0, 0.0, 0.892025, 0.451986)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.36231, 0.14669, 0.20365), rot=(0.0, 0.0, 0.391908, 0.920004)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.28133, 0.06535, 0.198), rot=(0.0, 0.0, 0.696726, 0.717337)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.38553, 0.25128, 0.20233), rot=(0.0, 0.0, 0.723138, 0.690704)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.49945, 0.20977, 0.27216), rot=(0.0, 0.0, -0.925907, 0.377753)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_eraser")],
                    "reference": SceneEntityCfg("av_book_77"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace007": HeterogeneousTaskCfg(
        instruction="place the spoon into the pan",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "av_pan_d",
                "empty_2",
                "empty_3",
                "empty_4",
                "canned_peaches",
                "spoon",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41098, 0.25704, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.42561, -0.12306, 0.06926), rot=(0.0, 0.0, -0.999842, 0.017766)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.30107, 0.25065, 0.09789), rot=(0.0, 0.0, -0.560793, 0.827956)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.41787, 0.22705, 0.07778), rot=(0.0, 0.0, -0.925828, 0.377945)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace008": HeterogeneousTaskCfg(
        instruction="place both the can of peaches and the can of green beans into the decorative bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "av_decorativebowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "canned_peaches",
                "measuring_cups",
                "green_beans_can",
                "empty_8",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.74801, -0.04179, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.39533, 0.0194, 0.07104), rot=(0.0, 0.0, 0.692445, 0.721471)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.35459, 0.22595, 0.09989), rot=(0.0, 0.0, 0.050577, 0.99872)
            ),
            "measuring_cups": RigidObjectCfg.InitialStateCfg(
                pos=(0.34335, -0.22576, 0.08995), rot=(0.0, 0.0, -0.999622, 0.027498)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.53765, 0.22625, 0.09979), rot=(0.0, 0.0, 0.844433, 0.535661)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_peaches"), SceneEntityCfg("green_beans_can")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace009": HeterogeneousTaskCfg(
        instruction="place both the Rubik's cube and the barbecue sauce bottle into the metal bowl",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "fstylejug_a02",
                "canned_tuna",
                "av_mustard_jar",
                "rubiks_cube",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.43713, -0.04354, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43025, 0.16591, 0.00725), rot=(0.0, 0.0, -0.658959, 0.752179)
            ),
            "fstylejug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.42317, -0.20978, 0.007), rot=(0.0, 0.0, 0.367219, 0.930135)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.56363, -0.0546, 0.02329), rot=(0.0, 0.0, -0.121142, 0.992635)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.30373, -0.01238, 0.00708), rot=(0.0, 0.0, 0.802621, 0.59649)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.38328, -0.01465, 0.03839), rot=(0.0, 0.0, -0.69359, 0.72037)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.46794, -0.09534, 0.00706), rot=(0.0, 0.0, -0.773361, 0.633966)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("rubiks_cube"), SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace010": HeterogeneousTaskCfg(
        instruction="put the eraser on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "empty_1",
                "av_book_66",
                "crabbypenholder",
                "empty_4",
                "av_p_glassware_tall",
                "spoon_2",
                "av_eraser",
                "lime01",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(
                pos=(0.91518, -0.01244, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_book_66": RigidObjectCfg.InitialStateCfg(
                pos=(0.41797, -0.13788, -0.058), rot=(0.0, 0.0, 0.87137, 0.490626)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.43496, 0.20795, -0.05803), rot=(0.0, 0.0, -0.984093, 0.177655)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.53092, 0.0965, -0.058), rot=(0.0, 0.0, -0.306477, 0.951878)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.52741, 0.23147, -0.05261), rot=(0.0, 0.0, -0.719327, 0.694671)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.40902, 0.15518, -0.05367), rot=(0.0, 0.0, -0.236965, 0.971518)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.33294, 0.16537, -0.05778), rot=(0.0, 0.0, 0.723683, 0.690133)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_eraser")],
                    "reference": SceneEntityCfg("av_book_66"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace011": HeterogeneousTaskCfg(
        instruction="set the avocado on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "empty_1",
                "av_book_66",
                "empty_3",
                "av_chess_queen",
                "empty_5",
                "av_chess_bishop",
                "frozen_vegetable_block",
                "avocado01",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.58698, -0.1273, -0.69705), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_66": RigidObjectCfg.InitialStateCfg(
                pos=(0.40184, 0.09676, 0.069), rot=(0.0, 0.0, 0.030231, 0.999543)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.5559, -0.08634, 0.069), rot=(0.0, 0.0, 0.141186, 0.989983)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.40241, -0.13814, 0.069), rot=(0.0, 0.0, -0.831029, 0.556228)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.34248, -0.17731, 0.069), rot=(0.0, 0.0, 0.383105, 0.923705)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.51141, -0.16177, 0.06883), rot=(0.0, 0.0, -0.445839, 0.895113)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("avocado01")],
                    "reference": SceneEntityCfg("av_book_66"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace012": HeterogeneousTaskCfg(
        instruction="put the raisin box and the ranch dressing bottle in the metal bin",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_b04",
                "empty_2",
                "empty_3",
                "ranch_dressing",
                "raisin_box",
                "empty_6",
                "green_beans_can",
                "empty_8",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41578, -0.29277, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.39955, 0.07049, 0.069), rot=(0.0, 0.0, -0.723459, 0.690368)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.29764, -0.22847, 0.14272), rot=(0.0, 0.0, -0.652057, 0.75817)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.42013, -0.20091, 0.13059), rot=(0.0, 0.0, -0.030306, 0.999541)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.51781, -0.16174, 0.09779), rot=(0.0, 0.0, 0.305417, 0.952219)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("raisin_box"), SceneEntityCfg("ranch_dressing")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace013": HeterogeneousTaskCfg(
        instruction="stack the green block on the wooden box",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "green_block",
                "raisin_box",
                "storage_box",
                "naturalbostonroundbottle_a03",
                "pitted_cherries",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.46663, 0.03152, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.38368, 0.10435, -0.03104), rot=(0.0, 0.0, 0.639952, 0.768415)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.53839, 0.0491, -0.0075), rot=(0.0, 0.0, 0.316999, 0.948426)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.292, -0.25207, 0.03059), rot=(0.0, 0.0, 0.241451, 0.970413)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.50576, -0.19995, -0.03113), rot=(0.0, 0.0, -0.743873, 0.668321)
            ),
            "naturalbostonroundbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.36989, -0.13409, -0.031), rot=(0.0, 0.0, -0.195709, 0.980662)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.49519, -0.1053, -0.00189), rot=(0.0, 0.0, 0.782678, 0.622427)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("green_block")],
                    "reference": SceneEntityCfg("storage_box"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace014": HeterogeneousTaskCfg(
        instruction="place the granola bar box on top of the clay plate",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "empty_1",
                "clay_plates",
                "empty_3",
                "orange_02",
                "granola_bars",
                "storage_box",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.42484, 0.0893, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "clay_plates": RigidObjectCfg.InitialStateCfg(
                pos=(0.41865, 0.00034, -0.03117), rot=(0.0, 0.0, -0.998922, 0.046426)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.40891, -0.24841, -0.03094), rot=(0.0, 0.0, -0.982781, 0.184774)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.37777, 0.25831, 0.031), rot=(0.0, 0.0, -0.639432, 0.768847)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.28994, -0.25823, -0.03113), rot=(0.0, 0.0, 0.682924, 0.730489)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.52347, -0.23494, -0.00626), rot=(0.0, 0.0, 0.999138, 0.041502)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("granola_bars")],
                    "reference": SceneEntityCfg("clay_plates"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace015": HeterogeneousTaskCfg(
        instruction="pick up the strawberry and put it in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_a05",
                "empty_2",
                "empty_3",
                "av_stawberry",
                "raisin_box",
                "empty_6",
                "empty_7",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.40937, 0.25939, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(pos=(0.3777, 0.0974, 0.069), rot=(0.0, 0.0, -0.649039, 0.760755)),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.44429, -0.23772, 0.06881), rot=(0.0, 0.0, 0.061849, 0.998086)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.32398, -0.2205, 0.13059), rot=(0.0, 0.0, 0.895523, 0.445016)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.52227, -0.20542, 0.14903), rot=(0.0, 0.0, 0.778077, 0.628169)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_stawberry")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace016": HeterogeneousTaskCfg(
        instruction="put the chess queen and the barbecue sauce bottle in the red bowl",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "bowl_ycb",
                "empty_2",
                "spatula_03",
                "av_chess_queen",
                "pumpkinlarge",
                "av_mustard_jar",
                "empty_7",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.46937, 0.11236, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.34284, -0.03269, 0.03448), rot=(0.0, 0.0, -0.72479, 0.68897)
            ),
            "spatula_03": RigidObjectCfg.InitialStateCfg(
                pos=(0.44927, -0.17767, 0.01373), rot=(0.0, 0.0, -0.943391, 0.331682)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.46801, -0.00274, 0.007), rot=(0.0, 0.0, -1.0, 0.000553)
            ),
            "pumpkinlarge": RigidObjectCfg.InitialStateCfg(
                pos=(0.50176, 0.19843, 0.00676), rot=(0.0, 0.0, -0.312783, 0.949825)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.34868, 0.16226, 0.00708), rot=(0.0, 0.0, -0.867665, 0.49715)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.45287, 0.09992, 0.08116), rot=(0.0, 0.0, 0.146688, 0.989183)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen"), SceneEntityCfg("bbq_sauce_bottle")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace017": HeterogeneousTaskCfg(
        instruction="put the chocolate pudding mix on the plate",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "empty_1",
                "plate_small",
                "crabbypenholder",
                "fstylejug_a01",
                "canned_tuna",
                "spoon",
                "chocolate_pudding_mix",
                "red_block",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.48729, -0.09306, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "plate_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.40679, -0.00553, -0.03109), rot=(0.0, 0.0, -0.015782, 0.999875)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.44659, 0.22972, -0.03103), rot=(0.0, 0.0, -0.418641, 0.908152)
            ),
            "fstylejug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.32602, 0.21646, -0.031), rot=(0.0, 0.0, 0.685007, 0.728537)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.41552, -0.25676, -0.01471), rot=(0.0, 0.0, 0.98486, 0.17335)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.30659, -0.23341, -0.02222), rot=(0.0, 0.0, -0.594605, 0.804018)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.52018, -0.2255, -0.00626), rot=(0.0, 0.0, -0.719002, 0.695008)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.51833, 0.22932, -0.0075), rot=(0.0, 0.0, 0.329367, 0.944202)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "reference": SceneEntityCfg("plate_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace018": HeterogeneousTaskCfg(
        instruction="put the popcorn box on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "empty_1",
                "av_book_99",
                "empty_3",
                "empty_4",
                "popcorn_box",
                "av_p_glassware_short",
                "mug_ycb",
                "brick",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.62377, 0.21294, -0.6376), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_book_99": RigidObjectCfg.InitialStateCfg(
                pos=(0.4389, -0.03071, 0.156), rot=(0.0, 0.0, 0.677434, 0.735584)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.36083, 0.16813, 0.19849), rot=(0.0, 0.0, -0.337355, 0.941377)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.5123, 0.2383, 0.156), rot=(0.0, 0.0, -0.930616, 0.365996)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.3818, -0.2176, 0.19669), rot=(0.0, 0.0, 0.221225, 0.975223)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.50889, 0.12645, 0.18157), rot=(0.0, 0.0, -0.903931, 0.427678)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("popcorn_box")],
                    "reference": SceneEntityCfg("av_book_99"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace019": HeterogeneousTaskCfg(
        instruction="place the measuring spoons into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_b04",
                "empty_2",
                "empty_3",
                "spam_can",
                "tomato_soup_can",
                "empty_6",
                "measuring_spoon",
                "empty_8",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41179, 0.35924, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(pos=(0.4278, -0.08273, 0.069), rot=(0.0, 0.0, 0.78851, 0.615022)),
            "spam_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.50674, 0.21646, 0.11082), rot=(0.0, 0.0, -0.113715, 0.993513)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.29676, -0.21389, 0.12002), rot=(0.0, 0.0, -0.985335, 0.170629)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.34622, 0.19811, 0.08334), rot=(0.0, 0.0, -0.301303, 0.953528)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("measuring_spoon")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace020": HeterogeneousTaskCfg(
        instruction="move the spoon into the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "bowl",
                "empty_2",
                "spatula_11",
                "ranch_dressing",
                "tuna_can",
                "spoon_1",
                "av_eraser",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.51563, 0.06531, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.46316, -0.12458, 0.03648), rot=(0.0, 0.0, -0.949502, 0.313761)
            ),
            "spatula_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.4595, 0.12662, 0.01573), rot=(0.0, 0.0, 0.962194, 0.272364)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.54213, 0.08461, 0.08272), rot=(0.0, 0.0, 0.925838, 0.377922)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.32422, -0.20776, 0.02576), rot=(0.0, 0.0, -0.249528, 0.968368)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.36085, 0.25489, 0.01966), rot=(0.0, 0.0, -0.023162, 0.999732)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.33636, 0.05297, 0.01333), rot=(0.0, 0.0, 0.579019, 0.815314)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.28663, 0.14939, 0.08316), rot=(0.0, 0.0, 0.695096, 0.718917)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_1")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace021": HeterogeneousTaskCfg(
        instruction="pick up the pumpkin and put it in the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "av_salt_bowl",
                "empty_2",
                "spatula_09",
                "empty_4",
                "pumpkinsmall",
                "av_chess_king",
                "rubiks_cube",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.48661, -0.06001, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.41963, -0.1915, -0.03094), rot=(0.0, 0.0, -0.719968, 0.694008)
            ),
            "spatula_09": RigidObjectCfg.InitialStateCfg(
                pos=(0.35502, 0.10313, -0.02427), rot=(0.0, 0.0, 0.733855, 0.679307)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.50906, 0.15147, -0.03126), rot=(0.0, 0.0, -0.981411, 0.191915)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.5648, 0.0235, -0.031), rot=(0.0, 0.0, 0.908613, 0.417639)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.52168, -0.06688, 0.00039), rot=(0.0, 0.0, 0.213532, 0.976936)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("pumpkinsmall")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace022": HeterogeneousTaskCfg(
        instruction="place the eraser on top of the clay plate",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "empty_1",
                "clay_plates",
                "crabbypenholder",
                "whitepackerbottle_a02",
                "milkjug_a02",
                "empty_6",
                "av_eraser",
                "empty_8",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(
                pos=(0.75559, -0.95548, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "clay_plates": RigidObjectCfg.InitialStateCfg(
                pos=(0.44751, 0.08788, 0.10883), rot=(0.0, 0.0, 0.785448, 0.618927)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.30838, -0.19133, 0.10897), rot=(0.0, 0.0, -0.416861, 0.90897)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.39111, -0.17986, 0.109), rot=(0.0, 0.0, -0.311669, 0.950191)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.49861, -0.17209, 0.109), rot=(0.0, 0.0, 0.729794, 0.683667)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.36958, -0.28088, 0.11333), rot=(0.0, 0.0, -0.994964, 0.100233)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_eraser")],
                    "reference": SceneEntityCfg("clay_plates"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace023": HeterogeneousTaskCfg(
        instruction="place the raisin box on top of the clay plate",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "empty_1",
                "clay_plates",
                "empty_3",
                "empty_4",
                "raisin_box",
                "empty_6",
                "naturalbostonroundbottle_a01",
                "pitted_cherries",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.49563, 0.13217, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "clay_plates": RigidObjectCfg.InitialStateCfg(
                pos=(0.40294, 0.12754, 0.00883), rot=(0.0, 0.0, -0.999063, 0.043288)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.38731, -0.14867, 0.07059), rot=(0.0, 0.0, 0.421142, 0.906995)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29039, -0.14086, 0.009), rot=(0.0, 0.0, -0.972369, 0.23345)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.51443, -0.17453, 0.03811), rot=(0.0, 0.0, -0.287273, 0.957849)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("raisin_box")],
                    "reference": SceneEntityCfg("clay_plates"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace024": HeterogeneousTaskCfg(
        instruction="pick up the crab pen holder and put it in the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "blackandbrassbowl_small",
                "empty_2",
                "crabbypenholder",
                "parmesan_cheese_canister",
                "tuna_can",
                "spoon_1",
                "frozen_vegetable_block",
                "pitcher_ycb",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.55623, 0.55802, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.46558, 0.16135, 0.07096), rot=(0.0, 0.0, 0.732819, 0.680423)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.56016, -0.05777, 0.07097), rot=(0.0, 0.0, -0.807447, 0.58994)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.31873, 0.03434, 0.12243), rot=(0.0, 0.0, 0.060928, 0.998142)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.3169, -0.23656, 0.08776), rot=(0.0, 0.0, 0.359867, 0.933004)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.39627, -0.03874, 0.08166), rot=(0.0, 0.0, 0.343886, 0.939011)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.29417, 0.14731, 0.071), rot=(0.0, 0.0, -0.979981, 0.199089)
            ),
            "pitcher_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.4646, -0.20654, 0.19232), rot=(0.0, 0.0, 0.169237, 0.985575)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("crabbypenholder")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace025": HeterogeneousTaskCfg(
        instruction="put the kiwi in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_maple",
                "bin_a03",
                "av_coaster_hexagon",
                "av_kiwi01",
                "ranch_dressing",
                "empty_5",
                "av_chess_bishop",
                "green_beans_can",
                "potato_masher",
            ]
        ),
        init_states={
            "table_maple": RigidObjectCfg.InitialStateCfg(pos=(0.545, 0.10895, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.4679, 0.19601, 0.009), rot=(0.0, 0.0, -0.081679, 0.996659)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.39471, -0.05925, 0.009), rot=(0.0, 0.0, -0.847193, 0.531285)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.34207, 0.05125, 0.00899), rot=(0.0, 0.0, -0.67297, 0.739669)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.31793, 0.18889, 0.08272), rot=(0.0, 0.0, 0.111773, 0.993734)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.52073, -0.2367, 0.009), rot=(0.0, 0.0, 0.882841, 0.469671)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.37481, -0.22181, 0.03779), rot=(0.0, 0.0, 0.396004, 0.918249)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.4856, -0.12782, 0.00942), rot=(0.0, 0.0, 0.966074, 0.258265)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace026": HeterogeneousTaskCfg(
        instruction="put the lime in the salt bowl, then the can of green beans",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "yogurt_cup",
                "granola_bars",
                "wood_block",
                "green_beans_can",
                "lime01",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.52618, 0.12911, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.40583, 0.06706, 0.00906), rot=(0.0, 0.0, -0.351131, 0.936326)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.38587, -0.26012, 0.03584), rot=(0.0, 0.0, -0.647375, 0.762172)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.45647, -0.19782, 0.071), rot=(0.0, 0.0, 0.529816, 0.848113)
            ),
            "wood_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.48495, 0.21366, 0.11194), rot=(0.0, 0.0, -0.876112, 0.482108)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.49922, -0.12964, 0.03779), rot=(0.0, 0.0, -0.925505, 0.378735)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.33595, -0.11467, 0.00922), rot=(0.0, 0.0, 0.542318, 0.840173)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("lime01"), SceneEntityCfg("green_beans_can")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                },
            ),
        },
    ),
    "PickPlace027": HeterogeneousTaskCfg(
        instruction="place both the butter and the crab pen holder into the pan",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "av_pan_d",
                "empty_2",
                "crabbypenholder",
                "butter",
                "popcorn_box",
                "empty_6",
                "empty_7",
                "red_block",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.45119, -0.02125, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.38227, -0.13794, -0.05574), rot=(0.0, 0.0, -0.999854, 0.017085)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.46967, 0.21744, -0.05603), rot=(0.0, 0.0, -0.986742, 0.162298)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(pos=(0.56939, -0.05935, -0.02959), rot=(0.0, 0.0, -0.000255, 1.0)),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.53697, 0.11908, -0.01351), rot=(0.0, 0.0, -0.008526, 0.999964)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.29944, 0.18353, -0.0325), rot=(0.0, 0.0, 0.86787, 0.496791)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("butter"), SceneEntityCfg("crabbypenholder")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace028": HeterogeneousTaskCfg(
        instruction="put the tomato sauce can in the black bowl, then the white bottle",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_sofatable",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "tomato_soup_can",
                "av_plant_succulent_01",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "appleseed_sofatable": RigidObjectCfg.InitialStateCfg(
                pos=(0.44969, -0.10898, -0.69682), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.46616, 0.17601, 0.05796), rot=(0.0, 0.0, 0.705341, 0.708868)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.58314, -0.00294, 0.058), rot=(0.0, 0.0, -0.047979, 0.998848)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.37389, -0.23337, 0.10902), rot=(0.0, 0.0, 0.783133, 0.621854)
            ),
            "av_plant_succulent_01": RigidObjectCfg.InitialStateCfg(
                pos=(0.42426, -0.08551, 0.058), rot=(0.0, 0.0, 0.94488, 0.327417)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.52468, -0.1937, 0.09942), rot=(0.0, 0.0, 0.452321, 0.891855)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("tomato_sauce_can"), SceneEntityCfg("whitepackerbottle_a03")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                },
            ),
        },
    ),
    "PickPlace029": HeterogeneousTaskCfg(
        instruction="move the yellow block into the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "bowl",
                "empty_2",
                "empty_3",
                "fstylejug_a01",
                "mug",
                "av_mustard_jar",
                "yellow_block",
                "potato_masher",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.42634, -0.48953, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.423, -0.09271, 0.13648), rot=(0.0, 0.0, -0.60546, 0.795876)),
            "fstylejug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.33335, -0.23985, 0.109), rot=(0.0, 0.0, 0.689403, 0.724378)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.37714, 0.21857, 0.10918), rot=(0.0, 0.0, -0.155721, 0.987801)),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.34529, 0.02437, 0.10908), rot=(0.0, 0.0, -0.998748, 0.050017)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.36244, 0.12374, 0.1325), rot=(0.0, 0.0, 0.065589, 0.997847)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.48891, 0.10406, 0.10942), rot=(0.0, 0.0, 0.169991, 0.985446)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("yellow_block")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace030": HeterogeneousTaskCfg(
        instruction="put the barbecue sauce bottle in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "bin_a02",
                "empty_2",
                "empty_3",
                "empty_4",
                "scissors",
                "av_mustard_jar",
                "clamp",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.61563, 0.38428, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.49489, -0.17854, 0.077), rot=(0.0, 0.0, 0.744809, 0.667278)
            ),
            "scissors": RigidObjectCfg.InitialStateCfg(
                pos=(0.37297, -0.07844, 0.08486), rot=(0.0, 0.0, 0.999399, 0.034655)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.32557, 0.14823, 0.07708), rot=(0.0, 0.0, 0.412332, 0.911034)
            ),
            "clamp": RigidObjectCfg.InitialStateCfg(
                pos=(0.47448, 0.18516, 0.09658), rot=(0.0, 0.0, -0.991705, 0.128537)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.56127, 0.01518, 0.07706), rot=(0.0, 0.0, 0.656181, 0.754604)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace031": HeterogeneousTaskCfg(
        instruction="move the can of tuna into the salt bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "canned_tuna",
                "measuring_cups_1",
                "mug_ycb",
                "pitted_cherries",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.60837, 0.34283, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.3721, 0.10591, 0.07106), rot=(0.0, 0.0, -0.733547, 0.679639)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.4037, -0.13632, 0.08729), rot=(0.0, 0.0, 0.90985, 0.414938)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.52847, -0.13739, 0.09703), rot=(0.0, 0.0, -0.842476, 0.538734)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.54057, 0.14385, 0.11169), rot=(0.0, 0.0, 0.830011, 0.557747)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.3249, -0.24647, 0.10011), rot=(0.0, 0.0, -0.251688, 0.967808)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_tuna")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace032": HeterogeneousTaskCfg(
        instruction="place the white bottle into the salt bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "mug",
                "av_apple",
                "mayonnaise_bottle",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.44638, -0.02362, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.40279, 0.02017, -0.03094), rot=(0.0, 0.0, 0.998369, 0.057087)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.41835, 0.23826, -0.031), rot=(0.0, 0.0, 0.98238, 0.186894)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.54737, 0.05852, -0.03082), rot=(0.0, 0.0, 0.658822, 0.752299)),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.32211, -0.18943, -0.03004), rot=(0.0, 0.0, -0.04784, 0.998855)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.54014, -0.19518, 0.04295), rot=(0.0, 0.0, -0.990069, 0.140584)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace033": HeterogeneousTaskCfg(
        instruction="pick up the marker and put it in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "bin_b04",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "empty_5",
                "empty_6",
                "dry_erase_marker",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.47467, 0.12226, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_b04": RigidObjectCfg.InitialStateCfg(pos=(0.39415, 0.00336, 0.09), rot=(0.0, 0.0, 0.686186, 0.727426)),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.51074, -0.2399, 0.09), rot=(0.0, 0.0, 0.088248, 0.996098)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.45749, 0.25263, 0.09944), rot=(0.0, 0.0, 0.561361, 0.827571)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.55773, 0.0162, 0.16416), rot=(0.0, 0.0, 0.080598, 0.996747)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("dry_erase_marker")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace034": HeterogeneousTaskCfg(
        instruction="remove the spoon from the metal bin",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_b04",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "empty_5",
                "spoon",
                "rubiks_cube",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41359, 0.25881, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.43709, -0.00792, 0.069), rot=(0.0, 0.0, 0.631171, 0.775643)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.32611, 0.26715, 0.069), rot=(0.0, 0.0, -0.999632, 0.027124)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.43709, -0.00792, 0.09178), rot=(0.0, 0.0, -0.805886, 0.592071)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.34361, -0.27524, 0.10039), rot=(0.0, 0.0, -0.740057, 0.672544)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.30713, 0.15976, 0.14903), rot=(0.0, 0.0, 0.699895, 0.714245)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("spoon")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace035": HeterogeneousTaskCfg(
        instruction="move the frozen waffles into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "bin_a05",
                "empty_2",
                "frozen_waffles",
                "empty_4",
                "empty_5",
                "ketchup_bottle",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.78855, -0.09558, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.45751, -0.10011, 0.071), rot=(0.0, 0.0, -0.822488, 0.568782)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.46308, 0.15109, 0.07098), rot=(0.0, 0.0, -0.649644, 0.760238)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.32933, -0.26181, 0.1453), rot=(0.0, 0.0, 0.689127, 0.724641)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.30866, -0.04399, 0.09574), rot=(0.0, 0.0, 0.25941, 0.965767)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("frozen_waffles")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace036": HeterogeneousTaskCfg(
        instruction="pick the white bottle out of the black bowl and put it down",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "blackandbrassbowl_small",
                "empty_2",
                "spatula_05",
                "whitepackerbottle_a01",
                "av_hippoorganizer",
                "milk_carton",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.60617, 0.02219, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.46981, 0.02805, 0.07096), rot=(0.0, 0.0, 0.428648, 0.903472)
            ),
            "spatula_05": RigidObjectCfg.InitialStateCfg(
                pos=(0.42168, -0.16402, 0.07471), rot=(0.0, 0.0, 0.989055, 0.147548)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.46981, 0.02805, 0.085), rot=(0.0, 0.0, -0.656198, 0.754588)
            ),
            "av_hippoorganizer": RigidObjectCfg.InitialStateCfg(
                pos=(0.37151, 0.17922, 0.071), rot=(0.0, 0.0, 0.492001, 0.870594)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.52716, 0.20737, 0.16618), rot=(0.0, 0.0, -0.819493, 0.573089)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a01")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace037": HeterogeneousTaskCfg(
        instruction="place the chess pawn upright on top of the book",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "empty_1",
                "av_book_51",
                "empty_3",
                "av_chess_pawn",
                "tomato_soup_can",
                "empty_6",
                "av_chess_rook",
                "mustard_bottle",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.50555, 0.12847, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_book_51": RigidObjectCfg.InitialStateCfg(
                pos=(0.40642, 0.02472, 0.009), rot=(0.0, 0.0, -0.058924, 0.998262)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.56535, -0.05444, 0.009), rot=(0.0, 0.0, 0.910559, 0.413379)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.37684, -0.20023, 0.06002), rot=(0.0, 0.0, 0.988827, 0.149065)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.48712, 0.26939, 0.009), rot=(0.0, 0.0, -0.201832, 0.97942)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.48208, -0.21956, 0.08903), rot=(0.0, 0.0, -0.965828, 0.259185)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_chess_pawn")],
                    "reference": SceneEntityCfg("av_book_51"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace038": HeterogeneousTaskCfg(
        instruction="move the ketchup bottle into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "bin_a02",
                "empty_2",
                "spatula_11",
                "yogurt_cup",
                "empty_5",
                "ketchup_bottle",
                "mayonnaise_bottle",
                "potato_masher",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.50623, -0.03323, -0.69705), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.49828, -0.15607, 0.069), rot=(0.0, 0.0, 0.660024, 0.751245)
            ),
            "spatula_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.39025, -0.08034, 0.07573), rot=(0.0, 0.0, -0.214615, 0.976699)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.3959, 0.16606, 0.09584), rot=(0.0, 0.0, -0.775041, 0.631911)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.46475, 0.24225, 0.1433), rot=(0.0, 0.0, -0.598416, 0.801186)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.30036, 0.19435, 0.14295), rot=(0.0, 0.0, 0.340636, 0.940195)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.52779, 0.07934, 0.06942), rot=(0.0, 0.0, 0.986298, 0.164974)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("ketchup_bottle")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace039": HeterogeneousTaskCfg(
        instruction="put the orange can and the tomato sauce can in the metal bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "serving_bowl",
                "av_coaster_hexagon",
                "empty_3",
                "whitepackerbottle_a01",
                "soup_can",
                "av_apple",
                "tomato_sauce_can",
                "soft_scrub",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.7233, 0.56238, -0.6971), rot=(0.0, 0.0, 0.0, 1.0)),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.37682, 0.13263, 0.12925), rot=(0.0, 0.0, -0.485916, 0.874006)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.37666, -0.04829, 0.129), rot=(0.0, 0.0, -0.00439, 0.99999)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.5378, 0.09058, 0.129), rot=(0.0, 0.0, 0.662673, 0.748909)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.46187, -0.21843, 0.12906), rot=(0.0, 0.0, -0.191175, 0.981556)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.31996, -0.23288, 0.12996), rot=(0.0, 0.0, -0.131641, 0.991298)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.52347, -0.07304, 0.17042), rot=(0.0, 0.0, -0.44462, 0.895719)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.51766, 0.20209, 0.25432), rot=(0.0, 0.0, 0.440653, 0.897677)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("soup_can"), SceneEntityCfg("tomato_sauce_can")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace040": HeterogeneousTaskCfg(
        instruction="take the can of mushrooms out of the red bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "bowl_ycb",
                "empty_2",
                "spatula_09",
                "parmesan_cheese_canister",
                "canned_peaches",
                "canned_mushrooms",
                "green_beans_can",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.43616, -0.00949, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.43928, 0.02015, -0.00352), rot=(0.0, 0.0, 0.616361, 0.787463)
            ),
            "spatula_09": RigidObjectCfg.InitialStateCfg(
                pos=(0.44897, -0.16101, -0.02427), rot=(0.0, 0.0, 0.987982, 0.154572)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.40688, -0.25507, 0.02043), rot=(0.0, 0.0, 0.991897, 0.127045)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.40937, 0.24625, -0.00211), rot=(0.0, 0.0, -0.767803, 0.640686)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.43928, 0.02015, -0.00034), rot=(0.0, 0.0, 0.6418, 0.766872)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.51632, 0.18779, -0.00221), rot=(0.0, 0.0, 0.813789, 0.581161)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("canned_mushrooms")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace041": HeterogeneousTaskCfg(
        instruction="put the spoon on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "empty_1",
                "av_book_01",
                "spatula_11",
                "empty_4",
                "empty_5",
                "spoon_1",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.50502, -0.15541, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_book_01": RigidObjectCfg.InitialStateCfg(
                pos=(0.41223, -0.01119, 0.009), rot=(0.0, 0.0, -0.073218, 0.997316)
            ),
            "spatula_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.39172, 0.22359, 0.01573), rot=(0.0, 0.0, 0.722697, 0.691164)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.57512, -0.12953, 0.01966), rot=(0.0, 0.0, -0.76446, 0.644671)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.41219, -0.25365, 0.009), rot=(0.0, 0.0, 0.699508, 0.714625)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("spoon_1")],
                    "reference": SceneEntityCfg("av_book_01"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace042": HeterogeneousTaskCfg(
        instruction="put a block in the metal bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "serving_bowl",
                "empty_2",
                "spatula_17",
                "green_block",
                "av_p_glassware_tall",
                "blue_block",
                "yellow_block",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.51427, -0.19626, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.4622, 0.09896, 0.00725), rot=(0.0, 0.0, 0.28506, 0.95851)
            ),
            "spatula_17": RigidObjectCfg.InitialStateCfg(
                pos=(0.36169, -0.09216, 0.01068), rot=(0.0, 0.0, -0.605446, 0.795886)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.30387, -0.18282, 0.0305), rot=(0.0, 0.0, -0.06623, 0.997804)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.35248, 0.21588, 0.007), rot=(0.0, 0.0, -0.93402, 0.35722)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.52018, -0.09974, 0.0305), rot=(0.0, 0.0, 0.741176, 0.67131)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.56683, -0.01712, 0.0305), rot=(0.0, 0.0, 0.959696, 0.281042)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("yellow_block"),
                        SceneEntityCfg("green_block"),
                        SceneEntityCfg("blue_block"),
                    ],
                    "container": SceneEntityCfg("serving_bowl"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace043": HeterogeneousTaskCfg(
        instruction="move the chess queen into the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "bowl",
                "empty_2",
                "spatula_01",
                "av_chess_queen",
                "empty_5",
                "storage_box",
                "av_woodenchsmall",
                "coffee_can",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.41982, 0.01126, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.36644, 0.0977, -0.02852), rot=(0.0, 0.0, 0.711457, 0.70273)),
            "spatula_01": RigidObjectCfg.InitialStateCfg(
                pos=(0.54053, -0.00898, -0.04925), rot=(0.0, 0.0, 0.73228, 0.681004)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.41318, 0.2208, -0.056), rot=(0.0, 0.0, 0.111914, 0.993718)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.52004, 0.21129, -0.05613), rot=(0.0, 0.0, 0.135865, 0.990727)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.30443, -0.04878, -0.056), rot=(0.0, 0.0, -0.994142, 0.108078)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.41881, -0.13161, 0.01407), rot=(0.0, 0.0, 0.980232, 0.197854)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace044": HeterogeneousTaskCfg(
        instruction="put the chocolate pudding mix on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "empty_1",
                "av_book_77",
                "empty_3",
                "parmesan_cheese_canister",
                "av_hippoorganizer",
                "av_chess_bishop",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.6041, 0.14157, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_77": RigidObjectCfg.InitialStateCfg(
                pos=(0.4577, 0.05045, 0.071), rot=(0.0, 0.0, 0.05728, 0.998358)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.3408, 0.24909, 0.12243), rot=(0.0, 0.0, 0.933439, 0.358736)
            ),
            "av_hippoorganizer": RigidObjectCfg.InitialStateCfg(
                pos=(0.3698, -0.17003, 0.071), rot=(0.0, 0.0, -0.808241, 0.588851)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.43106, 0.22868, 0.071), rot=(0.0, 0.0, -0.983702, 0.179804)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.55096, -0.19654, 0.09574), rot=(0.0, 0.0, 0.997268, 0.073862)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "reference": SceneEntityCfg("av_book_77"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace045": HeterogeneousTaskCfg(
        instruction="put the can of peaches in the decorative bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "av_decorativebowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "canned_peaches",
                "av_mustard_jar",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.4865, -0.34945, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.3919, 0.05954, 0.09004), rot=(0.0, 0.0, -0.99979, 0.020495)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.37408, -0.19422, 0.11889), rot=(0.0, 0.0, 0.975693, 0.219144)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.51634, -0.24945, 0.09008), rot=(0.0, 0.0, 0.681147, 0.732147)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace046": HeterogeneousTaskCfg(
        instruction="put the avocado in the metal bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "serving_bowl",
                "empty_2",
                "spatula_12",
                "cream_cheese",
                "soup_can",
                "ketchup_bottle",
                "rubiks_cube",
                "avocado01",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.49508, -0.4321, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.38737, 0.0481, 0.09025), rot=(0.0, 0.0, -0.947487, 0.319793)
            ),
            "spatula_12": RigidObjectCfg.InitialStateCfg(
                pos=(0.53886, 0.1049, 0.09367), rot=(0.0, 0.0, -0.619882, 0.784695)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.40579, -0.18669, 0.1166), rot=(0.0, 0.0, 0.696952, 0.717118)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.34515, 0.18268, 0.09006), rot=(0.0, 0.0, 0.284119, 0.958789)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.52035, -0.14553, 0.1643), rot=(0.0, 0.0, 0.739073, 0.673625)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.28686, -0.26037, 0.12139), rot=(0.0, 0.0, -0.826627, 0.56275)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4571, -0.0699, 0.08983), rot=(0.0, 0.0, 0.237954, 0.971277)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("avocado01")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace047": HeterogeneousTaskCfg(
        instruction="place the brick on the can of peaches",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "empty_1",
                "empty_2",
                "empty_3",
                "orange_02",
                "canned_peaches",
                "blue_block",
                "coffee_pot",
                "brick",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.29891, 0.24393, -0.6376), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.47493, -0.05949, 0.15606), rot=(0.0, 0.0, -0.483402, 0.875399)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.32925, 3e-05, 0.18489), rot=(0.0, 0.0, 0.121299, 0.992616)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.37629, -0.09122, 0.1795), rot=(0.0, 0.0, 0.57711, 0.816666)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.52282, 0.0718, 0.15596), rot=(0.0, 0.0, -0.393449, 0.919347)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.33058, 0.16667, 0.18157), rot=(0.0, 0.0, -0.326888, 0.945063)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("brick")],
                    "reference": SceneEntityCfg("canned_peaches"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace048": HeterogeneousTaskCfg(
        instruction="pick up the milk carton and put it in the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "av_salt_bowl",
                "empty_2",
                "ceramic_mug",
                "spaghetti",
                "pineapple_slices_can",
                "milk_carton",
                "mayonnaise_bottle",
                "empty_8",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(1.38683, -0.03695, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.49532, -0.00941, 0.07706), rot=(0.0, 0.0, -0.086761, 0.996229)
            ),
            "ceramic_mug": RigidObjectCfg.InitialStateCfg(
                pos=(0.32905, -0.0124, 0.07698), rot=(0.0, 0.0, 0.975591, 0.219597)
            ),
            "spaghetti": RigidObjectCfg.InitialStateCfg(
                pos=(0.42878, -0.15262, 0.10192), rot=(0.0, 0.0, 0.686229, 0.727386)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.37336, -0.23012, 0.10581), rot=(0.0, 0.0, 0.348845, 0.93718)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.40849, 0.21023, 0.17218), rot=(0.0, 0.0, 0.751092, 0.660198)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.55606, 0.17121, 0.15095), rot=(0.0, 0.0, -0.524599, 0.85135)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("milk_carton")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace049": HeterogeneousTaskCfg(
        instruction="put the spam can in the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "av_salt_bowl",
                "empty_2",
                "av_kiwi01",
                "spam_can",
                "granola_bars",
                "milk_carton",
                "naturalbostonroundbottle_a03",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(
                pos=(0.72545, 0.55678, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.42206, 0.12062, 0.10906), rot=(0.0, 0.0, -0.245395, 0.969423)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.27898, 0.22811, 0.10899), rot=(0.0, 0.0, 0.731988, 0.681317)
            ),
            "spam_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.50532, -0.1864, 0.15082), rot=(0.0, 0.0, -0.986003, 0.16673)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.27914, 0.08804, 0.171), rot=(0.0, 0.0, -0.056225, 0.998418)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.56869, 0.05648, 0.20418), rot=(0.0, 0.0, 0.783495, 0.621399)
            ),
            "naturalbostonroundbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.40383, -0.07814, 0.109), rot=(0.0, 0.0, 0.994866, 0.101203)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.37703, -0.22595, 0.19213), rot=(0.0, 0.0, 0.894742, 0.446584)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spam_can")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace050": HeterogeneousTaskCfg(
        instruction="put the spoon and the green block in the white bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "green_block",
                "canned_peaches",
                "spoon_2",
                "coffee_pot",
                "empty_8",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.49651, -0.15643, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.44455, 0.07729, 0.0088), rot=(0.0, 0.0, -0.25256, 0.967581)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.3007, -0.0988, 0.0325), rot=(0.0, 0.0, -0.120699, 0.992689)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.2933, 0.17803, 0.03789), rot=(0.0, 0.0, 0.993811, 0.111088)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.50431, -0.15423, 0.01439), rot=(0.0, 0.0, 0.174321, 0.984689)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.34586, -0.22445, 0.00896), rot=(0.0, 0.0, -0.971007, 0.239051)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_2"), SceneEntityCfg("green_block")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace051": HeterogeneousTaskCfg(
        instruction="put the butter on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "empty_1",
                "av_book_21",
                "empty_3",
                "butter",
                "empty_5",
                "blue_block",
                "av_woodenchmedium",
                "empty_8",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.55714, -0.06288, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_book_21": RigidObjectCfg.InitialStateCfg(
                pos=(0.43211, 0.08839, 0.009), rot=(0.0, 0.0, -0.101104, 0.994876)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.53885, -0.14717, 0.03541), rot=(0.0, 0.0, 0.999994, 0.003401)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41317, -0.26539, 0.0325), rot=(0.0, 0.0, -0.536293, 0.844032)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.31832, -0.16426, 0.00901), rot=(0.0, 0.0, -0.519475, 0.854486)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("butter")],
                    "reference": SceneEntityCfg("av_book_21"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace052": HeterogeneousTaskCfg(
        instruction="put a bottle in the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "yogurt_cup",
                "granola_bars",
                "ketchup_bottle",
                "naturalbostonroundbottle_a02",
                "red_block",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41264, 0.11423, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.40256, -0.21126, 0.09648), rot=(0.0, 0.0, 0.73855, 0.674199)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.50107, -0.03951, 0.09584), rot=(0.0, 0.0, 0.986204, 0.165533)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.37354, 0.00216, 0.131), rot=(0.0, 0.0, 0.483635, 0.87527)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.40197, 0.08475, 0.1433), rot=(0.0, 0.0, 0.629053, 0.777362)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.32391, -0.07399, 0.069), rot=(0.0, 0.0, -0.600896, 0.799327)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.50082, 0.21034, 0.0925), rot=(0.0, 0.0, -0.915277, 0.402825)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("ketchup_bottle"), SceneEntityCfg("naturalbostonroundbottle_a02")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace053": HeterogeneousTaskCfg(
        instruction="pick up the tomato sauce can and put it in the pan",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "av_pan_d",
                "empty_2",
                "empty_3",
                "empty_4",
                "empty_5",
                "empty_6",
                "tomato_sauce_can",
                "av_tincan",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.55572, 0.1348, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.42569, -0.15682, 0.00926), rot=(0.0, 0.0, 0.999735, 0.023024)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.31571, 0.19613, 0.05042), rot=(0.0, 0.0, -0.590436, 0.807085)
            ),
            "av_tincan": RigidObjectCfg.InitialStateCfg(
                pos=(0.50675, 0.21918, 0.009), rot=(0.0, 0.0, -0.476111, 0.879385)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("tomato_sauce_can")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace054": HeterogeneousTaskCfg(
        instruction="move the frozen waffles into the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "blackandbrassbowl_small",
                "av_coaster_hexagon",
                "frozen_waffles",
                "empty_4",
                "raisin_box",
                "peas_and_carrots",
                "mug_ycb",
                "pitted_cherries",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(
                pos=(1.38624, -0.19455, -0.6971), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.39396, -0.04695, 0.12896), rot=(0.0, 0.0, -0.970496, 0.241116)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.52783, 0.09948, 0.129), rot=(0.0, 0.0, 0.688808, 0.724944)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.36069, 0.18449, 0.12898), rot=(0.0, 0.0, -0.957949, 0.28694)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.53089, -0.12755, 0.19059), rot=(0.0, 0.0, 0.9154, 0.402545)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.29799, 0.10965, 0.15826), rot=(0.0, 0.0, -0.451156, 0.892445)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.43048, -0.23792, 0.16969), rot=(0.0, 0.0, -0.022019, 0.999758)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.53356, 0.21955, 0.15811), rot=(0.0, 0.0, 0.155157, 0.98789)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("frozen_waffles")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace055": HeterogeneousTaskCfg(
        instruction="place the crab pen holder into the pan",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "av_pan_d",
                "empty_2",
                "crabbypenholder",
                "whitepackerbottle_a02",
                "raisin_box",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.51374, 0.03644, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.40934, -0.09374, 0.00726), rot=(0.0, 0.0, -0.999805, 0.019725)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.41527, 0.27318, 0.00697), rot=(0.0, 0.0, -0.980806, 0.194985)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.50748, -0.26513, 0.007), rot=(0.0, 0.0, 0.198312, 0.980139)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.31078, 0.22092, 0.06859), rot=(0.0, 0.0, -0.640976, 0.767561)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("crabbypenholder")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace056": HeterogeneousTaskCfg(
        instruction="place the orange can upright on the cardboard box",
        episode_length_s=80.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "empty_1",
                "empty_2",
                "spatula_08",
                "cubebox_a02",
                "soup_can",
                "av_knife",
                "yellow_block",
                "potato_masher",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.43567, 0.18642, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "spatula_08": RigidObjectCfg.InitialStateCfg(
                pos=(0.32163, -0.08157, 0.11573), rot=(0.0, 0.0, 0.809001, 0.587807)
            ),
            "cubebox_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.48359, -0.07519, 0.109), rot=(0.0, 0.0, 0.724073, 0.689724)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.41769, 0.05744, 0.10906), rot=(0.0, 0.0, 0.004406, 0.99999)
            ),
            "av_knife": RigidObjectCfg.InitialStateCfg(
                pos=(0.42728, 0.22333, 0.11006), rot=(0.0, 0.0, 0.456251, 0.889851)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.51746, -0.2326, 0.1325), rot=(0.0, 0.0, 0.067256, 0.997736)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.53613, 0.10835, 0.10942), rot=(0.0, 0.0, 0.813306, 0.581836)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("soup_can")],
                    "reference": SceneEntityCfg("cubebox_a02"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace057": HeterogeneousTaskCfg(
        instruction="set the blue block on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "empty_1",
                "av_book_21",
                "empty_3",
                "parmesan_cheese_canister",
                "tuna_can",
                "blue_block",
                "measuring_spoon",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.53574, -0.03183, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_21": RigidObjectCfg.InitialStateCfg(
                pos=(0.45596, -0.02376, 0.077), rot=(0.0, 0.0, 0.999732, 0.023161)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.39741, -0.23214, 0.12843), rot=(0.0, 0.0, 0.800756, 0.598991)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.43453, 0.20701, 0.09376), rot=(0.0, 0.0, 0.400373, 0.916352)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.3355, 0.25122, 0.1005), rot=(0.0, 0.0, -0.944563, 0.328332)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.28629, -0.17252, 0.09134), rot=(0.0, 0.0, 0.712315, 0.70186)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.52041, -0.25887, 0.15703), rot=(0.0, 0.0, 0.581977, 0.813205)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("blue_block")],
                    "reference": SceneEntityCfg("av_book_21"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace058": HeterogeneousTaskCfg(
        instruction="set the raisin box on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "empty_1",
                "av_book_11",
                "empty_3",
                "whitepackerbottle_a04",
                "raisin_box",
                "storage_box",
                "empty_7",
                "red_block",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.97879, 0.14361, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_book_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.39814, -0.03795, 0.071), rot=(0.0, 0.0, 0.739436, 0.673227)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.48718, -0.23552, 0.071), rot=(0.0, 0.0, 0.999984, 0.005689)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.40392, 0.22039, 0.13259), rot=(0.0, 0.0, -0.788014, 0.615658)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.47756, 0.14948, 0.07087), rot=(0.0, 0.0, -0.236423, 0.97165)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.32272, 0.11665, 0.0945), rot=(0.0, 0.0, -0.468191, 0.883627)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("raisin_box")],
                    "reference": SceneEntityCfg("av_book_11"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace059": HeterogeneousTaskCfg(
        instruction="move the orange can into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "bin_a03",
                "empty_2",
                "empty_3",
                "empty_4",
                "soup_can",
                "spoon_big",
                "bagel",
                "red_block",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.42964, 0.00961, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.44318, -0.12534, -0.001), rot=(0.0, 0.0, -0.972527, 0.232789)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.49982, 0.12276, -0.00094), rot=(0.0, 0.0, -0.911281, 0.411786)
            ),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.40669, 0.01256, -0.00076), rot=(0.0, 0.0, -0.981113, 0.193436)
            ),
            "bagel": RigidObjectCfg.InitialStateCfg(pos=(0.32272, 0.1895, 0.01455), rot=(0.0, 0.0, 0.999851, 0.017238)),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.4327, 0.1896, 0.0225), rot=(0.0, 0.0, 0.64682, 0.762643)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("soup_can")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace060": HeterogeneousTaskCfg(
        instruction="first put the kiwi in the red bowl, then put the yellow block in it",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "bowl_ycb",
                "empty_2",
                "av_kiwi01",
                "av_chess_queen",
                "fork_big",
                "spoon_1",
                "yellow_block",
                "avocado01",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.40461, -0.08349, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.44304, 0.06847, 0.22548), rot=(0.0, 0.0, -0.516613, 0.856219)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29421, 0.19026, 0.19799), rot=(0.0, 0.0, -0.969501, 0.245089)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.32766, -0.14114, 0.198), rot=(0.0, 0.0, 0.865782, 0.500421)
            ),
            "fork_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.40842, -0.11511, 0.19835), rot=(0.0, 0.0, 0.960565, 0.278055)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.37922, 0.24162, 0.20866), rot=(0.0, 0.0, 0.950143, 0.311815)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.32847, 0.00534, 0.2215), rot=(0.0, 0.0, -0.833348, 0.552749)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29602, -0.23763, 0.19783), rot=(0.0, 0.0, -0.133122, 0.9911)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01"), SceneEntityCfg("yellow_block")],
                    "container": SceneEntityCfg("bowl_ycb"),
                },
            ),
        },
    ),
    "PickPlace061": HeterogeneousTaskCfg(
        instruction="put the chocolate pudding mix on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "empty_1",
                "av_book_51",
                "empty_3",
                "empty_4",
                "empty_5",
                "empty_6",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41323, 0.07372, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_51": RigidObjectCfg.InitialStateCfg(
                pos=(0.39587, -0.06054, 0.069), rot=(0.0, 0.0, 0.998974, 0.045279)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.45666, 0.22109, 0.09374), rot=(0.0, 0.0, -0.1583, 0.987391)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "reference": SceneEntityCfg("av_book_51"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace062": HeterogeneousTaskCfg(
        instruction="place the phone on top of the wooden tray",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "empty_1",
                "av_woodentray_medium",
                "empty_3",
                "cream_cheese",
                "smartphone",
                "empty_6",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(
                pos=(1.35651, -0.06325, -0.6971), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_woodentray_medium": RigidObjectCfg.InitialStateCfg(
                pos=(0.4462, -0.04215, 0.129), rot=(0.0, 0.0, 0.712469, 0.701703)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.42437, -0.27634, 0.1556), rot=(0.0, 0.0, -0.671425, 0.741073)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.46094, 0.20472, 0.13465), rot=(0.0, 0.0, 0.536649, 0.843805)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.32199, 0.20229, 0.15374), rot=(0.0, 0.0, 0.999851, 0.017247)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("smartphone")],
                    "reference": SceneEntityCfg("av_woodentray_medium"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace063": HeterogeneousTaskCfg(
        instruction="put the Rubik's cube on top of the can of pineapple",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "empty_1",
                "empty_2",
                "spatula_05",
                "yogurt_cup",
                "pineapple_slices_can",
                "av_mustard_jar",
                "rubiks_cube",
                "empty_8",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(1.33656, 0.03581, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "spatula_05": RigidObjectCfg.InitialStateCfg(
                pos=(0.45306, -0.03139, 0.07471), rot=(0.0, 0.0, 0.980569, 0.196174)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.43198, 0.14247, 0.09784), rot=(0.0, 0.0, 0.989155, 0.146877)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.49048, 0.05821, 0.09981), rot=(0.0, 0.0, 0.926942, 0.375205)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.38142, -0.161, 0.07108), rot=(0.0, 0.0, 0.883734, 0.467989)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.51985, -0.17139, 0.10239), rot=(0.0, 0.0, 0.689771, 0.724028)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("rubiks_cube")],
                    "reference": SceneEntityCfg("pineapple_slices_can"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace064": HeterogeneousTaskCfg(
        instruction="put the wooden box and the raisin box in the black bowl",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "blackandbrassbowl_small",
                "av_coaster_hexagon",
                "empty_3",
                "cream_cheese",
                "raisin_box",
                "storage_box",
                "dry_erase_marker",
                "empty_8",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(pos=(0.91172, 0.09635, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.44637, -0.15692, -0.05804), rot=(0.0, 0.0, -0.977049, 0.213015)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.5067, 0.12702, -0.058), rot=(0.0, 0.0, -0.966413, 0.256994)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.39075, 0.23784, -0.0314), rot=(0.0, 0.0, 0.999452, 0.033105)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.28626, 0.11598, 0.00359), rot=(0.0, 0.0, -0.153186, 0.988197)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.38826, 0.07992, -0.05813), rot=(0.0, 0.0, -0.907839, 0.41932)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.30231, 0.23848, -0.04856), rot=(0.0, 0.0, 0.286167, 0.95818)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("storage_box"), SceneEntityCfg("raisin_box")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace065": HeterogeneousTaskCfg(
        instruction="put the granola bar box in the wooden bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "wooden_bowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "granola_bars",
                "spoon_2",
                "empty_7",
                "avocado01",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.53747, 0.19487, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.40529, 0.10515, 0.00892), rot=(0.0, 0.0, 0.021709, 0.999764)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.36753, -0.16018, 0.071), rot=(0.0, 0.0, 0.984417, 0.175851)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.54321, -0.11739, 0.01439), rot=(0.0, 0.0, -0.461353, 0.887217)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.47971, -0.24589, 0.00883), rot=(0.0, 0.0, -0.592679, 0.805439)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("granola_bars")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace066": HeterogeneousTaskCfg(
        instruction="pick up the spam can and put it in the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "bowl",
                "empty_2",
                "empty_3",
                "spam_can",
                "av_p_glassware_tall",
                "av_chess_bishop",
                "green_beans_can",
                "empty_8",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.69751, -0.14768, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43468, -0.01013, 0.15648), rot=(0.0, 0.0, 0.873654, 0.486547)
            ),
            "spam_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.39519, 0.22249, 0.17082), rot=(0.0, 0.0, 0.818424, 0.574615)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.37853, -0.21045, 0.129), rot=(0.0, 0.0, 0.115995, 0.99325)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.54265, 0.0885, 0.129), rot=(0.0, 0.0, -0.656984, 0.753905)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30013, 0.17836, 0.15779), rot=(0.0, 0.0, 0.693433, 0.720521)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spam_can")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace067": HeterogeneousTaskCfg(
        instruction="set the green block on the plate",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "empty_1",
                "plate_small",
                "empty_3",
                "green_block",
                "empty_5",
                "empty_6",
                "bagel",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.47457, -0.06285, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "plate_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.41326, 0.04965, -0.03109), rot=(0.0, 0.0, -0.717787, 0.696263)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.46174, 0.26223, -0.0075), rot=(0.0, 0.0, 0.952818, 0.303542)
            ),
            "bagel": RigidObjectCfg.InitialStateCfg(
                pos=(0.38285, -0.18792, -0.01545), rot=(0.0, 0.0, -0.123256, 0.992375)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.50622, -0.20637, -0.03094), rot=(0.0, 0.0, 0.945915, 0.324416)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("green_block")],
                    "reference": SceneEntityCfg("plate_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace068": HeterogeneousTaskCfg(
        instruction="put the can of tuna in the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "butter",
                "canned_tuna",
                "av_chess_king",
                "mayonnaise_bottle",
                "potato_masher",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.4863, 0.00661, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.39243, 0.02976, -0.00352), rot=(0.0, 0.0, 0.768903, 0.639366)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.47073, 0.1687, -0.00459), rot=(0.0, 0.0, -0.824652, 0.56564)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.53296, -0.1401, -0.01471), rot=(0.0, 0.0, 0.984659, 0.174489)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.45839, -0.22816, -0.031), rot=(0.0, 0.0, -0.269041, 0.963129)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.30186, -0.2599, 0.04295), rot=(0.0, 0.0, 0.787813, 0.615915)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.31028, 0.18418, -0.03058), rot=(0.0, 0.0, 0.995115, 0.098724)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_tuna")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace069": HeterogeneousTaskCfg(
        instruction="place the orange on top of the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "empty_1",
                "av_book_01",
                "spatula_11",
                "orange_02",
                "soup_can",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.4274, 0.56358, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_book_01": RigidObjectCfg.InitialStateCfg(
                pos=(0.46266, 0.09801, 0.007), rot=(0.0, 0.0, -0.996804, 0.079885)
            ),
            "spatula_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.36722, -0.1668, 0.01373), rot=(0.0, 0.0, -0.945741, 0.32492)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.46963, -0.15569, 0.00706), rot=(0.0, 0.0, -0.92817, 0.372158)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.28453, 0.09276, 0.00706), rot=(0.0, 0.0, -0.691743, 0.722144)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("orange_02")],
                    "reference": SceneEntityCfg("av_book_01"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace070": HeterogeneousTaskCfg(
        instruction="place both the granola bar box and the tomato sauce can into the black bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "empty_4",
                "granola_bars",
                "av_apple",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.48613, 0.52518, -0.69646), rot=(0.0, 0.0, 0.0, 1.0)),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.43179, -0.08324, 0.08994), rot=(0.0, 0.0, 0.620985, 0.783822)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.37507, 0.201, 0.152), rot=(0.0, 0.0, -0.995238, 0.097473)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.54312, 0.13169, 0.09096), rot=(0.0, 0.0, 0.0396, 0.999216)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.46564, 0.22522, 0.13142), rot=(0.0, 0.0, 0.230018, 0.973186)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("granola_bars"), SceneEntityCfg("tomato_sauce_can")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace071": HeterogeneousTaskCfg(
        instruction="put the chess queen and the can of green beans in the black bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "blackandbrassbowl_small",
                "empty_2",
                "spatula_08",
                "av_chess_queen",
                "empty_5",
                "av_knife",
                "green_beans_can",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.44638, -0.12242, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.38334, -0.13273, -0.03104), rot=(0.0, 0.0, -0.024206, 0.999707)
            ),
            "spatula_08": RigidObjectCfg.InitialStateCfg(
                pos=(0.43832, 0.11856, -0.02427), rot=(0.0, 0.0, -0.71949, 0.694503)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.58031, 0.01349, -0.031), rot=(0.0, 0.0, 0.80347, 0.595345)
            ),
            "av_knife": RigidObjectCfg.InitialStateCfg(
                pos=(0.51832, -0.02532, -0.02994), rot=(0.0, 0.0, -0.999929, 0.011936)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.32166, 0.03822, -0.00221), rot=(0.0, 0.0, -0.814614, 0.580003)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.31608, 0.11773, 0.05213), rot=(0.0, 0.0, -0.773563, 0.633719)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen"), SceneEntityCfg("green_beans_can")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace072": HeterogeneousTaskCfg(
        instruction="put the sugar box on the sketchbook",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "empty_1",
                "av_sketchbook",
                "sugar_box",
                "empty_4",
                "milkjug_a01",
                "measuring_cups_1",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.4112, -0.30854, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_sketchbook": RigidObjectCfg.InitialStateCfg(
                pos=(0.39317, -0.01721, 0.20017), rot=(0.0, 0.0, 0.757824, 0.652459)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.36373, 0.21083, 0.28601), rot=(0.0, 0.0, 0.009009, 0.999959)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.51756, 0.205, 0.198), rot=(0.0, 0.0, 0.970233, 0.242173)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.44758, -0.22728, 0.22403), rot=(0.0, 0.0, -0.992456, 0.122601)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("sugar_box")],
                    "reference": SceneEntityCfg("av_sketchbook"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace073": HeterogeneousTaskCfg(
        instruction="put the can of peaches in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "gilbert",
                "bin_b04",
                "empty_2",
                "empty_3",
                "orange_02",
                "canned_peaches",
                "empty_6",
                "av_woodenchmedium",
                "mustard",
            ]
        ),
        init_states={
            "gilbert": RigidObjectCfg.InitialStateCfg(
                pos=(0.73573, 0.56783, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.36519, -0.00413, 0.119), rot=(0.0, 0.0, 0.695233, 0.718784)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.45693, -0.24934, 0.11906), rot=(0.0, 0.0, 0.462851, 0.886436)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.54624, 0.12934, 0.14789), rot=(0.0, 0.0, -0.072138, 0.997395)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.58066, -0.07619, 0.11901), rot=(0.0, 0.0, -0.999088, 0.042709)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.39175, 0.25619, 0.2147), rot=(0.0, 0.0, 0.054936, 0.99849)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace074": HeterogeneousTaskCfg(
        instruction="put the milk carton upright on the book",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "empty_1",
                "av_book_21",
                "empty_3",
                "empty_4",
                "pumpkinlarge",
                "milk_carton",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.40334, -0.01236, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_21": RigidObjectCfg.InitialStateCfg(
                pos=(0.43111, -0.06866, -0.001), rot=(0.0, 0.0, -0.999926, 0.01217)
            ),
            "pumpkinlarge": RigidObjectCfg.InitialStateCfg(
                pos=(0.52183, 0.14729, -0.00124), rot=(0.0, 0.0, -0.713153, 0.701009)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.32108, 0.16106, 0.09418), rot=(0.0, 0.0, 0.434141, 0.900845)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4118, 0.13582, -0.001), rot=(0.0, 0.0, 0.602133, 0.798396)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("milk_carton")],
                    "reference": SceneEntityCfg("av_book_21"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace075": HeterogeneousTaskCfg(
        instruction="move the green block into the salt bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "av_salt_bowl",
                "empty_2",
                "frozen_waffles",
                "green_block",
                "empty_5",
                "measuring_cups",
                "empty_7",
                "mustard_bottle",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.62645, 0.17637, -0.6376), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43453, 0.11341, 0.15606), rot=(0.0, 0.0, -0.997954, 0.063935)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.32358, -0.16071, 0.15598), rot=(0.0, 0.0, -0.908629, 0.417605)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.43989, -0.18695, 0.1795), rot=(0.0, 0.0, -0.243762, 0.969835)
            ),
            "measuring_cups": RigidObjectCfg.InitialStateCfg(
                pos=(0.45568, -0.046, 0.17495), rot=(0.0, 0.0, -0.999887, 0.015029)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.31298, -0.03787, 0.23603), rot=(0.0, 0.0, 0.530751, 0.847528)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("green_block")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace076": HeterogeneousTaskCfg(
        instruction="place the avocado into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "bin_a05",
                "empty_2",
                "empty_3",
                "spaghetti",
                "empty_5",
                "empty_6",
                "av_woodenchmedium",
                "avocado01",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(
                pos=(0.76679, -1.23583, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.44341, -0.07009, 0.109), rot=(0.0, 0.0, 0.628214, 0.778041)
            ),
            "spaghetti": RigidObjectCfg.InitialStateCfg(
                pos=(0.3403, 0.1093, 0.13392), rot=(0.0, 0.0, 0.962857, 0.270011)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.31111, 0.21575, 0.10901), rot=(0.0, 0.0, -0.914089, 0.405513)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.50734, 0.1862, 0.10883), rot=(0.0, 0.0, -0.983866, 0.178907)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("avocado01")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace077": HeterogeneousTaskCfg(
        instruction="move the orange into the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "av_salt_bowl",
                "av_woodentray_medium",
                "empty_3",
                "orange_02",
                "empty_5",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.54512, -0.02786, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.36888, -0.1961, 0.00906), rot=(0.0, 0.0, -0.735623, 0.677391)
            ),
            "av_woodentray_medium": RigidObjectCfg.InitialStateCfg(
                pos=(0.42782, 0.09363, 0.009), rot=(0.0, 0.0, 0.718942, 0.695071)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.5268, -0.1588, 0.00906), rot=(0.0, 0.0, 0.822779, 0.568361)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_02")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace078": HeterogeneousTaskCfg(
        instruction="move the orange juice carton into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "bin_a05",
                "empty_2",
                "empty_3",
                "ranch_dressing",
                "orange_juice_carton",
                "measuring_cups",
                "av_woodenchsmall",
                "empty_8",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.50562, 0.03154, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.46266, -0.05979, 0.009), rot=(0.0, 0.0, 0.545956, 0.837814)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.57973, 0.12517, 0.08272), rot=(0.0, 0.0, 0.033869, 0.999426)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.32849, 0.0686, 0.10524), rot=(0.0, 0.0, -0.990915, 0.134487)
            ),
            "measuring_cups": RigidObjectCfg.InitialStateCfg(
                pos=(0.42374, 0.23806, 0.02795), rot=(0.0, 0.0, -0.126679, 0.991944)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.2934, -0.26066, 0.009), rot=(0.0, 0.0, -0.524294, 0.851537)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_juice_carton")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace079": HeterogeneousTaskCfg(
        instruction="place the white bottle into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_maple",
                "bin_a03",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "tomato_soup_can",
                "spoon_2",
                "mug_ycb",
                "soft_scrub",
            ]
        ),
        init_states={
            "table_maple": RigidObjectCfg.InitialStateCfg(pos=(0.51501, -0.19272, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.39287, -0.11358, 0.009), rot=(0.0, 0.0, -0.21225, 0.977215)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.52058, 0.13768, 0.009), rot=(0.0, 0.0, 0.835898, 0.548886)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.41459, 0.02818, 0.06002), rot=(0.0, 0.0, 0.962757, 0.270369)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.40089, -0.23338, 0.01439), rot=(0.0, 0.0, 0.918542, 0.395324)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.55337, -0.04031, 0.04969), rot=(0.0, 0.0, 0.795128, 0.606441)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.35136, 0.15339, 0.13432), rot=(0.0, 0.0, -0.45076, 0.892645)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace080": HeterogeneousTaskCfg(
        instruction="take the red block out of the metal bin",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "bin_a02",
                "av_book_77",
                "empty_3",
                "ranch_dressing",
                "canned_peaches",
                "empty_6",
                "green_beans_can",
                "red_block",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.51508, 0.18044, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.41102, 0.17982, 0.009), rot=(0.0, 0.0, -0.00478, 0.999989)
            ),
            "av_book_77": RigidObjectCfg.InitialStateCfg(
                pos=(0.3953, -0.08076, 0.009), rot=(0.0, 0.0, 0.124368, 0.992236)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.29238, 0.18581, 0.08272), rot=(0.0, 0.0, 0.71938, 0.694617)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.57229, 0.03312, 0.03789), rot=(0.0, 0.0, -0.053623, 0.998561)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.55639, 0.11756, 0.03779), rot=(0.0, 0.0, 0.998072, 0.062068)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41102, 0.17982, 0.0465), rot=(0.0, 0.0, -0.986777, 0.162086)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("red_block")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace081": HeterogeneousTaskCfg(
        instruction="place a block into the black bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "green_block",
                "jello",
                "blue_block",
                "measuring_spoon",
                "red_block",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.45726, 0.0028, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.50588, 0.05343, -0.05604), rot=(0.0, 0.0, -0.989058, 0.147525)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41386, -0.21982, -0.0325), rot=(0.0, 0.0, -0.121387, 0.992605)
            ),
            "jello": RigidObjectCfg.InitialStateCfg(
                pos=(0.43748, -0.13662, -0.04094), rot=(0.0, 0.0, 0.643145, 0.765745)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.33022, -0.19389, -0.0325), rot=(0.0, 0.0, -0.935561, 0.353165)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.36901, 0.23057, -0.04166), rot=(0.0, 0.0, 0.999234, 0.03913)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.35361, 0.14214, -0.0325), rot=(0.0, 0.0, 0.484398, 0.874848)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("green_block"),
                        SceneEntityCfg("blue_block"),
                        SceneEntityCfg("red_block"),
                    ],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace082": HeterogeneousTaskCfg(
        instruction="pick up the tomato sauce can and put it in the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "blackandbrassbowl_small",
                "empty_2",
                "crabbypenholder",
                "oatmeal_raisin_cookies",
                "granola_bars",
                "storage_box",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.75781, 0.16442, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.39277, 0.02993, 0.12896), rot=(0.0, 0.0, -0.071428, 0.997446)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.37888, -0.16737, 0.12897), rot=(0.0, 0.0, 0.195796, 0.980645)
            ),
            "oatmeal_raisin_cookies": RigidObjectCfg.InitialStateCfg(
                pos=(0.49158, 0.19122, 0.21262), rot=(0.0, 0.0, 0.041536, 0.999137)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.55348, 0.09717, 0.191), rot=(0.0, 0.0, -0.044126, 0.999026)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.47967, -0.16925, 0.12887), rot=(0.0, 0.0, 0.635671, 0.77196)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.36558, -0.24321, 0.17042), rot=(0.0, 0.0, 0.709727, 0.704477)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("tomato_sauce_can")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace083": HeterogeneousTaskCfg(
        instruction="set the blue block on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "empty_1",
                "av_book_11",
                "empty_3",
                "empty_4",
                "empty_5",
                "blue_block",
                "av_woodenchsmall",
                "pitted_cherries",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.42561, -0.09907, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_book_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.41526, 0.05126, 0.109), rot=(0.0, 0.0, 0.045392, 0.998969)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.46179, -0.16217, 0.1325), rot=(0.0, 0.0, -0.987418, 0.158131)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.56583, -0.08067, 0.109), rot=(0.0, 0.0, -0.792058, 0.610446)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.48526, -0.2554, 0.13811), rot=(0.0, 0.0, 0.186722, 0.982413)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("blue_block")],
                    "reference": SceneEntityCfg("av_book_11"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace084": HeterogeneousTaskCfg(
        instruction="move the can of corn into the pan",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "av_pan_b",
                "empty_2",
                "empty_3",
                "empty_4",
                "corn_can",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.58615, 0.45857, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_pan_b": RigidObjectCfg.InitialStateCfg(
                pos=(0.40063, 0.09358, 0.07688), rot=(0.0, 0.0, -0.014089, 0.999901)
            ),
            "corn_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.33067, -0.2616, 0.10602), rot=(0.0, 0.0, 0.998445, 0.055752)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("corn_can")],
                    "container": SceneEntityCfg("av_pan_b"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace085": HeterogeneousTaskCfg(
        instruction="place the popcorn box on top of the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "empty_1",
                "av_book_11",
                "empty_3",
                "ranch_dressing",
                "popcorn_box",
                "peas_and_carrots",
                "naturalbostonroundbottle_a02",
                "empty_8",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.96886, -0.06876, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_book_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.43253, 0.03544, 0.071), rot=(0.0, 0.0, 0.363894, 0.93144)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.49981, 0.21931, 0.14472), rot=(0.0, 0.0, -0.548091, 0.836419)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.36042, -0.10761, 0.11349), rot=(0.0, 0.0, 0.385178, 0.922842)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.47094, -0.17831, 0.10026), rot=(0.0, 0.0, 0.889555, 0.456828)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.30386, 0.22556, 0.071), rot=(0.0, 0.0, -0.732967, 0.680265)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("popcorn_box")],
                    "reference": SceneEntityCfg("av_book_11"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace086": HeterogeneousTaskCfg(
        instruction="place the can of mushrooms into the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "raisin_box",
                "canned_mushrooms",
                "mug_ycb",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.46401, 0.00717, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.45578, 0.16706, -0.00094), rot=(0.0, 0.0, 0.999887, 0.015022)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.56738, -0.15058, 0.06059), rot=(0.0, 0.0, 0.963046, 0.269336)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.47931, -0.04383, 0.01566), rot=(0.0, 0.0, 0.831734, 0.555174)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.33429, -0.09673, 0.03969), rot=(0.0, 0.0, 0.993051, 0.117685)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_mushrooms")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace087": HeterogeneousTaskCfg(
        instruction="pick up the kiwi and put it in the wooden bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "gilbert",
                "wooden_bowl",
                "empty_2",
                "av_kiwi01",
                "empty_4",
                "canned_tuna",
                "empty_6",
                "coffee_pot",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "gilbert": RigidObjectCfg.InitialStateCfg(
                pos=(0.72607, -0.11559, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43594, 0.09715, 0.11892), rot=(0.0, 0.0, 0.096411, 0.995342)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.3077, -0.12687, 0.11899), rot=(0.0, 0.0, -0.254702, 0.96702)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.38598, -0.14585, 0.13529), rot=(0.0, 0.0, 0.999836, 0.018111)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.5015, -0.16117, 0.11896), rot=(0.0, 0.0, 0.670122, 0.742251)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.33581, -0.23261, 0.11906), rot=(0.0, 0.0, 0.803837, 0.59485)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace088": HeterogeneousTaskCfg(
        instruction="move the white bottle into the salt bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "av_p_glassware_tall",
                "ketchup_bottle",
                "mug_ycb",
                "brick",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41715, 0.177, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.45979, 0.05525, 0.06906), rot=(0.0, 0.0, -0.962678, 0.270649)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.34551, -0.23279, 0.069), rot=(0.0, 0.0, 0.221394, 0.975185)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.31099, 0.21317, 0.069), rot=(0.0, 0.0, 0.235614, 0.971847)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.35539, -0.08941, 0.1433), rot=(0.0, 0.0, 0.805833, 0.592144)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.46872, -0.17078, 0.10969), rot=(0.0, 0.0, -0.344735, 0.9387)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(pos=(0.44497, 0.20976, 0.09457), rot=(0.0, 0.0, 0.85384, 0.520536)),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace089": HeterogeneousTaskCfg(
        instruction="put the white bottle in the pan",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "av_pan_d",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a01",
                "empty_5",
                "empty_6",
                "green_beans_can",
                "mustard_bottle",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.47359, 0.30969, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.45721, -0.00305, 0.00726), rot=(0.0, 0.0, -0.997177, 0.075081)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.48193, -0.24039, 0.007), rot=(0.0, 0.0, 0.612197, 0.790705)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.33901, -0.19227, 0.03579), rot=(0.0, 0.0, -0.11097, 0.993824)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.5068, -0.16531, 0.08703), rot=(0.0, 0.0, 0.621271, 0.783596)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a01")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace090": HeterogeneousTaskCfg(
        instruction="place the marker on top of the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "empty_1",
                "av_book_21",
                "empty_3",
                "cream_cheese",
                "empty_5",
                "empty_6",
                "dry_erase_marker",
                "empty_8",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.57626, 0.73442, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_book_21": RigidObjectCfg.InitialStateCfg(
                pos=(0.42535, 0.00484, 0.071), rot=(0.0, 0.0, -0.13528, 0.990807)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.32086, 0.20666, 0.0976), rot=(0.0, 0.0, 0.954833, 0.297142)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.39142, -0.18786, 0.08044), rot=(0.0, 0.0, 0.473634, 0.880722)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("dry_erase_marker")],
                    "reference": SceneEntityCfg("av_book_21"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace091": HeterogeneousTaskCfg(
        instruction="move the chocolate pudding mix into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "bin_b04",
                "empty_2",
                "empty_3",
                "green_block",
                "empty_5",
                "av_plant_succulent_02",
                "chocolate_pudding_mix",
                "brick",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(
                pos=(0.91076, -0.00838, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.41139, -0.10333, -0.058), rot=(0.0, 0.0, -0.629542, 0.776967)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.53029, 0.03993, -0.0345), rot=(0.0, 0.0, -0.979049, 0.203623)
            ),
            "av_plant_succulent_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.37518, 0.202, -0.058), rot=(0.0, 0.0, -0.324064, 0.946035)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.5718, -0.03199, -0.03326), rot=(0.0, 0.0, -0.220332, 0.975425)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.49602, 0.12811, -0.03243), rot=(0.0, 0.0, 0.517519, 0.855672)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace092": HeterogeneousTaskCfg(
        instruction="move the foam roller into the decorative bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "av_decorativebowl",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "canned_peaches",
                "empty_6",
                "av_chess_rook",
                "foam_roller",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(
                pos=(0.93459, -0.03902, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43027, -0.08924, -0.05796), rot=(0.0, 0.0, 0.66032, 0.750984)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.5104, 0.14762, -0.058), rot=(0.0, 0.0, 0.780985, 0.62455)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.31279, 0.12274, -0.02911), rot=(0.0, 0.0, 0.208897, 0.977938)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.4391, 0.09611, -0.058), rot=(0.0, 0.0, 0.17286, 0.984946)
            ),
            "foam_roller": RigidObjectCfg.InitialStateCfg(
                pos=(0.42093, 0.20708, -0.05795), rot=(0.0, 0.0, -0.671484, 0.741019)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("foam_roller")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace093": HeterogeneousTaskCfg(
        instruction="first put the wooden box in the red bowl, then put the chocolate pudding mix in it",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "bowl",
                "av_coaster_hexagon",
                "crabbypenholder",
                "empty_4",
                "granola_bars",
                "storage_box",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.72759, -0.59797, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.41687, 0.15064, 0.15648), rot=(0.0, 0.0, 0.681609, 0.731717)),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.33179, -0.07969, 0.129), rot=(0.0, 0.0, 0.423097, 0.906085)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.55464, -0.15044, 0.12897), rot=(0.0, 0.0, 0.126002, 0.99203)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.45064, -0.20941, 0.191), rot=(0.0, 0.0, 0.534363, 0.845255)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.46047, -0.09457, 0.12887), rot=(0.0, 0.0, -0.889931, 0.456095)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.55547, 0.03569, 0.15374), rot=(0.0, 0.0, 0.897657, 0.440694)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("storage_box"), SceneEntityCfg("chocolate_pudding_mix")],
                    "container": SceneEntityCfg("bowl"),
                },
            ),
        },
    ),
    "PickPlace094": HeterogeneousTaskCfg(
        instruction="put the chess bishop in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "bin_b04",
                "empty_2",
                "crabbypenholder",
                "butter",
                "empty_5",
                "av_chess_bishop",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.4672, 0.02254, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.42875, 0.01483, -0.056), rot=(0.0, 0.0, 0.792633, 0.609699)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.30141, -0.06864, -0.05603), rot=(0.0, 0.0, -0.865014, 0.501748)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.32843, 0.27845, -0.02959), rot=(0.0, 0.0, -0.724437, 0.689341)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.43525, 0.23993, -0.056), rot=(0.0, 0.0, -0.682463, 0.730921)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.31402, 0.22337, -0.03126), rot=(0.0, 0.0, 0.796623, 0.604476)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_bishop")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace095": HeterogeneousTaskCfg(
        instruction="place a bottle into the salt bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "av_salt_bowl",
                "empty_2",
                "spatula_13",
                "ranch_dressing",
                "milkjug_a01",
                "av_chess_king",
                "naturalbostonroundbottle_a02",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.52727, 0.02256, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.48508, -0.05707, -0.03094), rot=(0.0, 0.0, -0.766631, 0.642088)
            ),
            "spatula_13": RigidObjectCfg.InitialStateCfg(
                pos=(0.42448, 0.15707, -0.02733), rot=(0.0, 0.0, 0.87752, 0.47954)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.34948, -0.01345, 0.04272), rot=(0.0, 0.0, 0.623166, 0.782089)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.47972, -0.23984, -0.031), rot=(0.0, 0.0, -0.913852, 0.406047)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.55695, 0.07206, -0.031), rot=(0.0, 0.0, 0.517474, 0.855699)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.31575, -0.24422, -0.031), rot=(0.0, 0.0, 0.063492, 0.997982)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("milkjug_a01"),
                        SceneEntityCfg("naturalbostonroundbottle_a02"),
                        SceneEntityCfg("ranch_dressing"),
                    ],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace096": HeterogeneousTaskCfg(
        instruction="first put the spoon in the decorative bowl, then put the mustard bottle in it",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "av_decorativebowl",
                "av_coaster_hexagon",
                "empty_3",
                "empty_4",
                "empty_5",
                "spoon",
                "empty_7",
                "mustard_bottle",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.42796, -0.31079, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.42903, 0.09719, 0.00704), rot=(0.0, 0.0, -0.988946, 0.148277)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.54536, -0.11851, 0.007), rot=(0.0, 0.0, 0.648004, 0.761637)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.36489, -0.15764, 0.01578), rot=(0.0, 0.0, 0.98978, 0.142601)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.43125, -0.24125, 0.08703), rot=(0.0, 0.0, -0.487637, 0.873046)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("spoon"), SceneEntityCfg("mustard_bottle")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                },
            ),
        },
    ),
    "PickPlace097": HeterogeneousTaskCfg(
        instruction="move the chess pawn into the wooden bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "wooden_bowl",
                "empty_2",
                "spatula_10",
                "av_chess_pawn",
                "empty_5",
                "empty_6",
                "naturalbostonroundbottle_a03",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.466, -0.0773, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.44146, -0.01051, -0.03108), rot=(0.0, 0.0, -0.082748, 0.996571)
            ),
            "spatula_10": RigidObjectCfg.InitialStateCfg(
                pos=(0.41808, 0.20677, -0.02427), rot=(0.0, 0.0, 0.029905, 0.999553)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.51892, -0.23098, -0.031), rot=(0.0, 0.0, -0.836859, 0.547419)
            ),
            "naturalbostonroundbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.4104, -0.23965, -0.031), rot=(0.0, 0.0, 0.998023, 0.062842)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_pawn")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace098": HeterogeneousTaskCfg(
        instruction="pick up the raisin box and put it in the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "bowl",
                "empty_2",
                "empty_3",
                "spam_can",
                "raisin_box",
                "av_mustard_jar",
                "measuring_spoon",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.49645, 0.01913, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.45642, 0.11888, -0.00352), rot=(0.0, 0.0, -0.999981, 0.006227)
            ),
            "spam_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.52694, -0.03185, 0.01082), rot=(0.0, 0.0, 0.984072, 0.177773)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.52612, -0.18846, 0.03059), rot=(0.0, 0.0, -0.626302, 0.77958)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.28736, 0.19215, -0.03092), rot=(0.0, 0.0, -0.127498, 0.991839)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.38267, -0.1468, -0.01666), rot=(0.0, 0.0, 0.747125, 0.664683)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("raisin_box")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace099": HeterogeneousTaskCfg(
        instruction="place a chess piece into the red bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "bowl",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "mug",
                "av_chess_bishop",
                "av_chess_rook",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.64514, 0.44175, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.48312, 0.02898, 0.10448), rot=(0.0, 0.0, 0.542976, 0.839748)),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.33077, 0.18361, 0.077), rot=(0.0, 0.0, -0.330384, 0.943847)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.32626, -0.16933, 0.07718), rot=(0.0, 0.0, 0.743664, 0.668554)),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.44956, -0.25801, 0.077), rot=(0.0, 0.0, -0.275551, 0.961287)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.49245, 0.27415, 0.077), rot=(0.0, 0.0, 0.996505, 0.083533)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.46911, -0.17747, 0.07706), rot=(0.0, 0.0, 0.687002, 0.726656)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("av_chess_rook"),
                        SceneEntityCfg("av_chess_bishop"),
                        SceneEntityCfg("av_chess_pawn"),
                    ],
                    "container": SceneEntityCfg("bowl"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace100": HeterogeneousTaskCfg(
        instruction="remove the wooden candle holder from the metal bin",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "gilbert",
                "bin_a05",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "mug",
                "spoon_big",
                "av_woodenchmedium",
                "empty_8",
            ]
        ),
        init_states={
            "gilbert": RigidObjectCfg.InitialStateCfg(pos=(1.64664, 0.08091, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.42849, -0.12845, 0.119), rot=(0.0, 0.0, -0.669449, 0.742858)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.29641, 0.24566, 0.119), rot=(0.0, 0.0, -0.752171, 0.658968)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.42354, 0.22613, 0.11918), rot=(0.0, 0.0, -0.041432, 0.999141)),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.51813, 0.10186, 0.11924), rot=(0.0, 0.0, 0.980438, 0.196826)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.42849, -0.12845, 0.13301), rot=(0.0, 0.0, 0.189047, 0.981968)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("av_woodenchmedium")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace101": HeterogeneousTaskCfg(
        instruction="put the green block in the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "green_block",
                "mug",
                "av_chess_bishop",
                "naturalbostonroundbottle_a02",
                "potato_masher",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(pos=(0.77704, 1.6845, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.42169, 0.00092, 0.13648), rot=(0.0, 0.0, -0.332865, 0.942974)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.51527, 0.20452, 0.1325), rot=(0.0, 0.0, 0.923829, 0.382807)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.32667, 0.22395, 0.10918), rot=(0.0, 0.0, 0.938893, 0.344208)),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.33781, -0.20168, 0.109), rot=(0.0, 0.0, -0.851321, 0.524645)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.44195, -0.22717, 0.109), rot=(0.0, 0.0, 0.532985, 0.846125)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.53068, -0.13715, 0.10942), rot=(0.0, 0.0, -0.582293, 0.812979)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("green_block")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace102": HeterogeneousTaskCfg(
        instruction="put the plastic bottle in the pan",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "av_pan_d",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "empty_5",
                "spoon_2",
                "naturalbostonroundbottle_a02",
                "empty_8",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.68734, -0.60028, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.39886, 0.01681, 0.12926), rot=(0.0, 0.0, -0.050815, 0.998708)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.57243, -0.00667, 0.129), rot=(0.0, 0.0, -0.299802, 0.954002)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.49416, 0.24162, 0.13439), rot=(0.0, 0.0, 0.988598, 0.15058)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.35165, 0.25159, 0.129), rot=(0.0, 0.0, -0.995412, 0.095683)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("naturalbostonroundbottle_a02")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace103": HeterogeneousTaskCfg(
        instruction="put the white bottle and the can of peaches in the metal bin",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "bin_a05",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "canned_peaches",
                "empty_6",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(pos=(2.2874, 0.12701, -0.697), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.43592, -0.04726, 0.109), rot=(0.0, 0.0, 0.723402, 0.690427)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.44884, -0.25241, 0.109), rot=(0.0, 0.0, -0.103497, 0.99463)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.32935, 0.20581, 0.13789), rot=(0.0, 0.0, -0.674868, 0.737939)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.45732, 0.1624, 0.13374), rot=(0.0, 0.0, 0.778857, 0.627202)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a03"), SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace104": HeterogeneousTaskCfg(
        instruction="put the barbecue sauce bottle in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "bin_a03",
                "empty_2",
                "spatula_17",
                "empty_4",
                "smartphone",
                "spoon_1",
                "coffee_pot",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.58619, -0.00783, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.46854, -0.03424, 0.009), rot=(0.0, 0.0, -0.837588, 0.546303)
            ),
            "spatula_17": RigidObjectCfg.InitialStateCfg(
                pos=(0.32837, 0.07619, 0.01268), rot=(0.0, 0.0, 0.748109, 0.663576)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.48186, -0.21385, 0.01465), rot=(0.0, 0.0, -0.834162, 0.551519)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.35059, -0.17891, 0.01966), rot=(0.0, 0.0, 0.763447, 0.645871)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.45048, 0.19764, 0.00896), rot=(0.0, 0.0, -0.453721, 0.891144)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.56734, 0.08946, 0.08316), rot=(0.0, 0.0, 0.999927, 0.012086)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("bbq_sauce_bottle")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace105": HeterogeneousTaskCfg(
        instruction="remove the chocolate pudding mix from the red bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "bowl",
                "empty_2",
                "spatula_08",
                "butter",
                "granola_bars",
                "av_mustard_jar",
                "chocolate_pudding_mix",
                "soft_scrub",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.50539, 0.07847, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.37305, 0.0048, 0.03648), rot=(0.0, 0.0, 0.987915, 0.154998)),
            "spatula_08": RigidObjectCfg.InitialStateCfg(
                pos=(0.51685, 0.09007, 0.01573), rot=(0.0, 0.0, -0.502943, 0.86432)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.31884, -0.27303, 0.03541), rot=(0.0, 0.0, 0.654259, 0.756271)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.42623, 0.26486, 0.071), rot=(0.0, 0.0, -0.700305, 0.713844)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.49262, -0.13335, 0.00908), rot=(0.0, 0.0, -0.015678, 0.999877)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.37305, 0.0048, 0.04774), rot=(0.0, 0.0, -0.240436, 0.970665)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.46983, -0.22959, 0.13432), rot=(0.0, 0.0, -0.051269, 0.998685)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace106": HeterogeneousTaskCfg(
        instruction="place the wooden candle holder into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_maple",
                "bin_a05",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "fork_big",
                "blue_block",
                "av_woodenchmedium",
                "empty_8",
            ]
        ),
        init_states={
            "table_maple": RigidObjectCfg.InitialStateCfg(pos=(0.47549, 0.13283, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.43353, 0.08645, 0.009), rot=(0.0, 0.0, 0.270274, 0.962783)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.38216, -0.13468, 0.009), rot=(0.0, 0.0, -0.127162, 0.991882)
            ),
            "fork_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.43719, -0.26775, 0.00935), rot=(0.0, 0.0, 0.058515, 0.998287)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.47586, -0.15225, 0.0325), rot=(0.0, 0.0, -0.442387, 0.896824)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.37549, 0.22476, 0.00901), rot=(0.0, 0.0, 0.414528, 0.910036)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_woodenchmedium")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace107": HeterogeneousTaskCfg(
        instruction="pick up the plastic bottle and put it in the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "av_salt_bowl",
                "av_photoalbum_closed",
                "empty_3",
                "whitepackerbottle_a02",
                "empty_5",
                "spoon_big",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(
                pos=(0.62503, -0.00017, -0.347), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.37465, 0.06965, 0.00906), rot=(0.0, 0.0, -0.053916, 0.998545)
            ),
            "av_photoalbum_closed": RigidObjectCfg.InitialStateCfg(
                pos=(0.46426, -0.14334, 0.009), rot=(0.0, 0.0, -0.949544, 0.313633)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.37427, 0.24433, 0.009), rot=(0.0, 0.0, 0.958302, 0.285756)
            ),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.296, -0.132, 0.00924), rot=(0.0, 0.0, -0.733466, 0.679726)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.51781, 0.14257, 0.009), rot=(0.0, 0.0, -0.977838, 0.209365)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace108": HeterogeneousTaskCfg(
        instruction="move the can of corn into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "bin_a05",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "corn_can",
                "empty_6",
                "empty_7",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.40829, -0.03374, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.41912, -0.02274, -0.001), rot=(0.0, 0.0, -0.55659, 0.830788)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.29903, -0.08418, -0.001), rot=(0.0, 0.0, -0.977233, 0.212167)
            ),
            "corn_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.5498, 0.09288, 0.02802), rot=(0.0, 0.0, 0.094985, 0.995479)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.30691, -0.17138, 0.07903), rot=(0.0, 0.0, 0.269862, 0.962899)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("corn_can")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace109": HeterogeneousTaskCfg(
        instruction="move the can of peaches into the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "butter",
                "canned_peaches",
                "av_plant_succulent_02",
                "bagel_00",
                "lime01",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.71738, 0.17355, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.48971, -0.0437, 0.12896), rot=(0.0, 0.0, 0.926179, 0.377084)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.31577, 0.24586, 0.15541), rot=(0.0, 0.0, 0.958216, 0.286047)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.50344, 0.19487, 0.15789), rot=(0.0, 0.0, 0.987144, 0.159832)
            ),
            "av_plant_succulent_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.38961, -0.21533, 0.129), rot=(0.0, 0.0, 0.940728, 0.339163)
            ),
            "bagel_00": RigidObjectCfg.InitialStateCfg(
                pos=(0.39746, 0.19592, 0.14458), rot=(0.0, 0.0, 0.601981, 0.798511)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.34205, -0.10808, 0.12922), rot=(0.0, 0.0, 0.397539, 0.917585)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace110": HeterogeneousTaskCfg(
        instruction="put the cream cheese on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "empty_1",
                "av_book_77",
                "spatula_11",
                "cream_cheese",
                "empty_5",
                "av_p_glassware_short",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.51612, -0.30698, -0.69646), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_book_77": RigidObjectCfg.InitialStateCfg(
                pos=(0.42533, 0.0416, 0.09), rot=(0.0, 0.0, -0.061728, 0.998093)
            ),
            "spatula_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.41285, -0.21476, 0.09673), rot=(0.0, 0.0, 0.82222, 0.569169)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.30922, -0.17639, 0.1166), rot=(0.0, 0.0, -0.662798, 0.748798)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.50382, 0.25016, 0.09), rot=(0.0, 0.0, -0.797407, 0.603442)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.31997, 0.24105, 0.13142), rot=(0.0, 0.0, 0.905562, 0.424214)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("cream_cheese")],
                    "reference": SceneEntityCfg("av_book_77"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace111": HeterogeneousTaskCfg(
        instruction="place the sugar box on top of the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "empty_1",
                "av_book_51",
                "sugar_box",
                "empty_4",
                "canned_peaches",
                "av_apple",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.6049, -0.45986, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_book_51": RigidObjectCfg.InitialStateCfg(
                pos=(0.40362, -0.10539, 0.077), rot=(0.0, 0.0, -0.997887, 0.064978)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.39366, 0.15185, 0.16501), rot=(0.0, 0.0, -0.99538, 0.09601)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.56102, 0.09254, 0.10589), rot=(0.0, 0.0, -0.960108, 0.279629)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.50556, 0.1968, 0.07796), rot=(0.0, 0.0, 0.099494, 0.995038)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29569, 0.2305, 0.077), rot=(0.0, 0.0, -0.99817, 0.060477)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("sugar_box")],
                    "reference": SceneEntityCfg("av_book_51"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace112": HeterogeneousTaskCfg(
        instruction="move the yellow block into the salt bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "av_salt_bowl",
                "empty_2",
                "av_kiwi01",
                "whitepackerbottle_a01",
                "mug",
                "spoon",
                "yellow_block",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.57749, -0.07961, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.35752, 0.07634, 0.00906), rot=(0.0, 0.0, -0.080384, 0.996764)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4665, 0.19284, 0.00899), rot=(0.0, 0.0, 0.997589, 0.069397)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.48122, 0.26553, 0.009), rot=(0.0, 0.0, -0.365764, 0.930708)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.54454, 0.00587, 0.00918), rot=(0.0, 0.0, 0.318331, 0.947979)),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.50304, -0.13661, 0.01778), rot=(0.0, 0.0, 0.999272, 0.038163)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41017, -0.23808, 0.0325), rot=(0.0, 0.0, -0.95821, 0.286067)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.33025, -0.21191, 0.09213), rot=(0.0, 0.0, 0.999805, 0.019741)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("yellow_block")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace113": HeterogeneousTaskCfg(
        instruction="place both the milk jug and the chess rook into the metal bin",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "bin_a05",
                "empty_2",
                "empty_3",
                "oatmeal_raisin_cookies",
                "milkjug_a01",
                "empty_6",
                "av_chess_rook",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.39882, 0.03972, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.40658, 0.04101, -0.001), rot=(0.0, 0.0, -0.999895, 0.014505)
            ),
            "oatmeal_raisin_cookies": RigidObjectCfg.InitialStateCfg(
                pos=(0.50615, -0.14806, 0.08262), rot=(0.0, 0.0, -0.504243, 0.863562)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.45731, 0.23451, -0.001), rot=(0.0, 0.0, -0.175617, 0.984459)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.30069, -0.16045, -0.001), rot=(0.0, 0.0, 0.271419, 0.962461)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("milkjug_a01"), SceneEntityCfg("av_chess_rook")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace114": HeterogeneousTaskCfg(
        instruction="place both the kiwi and the orange juice carton into the metal bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "serving_bowl",
                "empty_2",
                "av_kiwi01",
                "green_block",
                "orange_juice_carton",
                "av_apple",
                "empty_7",
                "soft_scrub",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.22922, -0.2144, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.45852, -0.10309, 0.15625), rot=(0.0, 0.0, -0.749074, 0.662487)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29494, -0.13545, 0.15599), rot=(0.0, 0.0, 0.40249, 0.915424)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.46632, -0.2351, 0.1795), rot=(0.0, 0.0, -0.479213, 0.877699)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.51821, 0.05567, 0.25224), rot=(0.0, 0.0, -0.736285, 0.676671)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.50382, 0.18151, 0.15696), rot=(0.0, 0.0, 0.566332, 0.824177)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.34224, 0.13632, 0.28132), rot=(0.0, 0.0, -0.995765, 0.09193)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01"), SceneEntityCfg("orange_juice_carton")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace115": HeterogeneousTaskCfg(
        instruction="put a block in the metal bin",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_a02",
                "empty_2",
                "spatula_10",
                "cream_cheese",
                "soup_can",
                "blue_block",
                "yellow_block",
                "brick",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41549, 0.1191, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.34256, 0.22214, 0.069), rot=(0.0, 0.0, -0.981187, 0.193061)
            ),
            "spatula_10": RigidObjectCfg.InitialStateCfg(
                pos=(0.31375, -0.00651, 0.07573), rot=(0.0, 0.0, -0.779123, 0.626871)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.54883, -0.21359, 0.0956), rot=(0.0, 0.0, 0.985421, 0.170134)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.52947, -0.0986, 0.06906), rot=(0.0, 0.0, 0.989117, 0.14713)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.53371, 0.06742, 0.0925), rot=(0.0, 0.0, -0.93555, 0.353194)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.51364, 0.20489, 0.0925), rot=(0.0, 0.0, -0.345714, 0.93834)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.40044, -0.12902, 0.09457), rot=(0.0, 0.0, -0.636448, 0.77132)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("blue_block"), SceneEntityCfg("brick"), SceneEntityCfg("yellow_block")],
                    "container": SceneEntityCfg("bin_a02"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace116": HeterogeneousTaskCfg(
        instruction="pick up the white bottle and put it in the decorative bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "av_decorativebowl",
                "empty_2",
                "spatula_05",
                "whitepackerbottle_a01",
                "canned_peaches",
                "empty_6",
                "empty_7",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(
                pos=(1.34497, -0.08278, -0.6971), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.42623, 0.11926, 0.12904), rot=(0.0, 0.0, -0.02012, 0.999798)
            ),
            "spatula_05": RigidObjectCfg.InitialStateCfg(
                pos=(0.41031, -0.07531, 0.13271), rot=(0.0, 0.0, -0.999308, 0.037206)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.54634, -0.15815, 0.129), rot=(0.0, 0.0, 0.952295, 0.305179)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.29752, -0.22109, 0.15789), rot=(0.0, 0.0, -0.995628, 0.093406)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.49559, -0.2334, 0.21213), rot=(0.0, 0.0, -0.905693, 0.423935)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a01")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace117": HeterogeneousTaskCfg(
        instruction="put the chocolate pudding mix in the metal bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "serving_bowl",
                "av_coaster_hexagon",
                "spatula_13",
                "whitepackerbottle_a02",
                "raisin_box",
                "storage_box",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.61452, 0.2904, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.38654, 0.03568, 0.07125), rot=(0.0, 0.0, 0.346425, 0.938078)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.36959, -0.21777, 0.071), rot=(0.0, 0.0, 0.073827, 0.997271)
            ),
            "spatula_13": RigidObjectCfg.InitialStateCfg(
                pos=(0.54549, 0.08231, 0.07467), rot=(0.0, 0.0, 0.16225, 0.98675)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.50391, -0.24024, 0.071), rot=(0.0, 0.0, -0.326256, 0.945281)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.38886, 0.23321, 0.13259), rot=(0.0, 0.0, 0.71968, 0.694306)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.48057, -0.14051, 0.07087), rot=(0.0, 0.0, 0.440348, 0.897827)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.32251, -0.06887, 0.09574), rot=(0.0, 0.0, 0.307173, 0.951654)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace118": HeterogeneousTaskCfg(
        instruction="pick up the can of corn and put it in the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "bowl_ycb",
                "empty_2",
                "frozen_waffles",
                "glasses",
                "corn_can",
                "canned_mushrooms",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(pos=(0.87585, 0.0617, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.50573, 0.01983, -0.03052), rot=(0.0, 0.0, 0.333868, 0.94262)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.35696, 0.01002, -0.05802), rot=(0.0, 0.0, 0.192699, 0.981258)
            ),
            "glasses": RigidObjectCfg.InitialStateCfg(
                pos=(0.45065, -0.16807, -0.02753), rot=(0.0, 0.0, 0.724602, 0.689167)
            ),
            "corn_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.29812, 0.13606, -0.02898), rot=(0.0, 0.0, 0.50825, 0.861209)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.50902, 0.22513, -0.04134), rot=(0.0, 0.0, 0.959702, 0.281021)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.30008, 0.24884, -0.058), rot=(0.0, 0.0, 0.115489, 0.993309)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("corn_can")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace119": HeterogeneousTaskCfg(
        instruction="move the can of mushrooms into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "bin_a05",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "empty_5",
                "canned_mushrooms",
                "frozen_vegetable_block",
                "empty_8",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.55591, 0.04381, -0.69705), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.43255, 0.02465, 0.069), rot=(0.0, 0.0, -0.753265, 0.657718)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.4759, -0.21581, 0.069), rot=(0.0, 0.0, -0.936059, 0.351844)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.3577, -0.18997, 0.08566), rot=(0.0, 0.0, 0.833965, 0.551818)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.57567, -0.13119, 0.069), rot=(0.0, 0.0, -0.130863, 0.9914)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_mushrooms")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace120": HeterogeneousTaskCfg(
        instruction="set the brick on the sketchbook",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "empty_1",
                "av_sketchbook",
                "av_kiwi01",
                "ranch_dressing",
                "granola_bars",
                "milk_carton",
                "mayonnaise_bottle",
                "brick",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.68776, 0.59843, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_sketchbook": RigidObjectCfg.InitialStateCfg(
                pos=(0.42613, 0.11578, 0.13117), rot=(0.0, 0.0, -0.414299, 0.910141)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4153, -0.19044, 0.12899), rot=(0.0, 0.0, 0.914255, 0.405139)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.29463, -0.06285, 0.20272), rot=(0.0, 0.0, -0.23131, 0.97288)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.33159, -0.14215, 0.191), rot=(0.0, 0.0, -0.383622, 0.92349)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.55111, -0.06993, 0.22418), rot=(0.0, 0.0, 0.366398, 0.930458)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.46914, -0.13132, 0.20295), rot=(0.0, 0.0, -0.735134, 0.677922)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.48668, -0.23873, 0.15457), rot=(0.0, 0.0, -0.174088, 0.98473)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("brick")],
                    "reference": SceneEntityCfg("av_sketchbook"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace121": HeterogeneousTaskCfg(
        instruction="set the can of peas and carrots upright on the plate",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "empty_1",
                "plate_small",
                "av_kiwi01",
                "parmesan_cheese_canister",
                "empty_5",
                "peas_and_carrots",
                "frozen_vegetable_block",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.50717, -0.12757, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "plate_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.42126, 0.01753, -0.03109), rot=(0.0, 0.0, -0.761462, 0.64821)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.39449, 0.25902, -0.03101), rot=(0.0, 0.0, -0.182306, 0.983242)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.33208, -0.26575, 0.02043), rot=(0.0, 0.0, 0.703272, 0.710921)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.51691, -0.24124, -0.00174), rot=(0.0, 0.0, -0.454555, 0.890719)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.4105, -0.2029, -0.031), rot=(0.0, 0.0, -0.790327, 0.612685)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("peas_and_carrots")],
                    "reference": SceneEntityCfg("plate_small"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace122": HeterogeneousTaskCfg(
        instruction="pick up the chess queen and put it in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "bin_a05",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "soup_can",
                "ketchup_bottle",
                "green_beans_can",
                "lime01",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.49479, -0.07245, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.41333, -0.0891, -0.031), rot=(0.0, 0.0, 0.772448, 0.635078)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.51195, 0.12339, -0.031), rot=(0.0, 0.0, 0.538087, 0.842889)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.3763, 0.15069, -0.03094), rot=(0.0, 0.0, 0.96491, 0.262582)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.38982, 0.251, 0.0433), rot=(0.0, 0.0, -0.938555, 0.34513)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.57213, 0.02581, -0.00221), rot=(0.0, 0.0, 0.61681, 0.787112)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.52013, 0.21347, -0.03078), rot=(0.0, 0.0, 0.996864, 0.07914)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace123": HeterogeneousTaskCfg(
        instruction="place the pumpkin into the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "bowl",
                "empty_2",
                "sugar_box",
                "empty_4",
                "pumpkinsmall",
                "spoon_big",
                "frozen_vegetable_block",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.47688, 0.17457, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.43414, 0.20083, 0.03448), rot=(0.0, 0.0, 0.676121, 0.73679)),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.31431, 0.06726, 0.09501), rot=(0.0, 0.0, 0.269962, 0.962871)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.42996, -0.07274, 0.00674), rot=(0.0, 0.0, -0.837767, 0.546027)
            ),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.33661, -0.1546, 0.00724), rot=(0.0, 0.0, -0.548494, 0.836155)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.53674, 0.08704, 0.007), rot=(0.0, 0.0, 0.950114, 0.311902)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("pumpkinsmall")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace124": HeterogeneousTaskCfg(
        instruction="move the barbecue sauce bottle into the pan",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "av_pan_b",
                "empty_2",
                "empty_3",
                "empty_4",
                "empty_5",
                "empty_6",
                "empty_7",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.51666, 0.13414, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_pan_b": RigidObjectCfg.InitialStateCfg(
                pos=(0.4089, -0.07586, 0.00888), rot=(0.0, 0.0, 0.999334, 0.036488)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.3358, -0.26343, 0.00906), rot=(0.0, 0.0, -0.779793, 0.626038)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                    "container": SceneEntityCfg("av_pan_b"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace125": HeterogeneousTaskCfg(
        instruction="place a chess piece into the white bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "smartphone",
                "av_chess_bishop",
                "av_chess_rook",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.41503, -0.03699, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.43816, 0.0969, -0.0312), rot=(0.0, 0.0, 0.889687, 0.45657)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.33967, -0.20446, -0.031), rot=(0.0, 0.0, -0.754442, 0.656366)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.49126, -0.13639, -0.02535), rot=(0.0, 0.0, -0.982519, 0.186162)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.30205, 0.24969, -0.031), rot=(0.0, 0.0, 0.779859, 0.625956)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.38819, -0.14225, -0.031), rot=(0.0, 0.0, 0.99029, 0.139017)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("av_chess_queen"),
                        SceneEntityCfg("av_chess_rook"),
                        SceneEntityCfg("av_chess_bishop"),
                    ],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace126": HeterogeneousTaskCfg(
        instruction="put the pumpkin and the kiwi in the metal bin",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "bin_a05",
                "empty_2",
                "av_kiwi01",
                "empty_4",
                "pumpkinsmall",
                "empty_6",
                "empty_7",
                "foam_roller",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.44519, 0.61302, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.45048, -0.02748, 0.09), rot=(0.0, 0.0, 0.825595, 0.564263)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.37847, 0.2227, 0.08999), rot=(0.0, 0.0, 0.089953, 0.995946)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.30301, -0.0942, 0.08974), rot=(0.0, 0.0, -0.977821, 0.209443)
            ),
            "foam_roller": RigidObjectCfg.InitialStateCfg(
                pos=(0.35054, -0.26538, 0.09005), rot=(0.0, 0.0, -0.730186, 0.683248)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("pumpkinsmall"), SceneEntityCfg("av_kiwi01")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace127": HeterogeneousTaskCfg(
        instruction="place the spoon into the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "orange_02",
                "pumpkinsmall",
                "spoon_2",
                "yellow_block",
                "empty_8",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.58519, 0.05788, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.43223, 0.03831, 0.00894), rot=(0.0, 0.0, -0.643293, 0.76562)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.31468, -0.2183, 0.00906), rot=(0.0, 0.0, -0.991932, 0.126772)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.52777, -0.21736, 0.00874), rot=(0.0, 0.0, -0.133856, 0.991001)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.36764, 0.21831, 0.01439), rot=(0.0, 0.0, 0.151244, 0.988496)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.40266, -0.16798, 0.0325), rot=(0.0, 0.0, -0.004301, 0.999991)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_2")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace128": HeterogeneousTaskCfg(
        instruction="place the popcorn box into the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "bowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "popcorn_box",
                "av_chess_king",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.22805, -0.20032, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.44673, -0.08885, 0.18348), rot=(0.0, 0.0, -0.111568, 0.993757)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.56139, 0.02678, 0.19849), rot=(0.0, 0.0, -0.969805, 0.243881)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.38077, -0.22123, 0.156), rot=(0.0, 0.0, 0.174929, 0.984581)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("popcorn_box")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace129": HeterogeneousTaskCfg(
        instruction="take the spoon out of the metal bin",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "bin_a05",
                "empty_2",
                "empty_3",
                "empty_4",
                "mug",
                "spoon",
                "mug_ycb",
                "pitted_cherries",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.516, 0.02593, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.43951, 0.0177, -0.031), rot=(0.0, 0.0, -0.383028, 0.923737)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.42253, -0.23296, -0.03082), rot=(0.0, 0.0, 0.034064, 0.99942)),
            "spoon": RigidObjectCfg.InitialStateCfg(pos=(0.43951, 0.0177, -0.00822), rot=(0.0, 0.0, 0.992555, 0.1218)),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.32828, 0.21957, 0.00969), rot=(0.0, 0.0, -0.882168, 0.470934)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.5475, 0.14132, -0.00189), rot=(0.0, 0.0, 0.368464, 0.929642)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("spoon")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace130": HeterogeneousTaskCfg(
        instruction="place the blue block into the white bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "empty_5",
                "blue_block",
                "clamp",
                "av_tincan",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.63804, -0.03401, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.45264, -0.04383, 0.0708), rot=(0.0, 0.0, 0.868416, 0.495837)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.41761, -0.26734, 0.071), rot=(0.0, 0.0, -0.713992, 0.700153)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.50376, -0.23801, 0.0945), rot=(0.0, 0.0, 0.276743, 0.960944)
            ),
            "clamp": RigidObjectCfg.InitialStateCfg(
                pos=(0.36524, 0.17312, 0.09058), rot=(0.0, 0.0, -0.993106, 0.117218)
            ),
            "av_tincan": RigidObjectCfg.InitialStateCfg(
                pos=(0.52627, 0.19402, 0.071), rot=(0.0, 0.0, -0.767307, 0.64128)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("blue_block")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace131": HeterogeneousTaskCfg(
        instruction="place both the spoon and the cookie box into the black bowl",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "oatmeal_raisin_cookies",
                "empty_5",
                "spoon_1",
                "av_woodenchmedium",
                "empty_8",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(
                pos=(0.64605, 0.02901, -0.347), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.43004, -0.02348, 0.00894), rot=(0.0, 0.0, -0.047662, 0.998864)
            ),
            "oatmeal_raisin_cookies": RigidObjectCfg.InitialStateCfg(
                pos=(0.40404, 0.22107, 0.09262), rot=(0.0, 0.0, -0.535856, 0.84431)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.36259, -0.23529, 0.01966), rot=(0.0, 0.0, 0.9817, 0.190436)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.32854, 0.26285, 0.00901), rot=(0.0, 0.0, 0.237583, 0.971367)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_1"), SceneEntityCfg("oatmeal_raisin_cookies")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace132": HeterogeneousTaskCfg(
        instruction="put the crab pen holder in the decorative bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "av_decorativebowl",
                "empty_2",
                "crabbypenholder",
                "empty_4",
                "orange_juice_carton",
                "empty_6",
                "frozen_vegetable_block",
                "empty_8",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.43695, 0.02955, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43452, -0.07578, 0.10904), rot=(0.0, 0.0, 0.545585, 0.838056)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.28225, 0.1287, 0.10897), rot=(0.0, 0.0, 0.166293, 0.986076)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.5586, 0.12409, 0.20524), rot=(0.0, 0.0, -0.981919, 0.1893)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.40251, 0.1927, 0.109), rot=(0.0, 0.0, -0.859303, 0.511467)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("crabbypenholder")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace133": HeterogeneousTaskCfg(
        instruction="pick up the white bottle and put it in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "bin_a03",
                "av_book_11",
                "empty_3",
                "whitepackerbottle_a03",
                "tuna_can",
                "empty_6",
                "empty_7",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.59735, -0.29278, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.46626, 0.16667, 0.071), rot=(0.0, 0.0, -0.766414, 0.642347)
            ),
            "av_book_11": RigidObjectCfg.InitialStateCfg(
                pos=(0.43911, -0.13404, 0.071), rot=(0.0, 0.0, -0.851137, 0.524944)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.53017, 0.00815, 0.071), rot=(0.0, 0.0, -0.074767, 0.997201)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.31812, 0.11521, 0.08776), rot=(0.0, 0.0, 0.472734, 0.881205)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.36002, 0.25523, 0.15103), rot=(0.0, 0.0, 0.824594, 0.565725)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a03")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace134": HeterogeneousTaskCfg(
        instruction="move the spoon into the wooden bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "wooden_bowl",
                "empty_2",
                "empty_3",
                "spaghetti",
                "tomato_soup_can",
                "spoon_2",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(pos=(0.76429, -0.03094, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.41681, -0.08242, 0.07092), rot=(0.0, 0.0, 0.99842, 0.056185)
            ),
            "spaghetti": RigidObjectCfg.InitialStateCfg(
                pos=(0.36878, 0.209, 0.09592), rot=(0.0, 0.0, -0.500766, 0.865583)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.47106, 0.17707, 0.12202), rot=(0.0, 0.0, 0.248775, 0.968561)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.54747, 0.18989, 0.07639), rot=(0.0, 0.0, -0.706147, 0.708066)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_2")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace135": HeterogeneousTaskCfg(
        instruction="put the can of mushrooms and the chess queen in the red bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "empty_5",
                "canned_mushrooms",
                "naturalbostonroundbottle_a03",
                "empty_8",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.43094, -0.02306, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.402, -0.06305, -0.02852), rot=(0.0, 0.0, -0.207436, 0.978249)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.44153, 0.20015, -0.056), rot=(0.0, 0.0, -0.999892, 0.014684)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.47693, 0.10181, -0.03934), rot=(0.0, 0.0, 0.202706, 0.97924)
            ),
            "naturalbostonroundbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.34514, -0.22309, -0.056), rot=(0.0, 0.0, 0.251457, 0.967869)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_mushrooms"), SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace136": HeterogeneousTaskCfg(
        instruction="place the chess rook into the decorative bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_sofatable",
                "av_decorativebowl",
                "av_coaster_hexagon",
                "empty_3",
                "empty_4",
                "canned_peaches",
                "empty_6",
                "av_chess_rook",
                "soft_scrub",
            ]
        ),
        init_states={
            "appleseed_sofatable": RigidObjectCfg.InitialStateCfg(
                pos=(0.43933, 0.25294, -0.69682), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.41247, -0.02346, 0.05804), rot=(0.0, 0.0, -0.993414, 0.114583)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.40204, 0.20435, 0.058), rot=(0.0, 0.0, -0.98288, 0.184245)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.4872, -0.22436, 0.08689), rot=(0.0, 0.0, -0.655197, 0.755458)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.30102, 0.13129, 0.058), rot=(0.0, 0.0, 0.359121, 0.933291)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.54445, 0.20763, 0.18332), rot=(0.0, 0.0, -0.556176, 0.831064)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_rook")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace137": HeterogeneousTaskCfg(
        instruction="place the white bottle upright on top of the photo album",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "bowl",
                "av_photoalbum_closed",
                "empty_3",
                "whitepackerbottle_a04",
                "fork_big",
                "blue_block",
                "av_chess_rook",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.42764, -0.06334, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.37171, 0.12555, -0.00352), rot=(0.0, 0.0, -0.001011, 0.999999)
            ),
            "av_photoalbum_closed": RigidObjectCfg.InitialStateCfg(
                pos=(0.41701, -0.09691, -0.031), rot=(0.0, 0.0, 0.583999, 0.811755)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.51019, 0.24452, -0.031), rot=(0.0, 0.0, 0.01073, 0.999942)
            ),
            "fork_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.26582, -0.11359, -0.03065), rot=(0.0, 0.0, 0.713067, 0.701096)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.32508, 0.25609, -0.0075), rot=(0.0, 0.0, 0.470691, 0.882298)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.54646, -0.21251, -0.031), rot=(0.0, 0.0, 0.975367, 0.220587)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                    "reference": SceneEntityCfg("av_photoalbum_closed"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace138": HeterogeneousTaskCfg(
        instruction="place the can of peaches into the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "bowl_ycb",
                "empty_2",
                "spatula_13",
                "empty_4",
                "canned_peaches",
                "av_mustard_jar",
                "bagel_00",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.64623, 0.59447, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.37239, 0.16088, 0.10448), rot=(0.0, 0.0, 0.983737, 0.179613)
            ),
            "spatula_13": RigidObjectCfg.InitialStateCfg(
                pos=(0.42786, -0.13385, 0.08067), rot=(0.0, 0.0, 0.249506, 0.968373)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.4835, -0.00794, 0.10589), rot=(0.0, 0.0, 0.352947, 0.935643)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.58399, -0.0338, 0.07708), rot=(0.0, 0.0, -0.715405, 0.69871)
            ),
            "bagel_00": RigidObjectCfg.InitialStateCfg(
                pos=(0.32483, -0.21033, 0.09258), rot=(0.0, 0.0, 0.064983, 0.997886)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.56727, 0.09381, 0.16013), rot=(0.0, 0.0, 0.161168, 0.986927)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace139": HeterogeneousTaskCfg(
        instruction="place the lime on top of the clay plate",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "empty_1",
                "clay_plates",
                "empty_3",
                "av_chess_queen",
                "tomato_soup_can",
                "av_mustard_jar",
                "empty_7",
                "lime01",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.66258, -0.39869, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "clay_plates": RigidObjectCfg.InitialStateCfg(
                pos=(0.41348, -0.01916, 0.07083), rot=(0.0, 0.0, -0.65629, 0.754509)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.28801, -0.27089, 0.071), rot=(0.0, 0.0, 0.991211, 0.132287)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.52048, 0.23189, 0.12202), rot=(0.0, 0.0, 0.49005, 0.871694)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.37525, 0.22432, 0.07108), rot=(0.0, 0.0, 0.027764, 0.999615)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4176, -0.22683, 0.07122), rot=(0.0, 0.0, 0.979956, 0.199213)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("lime01")],
                    "reference": SceneEntityCfg("clay_plates"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace140": HeterogeneousTaskCfg(
        instruction="move the pumpkin into the metal bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "ranch_dressing",
                "pumpkinsmall",
                "av_chess_king",
                "av_woodenchmedium",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.43062, 0.02995, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.44166, 0.05536, -0.00075), rot=(0.0, 0.0, 0.998438, 0.055867)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.54102, 0.21227, 0.07272), rot=(0.0, 0.0, 0.019783, 0.999804)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.35487, -0.1662, -0.00126), rot=(0.0, 0.0, -0.98127, 0.192636)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.52394, -0.10089, -0.001), rot=(0.0, 0.0, 0.605031, 0.796202)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.32564, 0.1966, -0.00099), rot=(0.0, 0.0, 0.082196, 0.996616)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("pumpkinsmall")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace141": HeterogeneousTaskCfg(
        instruction="place both the orange can and the red block into the metal bin",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "bin_b04",
                "empty_2",
                "empty_3",
                "empty_4",
                "soup_can",
                "empty_6",
                "dry_erase_marker",
                "red_block",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.4104, 0.00752, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.40289, 0.01023, 0.198), rot=(0.0, 0.0, -0.743459, 0.668781)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30601, -0.24931, 0.19806), rot=(0.0, 0.0, -0.357495, 0.933915)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.39011, 0.23115, 0.20744), rot=(0.0, 0.0, 0.768349, 0.640031)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.50274, -0.25504, 0.2215), rot=(0.0, 0.0, -0.269452, 0.963014)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("soup_can"), SceneEntityCfg("red_block")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace142": HeterogeneousTaskCfg(
        instruction="place the green block into the white bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "green_block",
                "canned_tuna",
                "empty_6",
                "yellow_block",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.43437, -0.04382, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.44289, -0.02583, -0.0312), rot=(0.0, 0.0, 0.909217, 0.416323)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.3064, 0.10713, -0.0075), rot=(0.0, 0.0, -0.776977, 0.629529)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.466, 0.21677, -0.01471), rot=(0.0, 0.0, 0.986247, 0.165276)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.3741, 0.19426, -0.0075), rot=(0.0, 0.0, -0.016056, 0.999871)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("green_block")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace143": HeterogeneousTaskCfg(
        instruction="pick up the blue block and put it in the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "cheez_it",
                "milkjug_a02",
                "blue_block",
                "frozen_vegetable_block",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.40532, -0.28405, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.43019, -0.04669, 0.19796), rot=(0.0, 0.0, 0.168222, 0.985749)
            ),
            "cheez_it": RigidObjectCfg.InitialStateCfg(
                pos=(0.41462, 0.16598, 0.30474), rot=(0.0, 0.0, -0.89188, 0.452272)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.47701, -0.23128, 0.198), rot=(0.0, 0.0, 0.698133, 0.715968)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.29094, -0.24356, 0.2215), rot=(0.0, 0.0, 0.630557, 0.776143)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.28783, -0.11219, 0.198), rot=(0.0, 0.0, -0.991799, 0.127808)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("blue_block")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace144": HeterogeneousTaskCfg(
        instruction="place the spoon on top of the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "empty_1",
                "av_book_33",
                "sugar_box",
                "empty_4",
                "orange_juice_carton",
                "spoon",
                "coffee_pot",
                "empty_8",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.56402, 0.76774, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_33": RigidObjectCfg.InitialStateCfg(
                pos=(0.45364, 0.07767, 0.071), rot=(0.0, 0.0, -0.998613, 0.052644)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.40737, 0.26266, 0.15901), rot=(0.0, 0.0, -0.684557, 0.72896)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.43314, -0.24402, 0.16724), rot=(0.0, 0.0, 0.993394, 0.114751)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.51149, -0.20563, 0.07978), rot=(0.0, 0.0, 0.847746, 0.530402)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.31151, -0.19599, 0.07096), rot=(0.0, 0.0, -0.774332, 0.63278)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("spoon")],
                    "reference": SceneEntityCfg("av_book_33"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace145": HeterogeneousTaskCfg(
        instruction="place the wooden box on top of the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "empty_1",
                "av_book_33",
                "empty_3",
                "ranch_dressing",
                "corn_can",
                "storage_box",
                "empty_7",
                "lime01",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(
                pos=(0.87849, 0.00658, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_33": RigidObjectCfg.InitialStateCfg(
                pos=(0.4491, -0.09127, -0.058), rot=(0.0, 0.0, 0.10664, 0.994298)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.32439, 0.10676, 0.01572), rot=(0.0, 0.0, 0.925993, 0.37754)
            ),
            "corn_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.31044, -0.16532, -0.02898), rot=(0.0, 0.0, 0.085782, 0.996314)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.51827, 0.1592, -0.05813), rot=(0.0, 0.0, 0.170608, 0.985339)
            ),
            "lime01": RigidObjectCfg.InitialStateCfg(
                pos=(0.41552, 0.23898, -0.05778), rot=(0.0, 0.0, -0.879986, 0.475)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("storage_box")],
                    "reference": SceneEntityCfg("av_book_33"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace146": HeterogeneousTaskCfg(
        instruction="place the Rubik's cube on top of the white tray",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "empty_1",
                "av_white_tray",
                "empty_3",
                "av_stawberry",
                "tuna_can",
                "spoon",
                "rubiks_cube",
                "chocolate_pudding",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.55582, -0.14855, -0.69705), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_white_tray": RigidObjectCfg.InitialStateCfg(
                pos=(0.43209, -0.12956, 0.06906), rot=(0.0, 0.0, -0.826687, 0.562663)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.56594, 0.18235, 0.06881), rot=(0.0, 0.0, -0.710917, 0.703276)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.54394, -0.14172, 0.08576), rot=(0.0, 0.0, -0.21088, 0.977512)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.45692, 0.18719, 0.07778), rot=(0.0, 0.0, -0.965303, 0.261131)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.35385, 0.2031, 0.10039), rot=(0.0, 0.0, -0.302307, 0.953211)
            ),
            "chocolate_pudding": RigidObjectCfg.InitialStateCfg(
                pos=(0.33376, 0.06828, 0.08841), rot=(0.0, 0.0, 0.756666, 0.653802)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("rubiks_cube")],
                    "reference": SceneEntityCfg("av_white_tray"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace147": HeterogeneousTaskCfg(
        instruction="put the mustard bottle in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "bin_a02",
                "av_book_33",
                "empty_3",
                "av_stawberry",
                "empty_5",
                "storage_box",
                "naturalbostonroundbottle_a02",
                "mustard_bottle",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.71731, -0.68065, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a02": RigidObjectCfg.InitialStateCfg(pos=(0.49565, 0.04339, 0.129), rot=(0.0, 0.0, -0.20314, 0.97915)),
            "av_book_33": RigidObjectCfg.InitialStateCfg(
                pos=(0.44371, -0.15045, 0.129), rot=(0.0, 0.0, 0.574966, 0.818177)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.27594, -0.05888, 0.12881), rot=(0.0, 0.0, 0.33254, 0.943089)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.39622, 0.2088, 0.12887), rot=(0.0, 0.0, 0.916299, 0.400496)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.3094, 0.09166, 0.129), rot=(0.0, 0.0, -0.242154, 0.970238)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.53487, 0.23497, 0.20903), rot=(0.0, 0.0, 0.844842, 0.535016)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("mustard_bottle")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace148": HeterogeneousTaskCfg(
        instruction="place the orange can into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "bin_a03",
                "empty_2",
                "empty_3",
                "glasses",
                "soup_can",
                "spoon_2",
                "frozen_vegetable_block",
                "pitted_cherries",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.51564, -0.18019, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a03": RigidObjectCfg.InitialStateCfg(pos=(0.49401, 0.05482, 0.009), rot=(0.0, 0.0, 0.837733, 0.54608)),
            "glasses": RigidObjectCfg.InitialStateCfg(
                pos=(0.40512, 0.20443, 0.03947), rot=(0.0, 0.0, 0.747495, 0.664267)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.35786, -0.25036, 0.00906), rot=(0.0, 0.0, -0.989184, 0.146682)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.45467, -0.24567, 0.01439), rot=(0.0, 0.0, 0.930277, 0.366858)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.33116, -0.05125, 0.009), rot=(0.0, 0.0, -0.603121, 0.797649)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.49784, -0.14852, 0.03811), rot=(0.0, 0.0, -0.958076, 0.286515)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("soup_can")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace149": HeterogeneousTaskCfg(
        instruction="put the white bottle in the metal bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "serving_bowl",
                "empty_2",
                "crabbypenholder",
                "whitepackerbottle_a04",
                "tuna_can",
                "empty_6",
                "coffee_pot",
                "coffee_can",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.46728, 0.05802, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.466, 0.11493, -0.03075), rot=(0.0, 0.0, 0.617641, 0.78646)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.33906, 0.19534, -0.03103), rot=(0.0, 0.0, -0.984144, 0.177373)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.44421, -0.20283, -0.031), rot=(0.0, 0.0, -0.882614, 0.470098)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30497, -0.17154, -0.01424), rot=(0.0, 0.0, 0.695474, 0.718551)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.35042, -0.03522, -0.03104), rot=(0.0, 0.0, -0.363974, 0.931409)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.54194, -0.0626, 0.03907), rot=(0.0, 0.0, 0.809075, 0.587706)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace150": HeterogeneousTaskCfg(
        instruction="remove the mustard bottle from the metal bin",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "bin_b04",
                "empty_2",
                "empty_3",
                "orange_02",
                "av_hippoorganizer",
                "milk_carton",
                "empty_7",
                "mustard",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.51377, -0.25512, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.43219, -0.09908, 0.007), rot=(0.0, 0.0, -0.885638, 0.464377)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.32619, 0.09167, 0.00706), rot=(0.0, 0.0, -0.692012, 0.721886)
            ),
            "av_hippoorganizer": RigidObjectCfg.InitialStateCfg(
                pos=(0.50689, 0.20959, 0.007), rot=(0.0, 0.0, -0.99993, 0.011817)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.32762, 0.20107, 0.10218), rot=(0.0, 0.0, -0.939136, 0.343546)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.43219, -0.09908, 0.1167), rot=(0.0, 0.0, 0.238873, 0.971051)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("mustard")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace151": HeterogeneousTaskCfg(
        instruction="stack the Rubik's cube on the sugar box",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "empty_1",
                "empty_2",
                "sugar_box",
                "cubebox_a02",
                "canned_peaches",
                "av_mustard_jar",
                "rubiks_cube",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.504, 0.0634, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.53557, 0.07638, 0.09501), rot=(0.0, 0.0, -0.989717, 0.143036)
            ),
            "cubebox_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.34169, 0.11799, 0.007), rot=(0.0, 0.0, 0.998539, 0.054027)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.30449, -0.05502, 0.03589), rot=(0.0, 0.0, 0.946911, 0.321497)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.36026, -0.18589, 0.00708), rot=(0.0, 0.0, 0.827939, 0.560818)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.49208, -0.12128, 0.03839), rot=(0.0, 0.0, 0.677713, 0.735327)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("rubiks_cube")],
                    "reference": SceneEntityCfg("sugar_box"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace152": HeterogeneousTaskCfg(
        instruction="pick up the chess queen and put it in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "bin_b04",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "av_p_glassware_tall",
                "empty_6",
                "green_beans_can",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.41709, -0.03862, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.42853, -0.02403, -0.031), rot=(0.0, 0.0, 0.481274, 0.87657)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.31396, 0.1359, -0.031), rot=(0.0, 0.0, -0.998952, 0.04578)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.3027, -0.25368, -0.031), rot=(0.0, 0.0, -0.682976, 0.730441)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.48921, -0.2504, -0.00221), rot=(0.0, 0.0, -0.630846, 0.775908)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace153": HeterogeneousTaskCfg(
        instruction="put the phone on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "empty_1",
                "av_book_21",
                "empty_3",
                "whitepackerbottle_a02",
                "smartphone",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.52583, 0.43409, -0.69646), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_book_21": RigidObjectCfg.InitialStateCfg(
                pos=(0.41851, -0.10044, 0.09), rot=(0.0, 0.0, -0.647684, 0.761909)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.54377, 0.21952, 0.09), rot=(0.0, 0.0, 0.021901, 0.99976)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.4027, 0.18813, 0.09565), rot=(0.0, 0.0, 0.686326, 0.727294)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("smartphone")],
                    "reference": SceneEntityCfg("av_book_21"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace154": HeterogeneousTaskCfg(
        instruction="put the white bottle in the white bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "av_bowl_plate",
                "empty_2",
                "spatula_06",
                "whitepackerbottle_a03",
                "empty_5",
                "empty_6",
                "av_woodenchsmall",
                "brick",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.50543, 0.15732, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.40127, -0.09698, 0.0088), rot=(0.0, 0.0, 0.08179, 0.99665)
            ),
            "spatula_06": RigidObjectCfg.InitialStateCfg(
                pos=(0.42131, 0.10735, 0.01271), rot=(0.0, 0.0, -0.0467, 0.998909)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.32406, 0.22914, 0.009), rot=(0.0, 0.0, 0.296532, 0.955023)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.57744, -0.07105, 0.009), rot=(0.0, 0.0, 0.217819, 0.975989)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.44568, 0.23002, 0.03457), rot=(0.0, 0.0, 0.638195, 0.769875)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a03")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace155": HeterogeneousTaskCfg(
        instruction="place the mustard jar into the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "fork_big",
                "av_mustard_jar",
                "bagel_00",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.6364, 0.11409, -0.6376), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.45667, 0.0244, 0.15606), rot=(0.0, 0.0, -0.94134, 0.337458)
            ),
            "fork_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.46854, -0.17755, 0.15635), rot=(0.0, 0.0, 0.99789, 0.06493)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.50764, -0.11162, 0.15608), rot=(0.0, 0.0, 0.195035, 0.980796)
            ),
            "bagel_00": RigidObjectCfg.InitialStateCfg(
                pos=(0.32002, -0.08732, 0.17158), rot=(0.0, 0.0, -0.620235, 0.784416)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_mustard_jar")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace156": HeterogeneousTaskCfg(
        instruction="move the cookie box into the pan",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "av_pan_d",
                "empty_2",
                "empty_3",
                "oatmeal_raisin_cookies",
                "empty_5",
                "wood_block",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.62451, 0.14376, -0.6376), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.44507, -0.00649, 0.15626), rot=(0.0, 0.0, -0.996384, 0.084959)
            ),
            "oatmeal_raisin_cookies": RigidObjectCfg.InitialStateCfg(
                pos=(0.29594, -0.2266, 0.23962), rot=(0.0, 0.0, -0.982152, 0.188089)
            ),
            "wood_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41504, -0.21151, 0.25894), rot=(0.0, 0.0, 0.753835, 0.657063)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("oatmeal_raisin_cookies")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace157": HeterogeneousTaskCfg(
        instruction="place the orange into the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "orange_02",
                "raisin_box",
                "peas_and_carrots",
                "av_eraser",
                "red_block",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(
                pos=(0.76521, 1.05568, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.46422, -0.10036, 0.10896), rot=(0.0, 0.0, -0.441817, 0.897105)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.41277, 0.17805, 0.10906), rot=(0.0, 0.0, -0.490685, 0.871337)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.55509, 0.10256, 0.17059), rot=(0.0, 0.0, -0.22817, 0.973621)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.36582, -0.23651, 0.13826), rot=(0.0, 0.0, -0.726154, 0.687532)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.29553, -0.02393, 0.11333), rot=(0.0, 0.0, -0.133994, 0.990982)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.29468, 0.08219, 0.1325), rot=(0.0, 0.0, 0.405321, 0.914174)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_02")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace158": HeterogeneousTaskCfg(
        instruction="move the orange juice carton into the pan",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "av_pan_b",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a01",
                "orange_juice_carton",
                "empty_6",
                "yellow_block",
                "red_block",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(
                pos=(0.71526, -0.00197, -0.347), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_pan_b": RigidObjectCfg.InitialStateCfg(
                pos=(0.40713, -0.12448, 0.00888), rot=(0.0, 0.0, -0.999998, 0.001819)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.3286, 0.26095, 0.009), rot=(0.0, 0.0, 0.178365, 0.983964)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.43454, 0.24251, 0.10524), rot=(0.0, 0.0, 0.189675, 0.981847)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.58991, 0.01649, 0.0325), rot=(0.0, 0.0, -0.999982, 0.006023)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.54048, 0.22774, 0.0325), rot=(0.0, 0.0, -0.999476, 0.032367)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_juice_carton")],
                    "container": SceneEntityCfg("av_pan_b"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace159": HeterogeneousTaskCfg(
        instruction="pick up the parmesan cheese and put it in the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "blackandbrassbowl_small",
                "empty_2",
                "spatula_03",
                "parmesan_cheese_canister",
                "milkjug_a02",
                "blue_block",
                "mayonnaise_bottle",
                "coffee_can",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(pos=(0.76049, 0.07402, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.36158, -0.05048, 0.07096), rot=(0.0, 0.0, -0.740806, 0.671719)
            ),
            "spatula_03": RigidObjectCfg.InitialStateCfg(
                pos=(0.5551, -0.01615, 0.07773), rot=(0.0, 0.0, 0.687093, 0.72657)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.46674, 0.11538, 0.12243), rot=(0.0, 0.0, -0.078567, 0.996909)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.49437, -0.21871, 0.071), rot=(0.0, 0.0, -0.26999, 0.962863)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.30959, 0.115, 0.0945), rot=(0.0, 0.0, -0.967739, 0.251956)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.47644, 0.22696, 0.14495), rot=(0.0, 0.0, 0.873395, 0.487012)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.37221, 0.23915, 0.14107), rot=(0.0, 0.0, 0.655594, 0.755114)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("parmesan_cheese_canister")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace160": HeterogeneousTaskCfg(
        instruction="put the tomato soup can upright on the book",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "empty_1",
                "av_book_66",
                "empty_3",
                "empty_4",
                "tomato_soup_can",
                "empty_6",
                "dry_erase_marker",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.24043, -0.11306, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_book_66": RigidObjectCfg.InitialStateCfg(
                pos=(0.44706, 0.0399, 0.156), rot=(0.0, 0.0, 0.644915, 0.764254)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.39453, -0.12339, 0.20702), rot=(0.0, 0.0, -0.721213, 0.692714)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.53457, 0.17213, 0.16544), rot=(0.0, 0.0, -0.80869, 0.588235)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.52984, -0.11446, 0.15606), rot=(0.0, 0.0, -0.07787, 0.996963)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("tomato_soup_can")],
                    "reference": SceneEntityCfg("av_book_66"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace161": HeterogeneousTaskCfg(
        instruction="put the crab pen holder in the decorative bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "av_decorativebowl",
                "empty_2",
                "crabbypenholder",
                "whitepackerbottle_a01",
                "empty_5",
                "av_apple",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.56513, -0.03312, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.41456, 0.05956, 0.00904), rot=(0.0, 0.0, 0.992954, 0.1185)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.46556, -0.24104, 0.00897), rot=(0.0, 0.0, 0.984241, 0.176832)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.40284, 0.25599, 0.009), rot=(0.0, 0.0, -0.409538, 0.912293)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.4002, -0.17193, 0.00996), rot=(0.0, 0.0, -0.459313, 0.888275)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.32303, -0.23016, 0.05042), rot=(0.0, 0.0, -0.941257, 0.337691)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("crabbypenholder")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace162": HeterogeneousTaskCfg(
        instruction="place the milk jug into the wooden bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "wooden_bowl",
                "empty_2",
                "av_kiwi01",
                "empty_4",
                "milkjug_a01",
                "empty_6",
                "rubiks_cube",
                "empty_8",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.44584, -0.01395, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.41362, -0.05534, -0.05608), rot=(0.0, 0.0, -0.717352, 0.696711)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.30805, 0.22313, -0.05601), rot=(0.0, 0.0, -0.999782, 0.020877)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.5105, 0.17377, -0.056), rot=(0.0, 0.0, -0.930167, 0.367136)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.31552, 0.16904, -0.02461), rot=(0.0, 0.0, -0.968168, 0.2503)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("milkjug_a01")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace163": HeterogeneousTaskCfg(
        instruction="place the macaroni and cheese box on top of the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_sofatable",
                "empty_1",
                "av_book_01",
                "frozen_waffles",
                "whitepackerbottle_a01",
                "milkjug_a01",
                "empty_6",
                "av_eraser",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "appleseed_sofatable": RigidObjectCfg.InitialStateCfg(
                pos=(0.4419, -0.2432, -0.69682), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_book_01": RigidObjectCfg.InitialStateCfg(
                pos=(0.41171, 0.09989, 0.058), rot=(0.0, 0.0, 0.993763, 0.111515)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.4256, -0.24699, 0.05798), rot=(0.0, 0.0, 0.688558, 0.725182)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.44434, -0.14169, 0.058), rot=(0.0, 0.0, -0.959938, 0.280214)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29482, -0.1456, 0.058), rot=(0.0, 0.0, 0.999141, 0.041451)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.2888, -0.23813, 0.06233), rot=(0.0, 0.0, -0.182313, 0.98324)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.53096, -0.15144, 0.14113), rot=(0.0, 0.0, -0.459394, 0.888233)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("macaroni_and_cheese")],
                    "reference": SceneEntityCfg("av_book_01"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace164": HeterogeneousTaskCfg(
        instruction="take the marker out of the white bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "av_bowl_plate",
                "empty_2",
                "spatula_18",
                "spam_can",
                "empty_5",
                "av_p_glassware_short",
                "dry_erase_marker",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.43923, -0.34899, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.38329, 0.01646, 0.1088), rot=(0.0, 0.0, -0.997277, 0.073745)
            ),
            "spatula_18": RigidObjectCfg.InitialStateCfg(
                pos=(0.43335, -0.20931, 0.11268), rot=(0.0, 0.0, 0.137231, 0.990539)
            ),
            "spam_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.31245, 0.22972, 0.15082), rot=(0.0, 0.0, -0.722717, 0.691144)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.43226, 0.22029, 0.109), rot=(0.0, 0.0, -0.851298, 0.524683)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.38343, 0.01648, 0.13244), rot=(0.0, 0.0, -0.330835, 0.943689)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.56324, 0.08439, 0.19213), rot=(0.0, 0.0, 0.158781, 0.987314)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("dry_erase_marker")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace165": HeterogeneousTaskCfg(
        instruction="move the marker into the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "blackandbrassbowl_small",
                "av_coaster_hexagon",
                "empty_3",
                "cheez_it",
                "jello",
                "empty_6",
                "dry_erase_marker",
                "empty_8",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.58561, 0.28758, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.42055, 0.16548, 0.07696), rot=(0.0, 0.0, -0.893484, 0.449094)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.35286, -0.11828, 0.077), rot=(0.0, 0.0, -0.926121, 0.377226)
            ),
            "cheez_it": RigidObjectCfg.InitialStateCfg(
                pos=(0.48848, -0.20695, 0.18374), rot=(0.0, 0.0, 0.285557, 0.958362)
            ),
            "jello": RigidObjectCfg.InitialStateCfg(
                pos=(0.54733, -0.02844, 0.09206), rot=(0.0, 0.0, -0.401444, 0.915883)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.41627, -0.03145, 0.08644), rot=(0.0, 0.0, 0.256999, 0.966412)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("dry_erase_marker")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace166": HeterogeneousTaskCfg(
        instruction="set the spoon on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "empty_1",
                "av_book_99",
                "spatula_18",
                "fstylejug_a02",
                "mug",
                "spoon",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.69355, -0.41182, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_book_99": RigidObjectCfg.InitialStateCfg(
                pos=(0.36283, -0.16156, 0.071), rot=(0.0, 0.0, 0.002104, 0.999998)
            ),
            "spatula_18": RigidObjectCfg.InitialStateCfg(
                pos=(0.40111, 0.09367, 0.07468), rot=(0.0, 0.0, 0.951209, 0.308546)
            ),
            "fstylejug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.53986, -0.10117, 0.071), rot=(0.0, 0.0, 0.997814, 0.06609)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.41202, 0.21883, 0.07118), rot=(0.0, 0.0, -0.787139, 0.616775)),
            "spoon": RigidObjectCfg.InitialStateCfg(pos=(0.50604, 0.1955, 0.07978), rot=(0.0, 0.0, 0.67633, 0.736599)),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("spoon")],
                    "reference": SceneEntityCfg("av_book_99"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace167": HeterogeneousTaskCfg(
        instruction="set the tomato soup can upright on the coaster",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "bin_a03",
                "av_coaster_hexagon",
                "crabbypenholder",
                "fstylejug_a02",
                "tomato_soup_can",
                "empty_6",
                "naturalbostonroundbottle_a02",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.47268, -0.03223, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.52198, 0.14158, -0.001), rot=(0.0, 0.0, 0.69833, 0.715776)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.36072, -0.19092, -0.001), rot=(0.0, 0.0, -0.657824, 0.753172)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.39411, -0.03657, -0.00103), rot=(0.0, 0.0, -0.915034, 0.403377)
            ),
            "fstylejug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.51634, -0.01718, -0.001), rot=(0.0, 0.0, -0.678832, 0.734294)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.49902, -0.16945, 0.05002), rot=(0.0, 0.0, 0.338149, 0.941092)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.37546, 0.08238, -0.001), rot=(0.0, 0.0, -0.395821, 0.918328)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.28712, 0.00738, 0.07316), rot=(0.0, 0.0, -0.988349, 0.152205)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("tomato_soup_can")],
                    "reference": SceneEntityCfg("av_coaster_hexagon"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace168": HeterogeneousTaskCfg(
        instruction="put the can of tuna in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "bin_a03",
                "empty_2",
                "spatula_03",
                "yogurt_cup",
                "canned_tuna",
                "measuring_cups_1",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.40611, 0.26655, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.38024, 0.13968, 0.198), rot=(0.0, 0.0, 0.793052, 0.609154)
            ),
            "spatula_03": RigidObjectCfg.InitialStateCfg(
                pos=(0.30965, -0.14097, 0.20473), rot=(0.0, 0.0, -0.777502, 0.62888)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.50371, 0.15725, 0.22484), rot=(0.0, 0.0, -0.557678, 0.830057)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.4549, -0.00909, 0.21429), rot=(0.0, 0.0, -0.414506, 0.910047)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.44899, -0.20287, 0.22403), rot=(0.0, 0.0, 0.766151, 0.64266)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_tuna")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace169": HeterogeneousTaskCfg(
        instruction="put a chess piece in the metal bin",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_a05",
                "empty_2",
                "av_kiwi01",
                "av_chess_pawn",
                "empty_5",
                "av_chess_bishop",
                "tomato_sauce_can",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41264, -0.19095, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.42961, 0.08486, 0.069), rot=(0.0, 0.0, -0.562536, 0.826773)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.39355, -0.26022, 0.06899), rot=(0.0, 0.0, 0.998405, 0.05646)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.44288, -0.14442, 0.069), rot=(0.0, 0.0, -0.984048, 0.177902)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.47694, -0.21928, 0.069), rot=(0.0, 0.0, 0.955219, 0.295899)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30149, -0.12374, 0.11042), rot=(0.0, 0.0, 0.99937, 0.035496)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.53821, -0.14886, 0.15213), rot=(0.0, 0.0, 0.077179, 0.997017)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_chess_pawn")],
                    "container": SceneEntityCfg("bin_a05"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace170": HeterogeneousTaskCfg(
        instruction="first put the chocolate pudding mix in the red bowl, then put the can of corn in it",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "gilbert",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "fstylejug_a01",
                "corn_can",
                "spoon",
                "chocolate_pudding_mix",
                "mustard",
            ]
        ),
        init_states={
            "gilbert": RigidObjectCfg.InitialStateCfg(
                pos=(0.78623, -0.08952, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.41571, -0.15894, 0.14648), rot=(0.0, 0.0, 0.855088, 0.518483)
            ),
            "fstylejug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.3057, 0.04982, 0.119), rot=(0.0, 0.0, -0.978385, 0.20679)
            ),
            "corn_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.57053, 0.05538, 0.14802), rot=(0.0, 0.0, 0.675905, 0.736989)
            ),
            "spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.45087, 0.05932, 0.12778), rot=(0.0, 0.0, 0.498237, 0.867041)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.3857, 0.25459, 0.14374), rot=(0.0, 0.0, -0.901011, 0.433797)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.38074, 0.15673, 0.2147), rot=(0.0, 0.0, -0.87201, 0.489488)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix"), SceneEntityCfg("corn_can")],
                    "container": SceneEntityCfg("bowl_ycb"),
                },
            ),
        },
    ),
    "PickPlace171": HeterogeneousTaskCfg(
        instruction="put the raisin box on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "bin_a02",
                "av_book_21",
                "empty_3",
                "whitepackerbottle_a03",
                "raisin_box",
                "av_knife",
                "empty_7",
                "potato_masher",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.59613, 0.60145, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(pos=(0.367, 0.2241, 0.071), rot=(0.0, 0.0, -0.98908, 0.147379)),
            "av_book_21": RigidObjectCfg.InitialStateCfg(
                pos=(0.4353, -0.03893, 0.071), rot=(0.0, 0.0, 0.790785, 0.612094)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.28981, -0.2323, 0.071), rot=(0.0, 0.0, 0.993907, 0.11022)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.46803, -0.23466, 0.13259), rot=(0.0, 0.0, 0.859247, 0.511562)
            ),
            "av_knife": RigidObjectCfg.InitialStateCfg(
                pos=(0.43327, 0.11361, 0.07206), rot=(0.0, 0.0, 0.779801, 0.626028)
            ),
            "potato_masher": RigidObjectCfg.InitialStateCfg(
                pos=(0.5149, 0.20924, 0.07142), rot=(0.0, 0.0, -0.61483, 0.78866)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("raisin_box")],
                    "reference": SceneEntityCfg("av_book_21"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace172": HeterogeneousTaskCfg(
        instruction="put the chess bishop and the chess pawn in the black bowl",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "granola_bars",
                "av_chess_bishop",
                "empty_7",
                "coffee_can",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(
                pos=(0.87751, 0.06047, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.46007, -0.05064, -0.05804), rot=(0.0, 0.0, 0.69016, 0.723657)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.35383, 0.16974, -0.058), rot=(0.0, 0.0, 0.1628, 0.986659)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.28787, -0.10958, 0.004), rot=(0.0, 0.0, 0.018505, 0.999829)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.45958, 0.08514, -0.058), rot=(0.0, 0.0, -0.945947, 0.324321)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.48286, 0.19733, 0.01207), rot=(0.0, 0.0, 0.999574, 0.02918)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_chess_pawn")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace173": HeterogeneousTaskCfg(
        instruction="first put the yellow block in the decorative bowl, then put the pumpkin in it",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "av_decorativebowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "pumpkinsmall",
                "empty_6",
                "yellow_block",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.57505, -0.21137, -0.69705), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.44586, -0.04814, 0.06904), rot=(0.0, 0.0, 0.685644, 0.727937)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.39827, 0.17947, 0.06874), rot=(0.0, 0.0, -0.311019, 0.950404)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41664, -0.22915, 0.0925), rot=(0.0, 0.0, 0.954842, 0.297114)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.54983, 0.18459, 0.06906), rot=(0.0, 0.0, -0.385369, 0.922762)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("yellow_block"), SceneEntityCfg("pumpkinsmall")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                },
            ),
        },
    ),
    "PickPlace174": HeterogeneousTaskCfg(
        instruction="set the avocado on the photo album",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "av_salt_bowl",
                "av_photoalbum_closed",
                "av_kiwi01",
                "av_chess_pawn",
                "pineapple_slices_can",
                "spoon_big",
                "empty_7",
                "avocado01",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(
                pos=(1.37566, -0.07556, -0.6971), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.46607, 0.18391, 0.12906), rot=(0.0, 0.0, 0.968851, 0.247644)
            ),
            "av_photoalbum_closed": RigidObjectCfg.InitialStateCfg(
                pos=(0.44151, -0.03953, 0.129), rot=(0.0, 0.0, -0.999915, 0.013022)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.28828, 0.04757, 0.12899), rot=(0.0, 0.0, -0.014663, 0.999892)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.28963, -0.09085, 0.129), rot=(0.0, 0.0, 0.032566, 0.99947)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.39519, -0.20605, 0.15781), rot=(0.0, 0.0, 0.846344, 0.532637)
            ),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.576, -0.05672, 0.12924), rot=(0.0, 0.0, -0.721567, 0.692345)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.29975, 0.15349, 0.12883), rot=(0.0, 0.0, -0.338997, 0.940788)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("avocado01")],
                    "reference": SceneEntityCfg("av_photoalbum_closed"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace175": HeterogeneousTaskCfg(
        instruction="place the plastic bottle into the white bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "mug",
                "av_p_glassware_short",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.5366, -0.01462, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.42024, -0.09842, -0.0312), rot=(0.0, 0.0, -0.986315, 0.164874)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.5616, 0.07928, -0.031), rot=(0.0, 0.0, -0.503671, 0.863895)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.34407, 0.22594, -0.03082), rot=(0.0, 0.0, 0.14902, 0.988834)),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.36409, 0.11442, -0.031), rot=(0.0, 0.0, -0.503157, 0.864195)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4602, 0.19392, -0.031), rot=(0.0, 0.0, -0.998149, 0.060815)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace176": HeterogeneousTaskCfg(
        instruction="pick up the frozen vegetables and put it in the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "empty_4",
                "canned_peaches",
                "av_p_glassware_short",
                "frozen_vegetable_block",
                "chocolate_pudding",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.53624, -0.08208, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.38167, -0.13701, -0.03104), rot=(0.0, 0.0, 0.669137, 0.743139)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.53546, -0.1871, -0.00211), rot=(0.0, 0.0, -0.050124, 0.998743)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.49101, 0.22244, -0.031), rot=(0.0, 0.0, 0.980546, 0.19629)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.55731, 0.01732, -0.031), rot=(0.0, 0.0, -0.789226, 0.614103)
            ),
            "chocolate_pudding": RigidObjectCfg.InitialStateCfg(
                pos=(0.37263, 0.06104, -0.01159), rot=(0.0, 0.0, 0.987653, 0.15666)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("frozen_vegetable_block")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace177": HeterogeneousTaskCfg(
        instruction="place both the yogurt cup and the mustard bottle into the red bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "yogurt_cup",
                "milkjug_a02",
                "canned_mushrooms",
                "naturalbostonroundbottle_a03",
                "mustard_bottle",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.49383, -0.04208, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.5007, 0.01951, 0.03448), rot=(0.0, 0.0, -0.992608, 0.121368)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.42019, -0.13196, 0.03384), rot=(0.0, 0.0, 0.389686, 0.920948)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.52909, -0.17414, 0.007), rot=(0.0, 0.0, 0.563584, 0.826059)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.35666, -0.23022, 0.02366), rot=(0.0, 0.0, 0.443551, 0.896249)
            ),
            "naturalbostonroundbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.30774, 0.12221, 0.007), rot=(0.0, 0.0, -0.153965, 0.988076)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.37175, 0.22365, 0.08703), rot=(0.0, 0.0, -0.121475, 0.992594)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("yogurt_cup"), SceneEntityCfg("mustard_bottle")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace178": HeterogeneousTaskCfg(
        instruction="remove the butter from the black bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "blackandbrassbowl_small",
                "empty_2",
                "spatula_19",
                "butter",
                "tuna_can",
                "milk_carton",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.45698, -0.01005, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.39468, 0.12717, 0.00696), rot=(0.0, 0.0, -0.206379, 0.978472)
            ),
            "spatula_19": RigidObjectCfg.InitialStateCfg(
                pos=(0.40499, -0.1162, 0.01068), rot=(0.0, 0.0, 0.960358, 0.278771)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.39468, 0.12717, 0.04741), rot=(0.0, 0.0, 0.64281, 0.766026)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.56392, -0.05654, 0.02376), rot=(0.0, 0.0, 0.646238, 0.763136)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.54583, 0.06732, 0.10218), rot=(0.0, 0.0, -0.939336, 0.342998)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30957, -0.20674, 0.04842), rot=(0.0, 0.0, -0.923842, 0.382774)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("butter")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace179": HeterogeneousTaskCfg(
        instruction="take the strawberry out of the red bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "bowl",
                "empty_2",
                "empty_3",
                "av_stawberry",
                "mug",
                "measuring_cups_1",
                "bagel_00",
                "mustard",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.4156, -0.07807, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.37922, 0.06242, -0.00352), rot=(0.0, 0.0, 0.934933, 0.354825)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.37922, 0.06242, -0.01719), rot=(0.0, 0.0, -0.043481, 0.999054)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(
                pos=(0.51623, -0.06317, -0.03082), rot=(0.0, 0.0, 0.077316, 0.997007)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.37437, -0.18136, -0.00497), rot=(0.0, 0.0, -0.148266, 0.988947)
            ),
            "bagel_00": RigidObjectCfg.InitialStateCfg(
                pos=(0.48682, 0.19809, -0.01542), rot=(0.0, 0.0, -0.957643, 0.287958)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.35893, 0.2557, 0.0647), rot=(0.0, 0.0, 0.996255, 0.086458)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("av_stawberry")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace180": HeterogeneousTaskCfg(
        instruction="put the chess queen in the metal bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "empty_5",
                "av_knife",
                "green_beans_can",
                "dumbbell",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.45692, 0.52505, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.45301, -0.17118, 0.09025), rot=(0.0, 0.0, -0.766799, 0.641887)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.43068, 0.21701, 0.09), rot=(0.0, 0.0, -0.663878, 0.747841)
            ),
            "av_knife": RigidObjectCfg.InitialStateCfg(
                pos=(0.30202, -0.16563, 0.09106), rot=(0.0, 0.0, 0.170162, 0.985416)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.32908, 0.10256, 0.11879), rot=(0.0, 0.0, -0.908113, 0.418725)
            ),
            "dumbbell": RigidObjectCfg.InitialStateCfg(
                pos=(0.40311, -0.00868, 0.12722), rot=(0.0, 0.0, -0.536387, 0.843972)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace181": HeterogeneousTaskCfg(
        instruction="pick the white bottle out of the wooden bowl and put it down",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "gilbert",
                "wooden_bowl",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "canned_tuna",
                "av_p_glassware_short",
                "av_woodenchmedium",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "gilbert": RigidObjectCfg.InitialStateCfg(
                pos=(0.76538, 0.00768, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.40505, 0.0375, 0.11892), rot=(0.0, 0.0, 0.705683, 0.708528)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.40692, 0.03853, 0.133), rot=(0.0, 0.0, 0.758349, 0.651849)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.50571, -0.20421, 0.13529), rot=(0.0, 0.0, 0.520544, 0.853835)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.41993, -0.17423, 0.119), rot=(0.0, 0.0, -0.841249, 0.540648)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.37395, 0.2411, 0.11901), rot=(0.0, 0.0, -0.320001, 0.947417)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.33704, -0.2431, 0.19316), rot=(0.0, 0.0, 0.959541, 0.281568)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a02")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace182": HeterogeneousTaskCfg(
        instruction="pick up the can of peaches and put it in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "bin_a02",
                "empty_2",
                "empty_3",
                "fstylejug_a01",
                "canned_peaches",
                "wood_block",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.23901, -0.12223, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.43021, -0.1122, 0.156), rot=(0.0, 0.0, -0.775089, 0.631852)
            ),
            "fstylejug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.52558, 0.09945, 0.156), rot=(0.0, 0.0, 0.295675, 0.955288)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.46438, 0.02598, 0.18489), rot=(0.0, 0.0, -0.987323, 0.158726)
            ),
            "wood_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.33403, 0.05642, 0.25894), rot=(0.0, 0.0, -0.983704, 0.179794)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace183": HeterogeneousTaskCfg(
        instruction="put the butter on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "empty_1",
                "av_book_99",
                "empty_3",
                "butter",
                "av_p_glassware_tall",
                "av_chess_bishop",
                "av_eraser",
                "mustard_bottle",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.22844, -0.19154, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_book_99": RigidObjectCfg.InitialStateCfg(
                pos=(0.42785, 0.13376, 0.156), rot=(0.0, 0.0, 0.987548, 0.157318)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.47166, -0.18217, 0.18241), rot=(0.0, 0.0, 0.48189, 0.876232)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.33691, -0.10665, 0.156), rot=(0.0, 0.0, -0.991886, 0.127132)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.29944, 0.20424, 0.156), rot=(0.0, 0.0, -0.033683, 0.999433)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.39259, -0.19287, 0.16033), rot=(0.0, 0.0, 0.823914, 0.566715)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.50148, -0.27141, 0.23603), rot=(0.0, 0.0, -0.679686, 0.733503)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("butter")],
                    "reference": SceneEntityCfg("av_book_99"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace184": HeterogeneousTaskCfg(
        instruction="move the white bottle into the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "canned_tuna",
                "empty_6",
                "bagel_00",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(pos=(2.33592, 0.09385, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.42849, 0.07219, 0.10894), rot=(0.0, 0.0, -0.634674, 0.77278)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.53542, -0.23432, 0.109), rot=(0.0, 0.0, -0.117145, 0.993115)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.5396, -0.1247, 0.12529), rot=(0.0, 0.0, -0.467228, 0.884137)
            ),
            "bagel_00": RigidObjectCfg.InitialStateCfg(
                pos=(0.3661, -0.20066, 0.12458), rot=(0.0, 0.0, -0.980917, 0.194426)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.29622, 0.24394, 0.10906), rot=(0.0, 0.0, -0.986595, 0.163186)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a03")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace185": HeterogeneousTaskCfg(
        instruction="pick up the raisin box and put it in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "bin_b04",
                "empty_2",
                "empty_3",
                "empty_4",
                "raisin_box",
                "blue_block",
                "empty_7",
                "av_tincan",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.51415, -0.1568, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.38529, 0.04575, 0.007), rot=(0.0, 0.0, 0.743142, 0.669134)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.40419, -0.21123, 0.06859), rot=(0.0, 0.0, -0.215848, 0.976427)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.48527, -0.24591, 0.0305), rot=(0.0, 0.0, -0.612206, 0.790698)
            ),
            "av_tincan": RigidObjectCfg.InitialStateCfg(
                pos=(0.55945, 0.03019, 0.007), rot=(0.0, 0.0, 0.246362, 0.969178)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("raisin_box")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace186": HeterogeneousTaskCfg(
        instruction="set the tomato soup can upright on the coaster",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "empty_1",
                "av_coaster_hexagon",
                "empty_3",
                "empty_4",
                "tomato_soup_can",
                "wood_block",
                "naturalbostonroundbottle_a01",
                "birdhouse",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.55779, 0.03555, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.40038, -0.10384, 0.009), rot=(0.0, 0.0, -0.324421, 0.945913)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.40642, 0.24502, 0.06002), rot=(0.0, 0.0, 0.933288, 0.359129)
            ),
            "wood_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.30954, 0.15442, 0.11194), rot=(0.0, 0.0, -0.731918, 0.681393)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.5126, -0.23015, 0.009), rot=(0.0, 0.0, -0.108774, 0.994067)
            ),
            "birdhouse": RigidObjectCfg.InitialStateCfg(
                pos=(0.45882, 0.0763, 0.00905), rot=(0.0, 0.0, 0.965926, 0.258817)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("tomato_soup_can")],
                    "reference": SceneEntityCfg("av_coaster_hexagon"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace187": HeterogeneousTaskCfg(
        instruction="place the kiwi on the popcorn box",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "bin_a02",
                "av_coaster_hexagon",
                "av_kiwi01",
                "parmesan_cheese_canister",
                "popcorn_box",
                "measuring_cups_1",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.58589, 0.38231, -0.69705), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.33843, -0.18341, 0.069), rot=(0.0, 0.0, 0.203284, 0.97912)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.48186, 0.05673, 0.069), rot=(0.0, 0.0, -0.149907, 0.9887)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.48071, 0.21297, 0.06899), rot=(0.0, 0.0, 0.875477, 0.48326)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.29782, -0.05583, 0.12043), rot=(0.0, 0.0, -0.284374, 0.958714)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.42603, -0.07405, 0.11149), rot=(0.0, 0.0, 0.556306, 0.830977)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.34359, 0.16155, 0.09503), rot=(0.0, 0.0, 0.577258, 0.816562)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.55438, -0.11701, 0.069), rot=(0.0, 0.0, 0.359592, 0.93311)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "reference": SceneEntityCfg("popcorn_box"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace188": HeterogeneousTaskCfg(
        instruction="pick up the popcorn box and put it in the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "popcorn_box",
                "peas_and_carrots",
                "coffee_pot",
                "coffee_can",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.41725, -0.3236, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.42062, 0.03071, 0.19796), rot=(0.0, 0.0, -0.554522, 0.832169)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.52846, -0.0701, 0.198), rot=(0.0, 0.0, -0.998332, 0.057727)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.47733, -0.18862, 0.24049), rot=(0.0, 0.0, 0.406695, 0.913564)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.49413, 0.23196, 0.22726), rot=(0.0, 0.0, -0.989803, 0.142442)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.33965, 0.17579, 0.19796), rot=(0.0, 0.0, 0.265764, 0.964038)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.33054, -0.2261, 0.26807), rot=(0.0, 0.0, 0.870916, 0.491432)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("popcorn_box")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace189": HeterogeneousTaskCfg(
        instruction="pick up the tomato soup can and put it in the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_endtable",
                "bowl",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "tomato_soup_can",
                "empty_6",
                "av_woodenchsmall",
                "red_block",
            ]
        ),
        init_states={
            "appleseed_endtable": RigidObjectCfg.InitialStateCfg(
                pos=(0.42556, -0.00621, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.43385, -0.1056, -0.02852), rot=(0.0, 0.0, 0.95839, 0.285462)),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.38115, 0.11379, -0.056), rot=(0.0, 0.0, 0.392464, 0.919767)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.53042, 0.1707, -0.00498), rot=(0.0, 0.0, -0.112382, 0.993665)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.29521, 0.06338, -0.056), rot=(0.0, 0.0, -0.999988, 0.004988)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.5743, -0.03415, -0.0325), rot=(0.0, 0.0, 0.930398, 0.36655)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("tomato_soup_can")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace190": HeterogeneousTaskCfg(
        instruction="move the milk jug into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_sofatable",
                "bin_a03",
                "empty_2",
                "empty_3",
                "parmesan_cheese_canister",
                "milkjug_a01",
                "peas_and_carrots",
                "empty_7",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "appleseed_sofatable": RigidObjectCfg.InitialStateCfg(
                pos=(0.4403, 0.30669, -0.69682), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.43907, -0.11024, 0.058), rot=(0.0, 0.0, -0.472013, 0.881592)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.30915, -0.01193, 0.10943), rot=(0.0, 0.0, -0.365359, 0.930867)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.482, 0.02732, 0.058), rot=(0.0, 0.0, 0.189162, 0.981946)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.56675, 0.09057, 0.08726), rot=(0.0, 0.0, 0.13798, 0.990435)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.37294, 0.15911, 0.14113), rot=(0.0, 0.0, 0.803097, 0.595848)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("milkjug_a01")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace191": HeterogeneousTaskCfg(
        instruction="place the strawberry into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_maple",
                "bin_a02",
                "empty_2",
                "empty_3",
                "av_stawberry",
                "granola_bars",
                "av_knife",
                "bagel",
                "birdhouse",
            ]
        ),
        init_states={
            "table_maple": RigidObjectCfg.InitialStateCfg(pos=(0.58509, -0.11722, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.52144, -0.03573, 0.009), rot=(0.0, 0.0, 0.958108, 0.286407)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.29404, 0.04698, 0.00881), rot=(0.0, 0.0, 0.999988, 0.004813)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.34172, 0.197, 0.071), rot=(0.0, 0.0, -0.610825, 0.791766)
            ),
            "av_knife": RigidObjectCfg.InitialStateCfg(
                pos=(0.53401, 0.16881, 0.01006), rot=(0.0, 0.0, 0.135959, 0.990715)
            ),
            "bagel": RigidObjectCfg.InitialStateCfg(
                pos=(0.40281, 0.07784, 0.02455), rot=(0.0, 0.0, 0.657831, 0.753165)
            ),
            "birdhouse": RigidObjectCfg.InitialStateCfg(
                pos=(0.3618, -0.15767, 0.00905), rot=(0.0, 0.0, 0.694552, 0.719442)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_stawberry")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace192": HeterogeneousTaskCfg(
        instruction="place both the orange juice carton and the avocado into the red bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "fstylejug_a02",
                "orange_juice_carton",
                "empty_6",
                "yellow_block",
                "avocado01",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.59527, 0.67429, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.45365, 0.12397, 0.10448), rot=(0.0, 0.0, 0.66716, 0.744915)
            ),
            "fstylejug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.50352, -0.06609, 0.077), rot=(0.0, 0.0, 0.917333, 0.398122)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.33552, -0.04459, 0.17324), rot=(0.0, 0.0, -0.665767, 0.74616)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.32577, 0.15766, 0.1005), rot=(0.0, 0.0, -0.231222, 0.972901)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4662, -0.25929, 0.07683), rot=(0.0, 0.0, -0.808088, 0.589062)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_juice_carton"), SceneEntityCfg("avocado01")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace193": HeterogeneousTaskCfg(
        instruction="place the ketchup bottle upright on top of the coaster",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "bowl",
                "av_coaster_hexagon",
                "spatula_09",
                "empty_4",
                "empty_5",
                "ketchup_bottle",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.62798, -0.44138, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.34132, -0.09159, 0.09848), rot=(0.0, 0.0, 0.997846, 0.065603)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.5033, -0.05172, 0.071), rot=(0.0, 0.0, 0.71867, 0.695352)
            ),
            "spatula_09": RigidObjectCfg.InitialStateCfg(
                pos=(0.43288, 0.08681, 0.07773), rot=(0.0, 0.0, -0.037048, 0.999313)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.43252, 0.19126, 0.1453), rot=(0.0, 0.0, 0.288591, 0.957452)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.50806, -0.17902, 0.11242), rot=(0.0, 0.0, 0.613064, 0.790033)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("ketchup_bottle")],
                    "reference": SceneEntityCfg("av_coaster_hexagon"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace194": HeterogeneousTaskCfg(
        instruction="put the frozen vegetables on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "empty_1",
                "av_book_66",
                "empty_3",
                "av_chess_queen",
                "pumpkinlarge",
                "canned_mushrooms",
                "frozen_vegetable_block",
                "pitted_cherries",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.414, 0.06633, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_66": RigidObjectCfg.InitialStateCfg(
                pos=(0.40055, 0.04554, 0.069), rot=(0.0, 0.0, -0.211377, 0.977405)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.53763, -0.02553, 0.069), rot=(0.0, 0.0, 0.776785, 0.629766)
            ),
            "pumpkinlarge": RigidObjectCfg.InitialStateCfg(
                pos=(0.48543, 0.22965, 0.06876), rot=(0.0, 0.0, -0.163684, 0.986513)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.50594, -0.15259, 0.08566), rot=(0.0, 0.0, 0.094054, 0.995567)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.41087, -0.20928, 0.069), rot=(0.0, 0.0, -0.372201, 0.928152)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.31951, 0.25626, 0.09811), rot=(0.0, 0.0, 0.999945, 0.010522)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("frozen_vegetable_block")],
                    "reference": SceneEntityCfg("av_book_66"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace195": HeterogeneousTaskCfg(
        instruction="put the spoon in the metal bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "mug",
                "spoon_2",
                "av_chess_rook",
                "pitted_cherries",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.41159, 0.28271, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.39388, -0.1355, 0.19825), rot=(0.0, 0.0, 0.682619, 0.730775)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.27674, -0.09423, 0.198), rot=(0.0, 0.0, 0.503986, 0.863712)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(pos=(0.33694, 0.16602, 0.19818), rot=(0.0, 0.0, 0.956175, 0.292794)),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.48577, 0.07921, 0.20339), rot=(0.0, 0.0, 0.808635, 0.588311)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.53731, -0.02969, 0.198), rot=(0.0, 0.0, -0.407573, 0.913172)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.31406, 0.05207, 0.22711), rot=(0.0, 0.0, 0.02582, 0.999667)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_2")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace196": HeterogeneousTaskCfg(
        instruction="pick up the tomato sauce can and put it in the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "av_chess_queen",
                "empty_5",
                "blue_block",
                "tomato_sauce_can",
                "red_block",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.55316, -0.16297, -0.6376), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.47635, -0.00206, 0.18348), rot=(0.0, 0.0, 0.586619, 0.809863)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.27946, 0.1894, 0.156), rot=(0.0, 0.0, -0.607286, 0.794484)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.43871, 0.18675, 0.1795), rot=(0.0, 0.0, 0.599898, 0.800077)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.35249, 0.1125, 0.19742), rot=(0.0, 0.0, -0.447811, 0.894128)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.57001, 0.13991, 0.1795), rot=(0.0, 0.0, 0.0387, 0.999251)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("tomato_sauce_can")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace197": HeterogeneousTaskCfg(
        instruction="put the sugar box on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "empty_1",
                "av_book_01",
                "sugar_box",
                "whitepackerbottle_a01",
                "empty_5",
                "av_mustard_jar",
                "rubiks_cube",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.75839, -0.06587, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_book_01": RigidObjectCfg.InitialStateCfg(
                pos=(0.43159, -0.02415, 0.071), rot=(0.0, 0.0, -0.99975, 0.022371)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.45428, -0.24831, 0.15901), rot=(0.0, 0.0, 0.667326, 0.744766)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.59248, -0.02448, 0.071), rot=(0.0, 0.0, 0.990866, 0.13485)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.4327, 0.2476, 0.07108), rot=(0.0, 0.0, -0.528312, 0.849051)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.3672, -0.28332, 0.10239), rot=(0.0, 0.0, 0.13347, 0.991053)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.36481, 0.18292, 0.07106), rot=(0.0, 0.0, -0.998134, 0.061059)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("sugar_box")],
                    "reference": SceneEntityCfg("av_book_01"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace198": HeterogeneousTaskCfg(
        instruction="move the green block into the white bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "av_bowl_plate",
                "empty_2",
                "frozen_waffles",
                "green_block",
                "pineapple_slices_can",
                "ketchup_bottle",
                "av_woodenchsmall",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.55481, -0.34466, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.41523, -0.11869, 0.0708), rot=(0.0, 0.0, 0.818437, 0.574596)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.53487, 0.08586, 0.07098), rot=(0.0, 0.0, -0.998648, 0.051985)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.42311, 0.1449, 0.0945), rot=(0.0, 0.0, -0.909779, 0.415093)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30152, 0.0834, 0.09981), rot=(0.0, 0.0, -0.896914, 0.442205)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.53499, 0.23462, 0.1453), rot=(0.0, 0.0, 0.879981, 0.475009)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.37916, 0.04878, 0.071), rot=(0.0, 0.0, 0.593341, 0.804951)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.36796, 0.20839, 0.15103), rot=(0.0, 0.0, -0.907736, 0.419542)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("green_block")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace199": HeterogeneousTaskCfg(
        instruction="place the plastic bottle upright on top of the book",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "empty_1",
                "av_book_51",
                "sugar_box",
                "empty_4",
                "milkjug_a02",
                "empty_6",
                "naturalbostonroundbottle_a01",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.42797, 0.40298, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_book_51": RigidObjectCfg.InitialStateCfg(
                pos=(0.43942, -0.07643, 0.007), rot=(0.0, 0.0, 0.995748, 0.09212)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.42204, 0.199, 0.09501), rot=(0.0, 0.0, 0.069159, 0.997606)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.31551, 0.20533, 0.007), rot=(0.0, 0.0, -0.999893, 0.014634)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.55719, 0.17447, 0.007), rot=(0.0, 0.0, 0.815755, 0.578397)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                    "reference": SceneEntityCfg("av_book_51"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace200": HeterogeneousTaskCfg(
        instruction="remove the blue block from the red bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "bowl",
                "empty_2",
                "spatula_07",
                "butter",
                "smartphone",
                "blue_block",
                "av_woodenchmedium",
                "pitted_cherries",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(pos=(0.87521, -0.05895, -0.69705), rot=(0.0, 0.0, 1.0, 0.0)),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.486, 0.09173, 0.09648), rot=(0.0, 0.0, 0.775222, 0.631689)),
            "spatula_07": RigidObjectCfg.InitialStateCfg(
                pos=(0.42554, -0.19632, 0.07214), rot=(0.0, 0.0, 0.221626, 0.975132)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.46123, 0.23319, 0.09541), rot=(0.0, 0.0, -0.39592, 0.918285)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.32796, 0.07536, 0.07465), rot=(0.0, 0.0, 0.999934, 0.011499)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.486, 0.09173, 0.1065), rot=(0.0, 0.0, 0.486207, 0.873844)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.34463, -0.0556, 0.06901), rot=(0.0, 0.0, -0.584522, 0.811378)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.30474, 0.24046, 0.09811), rot=(0.0, 0.0, -0.642813, 0.766023)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("blue_block")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace201": HeterogeneousTaskCfg(
        instruction="move the chess rook into the white bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "empty_5",
                "empty_6",
                "av_chess_rook",
                "av_tincan",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.46567, 0.01767, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.41435, 0.00815, -0.0312), rot=(0.0, 0.0, 0.3548, 0.934942)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.34148, 0.27426, -0.031), rot=(0.0, 0.0, -0.969784, 0.243964)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.31755, -0.24156, -0.031), rot=(0.0, 0.0, -0.999565, 0.029499)
            ),
            "av_tincan": RigidObjectCfg.InitialStateCfg(
                pos=(0.46217, -0.22953, -0.031), rot=(0.0, 0.0, -0.285143, 0.958485)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_rook")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace202": HeterogeneousTaskCfg(
        instruction="place both the milk jug and the mustard bottle into the pan",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "av_pan_d",
                "empty_2",
                "empty_3",
                "empty_4",
                "milkjug_a01",
                "empty_6",
                "empty_7",
                "mustard_bottle",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.42626, 0.11989, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.4484, 0.09931, -0.03074), rot=(0.0, 0.0, -0.055355, 0.998467)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.38603, -0.2436, -0.031), rot=(0.0, 0.0, 0.885436, 0.464762)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.51614, 0.26251, 0.04903), rot=(0.0, 0.0, 0.999988, 0.004827)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("milkjug_a01"), SceneEntityCfg("mustard_bottle")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace203": HeterogeneousTaskCfg(
        instruction="move the white bottle into the metal bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a04",
                "av_p_glassware_tall",
                "measuring_cups_1",
                "yellow_block",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.44471, 0.11961, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.49441, -0.03098, -0.03075), rot=(0.0, 0.0, 0.969535, 0.244953)
            ),
            "whitepackerbottle_a04": RigidObjectCfg.InitialStateCfg(
                pos=(0.43884, -0.1798, -0.031), rot=(0.0, 0.0, -0.908775, 0.417286)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.30053, -0.1366, -0.031), rot=(0.0, 0.0, -0.074502, 0.997221)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.40235, 0.16978, -0.00497), rot=(0.0, 0.0, -0.989548, 0.144206)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.3456, 0.04403, -0.0075), rot=(0.0, 0.0, 0.547002, 0.837131)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace204": HeterogeneousTaskCfg(
        instruction="place both the plastic bottle and the barbecue sauce bottle into the metal bin",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "bin_a05",
                "empty_2",
                "empty_3",
                "spaghetti",
                "empty_5",
                "empty_6",
                "naturalbostonroundbottle_a02",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.50381, 0.44344, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.43853, -0.08552, 0.007), rot=(0.0, 0.0, 0.44623, 0.894918)
            ),
            "spaghetti": RigidObjectCfg.InitialStateCfg(
                pos=(0.39436, 0.17435, 0.03192), rot=(0.0, 0.0, 0.439081, 0.898447)
            ),
            "naturalbostonroundbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.48577, 0.24501, 0.007), rot=(0.0, 0.0, 0.624331, 0.78116)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.34862, 0.07014, 0.00706), rot=(0.0, 0.0, -0.797176, 0.603747)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("naturalbostonroundbottle_a02"),
                        SceneEntityCfg("bbq_sauce_bottle_hot3d"),
                    ],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace205": HeterogeneousTaskCfg(
        instruction="pick the kiwi out of the metal bin and put it down",
        episode_length_s=45.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "bin_a02",
                "av_photoalbum_closed",
                "av_kiwi01",
                "yogurt_cup",
                "tuna_can",
                "empty_6",
                "dry_erase_marker",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.30996, 0.22021, -0.6376), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.33401, -0.0788, 0.156), rot=(0.0, 0.0, 0.073271, 0.997312)
            ),
            "av_photoalbum_closed": RigidObjectCfg.InitialStateCfg(
                pos=(0.49503, 0.06343, 0.156), rot=(0.0, 0.0, 0.866847, 0.498573)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.33401, -0.0788, 0.16999), rot=(0.0, 0.0, -0.793014, 0.609204)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.32492, 0.14956, 0.18284), rot=(0.0, 0.0, 0.99084, 0.135038)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.54544, -0.10995, 0.17276), rot=(0.0, 0.0, -0.108578, 0.994088)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.30643, 0.04945, 0.16544), rot=(0.0, 0.0, 0.370825, 0.928703)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace206": HeterogeneousTaskCfg(
        instruction="pick up the can of corn and put it in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "bin_b04",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "corn_can",
                "milk_carton",
                "mayonnaise_bottle",
                "empty_8",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(pos=(2.30517, 0.02027, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.43107, -0.03559, 0.109), rot=(0.0, 0.0, 0.662159, 0.749363)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.30801, -0.24693, 0.109), rot=(0.0, 0.0, 0.0064, 0.99998)
            ),
            "corn_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.45394, 0.26189, 0.13802), rot=(0.0, 0.0, -0.999883, 0.015284)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.3466, 0.2386, 0.20418), rot=(0.0, 0.0, 0.753928, 0.656957)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.53244, 0.20266, 0.18295), rot=(0.0, 0.0, 0.965164, 0.261645)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("corn_can")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace207": HeterogeneousTaskCfg(
        instruction="put the kiwi in the white bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "av_bowl_plate",
                "empty_2",
                "av_kiwi01",
                "fstylejug_a01",
                "tomato_soup_can",
                "empty_6",
                "empty_7",
                "brick",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.47495, 0.04467, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.426, -0.02492, 0.0898), rot=(0.0, 0.0, 0.980364, 0.197198)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.40925, 0.21332, 0.08999), rot=(0.0, 0.0, 0.276302, 0.961071)
            ),
            "fstylejug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.4948, -0.22147, 0.09), rot=(0.0, 0.0, -0.603039, 0.797712)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.49256, 0.13477, 0.14102), rot=(0.0, 0.0, -0.096399, 0.995343)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.31778, -0.19404, 0.11557), rot=(0.0, 0.0, -0.712292, 0.701883)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace208": HeterogeneousTaskCfg(
        instruction="place the kiwi into the wooden bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "gilbert",
                "wooden_bowl",
                "empty_2",
                "av_kiwi01",
                "av_stawberry",
                "empty_5",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "gilbert": RigidObjectCfg.InitialStateCfg(
                pos=(0.77784, 0.10032, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43859, 0.05503, 0.11892), rot=(0.0, 0.0, 0.144946, 0.98944)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.42452, -0.21539, 0.11899), rot=(0.0, 0.0, 0.07362, 0.997286)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.33324, -0.1878, 0.11881), rot=(0.0, 0.0, 0.549749, 0.83533)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace209": HeterogeneousTaskCfg(
        instruction="take the strawberry out of the salt bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "av_salt_bowl",
                "empty_2",
                "spatula_10",
                "av_stawberry",
                "empty_5",
                "ketchup_bottle",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.62319, 0.07574, -0.6376), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.49573, -0.01769, 0.15606), rot=(0.0, 0.0, -0.781901, 0.623403)
            ),
            "spatula_10": RigidObjectCfg.InitialStateCfg(
                pos=(0.42169, -0.16687, 0.16273), rot=(0.0, 0.0, 0.98927, 0.146099)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.49574, -0.01769, 0.16981), rot=(0.0, 0.0, 0.826819, 0.562468)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.32443, 0.04479, 0.2303), rot=(0.0, 0.0, -0.987999, 0.154463)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("av_stawberry")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace210": HeterogeneousTaskCfg(
        instruction="put the frozen vegetables in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "bin_b04",
                "empty_2",
                "empty_3",
                "green_block",
                "canned_tuna",
                "av_apple",
                "frozen_vegetable_block",
                "empty_8",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.24426, -0.20654, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.42986, 0.00392, 0.156), rot=(0.0, 0.0, 0.583786, 0.811908)
            ),
            "green_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.52038, 0.21257, 0.1795), rot=(0.0, 0.0, -0.635061, 0.772462)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.524, -0.21053, 0.17229), rot=(0.0, 0.0, 0.961577, 0.274534)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.39266, -0.23874, 0.15696), rot=(0.0, 0.0, -0.829194, 0.558961)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.31279, 0.18974, 0.156), rot=(0.0, 0.0, 0.090673, 0.995881)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("frozen_vegetable_block")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace211": HeterogeneousTaskCfg(
        instruction="put a bottle in the red bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "bowl",
                "empty_2",
                "sugar_box",
                "empty_4",
                "milkjug_a01",
                "ketchup_bottle",
                "av_woodenchmedium",
                "mustard_bottle",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.52647, 0.17149, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.46334, 0.08106, 0.03648), rot=(0.0, 0.0, 0.71163, 0.702554)),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.4456, 0.22878, 0.09701), rot=(0.0, 0.0, -0.888674, 0.458539)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.52261, -0.08551, 0.009), rot=(0.0, 0.0, 0.011301, 0.999936)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.30207, 0.11255, 0.0833), rot=(0.0, 0.0, 0.807025, 0.590517)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.4099, -0.25314, 0.00901), rot=(0.0, 0.0, 0.629991, 0.776602)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.37245, -0.09228, 0.08903), rot=(0.0, 0.0, 0.91061, 0.413266)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("milkjug_a01"),
                        SceneEntityCfg("ketchup_bottle"),
                        SceneEntityCfg("mustard_bottle"),
                    ],
                    "container": SceneEntityCfg("bowl"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace212": HeterogeneousTaskCfg(
        instruction="move the yogurt cup into the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "bin_a02",
                "av_coaster_hexagon",
                "sugar_box",
                "yogurt_cup",
                "milkjug_a03",
                "empty_6",
                "av_eraser",
                "empty_8",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(pos=(0.55533, -0.06202, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.36134, -0.04602, 0.009), rot=(0.0, 0.0, -0.999851, 0.01728)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.53815, 0.04421, 0.009), rot=(0.0, 0.0, -0.424529, 0.905415)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.36295, -0.15756, 0.09701), rot=(0.0, 0.0, -0.717497, 0.696562)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.47159, -0.20209, 0.03584), rot=(0.0, 0.0, -0.934062, 0.357111)
            ),
            "milkjug_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.37676, 0.11323, 0.009), rot=(0.0, 0.0, -0.756339, 0.65418)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.51671, -0.1046, 0.01333), rot=(0.0, 0.0, -0.982537, 0.186069)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("yogurt_cup")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace213": HeterogeneousTaskCfg(
        instruction="pick up the red block and put it in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_black",
                "bin_a05",
                "empty_2",
                "spatula_12",
                "empty_4",
                "empty_5",
                "empty_6",
                "mug_ycb",
                "red_block",
            ]
        ),
        init_states={
            "table_black": RigidObjectCfg.InitialStateCfg(pos=(0.48618, -0.14997, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bin_a05": RigidObjectCfg.InitialStateCfg(pos=(0.38378, -0.10354, 0.009), rot=(0.0, 0.0, 0.7386, 0.674144)),
            "spatula_12": RigidObjectCfg.InitialStateCfg(
                pos=(0.52887, 0.10145, 0.01267), rot=(0.0, 0.0, 0.812213, 0.583361)
            ),
            "mug_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.3435, 0.22596, 0.04969), rot=(0.0, 0.0, 0.993658, 0.112447)
            ),
            "red_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.38493, 0.12116, 0.0325), rot=(0.0, 0.0, 0.701915, 0.71226)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("red_block")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace214": HeterogeneousTaskCfg(
        instruction="put the wooden candle holder in the decorative bowl, then the raisin box",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "table_maple",
                "av_decorativebowl",
                "empty_2",
                "empty_3",
                "empty_4",
                "raisin_box",
                "av_p_glassware_short",
                "av_woodenchmedium",
                "empty_8",
            ]
        ),
        init_states={
            "table_maple": RigidObjectCfg.InitialStateCfg(
                pos=(0.66603, -0.01217, -0.347), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.42369, 0.00439, 0.00904), rot=(0.0, 0.0, -0.01696, 0.999856)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.3325, 0.24395, 0.07059), rot=(0.0, 0.0, -0.168086, 0.985772)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.44351, 0.24003, 0.009), rot=(0.0, 0.0, -0.998003, 0.063163)
            ),
            "av_woodenchmedium": RigidObjectCfg.InitialStateCfg(
                pos=(0.32975, -0.19752, 0.00901), rot=(0.0, 0.0, -0.639147, 0.769085)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("av_woodenchmedium"), SceneEntityCfg("raisin_box")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                },
            ),
        },
    ),
    "PickPlace215": HeterogeneousTaskCfg(
        instruction="pick the can of pineapple out of the black bowl and put it down",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "blackandbrassbowl_small",
                "empty_2",
                "empty_3",
                "empty_4",
                "pineapple_slices_can",
                "spoon_big",
                "naturalbostonroundbottle_a03",
                "mustard",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.49635, -0.18411, -0.347), rot=(0.0, 0.0, 1.0, 0.0)),
            "blackandbrassbowl_small": RigidObjectCfg.InitialStateCfg(
                pos=(0.39361, 0.06062, 0.00896), rot=(0.0, 0.0, -0.480595, 0.876942)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.39361, 0.06062, 0.05181), rot=(0.0, 0.0, 0.582234, 0.813021)
            ),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.35137, 0.23356, 0.00924), rot=(0.0, 0.0, 0.036913, 0.999318)
            ),
            "naturalbostonroundbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.51454, -0.11288, 0.009), rot=(0.0, 0.0, 0.92636, 0.37664)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.37912, -0.23121, 0.1047), rot=(0.0, 0.0, -0.637432, 0.770507)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("pineapple_slices_can")],
                    "container": SceneEntityCfg("blackandbrassbowl_small"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace216": HeterogeneousTaskCfg(
        instruction="take the chess bishop out of the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "bin_b04",
                "empty_2",
                "empty_3",
                "ranch_dressing",
                "empty_5",
                "av_chess_bishop",
                "green_beans_can",
                "pitted_cherries",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.47397, 0.2536, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(pos=(0.477, 0.09922, 0.007), rot=(0.0, 0.0, -0.611841, 0.790981)),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.35847, -0.10345, 0.08072), rot=(0.0, 0.0, -0.643084, 0.765795)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.477, 0.09922, 0.021), rot=(0.0, 0.0, -0.349438, 0.93696)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.49132, -0.16743, 0.03579), rot=(0.0, 0.0, 0.183072, 0.9831)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.39278, -0.20515, 0.03611), rot=(0.0, 0.0, -0.064814, 0.997897)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("av_chess_bishop")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace217": HeterogeneousTaskCfg(
        instruction="place both the chess bishop and the eraser into the white bowl",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "appleseed_sofatable",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "empty_5",
                "av_chess_bishop",
                "av_eraser",
                "pitcher",
            ]
        ),
        init_states={
            "appleseed_sofatable": RigidObjectCfg.InitialStateCfg(
                pos=(0.44704, 0.2156, -0.69682), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.44545, 0.04254, 0.0578), rot=(0.0, 0.0, -0.986609, 0.163102)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.5169, -0.21691, 0.058), rot=(0.0, 0.0, 0.980294, 0.197544)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.38181, 0.22153, 0.058), rot=(0.0, 0.0, 0.968147, 0.250384)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.55635, -0.11532, 0.06233), rot=(0.0, 0.0, -0.991909, 0.126947)
            ),
            "pitcher": RigidObjectCfg.InitialStateCfg(
                pos=(0.36401, -0.2229, 0.05771), rot=(0.0, 0.0, -0.94903, 0.315186)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_eraser")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace218": HeterogeneousTaskCfg(
        instruction="place the cream cheese into the salt bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "av_salt_bowl",
                "empty_2",
                "empty_3",
                "cream_cheese",
                "tomato_soup_can",
                "av_mustard_jar",
                "empty_7",
                "foam_roller",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.43919, -0.37223, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.37109, 0.12436, 0.10906), rot=(0.0, 0.0, -0.239454, 0.970908)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.4638, 0.27286, 0.1356), rot=(0.0, 0.0, -0.783473, 0.621426)
            ),
            "tomato_soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.5146, -0.17039, 0.16002), rot=(0.0, 0.0, 0.88935, 0.457226)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.2968, -0.0064, 0.10908), rot=(0.0, 0.0, -0.596682, 0.802478)
            ),
            "foam_roller": RigidObjectCfg.InitialStateCfg(
                pos=(0.43554, -0.10467, 0.10905), rot=(0.0, 0.0, -0.363909, 0.931435)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("cream_cheese")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace219": HeterogeneousTaskCfg(
        instruction="put a chess piece in the black bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "milkjug_a02",
                "av_chess_king",
                "av_chess_rook",
                "empty_8",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.5758, -0.47788, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.42491, -0.06642, 0.07694), rot=(0.0, 0.0, -0.017548, 0.999846)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.38546, -0.25564, 0.077), rot=(0.0, 0.0, 0.832083, 0.554651)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.4279, 0.18024, 0.077), rot=(0.0, 0.0, 0.914581, 0.404404)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.30412, 0.15623, 0.077), rot=(0.0, 0.0, -0.965439, 0.260629)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.55879, 0.20851, 0.077), rot=(0.0, 0.0, -0.613189, 0.789936)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("av_chess_pawn"),
                        SceneEntityCfg("av_chess_king"),
                        SceneEntityCfg("av_chess_rook"),
                    ],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace220": HeterogeneousTaskCfg(
        instruction="pick up the chess queen and put it in the white bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_eastrural_table",
                "av_bowl_plate",
                "empty_2",
                "spatula_12",
                "av_chess_queen",
                "jello",
                "empty_6",
                "empty_7",
                "empty_8",
            ]
        ),
        init_states={
            "av_eastrural_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.95148, -0.00571, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.40824, -0.12649, 0.0708), rot=(0.0, 0.0, -0.996433, 0.084389)
            ),
            "spatula_12": RigidObjectCfg.InitialStateCfg(
                pos=(0.41022, 0.06046, 0.07467), rot=(0.0, 0.0, -0.998434, 0.05594)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.29198, 0.17427, 0.071), rot=(0.0, 0.0, 0.619654, 0.784875)
            ),
            "jello": RigidObjectCfg.InitialStateCfg(
                pos=(0.38915, 0.21244, 0.08606), rot=(0.0, 0.0, 0.148613, 0.988895)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace221": HeterogeneousTaskCfg(
        instruction="pick the white bottle out of the metal bin and put it down",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "bin_b04",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a02",
                "tuna_can",
                "av_chess_bishop",
                "av_woodenchsmall",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.44224, 0.03768, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_b04": RigidObjectCfg.InitialStateCfg(
                pos=(0.38615, 0.07695, -0.001), rot=(0.0, 0.0, -0.636447, 0.771321)
            ),
            "whitepackerbottle_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.38615, 0.07695, 0.013), rot=(0.0, 0.0, 0.72671, 0.686944)
            ),
            "tuna_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.54816, -0.13984, 0.01576), rot=(0.0, 0.0, -0.869144, 0.49456)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.53368, 0.18446, -0.001), rot=(0.0, 0.0, -0.953788, 0.300481)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.4445, -0.15605, -0.001), rot=(0.0, 0.0, -0.214192, 0.976792)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.58737, 0.07745, 0.08213), rot=(0.0, 0.0, -0.997351, 0.072738)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("whitepackerbottle_a02")],
                    "container": SceneEntityCfg("bin_b04"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace222": HeterogeneousTaskCfg(
        instruction="put the avocado in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "bin_a02",
                "empty_2",
                "empty_3",
                "cream_cheese",
                "orange_juice_carton",
                "wood_block",
                "bagel_00",
                "avocado01",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.41164, 0.08992, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.37111, 0.01489, 0.198), rot=(0.0, 0.0, 0.997246, 0.074168)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.47347, -0.21352, 0.2246), rot=(0.0, 0.0, -0.193738, 0.981053)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.49671, -0.00866, 0.29424), rot=(0.0, 0.0, -0.51053, 0.85986)
            ),
            "wood_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.33386, -0.20333, 0.30094), rot=(0.0, 0.0, 0.784195, 0.620514)
            ),
            "bagel_00": RigidObjectCfg.InitialStateCfg(
                pos=(0.35255, 0.19991, 0.21358), rot=(0.0, 0.0, -0.213536, 0.976935)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.50662, 0.16331, 0.19783), rot=(0.0, 0.0, -0.261119, 0.965307)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("avocado01")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace223": HeterogeneousTaskCfg(
        instruction="remove the can of mushrooms from the red bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bowl",
                "av_white_tray",
                "empty_3",
                "av_chess_queen",
                "popcorn_box",
                "canned_mushrooms",
                "av_eraser",
                "bbq_sauce_bottle_hot3d",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.41301, 0.06356, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bowl": RigidObjectCfg.InitialStateCfg(pos=(0.36179, -0.17228, 0.09648), rot=(0.0, 0.0, 0.87799, 0.478679)),
            "av_white_tray": RigidObjectCfg.InitialStateCfg(
                pos=(0.42322, 0.12967, 0.06906), rot=(0.0, 0.0, -0.922862, 0.385131)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.53983, 0.10819, 0.069), rot=(0.0, 0.0, 0.809415, 0.587238)
            ),
            "popcorn_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.52151, -0.12865, 0.11149), rot=(0.0, 0.0, -0.993763, 0.111515)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.36179, -0.17228, 0.09966), rot=(0.0, 0.0, -0.995948, 0.089933)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.45311, 0.02423, 0.07333), rot=(0.0, 0.0, -0.999416, 0.034163)
            ),
            "bbq_sauce_bottle_hot3d": RigidObjectCfg.InitialStateCfg(
                pos=(0.29973, 0.14842, 0.06906), rot=(0.0, 0.0, -0.958274, 0.285852)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("canned_mushrooms")],
                    "container": SceneEntityCfg("bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace224": HeterogeneousTaskCfg(
        instruction="put the milk jug in the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "bin_a05",
                "av_coaster_hexagon",
                "empty_3",
                "yogurt_cup",
                "milkjug_a01",
                "empty_6",
                "av_eraser",
                "mustard_bottle",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.53571, 0.81506, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(pos=(0.42951, 0.03873, 0.077), rot=(0.0, 0.0, 0.429851, 0.9029)),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.48278, -0.24764, 0.077), rot=(0.0, 0.0, 0.0144, 0.999896)
            ),
            "yogurt_cup": RigidObjectCfg.InitialStateCfg(
                pos=(0.33615, -0.22707, 0.10384), rot=(0.0, 0.0, 0.915158, 0.403096)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.33867, 0.2341, 0.077), rot=(0.0, 0.0, -0.992735, 0.120325)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.29424, -0.10792, 0.08133), rot=(0.0, 0.0, 0.474892, 0.880044)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.54734, -0.08076, 0.15703), rot=(0.0, 0.0, 0.125575, 0.992084)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("milkjug_a01")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace225": HeterogeneousTaskCfg(
        instruction="stack the Rubik's cube on the orange can",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "whitepackerbottle_a03",
                "soup_can",
                "storage_box",
                "rubiks_cube",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.54644, 0.84858, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.36366, -0.12359, 0.07125), rot=(0.0, 0.0, 0.441523, 0.89725)
            ),
            "whitepackerbottle_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.29509, 0.00306, 0.071), rot=(0.0, 0.0, -0.888226, 0.459407)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.48495, 0.11901, 0.07106), rot=(0.0, 0.0, -0.400866, 0.916137)
            ),
            "storage_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.51121, 0.21468, 0.07087), rot=(0.0, 0.0, 0.642703, 0.766115)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.3635, 0.1887, 0.10239), rot=(0.0, 0.0, 0.795004, 0.606604)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.3957, 0.23153, 0.14516), rot=(0.0, 0.0, 0.278015, 0.960577)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("rubiks_cube")],
                    "reference": SceneEntityCfg("soup_can"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace226": HeterogeneousTaskCfg(
        instruction="move the cleanser bottle into the decorative bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "steam",
                "av_decorativebowl",
                "empty_2",
                "empty_3",
                "parmesan_cheese_canister",
                "canned_peaches",
                "av_mustard_jar",
                "dry_erase_marker",
                "soft_scrub",
            ]
        ),
        init_states={
            "steam": RigidObjectCfg.InitialStateCfg(
                pos=(0.76604, -0.38265, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "av_decorativebowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.4187, 0.03053, 0.10904), rot=(0.0, 0.0, 0.604375, 0.7967)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.48368, 0.21769, 0.16043), rot=(0.0, 0.0, -0.48576, 0.874092)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.37926, 0.25345, 0.13789), rot=(0.0, 0.0, -0.846784, 0.531937)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.33144, -0.15446, 0.10908), rot=(0.0, 0.0, -0.938958, 0.344031)
            ),
            "dry_erase_marker": RigidObjectCfg.InitialStateCfg(
                pos=(0.57003, -0.17565, 0.11844), rot=(0.0, 0.0, -0.192443, 0.981308)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.43613, -0.236, 0.23432), rot=(0.0, 0.0, -0.968291, 0.249825)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("soft_scrub")],
                    "container": SceneEntityCfg("av_decorativebowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace227": HeterogeneousTaskCfg(
        instruction="move the brick into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "bin_a02",
                "av_coaster_hexagon",
                "spatula_18",
                "butter",
                "empty_5",
                "empty_6",
                "measuring_spoon",
                "brick",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.585, -0.18511, -0.69705), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.42569, -0.12383, 0.069), rot=(0.0, 0.0, -0.655432, 0.755254)
            ),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.46229, 0.11169, 0.069), rot=(0.0, 0.0, 0.523246, 0.852182)
            ),
            "spatula_18": RigidObjectCfg.InitialStateCfg(
                pos=(0.40262, -0.24531, 0.07268), rot=(0.0, 0.0, 0.057428, 0.99835)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.28054, 0.19234, 0.09541), rot=(0.0, 0.0, -0.995034, 0.099535)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.54236, -0.05368, 0.08334), rot=(0.0, 0.0, 0.70506, 0.709147)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.31471, 0.01356, 0.09457), rot=(0.0, 0.0, -0.941517, 0.336967)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("brick")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace228": HeterogeneousTaskCfg(
        instruction="pick the foam roller out of the white bowl and put it down",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "av_bowl_plate",
                "empty_2",
                "spatula_09",
                "orange_02",
                "empty_5",
                "ketchup_bottle",
                "yellow_block",
                "foam_roller",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.69787, -0.17885, -0.6971), rot=(0.0, 0.0, 1.0, 0.0)),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.42612, -0.08181, 0.1288), rot=(0.0, 0.0, 0.71521, 0.698909)
            ),
            "spatula_09": RigidObjectCfg.InitialStateCfg(
                pos=(0.46324, 0.17198, 0.13573), rot=(0.0, 0.0, 0.983986, 0.178247)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.35338, 0.1013, 0.12906), rot=(0.0, 0.0, 0.997166, 0.075239)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.59244, -0.01626, 0.2033), rot=(0.0, 0.0, 0.998953, 0.045745)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.44595, -0.23987, 0.1525), rot=(0.0, 0.0, 0.995141, 0.098463)
            ),
            "foam_roller": RigidObjectCfg.InitialStateCfg(
                pos=(0.42612, -0.08195, 0.14305), rot=(0.0, 0.0, 0.744032, 0.668144)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("foam_roller")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace229": HeterogeneousTaskCfg(
        instruction="place a piece of fruit into the white bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_laterialfile",
                "av_bowl_plate",
                "empty_2",
                "av_kiwi01",
                "av_stawberry",
                "pumpkinlarge",
                "empty_6",
                "empty_7",
                "avocado01",
            ]
        ),
        init_states={
            "contemporary_laterialfile": RigidObjectCfg.InitialStateCfg(
                pos=(0.43547, 0.09164, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.44429, -0.06319, -0.0312), rot=(0.0, 0.0, 0.463414, 0.886142)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.38963, 0.14309, -0.03101), rot=(0.0, 0.0, 0.888763, 0.458366)
            ),
            "av_stawberry": RigidObjectCfg.InitialStateCfg(
                pos=(0.44832, 0.23254, -0.03119), rot=(0.0, 0.0, 0.089834, 0.995957)
            ),
            "pumpkinlarge": RigidObjectCfg.InitialStateCfg(
                pos=(0.31594, 0.08085, -0.03124), rot=(0.0, 0.0, 0.324649, 0.945834)
            ),
            "avocado01": RigidObjectCfg.InitialStateCfg(
                pos=(0.35628, -0.20478, -0.03117), rot=(0.0, 0.0, -0.883553, 0.468332)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [
                        SceneEntityCfg("av_stawberry"),
                        SceneEntityCfg("av_kiwi01"),
                        SceneEntityCfg("avocado01"),
                    ],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace230": HeterogeneousTaskCfg(
        instruction="place a block into the red bowl",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "bowl_ycb",
                "empty_2",
                "empty_3",
                "cordless_drill",
                "pineapple_slices_can",
                "blue_block",
                "yellow_block",
                "coffee_can",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.5262, -0.11036, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.40424, 0.17852, -0.00352), rot=(0.0, 0.0, -0.482229, 0.876045)
            ),
            "cordless_drill": RigidObjectCfg.InitialStateCfg(
                pos=(0.47299, -0.00765, -0.00234), rot=(0.0, 0.0, 0.86072, 0.509079)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.30435, 0.05985, -0.00219), rot=(0.0, 0.0, 0.979931, 0.199337)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.48393, -0.18218, -0.0075), rot=(0.0, 0.0, 0.707978, 0.706235)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.28983, -0.07168, -0.0075), rot=(0.0, 0.0, -0.178003, 0.98403)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.39138, -0.22726, 0.03907), rot=(0.0, 0.0, -0.992014, 0.126126)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("blue_block"), SceneEntityCfg("yellow_block")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace231": HeterogeneousTaskCfg(
        instruction="put the mustard bottle in the black bowl, then the pumpkin",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "empty_4",
                "pumpkinsmall",
                "empty_6",
                "empty_7",
                "mustard",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.47358, 0.55221, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.42828, -0.00531, 0.00694), rot=(0.0, 0.0, 0.082428, 0.996597)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.51631, -0.21891, 0.00674), rot=(0.0, 0.0, -0.320547, 0.947233)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.30899, 0.22831, 0.1027), rot=(0.0, 0.0, -0.766328, 0.64245)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("mustard"), SceneEntityCfg("pumpkinsmall")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                },
            ),
        },
    ),
    "PickPlace232": HeterogeneousTaskCfg(
        instruction="remove the can of peaches from the white bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowhalfdesk",
                "av_bowl_plate",
                "empty_2",
                "empty_3",
                "cheez_it",
                "canned_peaches",
                "milk_carton",
                "empty_7",
                "mustard",
            ]
        ),
        init_states={
            "contemporary_lowhalfdesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.43657, -0.08132, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.41711, 0.01751, 0.0068), rot=(0.0, 0.0, -0.984574, 0.174971)
            ),
            "cheez_it": RigidObjectCfg.InitialStateCfg(
                pos=(0.33291, -0.20108, 0.11374), rot=(0.0, 0.0, -0.697977, 0.71612)
            ),
            "canned_peaches": RigidObjectCfg.InitialStateCfg(
                pos=(0.41724, 0.01756, 0.04989), rot=(0.0, 0.0, -0.024607, 0.999697)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.31322, 0.16127, 0.10218), rot=(0.0, 0.0, 0.175619, 0.984458)
            ),
            "mustard": RigidObjectCfg.InitialStateCfg(
                pos=(0.52398, -0.20474, 0.1027), rot=(0.0, 0.0, 0.706509, 0.707704)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("canned_peaches")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace233": HeterogeneousTaskCfg(
        instruction="move the can of pineapple into the red bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_openstorage",
                "bowl_ycb",
                "empty_2",
                "frozen_waffles",
                "whitepackerbottle_a01",
                "pineapple_slices_can",
                "av_p_glassware_short",
                "coffee_pot",
                "empty_8",
            ]
        ),
        init_states={
            "contemporary_openstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.43631, -0.0095, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.36062, 0.1521, -0.00352), rot=(0.0, 0.0, -0.512054, 0.858953)
            ),
            "frozen_waffles": RigidObjectCfg.InitialStateCfg(
                pos=(0.52319, -0.07291, -0.03102), rot=(0.0, 0.0, 0.237588, 0.971366)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.42598, -0.26299, -0.031), rot=(0.0, 0.0, 0.544318, 0.838879)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.54985, 0.07844, -0.00219), rot=(0.0, 0.0, -0.777162, 0.629301)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.48649, 0.2412, -0.031), rot=(0.0, 0.0, -0.828902, 0.559394)
            ),
            "coffee_pot": RigidObjectCfg.InitialStateCfg(
                pos=(0.36417, -0.17176, -0.03104), rot=(0.0, 0.0, 0.049975, 0.99875)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("pineapple_slices_can")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace234": HeterogeneousTaskCfg(
        instruction="place the blue block on top of the coaster",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "desk_01",
                "bin_a03",
                "av_coaster_hexagon",
                "empty_3",
                "cream_cheese",
                "empty_5",
                "blue_block",
                "frozen_vegetable_block",
                "soft_scrub",
            ]
        ),
        init_states={
            "desk_01": RigidObjectCfg.InitialStateCfg(pos=(0.50539, 0.19744, -0.69646), rot=(0.0, 0.0, 1.0, 0.0)),
            "bin_a03": RigidObjectCfg.InitialStateCfg(pos=(0.51047, 0.13392, 0.09), rot=(0.0, 0.0, 0.807943, 0.58926)),
            "av_coaster_hexagon": RigidObjectCfg.InitialStateCfg(
                pos=(0.4659, -0.18495, 0.09), rot=(0.0, 0.0, -0.080318, 0.996769)
            ),
            "cream_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.308, -0.15289, 0.1166), rot=(0.0, 0.0, 0.19346, 0.981108)
            ),
            "blue_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.5622, -0.09704, 0.1135), rot=(0.0, 0.0, -0.589495, 0.807772)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.36819, 0.23102, 0.09), rot=(0.0, 0.0, -0.232978, 0.972482)
            ),
            "soft_scrub": RigidObjectCfg.InitialStateCfg(
                pos=(0.34073, 0.09461, 0.21532), rot=(0.0, 0.0, 0.851469, 0.524404)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("blue_block")],
                    "reference": SceneEntityCfg("av_coaster_hexagon"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace235": HeterogeneousTaskCfg(
        instruction="move the orange juice carton into the metal bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "serving_bowl",
                "empty_2",
                "spatula_13",
                "butter",
                "orange_juice_carton",
                "empty_6",
                "yellow_block",
                "empty_8",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(pos=(0.74362, -0.04186, -0.697), rot=(0.0, 0.0, 1.0, 0.0)),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43357, 0.16841, 0.07125), rot=(0.0, 0.0, -0.032992, 0.999456)
            ),
            "spatula_13": RigidObjectCfg.InitialStateCfg(
                pos=(0.40472, -0.06189, 0.07467), rot=(0.0, 0.0, -0.896482, 0.44308)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.29279, -0.13705, 0.09741), rot=(0.0, 0.0, -0.291454, 0.956585)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.401, -0.20287, 0.16724), rot=(0.0, 0.0, -0.215482, 0.976508)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.56927, 0.0212, 0.0945), rot=(0.0, 0.0, 0.924958, 0.38007)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_juice_carton")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace236": HeterogeneousTaskCfg(
        instruction="first put the chess rook in the metal bin, then put the chess pawn in it",
        episode_length_s=110.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "bin_a03",
                "empty_2",
                "spatula_16",
                "av_chess_pawn",
                "empty_5",
                "empty_6",
                "av_chess_rook",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.55345, -0.20187, -0.6376), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.46638, -0.04677, 0.156), rot=(0.0, 0.0, 0.329512, 0.944152)
            ),
            "spatula_16": RigidObjectCfg.InitialStateCfg(
                pos=(0.33768, -0.01207, 0.15967), rot=(0.0, 0.0, -0.119993, 0.992775)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.39273, -0.15868, 0.156), rot=(0.0, 0.0, 0.911875, 0.410468)
            ),
            "av_chess_rook": RigidObjectCfg.InitialStateCfg(
                pos=(0.52506, 0.10769, 0.156), rot=(0.0, 0.0, 0.998507, 0.054618)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.objects_placed_in_container_in_order,
                params={
                    "objects": [SceneEntityCfg("av_chess_rook"), SceneEntityCfg("av_chess_pawn")],
                    "container": SceneEntityCfg("bin_a03"),
                },
            ),
        },
    ),
    "PickPlace237": HeterogeneousTaskCfg(
        instruction="put the barbecue sauce bottle in the metal bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "standard_smallunit",
                "serving_bowl",
                "empty_2",
                "empty_3",
                "butter",
                "soup_can",
                "spoon_big",
                "tomato_sauce_can",
                "bbq_sauce_bottle",
            ]
        ),
        init_states={
            "standard_smallunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.61388, 0.15443, -0.6376), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.43431, 0.04008, 0.15625), rot=(0.0, 0.0, -0.382123, 0.924111)
            ),
            "butter": RigidObjectCfg.InitialStateCfg(
                pos=(0.31292, 0.16213, 0.18241), rot=(0.0, 0.0, 0.666006, 0.745946)
            ),
            "soup_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.31203, -0.07995, 0.15606), rot=(0.0, 0.0, 0.846199, 0.532866)
            ),
            "spoon_big": RigidObjectCfg.InitialStateCfg(
                pos=(0.38302, -0.26315, 0.15624), rot=(0.0, 0.0, -0.997833, 0.065802)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.42259, 0.24656, 0.19742), rot=(0.0, 0.0, -0.102168, 0.994767)
            ),
            "bbq_sauce_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.46072, -0.15377, 0.23016), rot=(0.0, 0.0, 0.810116, 0.58627)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("bbq_sauce_bottle")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace238": HeterogeneousTaskCfg(
        instruction="pick the chocolate pudding mix out of the salt bowl and put it down",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "standard_halfunit",
                "av_salt_bowl",
                "empty_2",
                "spatula_12",
                "empty_4",
                "pumpkinlarge",
                "empty_6",
                "chocolate_pudding_mix",
                "empty_8",
            ]
        ),
        init_states={
            "standard_halfunit": RigidObjectCfg.InitialStateCfg(
                pos=(0.23829, -0.11929, -0.6376), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_salt_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.50627, -0.03753, 0.15606), rot=(0.0, 0.0, -0.524411, 0.851465)
            ),
            "spatula_12": RigidObjectCfg.InitialStateCfg(
                pos=(0.35709, -0.03155, 0.15967), rot=(0.0, 0.0, 0.758911, 0.651194)
            ),
            "pumpkinlarge": RigidObjectCfg.InitialStateCfg(
                pos=(0.49767, 0.12677, 0.15576), rot=(0.0, 0.0, -0.935961, 0.352103)
            ),
            "chocolate_pudding_mix": RigidObjectCfg.InitialStateCfg(
                pos=(0.50627, -0.03752, 0.19474), rot=(0.0, 0.0, -0.302792, 0.953057)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                    "container": SceneEntityCfg("av_salt_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace239": HeterogeneousTaskCfg(
        instruction="move the chess king into the pan",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "av_pan_d",
                "empty_2",
                "sugar_box",
                "oatmeal_raisin_cookies",
                "empty_5",
                "av_chess_king",
                "empty_7",
                "pitted_cherries",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(
                pos=(0.69592, -0.02642, -0.347), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_pan_d": RigidObjectCfg.InitialStateCfg(
                pos=(0.37621, -0.12773, 0.00926), rot=(0.0, 0.0, -0.999925, 0.012253)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.5627, 0.01889, 0.09701), rot=(0.0, 0.0, -0.013503, 0.999909)
            ),
            "oatmeal_raisin_cookies": RigidObjectCfg.InitialStateCfg(
                pos=(0.43372, 0.23629, 0.09262), rot=(0.0, 0.0, -0.957045, 0.289939)
            ),
            "av_chess_king": RigidObjectCfg.InitialStateCfg(
                pos=(0.56119, -0.15241, 0.009), rot=(0.0, 0.0, 0.326381, 0.945238)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.55712, 0.12629, 0.03811), rot=(0.0, 0.0, 0.993403, 0.114673)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_king")],
                    "container": SceneEntityCfg("av_pan_d"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace240": HeterogeneousTaskCfg(
        instruction="place the spoon into the white bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "table_bamboo",
                "av_bowl_plate",
                "empty_2",
                "spatula_03",
                "empty_4",
                "empty_5",
                "spoon_1",
                "rubiks_cube",
                "pitted_cherries",
            ]
        ),
        init_states={
            "table_bamboo": RigidObjectCfg.InitialStateCfg(
                pos=(0.70584, 0.00721, -0.347), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.38955, -0.16128, 0.0088), rot=(0.0, 0.0, 0.00545, 0.999985)
            ),
            "spatula_03": RigidObjectCfg.InitialStateCfg(
                pos=(0.3828, 0.18201, 0.01573), rot=(0.0, 0.0, 0.365093, 0.930971)
            ),
            "spoon_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.55432, -0.04161, 0.01966), rot=(0.0, 0.0, -0.575565, 0.817756)
            ),
            "rubiks_cube": RigidObjectCfg.InitialStateCfg(
                pos=(0.50155, 0.07046, 0.04039), rot=(0.0, 0.0, -0.660668, 0.750678)
            ),
            "pitted_cherries": RigidObjectCfg.InitialStateCfg(
                pos=(0.50672, 0.17329, 0.03811), rot=(0.0, 0.0, -0.944155, 0.329503)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("spoon_1")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace241": HeterogeneousTaskCfg(
        instruction="place the wooden candle holder into the metal bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "table_maple",
                "serving_bowl",
                "empty_2",
                "spatula_09",
                "fstylejug_a01",
                "scissors",
                "milk_carton",
                "av_woodenchsmall",
                "empty_8",
            ]
        ),
        init_states={
            "table_maple": RigidObjectCfg.InitialStateCfg(
                pos=(0.6753, -0.0315, -0.347), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "serving_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.37251, -0.03759, 0.00925), rot=(0.0, 0.0, -0.999857, 0.016886)
            ),
            "spatula_09": RigidObjectCfg.InitialStateCfg(
                pos=(0.41277, -0.20002, 0.01573), rot=(0.0, 0.0, -0.020539, 0.999789)
            ),
            "fstylejug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.40606, 0.16702, 0.009), rot=(0.0, 0.0, 0.223403, 0.974726)
            ),
            "scissors": RigidObjectCfg.InitialStateCfg(
                pos=(0.5355, 0.11334, 0.01686), rot=(0.0, 0.0, 0.017936, 0.999839)
            ),
            "milk_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.55401, -0.06741, 0.10418), rot=(0.0, 0.0, 0.233559, 0.972343)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.30937, 0.26687, 0.009), rot=(0.0, 0.0, 0.0613, 0.998119)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_woodenchsmall")],
                    "container": SceneEntityCfg("serving_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace242": HeterogeneousTaskCfg(
        instruction="place both the orange and the mayonnaise bottle into the red bowl",
        episode_length_s=105.0,
        inclusion=InclusionSet(
            assets=[
                "table_oak",
                "bowl_ycb",
                "empty_2",
                "sugar_box",
                "orange_02",
                "milkjug_a01",
                "canned_mushrooms",
                "mayonnaise_bottle",
                "empty_8",
            ]
        ),
        init_states={
            "table_oak": RigidObjectCfg.InitialStateCfg(pos=(0.54507, 0.05578, -0.347), rot=(0.0, 0.0, 0.0, 1.0)),
            "bowl_ycb": RigidObjectCfg.InitialStateCfg(
                pos=(0.47738, 0.00706, 0.03648), rot=(0.0, 0.0, -0.880399, 0.474234)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.43493, 0.20249, 0.09701), rot=(0.0, 0.0, 0.457724, 0.889095)
            ),
            "orange_02": RigidObjectCfg.InitialStateCfg(
                pos=(0.2967, -0.1287, 0.00906), rot=(0.0, 0.0, 0.188765, 0.982022)
            ),
            "milkjug_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.35819, 0.13368, 0.009), rot=(0.0, 0.0, -0.137535, 0.990497)
            ),
            "canned_mushrooms": RigidObjectCfg.InitialStateCfg(
                pos=(0.46471, -0.23531, 0.02566), rot=(0.0, 0.0, -0.499757, 0.866165)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.3923, -0.14356, 0.08295), rot=(0.0, 0.0, 0.647629, 0.761956)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("orange_02"), SceneEntityCfg("mayonnaise_bottle")],
                    "container": SceneEntityCfg("bowl_ycb"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace243": HeterogeneousTaskCfg(
        instruction="put the chess queen in the metal bin",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "bin_a02",
                "empty_2",
                "spatula_19",
                "av_chess_queen",
                "mug",
                "spoon_2",
                "measuring_spoon",
                "empty_8",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.52518, -0.3767, -0.69705), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.47504, 0.18097, 0.069), rot=(0.0, 0.0, -0.852138, 0.523317)
            ),
            "spatula_19": RigidObjectCfg.InitialStateCfg(
                pos=(0.45387, -0.12523, 0.07268), rot=(0.0, 0.0, -0.127026, 0.991899)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.30406, 0.00625, 0.069), rot=(0.0, 0.0, 0.683977, 0.729503)
            ),
            "mug": RigidObjectCfg.InitialStateCfg(
                pos=(0.35781, -0.22736, 0.06918), rot=(0.0, 0.0, -0.424583, 0.905389)
            ),
            "spoon_2": RigidObjectCfg.InitialStateCfg(
                pos=(0.32742, 0.16912, 0.07439), rot=(0.0, 0.0, -0.314961, 0.949105)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.41642, 0.00176, 0.08334), rot=(0.0, 0.0, 0.916284, 0.400528)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("bin_a02"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace244": HeterogeneousTaskCfg(
        instruction="set the ranch dressing bottle upright on the white tray",
        episode_length_s=75.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lowfulldesk",
                "empty_1",
                "av_white_tray",
                "empty_3",
                "ranch_dressing",
                "raisin_box",
                "ketchup_bottle",
                "measuring_spoon",
                "coffee_can",
            ]
        ),
        init_states={
            "contemporary_lowfulldesk": RigidObjectCfg.InitialStateCfg(
                pos=(0.53704, 0.36658, -0.697), rot=(0.0, 0.0, 1.0, 0.0)
            ),
            "av_white_tray": RigidObjectCfg.InitialStateCfg(
                pos=(0.40232, 0.12558, 0.00706), rot=(0.0, 0.0, -0.651443, 0.758698)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.51801, 0.05927, 0.08072), rot=(0.0, 0.0, -0.589355, 0.807874)
            ),
            "raisin_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.29897, 0.12796, 0.06859), rot=(0.0, 0.0, -0.276749, 0.960942)
            ),
            "ketchup_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.54174, -0.16756, 0.0813), rot=(0.0, 0.0, -0.84125, 0.540646)
            ),
            "measuring_spoon": RigidObjectCfg.InitialStateCfg(
                pos=(0.50765, 0.21584, 0.02134), rot=(0.0, 0.0, -0.570069, 0.821597)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.36045, -0.15986, 0.07707), rot=(0.0, 0.0, 0.751397, 0.659851)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("ranch_dressing")],
                    "reference": SceneEntityCfg("av_white_tray"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
    "PickPlace245": HeterogeneousTaskCfg(
        instruction="put the kiwi on the book",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_aftonconsole",
                "empty_1",
                "av_book_66",
                "av_kiwi01",
                "ranch_dressing",
                "pineapple_slices_can",
                "empty_6",
                "empty_7",
                "av_tincan",
            ]
        ),
        init_states={
            "av_aftonconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.42518, -0.30968, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_66": RigidObjectCfg.InitialStateCfg(
                pos=(0.43005, -0.12739, 0.109), rot=(0.0, 0.0, 0.968345, 0.249617)
            ),
            "av_kiwi01": RigidObjectCfg.InitialStateCfg(
                pos=(0.50795, 0.15395, 0.10899), rot=(0.0, 0.0, 0.465412, 0.885094)
            ),
            "ranch_dressing": RigidObjectCfg.InitialStateCfg(
                pos=(0.51704, 0.06415, 0.18272), rot=(0.0, 0.0, -0.941585, 0.336776)
            ),
            "pineapple_slices_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.32497, 0.00536, 0.13781), rot=(0.0, 0.0, -0.837376, 0.546627)
            ),
            "av_tincan": RigidObjectCfg.InitialStateCfg(
                pos=(0.33127, 0.17202, 0.109), rot=(0.0, 0.0, 0.448122, 0.893972)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("av_kiwi01")],
                    "reference": SceneEntityCfg("av_book_66"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace246": HeterogeneousTaskCfg(
        instruction="pick up the chess pawn and put it in the black bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_desperes_table",
                "blackandbrassbowl_large",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "orange_juice_carton",
                "empty_6",
                "green_beans_can",
                "brick",
            ]
        ),
        init_states={
            "av_desperes_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.57621, 0.43, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.43497, -0.00449, 0.07694), rot=(0.0, 0.0, -0.682002, 0.73135)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.35685, -0.2027, 0.077), rot=(0.0, 0.0, 0.397104, 0.917774)
            ),
            "orange_juice_carton": RigidObjectCfg.InitialStateCfg(
                pos=(0.52407, -0.21711, 0.17324), rot=(0.0, 0.0, 0.799739, 0.600348)
            ),
            "green_beans_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.33247, 0.26328, 0.10579), rot=(0.0, 0.0, 0.70727, 0.706944)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.44713, 0.19841, 0.10257), rot=(0.0, 0.0, 0.775968, 0.630772)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_pawn")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace247": HeterogeneousTaskCfg(
        instruction="place the yellow block into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "contemporary_lockedstorage",
                "bin_a05",
                "empty_2",
                "sugar_box",
                "oatmeal_raisin_cookies",
                "canned_tuna",
                "empty_6",
                "yellow_block",
                "brick",
            ]
        ),
        init_states={
            "contemporary_lockedstorage": RigidObjectCfg.InitialStateCfg(
                pos=(0.45588, -0.0753, -0.697), rot=(0.0, 0.0, 0.0, 1.0)
            ),
            "bin_a05": RigidObjectCfg.InitialStateCfg(
                pos=(0.45876, 0.04938, -0.031), rot=(0.0, 0.0, -0.740629, 0.671914)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.28677, -0.01083, 0.05701), rot=(0.0, 0.0, 0.038131, 0.999273)
            ),
            "oatmeal_raisin_cookies": RigidObjectCfg.InitialStateCfg(
                pos=(0.29654, 0.14275, 0.05262), rot=(0.0, 0.0, 0.193703, 0.98106)
            ),
            "canned_tuna": RigidObjectCfg.InitialStateCfg(
                pos=(0.31871, -0.25482, -0.01471), rot=(0.0, 0.0, 0.994459, 0.105121)
            ),
            "yellow_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.33772, 0.26201, -0.0075), rot=(0.0, 0.0, -0.672512, 0.740086)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.4593, -0.21101, -0.00543), rot=(0.0, 0.0, -0.896594, 0.442853)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("yellow_block")],
                    "container": SceneEntityCfg("bin_a05"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace248": HeterogeneousTaskCfg(
        instruction="put the mayonnaise bottle in the wooden bowl",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellendale",
                "wooden_bowl",
                "empty_2",
                "empty_3",
                "parmesan_cheese_canister",
                "empty_5",
                "empty_6",
                "mayonnaise_bottle",
                "empty_8",
            ]
        ),
        init_states={
            "av_ellendale": RigidObjectCfg.InitialStateCfg(
                pos=(0.38958, 0.00637, -0.697), rot=(0.0, 0.0, -0.707107, 0.707107)
            ),
            "wooden_bowl": RigidObjectCfg.InitialStateCfg(
                pos=(0.41401, -0.00942, -0.00108), rot=(0.0, 0.0, 0.999936, 0.01132)
            ),
            "parmesan_cheese_canister": RigidObjectCfg.InitialStateCfg(
                pos=(0.29372, 0.22294, 0.05043), rot=(0.0, 0.0, 0.740173, 0.672417)
            ),
            "mayonnaise_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.3898, 0.21013, 0.07295), rot=(0.0, 0.0, 0.054798, 0.998497)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("mayonnaise_bottle")],
                    "container": SceneEntityCfg("wooden_bowl"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace249": HeterogeneousTaskCfg(
        instruction="set the macaroni and cheese box on the book",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_ellisville",
                "empty_1",
                "av_book_99",
                "empty_3",
                "whitepackerbottle_a01",
                "empty_5",
                "peas_and_carrots",
                "av_woodenchsmall",
                "macaroni_and_cheese",
            ]
        ),
        init_states={
            "av_ellisville": RigidObjectCfg.InitialStateCfg(
                pos=(0.40543, 0.08988, -0.69599), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "av_book_99": RigidObjectCfg.InitialStateCfg(
                pos=(0.42845, 0.12239, 0.198), rot=(0.0, 0.0, 0.994252, 0.107062)
            ),
            "whitepackerbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.31242, -0.10464, 0.198), rot=(0.0, 0.0, -0.999335, 0.036476)
            ),
            "peas_and_carrots": RigidObjectCfg.InitialStateCfg(
                pos=(0.47805, -0.12138, 0.22726), rot=(0.0, 0.0, 0.645293, 0.763935)
            ),
            "av_woodenchsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.50895, -0.27187, 0.198), rot=(0.0, 0.0, 0.999996, 0.002729)
            ),
            "macaroni_and_cheese": RigidObjectCfg.InitialStateCfg(
                pos=(0.3891, -0.19019, 0.28113), rot=(0.0, 0.0, 0.450287, 0.892884)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("macaroni_and_cheese")],
                    "reference": SceneEntityCfg("av_book_99"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace250": HeterogeneousTaskCfg(
        instruction="place the plastic bottle into the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_eurekaconsole",
                "bin_a03",
                "empty_2",
                "spatula_19",
                "av_chess_queen",
                "pumpkinsmall",
                "av_mustard_jar",
                "naturalbostonroundbottle_a01",
                "coffee_can",
            ]
        ),
        init_states={
            "av_eurekaconsole": RigidObjectCfg.InitialStateCfg(
                pos=(0.4151, 0.23851, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(pos=(0.44373, -0.12516, 0.069), rot=(0.0, 0.0, 0.921069, 0.3894)),
            "spatula_19": RigidObjectCfg.InitialStateCfg(
                pos=(0.39167, 0.04943, 0.07268), rot=(0.0, 0.0, 0.991577, 0.129522)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.37575, -0.02131, 0.069), rot=(0.0, 0.0, -0.134595, 0.990901)
            ),
            "pumpkinsmall": RigidObjectCfg.InitialStateCfg(
                pos=(0.39645, -0.25601, 0.06874), rot=(0.0, 0.0, -0.050132, 0.998743)
            ),
            "av_mustard_jar": RigidObjectCfg.InitialStateCfg(
                pos=(0.51285, 0.13751, 0.06908), rot=(0.0, 0.0, -0.836933, 0.547305)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.28816, -0.16816, 0.069), rot=(0.0, 0.0, -0.654684, 0.755902)
            ),
            "coffee_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.36481, 0.21846, 0.13907), rot=(0.0, 0.0, 0.200825, 0.979627)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace251": HeterogeneousTaskCfg(
        instruction="take the crab pen holder out of the white bowl",
        episode_length_s=50.0,
        inclusion=InclusionSet(
            assets=[
                "av_hexagon",
                "av_bowl_plate",
                "empty_2",
                "crabbypenholder",
                "empty_4",
                "granola_bars",
                "measuring_cups_1",
                "clamp",
                "empty_8",
            ]
        ),
        init_states={
            "av_hexagon": RigidObjectCfg.InitialStateCfg(pos=(0.74307, 0.00114, -0.697), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_bowl_plate": RigidObjectCfg.InitialStateCfg(
                pos=(0.41022, 0.04277, 0.0708), rot=(0.0, 0.0, 0.998688, 0.051208)
            ),
            "crabbypenholder": RigidObjectCfg.InitialStateCfg(
                pos=(0.41036, 0.04276, 0.08497), rot=(0.0, 0.0, -0.802611, 0.596503)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.53579, -0.15087, 0.133), rot=(0.0, 0.0, -0.366807, 0.930297)
            ),
            "measuring_cups_1": RigidObjectCfg.InitialStateCfg(
                pos=(0.43105, 0.23145, 0.09703), rot=(0.0, 0.0, -0.000543, 1.0)
            ),
            "clamp": RigidObjectCfg.InitialStateCfg(
                pos=(0.3563, -0.17158, 0.09058), rot=(0.0, 0.0, -0.715976, 0.698125)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("crabbypenholder")],
                    "container": SceneEntityCfg("av_bowl_plate"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace252": HeterogeneousTaskCfg(
        instruction="remove the eraser from the metal bin",
        episode_length_s=55.0,
        inclusion=InclusionSet(
            assets=[
                "av_jennings_table",
                "bin_a03",
                "empty_2",
                "empty_3",
                "av_chess_pawn",
                "milkjug_a02",
                "av_apple",
                "av_eraser",
                "birdhouse",
            ]
        ),
        init_states={
            "av_jennings_table": RigidObjectCfg.InitialStateCfg(
                pos=(0.59529, -0.32853, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.50648, -0.11917, 0.071), rot=(0.0, 0.0, 0.953403, 0.301699)
            ),
            "av_chess_pawn": RigidObjectCfg.InitialStateCfg(
                pos=(0.29617, 0.20274, 0.071), rot=(0.0, 0.0, 0.994246, 0.107117)
            ),
            "milkjug_a02": RigidObjectCfg.InitialStateCfg(
                pos=(0.3509, -0.042, 0.071), rot=(0.0, 0.0, 0.862375, 0.506271)
            ),
            "av_apple": RigidObjectCfg.InitialStateCfg(
                pos=(0.34035, -0.19989, 0.07196), rot=(0.0, 0.0, 0.887198, 0.461388)
            ),
            "av_eraser": RigidObjectCfg.InitialStateCfg(
                pos=(0.50648, -0.11917, 0.08933), rot=(0.0, 0.0, -0.960835, 0.277122)
            ),
            "birdhouse": RigidObjectCfg.InitialStateCfg(
                pos=(0.46865, 0.12394, 0.07105), rot=(0.0, 0.0, -0.355426, 0.934704)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_outside_of,
                params={
                    "objects": [SceneEntityCfg("av_eraser")],
                    "container": SceneEntityCfg("bin_a03"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace253": HeterogeneousTaskCfg(
        instruction="put a chess piece in the metal bin",
        episode_length_s=65.0,
        inclusion=InclusionSet(
            assets=[
                "av_whittershins",
                "bin_a03",
                "empty_2",
                "spatula_09",
                "av_chess_queen",
                "granola_bars",
                "av_chess_bishop",
                "frozen_vegetable_block",
                "brick",
            ]
        ),
        init_states={
            "av_whittershins": RigidObjectCfg.InitialStateCfg(
                pos=(0.92018, 0.01698, -0.697), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "bin_a03": RigidObjectCfg.InitialStateCfg(
                pos=(0.36708, -0.02931, -0.058), rot=(0.0, 0.0, 0.769312, 0.638873)
            ),
            "spatula_09": RigidObjectCfg.InitialStateCfg(
                pos=(0.52894, 0.01911, -0.05127), rot=(0.0, 0.0, 0.756174, 0.654371)
            ),
            "av_chess_queen": RigidObjectCfg.InitialStateCfg(
                pos=(0.48813, 0.27045, -0.058), rot=(0.0, 0.0, 0.978983, 0.203942)
            ),
            "granola_bars": RigidObjectCfg.InitialStateCfg(
                pos=(0.35543, 0.25818, 0.004), rot=(0.0, 0.0, 0.790531, 0.612421)
            ),
            "av_chess_bishop": RigidObjectCfg.InitialStateCfg(
                pos=(0.30548, -0.15134, -0.058), rot=(0.0, 0.0, -0.264638, 0.964348)
            ),
            "frozen_vegetable_block": RigidObjectCfg.InitialStateCfg(
                pos=(0.48385, -0.17023, -0.058), rot=(0.0, 0.0, 0.37929, 0.925278)
            ),
            "brick": RigidObjectCfg.InitialStateCfg(
                pos=(0.41087, 0.13708, -0.03243), rot=(0.0, 0.0, 0.937211, 0.348762)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_chess_queen")],
                    "container": SceneEntityCfg("bin_a03"),
                    "logical": "any",
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace254": HeterogeneousTaskCfg(
        instruction="place the phone into the black bowl",
        episode_length_s=60.0,
        inclusion=InclusionSet(
            assets=[
                "av_willow",
                "blackandbrassbowl_large",
                "empty_2",
                "sugar_box",
                "empty_4",
                "smartphone",
                "empty_6",
                "tomato_sauce_can",
                "empty_8",
            ]
        ),
        init_states={
            "av_willow": RigidObjectCfg.InitialStateCfg(
                pos=(0.53625, -0.13093, -0.69705), rot=(0.0, 0.0, 0.707107, 0.707107)
            ),
            "blackandbrassbowl_large": RigidObjectCfg.InitialStateCfg(
                pos=(0.42975, -0.07853, 0.06894), rot=(0.0, 0.0, -0.060939, 0.998142)
            ),
            "sugar_box": RigidObjectCfg.InitialStateCfg(
                pos=(0.2834, 0.15636, 0.15701), rot=(0.0, 0.0, 0.999177, 0.040555)
            ),
            "smartphone": RigidObjectCfg.InitialStateCfg(
                pos=(0.47346, 0.19466, 0.07465), rot=(0.0, 0.0, 0.570351, 0.821401)
            ),
            "tomato_sauce_can": RigidObjectCfg.InitialStateCfg(
                pos=(0.36313, 0.13503, 0.11042), rot=(0.0, 0.0, 0.675515, 0.737346)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_in_container,
                params={
                    "objects": [SceneEntityCfg("smartphone")],
                    "container": SceneEntityCfg("blackandbrassbowl_large"),
                    "require_gripper_detached": True,
                },
            ),
        },
    ),
    "PickPlace255": HeterogeneousTaskCfg(
        instruction="place the plastic bottle upright on top of the book",
        episode_length_s=70.0,
        inclusion=InclusionSet(
            assets=[
                "cline",
                "empty_1",
                "av_book_77",
                "empty_3",
                "cheez_it",
                "av_p_glassware_tall",
                "av_p_glassware_short",
                "naturalbostonroundbottle_a01",
                "mustard_bottle",
            ]
        ),
        init_states={
            "cline": RigidObjectCfg.InitialStateCfg(pos=(0.75375, 0.69264, -0.6971), rot=(0.0, 0.0, 0.0, 1.0)),
            "av_book_77": RigidObjectCfg.InitialStateCfg(
                pos=(0.44804, -0.00678, 0.129), rot=(0.0, 0.0, -0.808633, 0.588314)
            ),
            "cheez_it": RigidObjectCfg.InitialStateCfg(
                pos=(0.43388, 0.21371, 0.23574), rot=(0.0, 0.0, 0.672841, 0.739787)
            ),
            "av_p_glassware_tall": RigidObjectCfg.InitialStateCfg(
                pos=(0.48463, -0.21905, 0.129), rot=(0.0, 0.0, 0.756919, 0.653509)
            ),
            "av_p_glassware_short": RigidObjectCfg.InitialStateCfg(
                pos=(0.29221, -0.21461, 0.129), rot=(0.0, 0.0, 0.71195, 0.70223)
            ),
            "naturalbostonroundbottle_a01": RigidObjectCfg.InitialStateCfg(
                pos=(0.39936, -0.26214, 0.129), rot=(0.0, 0.0, -0.62198, 0.783033)
            ),
            "mustard_bottle": RigidObjectCfg.InitialStateCfg(
                pos=(0.30495, 0.15817, 0.20903), rot=(0.0, 0.0, -0.836083, 0.548603)
            ),
        },
        terminations={
            "success": DoneTerm(
                func=mdp.object_on_top,
                params={
                    "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                    "reference": SceneEntityCfg("av_book_77"),
                    "require_gripper_detached": True,
                    "require_upright": True,
                },
            ),
        },
    ),
}


@configclass
class VLAHeterogeneousEnvCfg(ManagerBasedRLEnvCfg):
    """One shared scene and declarative records for all benchmark tasks.

    ``task_names`` selects a subset without constructing task-specific environment configurations.
    Individual benchmark registrations bind this selector; the heterogeneous registration uses all tasks.
    """

    scene: VLABenchmarkSceneCfg = VLABenchmarkSceneCfg(num_envs=1, env_spacing=10.0, lazy_sensor_update=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards = {}
    decimation = 8
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=8,
        # Newton preset of the Franka stack task, with sweep-and-prune broad phase.
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=4096,
                nconmax=4096,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
            num_substeps=2,
            # Thousands of small hull meshes: cuBQL takes about 0.1 s to build each one's BVH, LBVH a few ms.
            bvh_constructor_geometry="lbvh",
        ),
    )

    tasks: dict[str, HeterogeneousTaskCfg] = TASKS
    task_names: tuple[str, ...] = ()
    instruction: str | None = None
    episode_length_s = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.task_names:
            self.tasks = {name: self.tasks[name] for name in self.task_names}
        for name in dict.fromkeys(name for task in self.tasks.values() for name in task.inclusion.assets):
            if name not in vars(self.scene):
                setattr(self.scene, name, ASSET_CATALOG[name].copy())
        self.instruction = next(iter(self.tasks.values())).instruction if len(self.tasks) == 1 else None
        self.terminations.success = DoneTerm(func=mdp.task_success)
        HeterogeneousTaskManager.compose_tasks(self, self.tasks)
