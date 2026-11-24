# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A flat tabletop scene: bounded object combinations, geometric reset and available task families."""

import math
from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.cloner.cloner_strategies import random
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

from . import VLA_BENCHMARKS_ASSETS_DIR, mdp

GRIPPER_BODY = "robot/Robotiq_2F_85_edit/Robotiq_2F_85/left_inner_finger"
"""Gripper link, relative to the environment prim, whose contacts decide whether an object is held."""


CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/camera",
    width=448,
    height=252,
    data_types=["rgb"],
    update_period=0.0,
    renderer_cfg=NewtonWarpRendererCfg(ambient_light_color=(1.0, 1.0, 1.0), default_light_color=(0.5, 0.5, 0.5)),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.1,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
        clipping_range=(0.01, 1.0e6),
    ),
)

OBJECT_COUNT = (2, 8)
"""Inclusive cardinality bounds, excluding the robot and table."""


@configclass
class TableCfg(RigidObjectCfg):
    """A real table asset, its initial pose and its measured usable support region in local metres."""

    surface_outline: tuple[tuple[float, float], ...] = MISSING
    surface_height: float = MISSING
    surface_holes: tuple[tuple[tuple[float, float], ...], ...] = ()


EASTRURAL_TABLE_CFG = TableCfg(
    prim_path="{ENV_REGEX_NS}/work_table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/objects/av_eastrural_table/av_eastrural_table.usd",
        func=mdp.spawn_table,
        semantic_tags=[
            ("category", "table"),
            *[("object_count", str(count)) for count in range(OBJECT_COUNT[0], OBJECT_COUNT[1] + 1)],
        ],
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, -0.697)),
    surface_outline=((-0.4557, -0.76215), (0.4557, -0.76215), (0.4557, 0.76215), (-0.4557, 0.76215)),
    surface_height=0.762,
)
DESPERES_TABLE_CFG = EASTRURAL_TABLE_CFG.replace(
    spawn=EASTRURAL_TABLE_CFG.spawn.replace(
        usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/objects/av_desperes_table/av_desperes_table.usd"
    ),
    init_state=EASTRURAL_TABLE_CFG.init_state.replace(rot=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))),
    # The measured fixed centre panel; the extension leaves are not part of the placement region.
    surface_outline=((-0.760438, -0.406075), (0.760438, -0.406075), (0.760438, 0.406075), (-0.760438, 0.406075)),
    surface_height=0.76835005,
)
TABLE_VARIANTS = (EASTRURAL_TABLE_CFG, DESPERES_TABLE_CFG)
"""Variable real table assets, including their own measured usable surfaces and poses."""

EMPTY_OBJECT_CFG = sim_utils.SphereCfg(
    radius=0.001,
    visible=False,
    mass_props=sim_utils.MassCfg(mass=0.001),
    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
    collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
)


# Each movable asset has one sampling category. Task families use these same semantic tags.
ASSET_CATALOG = {
    name: sim_utils.UsdFileCfg(
        usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/objects/{name}/{name}.usd",
        semantic_tags=[("category", category)],
    )
    for name, category in {
        "av_apple": "food",
        "av_book_01": "book",
        "av_book_11": "book",
        "av_book_21": "book",
        "av_book_33": "book",
        "av_book_51": "book",
        "av_book_66": "book",
        "av_book_77": "book",
        "av_book_99": "book",
        "av_bowl_plate": "plate",
        "av_chess_bishop": "toy",
        "av_chess_king": "toy",
        "av_chess_pawn": "toy",
        "av_chess_queen": "toy",
        "av_chess_rook": "toy",
        "av_coaster_hexagon": "object",
        "av_decorativebowl": "container",
        "av_eraser": "object",
        "av_hippoorganizer": "object",
        "av_kiwi01": "food",
        "av_knife": "utensil",
        "av_mustard_jar": "bottle",
        "av_p_glassware_short": "object",
        "av_p_glassware_tall": "object",
        "av_pan_b": "utensil",
        "av_pan_d": "utensil",
        "av_photoalbum_closed": "book",
        "av_plant_succulent_01": "object",
        "av_plant_succulent_02": "object",
        "av_salt_bowl": "container",
        "av_sketchbook": "book",
        "av_stawberry": "food",
        "av_tincan": "can",
        "av_white_tray": "plate",
        "av_woodenchmedium": "object",
        "av_woodenchsmall": "object",
        "av_woodentray_medium": "plate",
        "avocado01": "food",
        "bagel": "food",
        "bagel_00": "food",
        "bbq_sauce_bottle": "bottle",
        "bbq_sauce_bottle_hot3d": "bottle",
        "bin_a02": "container",
        "bin_a03": "container",
        "bin_a05": "container",
        "bin_b04": "container",
        "birdhouse": "object",
        "blackandbrassbowl_large": "container",
        "blackandbrassbowl_small": "container",
        "blue_block": "toy",
        "bowl": "container",
        "bowl_ycb": "container",
        "brick": "object",
        "butter": "food",
        "canned_mushrooms": "can",
        "canned_peaches": "can",
        "canned_tuna": "can",
        "ceramic_mug": "object",
        "cheez_it": "food",
        "chocolate_pudding": "food",
        "chocolate_pudding_mix": "food",
        "clamp": "object",
        "clay_plates": "plate",
        "coffee_can": "can",
        "coffee_pot": "object",
        "cordless_drill": "object",
        "corn_can": "can",
        "crabbypenholder": "object",
        "cream_cheese": "food",
        "cubebox_a02": "object",
        "dry_erase_marker": "object",
        "dumbbell": "object",
        "foam_roller": "object",
        "fork_big": "utensil",
        "frozen_vegetable_block": "toy",
        "frozen_waffles": "food",
        "fstylejug_a01": "bottle",
        "fstylejug_a02": "bottle",
        "glasses": "object",
        "granola_bars": "food",
        "green_beans_can": "can",
        "green_block": "toy",
        "jello": "food",
        "ketchup_bottle": "bottle",
        "lime01": "food",
        "macaroni_and_cheese": "food",
        "mayonnaise_bottle": "bottle",
        "measuring_cups": "utensil",
        "measuring_cups_1": "utensil",
        "measuring_spoon": "utensil",
        "milk_carton": "food",
        "milkjug_a01": "bottle",
        "milkjug_a02": "bottle",
        "milkjug_a03": "bottle",
        "mug": "object",
        "mug_ycb": "object",
        "mustard": "bottle",
        "mustard_bottle": "bottle",
        "naturalbostonroundbottle_a01": "bottle",
        "naturalbostonroundbottle_a02": "bottle",
        "naturalbostonroundbottle_a03": "bottle",
        "oatmeal_raisin_cookies": "food",
        "orange_02": "food",
        "orange_juice_carton": "food",
        "parmesan_cheese_canister": "can",
        "peas_and_carrots": "can",
        "pineapple_slices_can": "can",
        "pitcher": "object",
        "pitcher_ycb": "object",
        "pitted_cherries": "food",
        "plate_small": "plate",
        "popcorn_box": "food",
        "potato_masher": "utensil",
        "pumpkinlarge": "food",
        "pumpkinsmall": "food",
        "raisin_box": "food",
        "ranch_dressing": "bottle",
        "red_block": "toy",
        "rubiks_cube": "toy",
        "scissors": "object",
        "serving_bowl": "container",
        "smartphone": "object",
        "soft_scrub": "bottle",
        "soup_can": "can",
        "spaghetti": "food",
        "spam_can": "can",
        "spatula_01": "utensil",
        "spatula_03": "utensil",
        "spatula_04": "utensil",
        "spatula_05": "utensil",
        "spatula_06": "utensil",
        "spatula_07": "utensil",
        "spatula_08": "utensil",
        "spatula_09": "utensil",
        "spatula_10": "utensil",
        "spatula_11": "utensil",
        "spatula_12": "utensil",
        "spatula_13": "utensil",
        "spatula_16": "utensil",
        "spatula_17": "utensil",
        "spatula_18": "utensil",
        "spatula_19": "utensil",
        "spoon": "utensil",
        "spoon_1": "utensil",
        "spoon_2": "utensil",
        "spoon_big": "utensil",
        "storage_box": "object",
        "sugar_box": "food",
        "tomato_sauce_can": "can",
        "tomato_soup_can": "can",
        "tuna_can": "can",
        "whitepackerbottle_a01": "bottle",
        "whitepackerbottle_a02": "bottle",
        "whitepackerbottle_a03": "bottle",
        "whitepackerbottle_a04": "bottle",
        "wood_block": "toy",
        "wooden_bowl": "container",
        "yellow_block": "toy",
        "yogurt_cup": "food",
    }.items()
}

OBJECTS_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"object_{slot}": RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/object_{slot}",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    spawn.replace(
                        semantic_tags=[
                            *(spawn.semantic_tags or ()),
                            ("_catalog_key", label),
                            *[
                                ("object_count", str(count))
                                for count in range(max(OBJECT_COUNT[0], slot + 1), OBJECT_COUNT[1] + 1)
                            ],
                        ]
                    )
                    for label, spawn in ASSET_CATALOG.items()
                ]
                + [
                    EMPTY_OBJECT_CFG.replace(
                        semantic_tags=[
                            ("_catalog_key", "__empty__"),
                            ("category", "__empty__"),
                            *[("object_count", str(count)) for count in range(OBJECT_COUNT[0], slot + 1)],
                        ]
                    )
                ],
                static=False,
                random_choice=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -10.0, 0.5)),
        )
        for slot in range(OBJECT_COUNT[1])
    },
)
"""Complete object group; inclusion matching selects count, then category and asset per occupied slot."""


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Robot, robot table, ground, lights, cameras and contact sensor of the VLA benchmark tasks.

    An inclusion set samples bounded combinations from the tagged object catalog. The contact sensor
    senses every rigid object and the gripper link against each other.
    """

    clone_cfg = CloneCfg(
        clone_strategy=random,
        clone_combinations=[
            InclusionSet(assets=["objects", "work_table"], match_tags=["object_count"], sample_by="category")
        ],
    )
    objects = OBJECTS_CFG

    robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot",
        init_state=FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.replace(pos=(0.0, 0.0, 0.0)),
    )
    table_fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/scene/franka_table/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.087, 0.0, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
    )
    work_table = EASTRURAL_TABLE_CFG.replace(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[table.spawn for table in TABLE_VARIANTS], static=False, random_choice=False
        )
    )
    # Ground under the shared work surface.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
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
    over_shoulder_left_camera = CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
        spawn=CAMERA_CFG.spawn.replace(focal_length=4.2),
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, 1.4, 1.5), rot=(0.07322545, 0.38665175, 0.90325868, 0.17106223), convention="opengl"
        ),
    )
    wrist_cam = CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/wrist_cam",
        update_latest_camera_pose=True,
        spawn=CAMERA_CFG.spawn.replace(focal_length=2.8),
        offset=CameraCfg.OffsetCfg(pos=(-0.074, 0.031, 0.011), rot=(0.704, -0.692, -0.11, 0.114), convention="opengl"),
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

        concatenate_terms = False

    policy = PolicyCfg()


@configclass
class EventsCfg:
    """Reset the robot and the objects to their start states."""

    reset = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    place_objects = EventTerm(
        func=mdp.reset_tabletop,
        mode="reset",
        params={
            "objects": SceneEntityCfg("objects"),
            "table": SceneEntityCfg("work_table"),
            "table_variants": TABLE_VARIANTS,
            "workspace": ((0.25, -0.4, 0.0), (0.85, 0.4, 0.6)),
            "gripper_body": GRIPPER_BODY.rsplit("/", 1)[-1],
            "clearance": 0.005,
            "max_attempts": 128,
        },
    )


@configclass
class TerminationsCfg:
    """Episode timeout and success of the selected available task."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_success)


@configclass
class VLAHeterogeneousEnvCfg(ManagerBasedRLEnvCfg):
    """Sample any bounded object combination; tasks become available from categories and geometric fit.

    ``OBJECTS_CFG`` declares the object collection before environment construction. Empty placeholders
    keep runtime views rectangular; inclusion matching selects the count, category and concrete assets.
    """

    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1, env_spacing=10.0, lazy_sensor_update=True)
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
            # Explicit shared budgets avoid estimates dominated by inactive variant shape slots.
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="sap", rigid_contact_max=524288, max_triangle_pairs=2097152
            ),
            num_substeps=2,
            # Thousands of small hull meshes: cuBQL takes about 0.1 s to build each one's BVH, LBVH a few ms.
            bvh_constructor_geometry="lbvh",
        ),
    )

    tasks: dict[str, dict[str, str]] = {
        "put_on_plate": {"target": "plate", "relation": "on", "instruction": "Place {object} on {target}."},
        "put_in_container": {"target": "container", "relation": "in", "instruction": "Put {object} in {target}."},
    }
    episode_length_s = 60.0
