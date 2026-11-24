# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A flat tabletop scene: bounded object combinations, geometric reset and available task families."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
class TabletopSceneCfg(InteractiveSceneCfg):
    """Robot, robot table, ground, lights, cameras and contact sensor of the VLA benchmark tasks.

    An inclusion set samples bounded combinations from the tagged object catalog below. The contact sensor
    senses every rigid object and the gripper link against each other.
    """

    clone_cfg = CloneCfg(clone_strategy=random)

    robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot",
        init_state=FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.replace(pos=(0.0, 0.0, 0.0)),
    )
    table_fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/scene/franka_table/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.087, 0.0, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
    )
    work_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/work_table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.0, 0.06),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.28, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, -0.03)),
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
    place_objects = EventTerm(
        func=mdp.reset_tabletop,
        mode="reset",
        params={
            "objects": [],
            "table": SceneEntityCfg("work_table"),
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


@configclass
class VLAHeterogeneousEnvCfg(ManagerBasedRLEnvCfg):
    """Sample any bounded object combination; tasks become available from categories and geometric fit.

    ``object_count`` bounds the number of real objects, excluding the table and robot. Empty
    placeholders keep runtime views rectangular. Inclusion matching encodes the permitted counts
    as factorized groups; category-first inclusion sampling selects only the requested worlds.
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
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
            num_substeps=2,
            # Thousands of small hull meshes: cuBQL takes about 0.1 s to build each one's BVH, LBVH a few ms.
            bvh_constructor_geometry="lbvh",
        ),
    )

    objects: dict[str, sim_utils.SpawnerCfg] = ASSET_CATALOG
    object_count: tuple[int, int] = (2, 8)
    tasks: dict[str, dict[str, str]] = {
        "put_on_plate": {"target": "plate", "relation": "on", "instruction": "Place {object} on {target}."},
        "put_in_container": {"target": "container", "relation": "in", "instruction": "Put {object} in {target}."},
    }
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        lower, upper = self.object_count
        if not self.objects or not 1 <= lower <= upper:
            raise ValueError("Select a nonempty object catalog and positive, ordered object counts.")
        for name, rule in self.tasks.items():
            if set(rule) != {"target", "relation", "instruction"} or rule["relation"] not in ("on", "in"):
                raise ValueError(
                    f"Task family {name!r} requires a target category, relation ('on' or 'in') and instruction."
                )
        previous = self.events.place_objects.params["objects"]
        for name in previous:
            vars(self.scene).pop(name, None)
        names = []
        for slot in range(upper):
            name = f"object_{slot}"
            names.append(name)
            candidates = []
            for label, spawn in self.objects.items():
                spawn = spawn.copy()
                spawn.semantic_tags = [
                    *(spawn.semantic_tags or ()),
                    ("_catalog_key", label),
                    *[("object_count", str(count)) for count in range(max(lower, slot + 1), upper + 1)],
                ]
                candidates.append(spawn)
            candidates.append(
                sim_utils.SphereCfg(
                    radius=0.001,
                    visible=False,
                    mass_props=sim_utils.MassCfg(mass=0.001),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    semantic_tags=[
                        ("_catalog_key", "__empty__"),
                        ("category", "__empty__"),
                        *[("object_count", str(count)) for count in range(lower, slot + 1)],
                    ],
                )
            )
            setattr(
                self.scene,
                name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=candidates, static=False, random_choice=False),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -10.0, 0.5)),
                ),
            )
        self.events.place_objects.params["objects"] = names
        self.scene.clone_cfg.clone_combinations = [
            InclusionSet(assets=names, match_tags=["object_count"], sample_by="category")
        ]
