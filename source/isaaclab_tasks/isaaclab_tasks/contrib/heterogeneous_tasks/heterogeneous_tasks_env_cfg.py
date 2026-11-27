# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A flat tabletop scene: bounded object combinations, geometric reset and available task families."""

import math
from dataclasses import MISSING

from isaaclab_isaacsim.sim.schemas import PhysxArticulationCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonArticulationCfg, NewtonSDFCollisionCfg

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

from . import mdp

GRIPPER_BODY = "robot/Gripper/Robotiq_2F_85/left_inner_finger"
"""Gripper link, relative to the environment prim, whose contacts decide whether an object is held."""


CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/camera",
    width=448,
    height=252,
    data_types=["rgb"],
    update_period=0.0,
    renderer_cfg=NewtonWarpRendererCfg(),
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

ROBOT_WORKSPACE = ((0.25, -0.35, -0.10), (0.75, 0.35, 0.50))
"""Hand-defined placement bounds in each environment's frame, intersected with the measured tabletop."""


@configclass
class TableCfg(RigidObjectCfg):
    """A real table asset, its initial pose and its measured usable support region in local metres."""

    surface_outline: tuple[tuple[float, float], ...] = MISSING
    surface_height: float = MISSING
    surface_holes: tuple[tuple[tuple[float, float], ...], ...] = ()


FLOOR_HEIGHT = -0.850005
"""Measured floor height when the stainless table supports the robot base at Z=0."""

# Measured in each USD root frame, in metres. Regions stay on the flat worktop, at least 5 mm inside its
# edge. Packing/metal tables and the desk use their uninterrupted central panels, excluding mounting holes,
# cable grommets and elevated shelves. The bedside table needs a 10 mm inset around its rounded corners.

STAINLESS_TABLE_CFG = TableCfg(
    prim_path="{ENV_REGEX_NS}/work_table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Warehouse/Equipment/"
            "Stainless_Metal_Table_A03/sm_table_metal_stainless_a03_01.usd"
        ),
        func=mdp.spawn_table,
        make_uninstanceable=True,
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        semantic_tags=[
            ("category", "table"),
        ],
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.59844, 0.000565, -0.055237), rot=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    ),
    surface_outline=((-0.891539, -0.341436), (0.890409, -0.341436), (0.890409, 0.339439), (-0.891539, 0.339439)),
    surface_height=0.055237,
)

PACKING_TABLE_C01_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Warehouse/Equipment/"
            "Heavy_Duty_Packing_Table_C01/sm_table_packing_heavyduty_c01_01.usd"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.859741, 0.0, -0.850005), rot=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    ),
    surface_outline=((-0.90807, -0.581883), (0.90807, -0.581883), (0.90807, 0.581882), (-0.90807, 0.581882)),
    surface_height=0.994052,
)

PACKING_TABLE_C02_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Warehouse/Equipment/"
            "Heavy_Duty_Packing_Table_C02/sm_table_packing_heavyduty_c02_01.usd"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.631, 0.0, -0.850005), rot=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    ),
    surface_outline=((-1.213215, -0.353142), (1.213215, -0.353142), (1.213215, 0.353142), (-1.213215, 0.353142)),
    surface_height=0.994051,
)

DELUXE_PACKING_TABLE_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Warehouse/Equipment/"
            "Deluxe_Packing_Table_A01/sm_table_deluxe_packing_a01_01.usd"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.737395, 0.0, -0.850005), rot=(0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
    ),
    surface_outline=((-0.8, -0.469906), (0.8, -0.469906), (0.8, 0.15), (-0.8, 0.15)),
    surface_height=0.927011,
)

SCISSOR_LIFT_TABLE_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Equipment/"
            "Scissor_Mobile_Lift_Table_A01/sm_equipment_liftTable_scissorMobile_a01_01.usd"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.620413, 0.095614, -0.850005), rot=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    ),
    surface_outline=((-0.430533, -0.325908), (0.545674, -0.325908), (0.545674, 0.325908), (-0.430533, 0.325908)),
    surface_height=1.133782,
)

METAL_TABLE_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Warehouse/Equipment/Metal_Table_A02/sm_table_metal_a02_01.usd"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.680092, -0.000723, -0.850005), rot=(0.0, 0.0, 1.0, 0.0)),
    surface_outline=((-0.429456, -0.8), (0.423137, -0.8), (0.423137, 0.8), (-0.429456, 0.8)),
    surface_height=0.867673,
)

BEDSIDE_TABLE_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/SimReady/Hospital/Room_and_Furniture/Table_Bedside_B01/sm_table_bedside_b01_01.usd"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.47609, -0.009427, -0.850184), rot=(0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
    ),
    surface_outline=((-0.207442, -0.215942), (0.207442, -0.215942), (0.207442, 0.215942), (-0.207442, 0.215942)),
    surface_height=0.875,
)

KITCHEN_ISLAND_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Counters/Island_A01/sm_fixture_island_a01_01.usd")
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.565646, 0.0, -0.850005), rot=(0.0, 0.0, 0.0, 1.0)),
    surface_outline=((-0.305646, -0.610446), (0.305646, -0.610446), (0.305646, 0.610446), (-0.305646, 0.610446)),
    surface_height=0.900409,
)

DESK_CFG = STAINLESS_TABLE_CFG.replace(
    spawn=STAINLESS_TABLE_CFG.spawn.replace(
        usd_path=(f"{ISAAC_NUCLEUS_DIR}/SimReady/Hospital/Room_and_Furniture/Desk_A01/sm_desk_a01_01.usd")
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.651, 0.0, -0.85054), rot=(0.0, 0.0, 0.0, 1.0)),
    surface_outline=((-0.39445, -0.5), (0.39445, -0.5), (0.39445, 0.5), (-0.39445, 0.5)),
    surface_height=0.758063,
)

TABLE_VARIANTS = (
    STAINLESS_TABLE_CFG,
    PACKING_TABLE_C01_CFG,
    PACKING_TABLE_C02_CFG,
    DELUXE_PACKING_TABLE_CFG,
    SCISSOR_LIFT_TABLE_CFG,
    METAL_TABLE_CFG,
    BEDSIDE_TABLE_CFG,
    KITCHEN_ISLAND_CFG,
    DESK_CFG,
)
"""Nine explicit NVIDIA-hosted SimReady tables, with measured worktops and floor-aligned poses."""

EMPTY_OBJECT_CFG = sim_utils.SphereCfg(
    radius=0.001,
    visible=False,
    mass_props=sim_utils.MassCfg(mass=0.001),
    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
    collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
)


ASSET_NAMES = {
    "Bowl_F01": "black bowl",
    "Bowl_H01": "beige bowl",
    "Bowl_I01": "brown bowl",
    "Bowl_J01": "brown bowl",
    "Bowl_K01": "brown bowl",
    "Bowl_L01": "white bowl",
    "Bowl_M01": "beige bowl",
    "Bowl_N01": "white bowl",
    "Bowl_O01": "grey bowl",
    "Bowl_P01": "brown bowl",
    "Bowl_Q01": "white bowl",
    "Bowl_R01": "white bowl",
    "Bowl_S01": "green bowl",
    "Bowl_U01": "orange bowl",
    "Kitchenware_Bowl_B01": "beige bowl",
    "Kitchenware_Bowl_C01": "brown bowl",
    "Kitchenware_Bowl_D01": "blue bowl",
    "Kitchenware_Bowl_E01": "beige bowl",
}
"""Names for instructions where the catalog label does not tell assets apart: the bowls, by their color."""

CONTACT_CFG = [sim_utils.UsdPhysicsCollisionCfg(), MujocoCollisionCfg(solref=(0.02, 1.0))]
"""A 20 ms contact response for the gripper and props, resolved with contacts refreshed at each solver substep."""

OBJECT_COLLISION_CFG = [
    sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
    NewtonSDFCollisionCfg(sdf_target_voxel_size=0.00075, sdf_narrow_band_inner=-0.01, sdf_narrow_band_outer=0.01),
]
"""Native SDF collision with 0.75 mm voxels, preserving receiver cavities."""


ASSET_CATALOG = {
    path.split("/")[-2]: sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/SimReady/{path}",
        make_uninstanceable=True,
        move_rigid_body_to_root=True,
        collision_props=CONTACT_CFG,
        mesh_collision_props=OBJECT_COLLISION_CFG,
        semantic_tags=[
            ("category", category),
            *[("name", name) for name in (ASSET_NAMES.get(path.split("/")[-2]),) if name],
        ],
    )
    for path, category in {
        "Residential/Kitchen/Food/Fruit/Granny_Smith_Apple_A01/sm_food_produce_apple_grannySmith_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Red_Delicious_Apple_A01/sm_food_produce_apple_redDelicous_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Banana_A01/sm_food_produce_banana_a01_01.usd": "food",
        "Residential/Kitchen/Dishware/Plastic_Small_Plate_A01/sm_dishware_plastic_smallPlate_a01_01.usd": "plate",
        "Residential/Kitchen/Dishware/Bowl_U01/sm_kitchenware_bowl_u01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_H01/sm_kitchenware_bowl_h01_01.usd": "container",
        "Industrial/Tools/Hand/Socket_B01_03/sm_handTool_socket_5_8in_b01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_1_2in_B01/sm_handTool_socket_1_2in_b01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_B01_07/sm_handTool_socket_9_16in_b01_01.usd": "tool",
        "Residential/Kitchen/Cooking_Gear/Tray_A01/sm_kitchenware_tray_a01_01.usd": "tray",
        "Residential/Kitchen/Dishware/Bowl_F01/sm_kitchenware_bowl_f01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_N01/sm_kitchenware_bowl_n01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_Q01/sm_kitchenware_bowl_q01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_S01/sm_kitchenware_bowl_s01_01.usd": "container",
        "Residential/Kitchen/Dishware/Kitchenware_Bowl_B01/sm_kitchenware_bowl_b01_01.usd": "container",
        "Residential/Kitchen/Dishware/Kitchenware_Bowl_D01/sm_kitchenware_bowl_d01_01.usd": "container",
        "Residential/Kitchen/Dishware/Plastic_Tray_A01/sm_dishware_plastic_tray_a01_01.usd": "tray",
        "Residential/Kitchen/Food/Fruit/Orange_A01_02/sm_food_produce_orange_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Banana_Bunch_A01/sm_food_produce_banana_bunch_a01_01.usd": "food",
        "Residential/Kitchen/Food/Vegetables/Eggplant_A01/sm_food_produce_eggplant_a01_01.usd": "food",
        "Residential/Kitchen/Food/Vegetables/Garlic_Bulb_A01/sm_food_produce_garlic_bulb_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Lime_A01/sm_food_produce_lime_a01_01.usd": "food",
        "Residential/Kitchen/Food/Vegetables/Onion_A01/sm_food_produce_onion_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Pear_A01/sm_food_produce_pear_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Strawberry_A01/sm_food_produce_strawberry_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Tangerine_A01/sm_food_produce_tangerine_a01_01.usd": "food",
        "Residential/Kitchen/Food/Vegetables/Zucchini_A01/sm_food_produce_zucchini_a01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Lime_A02/sm_food_produce_lime_a02_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_B01/sm_food_prepared_bagel_b01_01.usd": "food",
        "Residential/Kitchen/Food/Vegetables/Onion_B01/sm_food_produce_onion_b01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_C01/sm_food_prepared_bagel_c01_01.usd": "food",
        "Residential/Kitchen/Food/Fruit/Mango_C01/sm_food_produce_mango_c01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_D01/sm_food_prepared_bagel_d01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_E01/sm_food_prepared_bagel_e01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_F01/sm_food_prepared_bagel_f01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_G01/sm_food_prepared_bagel_g01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_H01/sm_food_prepared_bagel_h01_01.usd": "food",
        "Residential/Kitchen/Food/Breads/Bagel_I01/sm_food_prepared_bagel_i01_01.usd": "food",
        "Residential/Kitchen/Food/Beverages/Beer_Bottle_A01/sm_food_beverage_beerBottle_a01_01.usd": "bottle",
        "Residential/Kitchen/Food/Condiments/Ketchup_A01/sm_food_condiment_ketchup_a01_01.usd": "bottle",
        "Residential/Kitchen/Food/Condiments/Ketchup_A02/sm_food_condiment_ketchup_a02_01.usd": "bottle",
        "Residential/Kitchen/Food/Condiments/Ketchup_A03/sm_food_condiment_ketchup_a03_01.usd": "bottle",
        "Residential/Kitchen/Food/Condiments/Jam_C01/sm_food_processed_jam_c01_01.usd": "bottle",
        "Residential/Kitchen/Food/Beverages/Beer_Bottle_O01/sm_food_beverage_beerBottle_o01_01.usd": "bottle",
        "Residential/Kitchen/Food/Canned_Goods/Can_B01/sm_food_beverage_can_b01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_B01/sm_food_processed_cannedFood_b01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_C01/sm_food_beverage_can_c01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_C01/sm_food_processed_cannedFood_c01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_D01/sm_food_beverage_can_d01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_D01/sm_food_processed_cannedFood_d01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_E01/sm_food_beverage_can_e01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_E01/sm_food_processed_cannedFood_e01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_F01/sm_food_processed_cannedFood_f01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_G01/sm_food_beverage_can_g01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_H01/sm_food_beverage_can_h01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_I01/sm_food_beverage_can_i01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_J01/sm_food_beverage_can_j01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_J01/sm_food_processed_cannedFood_j01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_K01/sm_food_beverage_can_k01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_K01/sm_food_processed_cannedFood_k01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_L01/sm_food_beverage_can_l01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_L01/sm_food_processed_cannedFood_l01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_M01/sm_food_beverage_can_m01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_M01/sm_food_processed_cannedFood_m01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Can_N01/sm_food_beverage_can_n01_01.usd": "can",
        "Residential/Kitchen/Food/Canned_Goods/Canned_Food_N01/sm_food_processed_cannedFood_n01_01.usd": "can",
        "Residential/Kitchen/Utensils/Cup_A01/sm_kitchenware_cup_a01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_A01/sm_kitchenware_fork_a01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Ladle_A01/sm_kitchenware_ladle_a01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Spoon_Wooden_A01/sm_utensil_spoon_wooden_a01_01.usda": "utensil",
        "Residential/Kitchen/Utensils/Fork_B01/sm_kitchenware_fork_b01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_C01/sm_kitchenware_fork_c01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_D01/sm_kitchenware_fork_d01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_E01/sm_kitchenware_fork_e01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_F01/sm_kitchenware_fork_f01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_G01/sm_kitchenware_fork_g01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_H01/sm_kitchenware_fork_h01_01.usd": "utensil",
        "Residential/Kitchen/Utensils/Fork_I01/sm_kitchenware_fork_i01_01.usd": "utensil",
        "Industrial/Tools/Hand/Socket_B01_04/sm_handTool_socket_11_32in_b01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_B01_05/sm_handTool_socket_15_32in_b01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_B01_06/sm_handTool_socket_5_16in_b01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_B01_08/sm_handTool_socket_21_32in_b01_01.usd": "tool",
        "Industrial/Tools/Hand/Screwdriver_Phillips_Ph1_A01/sm_handTool_screwdriver_phillips_ph1_a01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_1_3_16in_A01/sm_handTool_socket_1_3_16in_a01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_1_5_16in_A01/sm_handTool_socket_1_5_16in_a01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_3_4in_A01/sm_handTool_socket_3_4in_a01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_3_8in_A01/sm_handTool_socket_3_8in_a01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_5_16in_A01/sm_handTool_socket_5_16in_a01_01.usd": "tool",
        "Industrial/Tools/Hand/Socket_5_8in_A01/sm_handTool_socket_5_8in_a01_01.usd": "tool",
        "Residential/Kitchen/Dishware/Kitchenware_Bowl_C01/sm_kitchenware_bowl_c01_01.usd": "container",
        "Residential/Kitchen/Dishware/Kitchenware_Bowl_E01/sm_kitchenware_bowl_e01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_I01/sm_kitchenware_bowl_i01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_J01/sm_kitchenware_bowl_j01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_K01/sm_kitchenware_bowl_k01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_L01/sm_kitchenware_bowl_l01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_M01/sm_kitchenware_bowl_m01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_O01/sm_kitchenware_bowl_o01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_P01/sm_kitchenware_bowl_p01_01.usd": "container",
        "Residential/Kitchen/Dishware/Bowl_R01/sm_kitchenware_bowl_r01_01.usd": "container",
        "Residential/Personal_Items/Notebook_A02/sm_misc_notebook_a02_01.usd": "object",
    }.items()
}
"""100 NVIDIA-hosted SimReady props: relative USD paths, sampling categories and maximum world counts.

The known failing large plastic plate is excluded. Reduced SDF collision meshes preserve receiver cavities;
SimReady's PhysX validation alone does not certify contact behavior on Newton.
"""

OBJECTS_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"object_{slot}": RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/object_{slot}",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    spawn.replace(semantic_tags=[*spawn.semantic_tags, ("_catalog_key", label)])
                    for label, spawn in ASSET_CATALOG.items()
                ]
                + [EMPTY_OBJECT_CFG.replace(semantic_tags=[("_catalog_key", "__empty__"), ("category", "__empty__")])],
                static=False,
                random_choice=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -10.0, 0.5)),
        )
        for slot in range(OBJECT_COUNT[1])
    },
)
"""Identical slots: every catalog asset and a non-colliding placeholder are available in every world."""


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Robot, robot table, ground, lights, cameras and contact sensor of the VLA benchmark tasks.

    An inclusion set samples bounded combinations from the tagged object catalog. The contact sensor
    senses every rigid object and the gripper link against each other.
    """

    clone_cfg = CloneCfg(
        clone_strategy=random,
        clone_combinations=[InclusionSet(assets=["objects", "work_table"], sample_by="category")],
    )
    objects = OBJECTS_CFG

    robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot",
        # Preserve the original RoboLab home pose, collision setting and arm impedance.
        init_state=FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                **FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos,
                "panda_joint2": -math.pi / 5,
                "panda_joint4": -4 * math.pi / 5,
                "panda_joint6": 3 * math.pi / 5,
                "panda_joint7": 0.0,
            },
        ),
        spawn=FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.replace(
            activate_contact_sensors=True,
            collision_props={"/.*": CONTACT_CFG},
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=64, solver_velocity_iteration_count=0
                ),
                NewtonArticulationCfg(self_collision_enabled=False),
            ],
        ),
        actuators={
            **FRANKA_ROBOTIQ_GRIPPER_CFG.actuators,
            "panda_shoulder": FRANKA_ROBOTIQ_GRIPPER_CFG.actuators["panda_shoulder"].replace(
                stiffness=400.0, damping=80.0, armature=0.1
            ),
            "panda_forearm": FRANKA_ROBOTIQ_GRIPPER_CFG.actuators["panda_forearm"].replace(
                stiffness=400.0, damping=80.0, armature=0.1
            ),
        },
    )
    # Reuse the stainless SimReady table as the fixed robot mount; its top is exactly at the base frame.
    table_fixture = STAINLESS_TABLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        init_state=STAINLESS_TABLE_CFG.init_state.replace(pos=(-0.25, 0.000565, -0.055237)),
    )
    work_table = STAINLESS_TABLE_CFG.replace(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[table.spawn for table in TABLE_VARIANTS], static=False, random_choice=False
        )
    )
    # Ground under the shared work surface.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, FLOOR_HEIGHT)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(intensity=1.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/key_light",
        spawn=sim_utils.DistantLightCfg(intensity=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3250576, 0.3250576, 0.0, 0.8880738)),
    )
    # The exterior view of the DROID setup: over the robot's left shoulder, looking at the workspace.
    over_shoulder_left_camera = CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.57, 0.66), rot=(-0.195, 0.399, 0.805, -0.393), convention="opengl"),
    )
    wrist_cam = CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        update_latest_camera_pose=True,
        spawn=CAMERA_CFG.spawn.replace(focal_length=2.8),
        offset=CameraCfg.OffsetCfg(pos=(0.011, -0.031, -0.074), rot=(0.57, 0.576, -0.409, -0.42), convention="opengl"),
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
            "object_count": OBJECT_COUNT,
            "table": SceneEntityCfg("work_table"),
            "table_variants": TABLE_VARIANTS,
            "workspace": ROBOT_WORKSPACE,
            "gripper_body": GRIPPER_BODY.rsplit("/", 1)[-1],
            "clearance": 0.005,
            "max_attempts": 128,
            # The Robotiq 2F-85 opens 85 mm and cannot pinch anything thinner than about 2 cm off the table.
            "graspable_size": (0.02, 0.085),
        },
    )


@configclass
class TerminationsCfg:
    """Episode timeout and success of the selected available task."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_success)


@configclass
class TabletopPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Sample bounded object combinations; tasks become available from categories and geometric fit.

    ``OBJECTS_CFG`` declares the object collection before environment construction. Empty placeholders
    keep runtime views rectangular. Every reset samples the count, category and concrete assets anew.
    Any environment can select any task family; its composition includes a compatible object and receiver.
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
                broad_phase="sap", rigid_contact_max=1048576, max_triangle_pairs=33554432
            ),
            sdf_mesh_reduction=0.75,
            num_substeps=4,
            collision_decimation=1,
            # SAH keeps dense mesh-overlap queries stable across reset-time asset changes.
            bvh_constructor_geometry="sah",
        ),
    )

    tasks: dict[str, dict[str, str]] = {
        # No plate placement yet: the only plate in the catalog is a 10 cm dish that does not look like a plate.
        "put_in_container": {
            "target": "container",
            "relation": "in",
            "instruction": "Put the {object} in the {target}.",
        },
        "put_on_tray": {"target": "tray", "relation": "on", "instruction": "Place the {object} on the {target}."},
    }
    episode_length_s = 60.0
