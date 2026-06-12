# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Demonstrate a highly heterogeneous multi-robot scene with Selector.

This script shows the core :class:`Selector` workflow without any RL
machinery (no observations, rewards, or training loop).  Each robot
type lives in its own environment group, so a single parallel scene
contains tabletop manipulators, quadrupeds, humanoids, and mobile
manipulators all stepping the same physics together.

Groups (22 total):

* tabletop manipulation (7)
    - ``franka_stack``    -- Franka Panda    + three coloured cubes
    - ``ur10_reach``      -- UR10            (no objects)
    - ``ur10e_reach``     -- UR10e
    - ``openarm_lift``    -- OpenArm         + one DexCube
    - ``kinova_gen3_reach`` -- Kinova Gen3
    - ``sawyer_reach``    -- Sawyer
    - ``flexiv_reach``    -- Flexiv Rizon4s
* quadruped locomotion (6)
    - ``anymal_c_walk``, ``anymal_d_walk``
    - ``unitree_a1_walk``, ``unitree_go1_walk``, ``unitree_go2_walk``
    - ``spot_walk``
* humanoid locomotion (6)
    - ``h1_stand``, ``g1_stand``
    - ``cassie_stand``, ``digit_stand``
    - ``gr1t2_stand``, ``humanoid28_stand``
* mobile manipulation (3)
    - ``ridgeback_franka_mobile``
    - ``agibot_mobile``
    - ``galbot_mobile``

The ``clone_cfg`` on the scene config declares which assets belong to
each group, and ``selector_cfg`` exposes the same groups at runtime.  The
simulation loop uses :class:`Selector` to dispatch per-group resets and
joint targets.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene_for_paper.py --num_envs 88

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Demo: highly heterogeneous multi-robot scene.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=88,
    help="Number of environments to spawn (default 88 -> ~4 envs per group).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg, SelectorCfg, SelectorTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG
from isaaclab_assets.robots.agility import DIGIT_V4_CFG
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG, ANYMAL_D_CFG
from isaaclab_assets.robots.cassie import CASSIE_CFG
from isaaclab_assets.robots.flexiv import FLEXIV_RIZON4S_CFG
from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.galbot import GALBOT_ONE_CHARLIE_CFG
from isaaclab_assets.robots.humanoid_28 import HUMANOID_28_CFG
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from isaaclab_assets.robots.sawyer import SAWYER_CFG
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import (
    G1_MINIMAL_CFG,
    H1_MINIMAL_CFG,
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
)
from isaaclab_assets.robots.universal_robots import UR10_CFG, UR10e_CFG

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Tabletop manipulation
TASK_FRANKA_STACK = "franka_stack"
TASK_UR10_REACH = "ur10_reach"
TASK_UR10E_REACH = "ur10e_reach"
TASK_OPENARM_LIFT = "openarm_lift"
TASK_KINOVA_GEN3_REACH = "kinova_gen3_reach"
TASK_SAWYER_REACH = "sawyer_reach"
TASK_FLEXIV_REACH = "flexiv_reach"

# Quadruped locomotion
TASK_ANYMAL_C_WALK = "anymal_c_walk"
TASK_ANYMAL_D_WALK = "anymal_d_walk"
TASK_UNITREE_A1_WALK = "unitree_a1_walk"
TASK_UNITREE_GO1_WALK = "unitree_go1_walk"
TASK_UNITREE_GO2_WALK = "unitree_go2_walk"
TASK_SPOT_WALK = "spot_walk"

# Humanoid locomotion
TASK_H1_STAND = "h1_stand"
TASK_G1_STAND = "g1_stand"
TASK_CASSIE_STAND = "cassie_stand"
TASK_DIGIT_STAND = "digit_stand"
TASK_GR1T2_STAND = "gr1t2_stand"
TASK_HUMANOID28_STAND = "humanoid28_stand"

# Mobile manipulation
TASK_RIDGEBACK_FRANKA = "ridgeback_franka_mobile"
TASK_AGIBOT_MOBILE = "agibot_mobile"
TASK_GALBOT_MOBILE = "galbot_mobile"

# Floor sits 1.05 m below the world origin so tabletop assets at z=0
# rest on a Seattle-Lab table that extends down to the floor.  Non-
# tabletop robots therefore need their root z shifted onto the floor.
_FLOOR_Z = -1.05

# Per-group joint-perturbation magnitude.  Tabletop arms swing harder
# than legged robots so they remain visually expressive without
# tipping legged platforms over.
_NOISE_SCALE: dict[str, float] = {
    # arms
    TASK_FRANKA_STACK: 0.4,
    TASK_UR10_REACH: 0.4,
    TASK_UR10E_REACH: 0.4,
    TASK_OPENARM_LIFT: 0.4,
    TASK_KINOVA_GEN3_REACH: 0.4,
    TASK_SAWYER_REACH: 0.4,
    TASK_FLEXIV_REACH: 0.4,
    # quadrupeds
    TASK_ANYMAL_C_WALK: 0.08,
    TASK_ANYMAL_D_WALK: 0.08,
    TASK_UNITREE_A1_WALK: 0.08,
    TASK_UNITREE_GO1_WALK: 0.08,
    TASK_UNITREE_GO2_WALK: 0.08,
    TASK_SPOT_WALK: 0.08,
    # humanoids
    TASK_H1_STAND: 0.05,
    TASK_G1_STAND: 0.05,
    TASK_CASSIE_STAND: 0.05,
    TASK_DIGIT_STAND: 0.05,
    TASK_GR1T2_STAND: 0.05,
    TASK_HUMANOID28_STAND: 0.05,
    # mobile manip
    TASK_RIDGEBACK_FRANKA: 0.1,
    TASK_AGIBOT_MOBILE: 0.1,
    TASK_GALBOT_MOBILE: 0.1,
}
_DEFAULT_NOISE = 0.1

_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_TABLE_SEATTLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
_TABLE_PACKING_USD = f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"
_BLOCKS_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks"

# YCB models live on the public Nucleus server. Per-env variety is achieved by
# spawning all four templates via MultiUsdFileCfg and letting the random cloner
# pick one per env.
_YCB_USDS: list[str] = [
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
]

# HDR textures sampled on every env reset (mirrors the reset-mode event
# `randomize_scene_lighting_domelight` from franka_stack_events).
_HDR_TEXTURES: list[str] = [
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
]
_LIGHT_INTENSITY_RANGE: tuple[float, float] = (1500.0, 6000.0)
_LIGHT_COLOR_VARIATION: float = 0.3


def _shift_to_floor(cfg: ArticulationCfg, native_z: float) -> ArticulationCfg:
    """Re-anchor a robot's init root z onto the shared floor.

    Locomotion robots ship with init z = "height above ground"; with
    the floor at ``_FLOOR_Z`` we shift that height by ``_FLOOR_Z`` so
    the feet/wheels rest on the ground plane.
    """
    new_init = cfg.init_state.replace(pos=(0.0, 0.0, _FLOOR_Z + native_z))
    return cfg.replace(init_state=new_init)


# Two table styles, each tuned so the top surface lands at z=0 and is centred
# in front of the robot at (0, 0, 0).
#
# Note: ``stand_instanceable.usd`` (used in ``gear_assembly_env_cfg`` /
# ``deploy/reach/reach_env_cfg``) is NOT a table -- it is a vertical mount
# column placed *under* the robot at (0, 0, 0).  It has no horizontal top for
# objects, so it is omitted here.
_TABLE_PLACEMENT: dict[str, dict] = {
    "seattle": {
        "pos": (0.5, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.707, 0.707),
    },
    # PackingTable origin is at the base; the top is ~1.0 m above (per
    # ``pickplace_unitree_g1_inspire_hand_env_cfg.py`` where objects on
    # the table sit at z=0.9996).  No 90° twist needed.
    "packing": {
        "pos": (0.55, 0.0, -1.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
    },
}


def _table(prim_path: str, style: str = "seattle") -> AssetBaseCfg:
    """Spawn a table in one of two styles: ``seattle`` or ``packing``."""
    if style == "seattle":
        spawn = UsdFileCfg(usd_path=_TABLE_SEATTLE_USD)
    elif style == "packing":
        spawn = UsdFileCfg(usd_path=_TABLE_PACKING_USD)
    else:
        raise ValueError(f"unknown table style: {style!r}")
    placement = _TABLE_PLACEMENT[style]
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=placement["pos"],
            rot=placement["rot"],
        ),
        spawn=spawn,
    )


def _ycb_on_table(
    prim_path: str,
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Random YCB object placed on the table.

    Uses :class:`sim_utils.MultiAssetSpawnerCfg` (which the scene recognises
    as multi-variant) so the random cloner picks one YCB per env.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[UsdFileCfg(usd_path=path) for path in _YCB_USDS],
            rigid_props=_CUBE_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
    )


# ------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------


# Single source of truth: task group -> the scene-asset names active in that group.
# Drives both ``clone_combinations`` (which assets are cloned together per env) and the
# runtime ``SelectorCfg`` terms (how those envs are grouped for resets / actions).
_GROUP_ASSETS: dict[str, list[str]] = {
    # --- tabletop manipulation ---
    TASK_FRANKA_STACK: [
        "franka_robot",
        "franka_table",
        "franka_cube_blue",
        "franka_cube_red",
        "franka_cube_green",
        "franka_ycb_a",
    ],
    TASK_UR10_REACH: ["ur10_robot", "ur10_table", "ur10_ycb_a", "ur10_ycb_b"],
    TASK_UR10E_REACH: ["ur10e_robot", "ur10e_table", "ur10e_ycb_a", "ur10e_ycb_b"],
    TASK_OPENARM_LIFT: ["openarm_robot", "openarm_table", "openarm_cube", "openarm_ycb_a"],
    TASK_KINOVA_GEN3_REACH: ["kinova_robot", "kinova_table", "kinova_ycb_a", "kinova_ycb_b"],
    TASK_SAWYER_REACH: ["sawyer_robot", "sawyer_table", "sawyer_ycb_a", "sawyer_ycb_b"],
    TASK_FLEXIV_REACH: ["flexiv_robot", "flexiv_table", "flexiv_ycb_a", "flexiv_ycb_b"],
    # --- quadruped locomotion ---
    TASK_ANYMAL_C_WALK: ["anymal_c_robot"],
    TASK_ANYMAL_D_WALK: ["anymal_d_robot"],
    TASK_UNITREE_A1_WALK: ["unitree_a1_robot"],
    TASK_UNITREE_GO1_WALK: ["unitree_go1_robot"],
    TASK_UNITREE_GO2_WALK: ["unitree_go2_robot"],
    TASK_SPOT_WALK: ["spot_robot"],
    # --- humanoid locomotion ---
    TASK_H1_STAND: ["h1_robot"],
    TASK_G1_STAND: ["g1_robot"],
    TASK_CASSIE_STAND: ["cassie_robot"],
    TASK_DIGIT_STAND: ["digit_robot"],
    TASK_GR1T2_STAND: ["gr1t2_robot"],
    TASK_HUMANOID28_STAND: ["humanoid28_robot"],
    # --- mobile manipulation ---
    TASK_RIDGEBACK_FRANKA: ["ridgeback_franka_robot"],
    TASK_AGIBOT_MOBILE: ["agibot_robot"],
    TASK_GALBOT_MOBILE: ["galbot_robot"],
}


def _asset_names(asset_cfgs: dict[str, object], names: list[str]) -> tuple[str, ...]:
    """Return the configured asset names from an explicit list (those present in the scene)."""
    return tuple(name for name in names if name in asset_cfgs)


def _term(task: str) -> SelectorTermCfg:
    """Build the selector term selecting the envs that own ``task``'s assets."""
    return SelectorTermCfg(func=_asset_names, params={"names": _GROUP_ASSETS[task]})


@configclass
class MultiRobotSelectorCfg(SelectorCfg):
    """Runtime selector terms, one per task group (names mirror ``_GROUP_ASSETS`` keys)."""

    # --- tabletop manipulation ---
    franka_stack = _term(TASK_FRANKA_STACK)
    ur10_reach = _term(TASK_UR10_REACH)
    ur10e_reach = _term(TASK_UR10E_REACH)
    openarm_lift = _term(TASK_OPENARM_LIFT)
    kinova_gen3_reach = _term(TASK_KINOVA_GEN3_REACH)
    sawyer_reach = _term(TASK_SAWYER_REACH)
    flexiv_reach = _term(TASK_FLEXIV_REACH)
    # --- quadruped locomotion ---
    anymal_c_walk = _term(TASK_ANYMAL_C_WALK)
    anymal_d_walk = _term(TASK_ANYMAL_D_WALK)
    unitree_a1_walk = _term(TASK_UNITREE_A1_WALK)
    unitree_go1_walk = _term(TASK_UNITREE_GO1_WALK)
    unitree_go2_walk = _term(TASK_UNITREE_GO2_WALK)
    spot_walk = _term(TASK_SPOT_WALK)
    # --- humanoid locomotion ---
    h1_stand = _term(TASK_H1_STAND)
    g1_stand = _term(TASK_G1_STAND)
    cassie_stand = _term(TASK_CASSIE_STAND)
    digit_stand = _term(TASK_DIGIT_STAND)
    gr1t2_stand = _term(TASK_GR1T2_STAND)
    humanoid28_stand = _term(TASK_HUMANOID28_STAND)
    # --- mobile manipulation ---
    ridgeback_franka_mobile = _term(TASK_RIDGEBACK_FRANKA)
    agibot_mobile = _term(TASK_AGIBOT_MOBILE)
    galbot_mobile = _term(TASK_GALBOT_MOBILE)


@configclass
class MultiRobotSceneCfg(InteractiveSceneCfg):
    """Scene with 22 heterogeneous robot groups."""

    # Each :class:`InclusionSet` lists the assets cloned together into one env; the default
    # random strategy assigns each env one combination. ``selector_cfg`` re-exposes the same
    # groups for runtime dispatch.
    clone_cfg = CloneCfg(
        clone_combinations=[InclusionSet(assets=list(assets), weight=1) for assets in _GROUP_ASSETS.values()],
    )
    selector_cfg = MultiRobotSelectorCfg()

    # -- shared across ALL envs --------------------------------
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _FLOOR_Z)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # ============================================================
    # Tabletop manipulation
    # ============================================================

    # -- Franka stack (SeattleLab table + 3 stacking cubes + 1 YCB) ---
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka_Robot")
    franka_table = _table("{ENV_REGEX_NS}/Franka_Table", style="seattle")
    franka_cube_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_CubeBlue",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.02)),
        spawn=UsdFileCfg(usd_path=f"{_BLOCKS_DIR}/blue_block.usd", rigid_props=_CUBE_RIGID_PROPS),
    )
    franka_cube_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_CubeRed",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.02)),
        spawn=UsdFileCfg(usd_path=f"{_BLOCKS_DIR}/red_block.usd", rigid_props=_CUBE_RIGID_PROPS),
    )
    franka_cube_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_CubeGreen",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.1, 0.02)),
        spawn=UsdFileCfg(usd_path=f"{_BLOCKS_DIR}/green_block.usd", rigid_props=_CUBE_RIGID_PROPS),
    )
    franka_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/Franka_YCB_A", pos=(0.45, 0.2, 0.15))

    # -- UR10 reach (Packing table + 2 YCB) ----------------------
    ur10_robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10_Robot")
    ur10_table = _table("{ENV_REGEX_NS}/UR10_Table", style="packing")
    ur10_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/UR10_YCB_A", pos=(0.5, 0.15, 0.15))
    ur10_ycb_b = _ycb_on_table("{ENV_REGEX_NS}/UR10_YCB_B", pos=(0.5, -0.15, 0.15))

    # -- UR10e reach (SeattleLab + 2 YCB) ------------------------
    ur10e_robot = UR10e_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10e_Robot")
    ur10e_table = _table("{ENV_REGEX_NS}/UR10e_Table", style="seattle")
    ur10e_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/UR10e_YCB_A", pos=(0.5, 0.15, 0.15))
    ur10e_ycb_b = _ycb_on_table("{ENV_REGEX_NS}/UR10e_YCB_B", pos=(0.5, -0.15, 0.15))

    # -- OpenArm lift (SeattleLab + DexCube + 1 YCB) -------------
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/OpenArm_Robot")
    openarm_table = _table("{ENV_REGEX_NS}/OpenArm_Table", style="seattle")
    openarm_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055)),
        spawn=UsdFileCfg(
            usd_path=f"{_BLOCKS_DIR}/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=_CUBE_RIGID_PROPS,
        ),
    )
    openarm_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/OpenArm_YCB_A", pos=(0.55, 0.15, 0.15))

    # -- Kinova Gen3 reach (Packing table + 2 YCB) ---------------
    kinova_robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Kinova_Robot")
    kinova_table = _table("{ENV_REGEX_NS}/Kinova_Table", style="packing")
    kinova_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/Kinova_YCB_A", pos=(0.5, 0.15, 0.15))
    kinova_ycb_b = _ycb_on_table("{ENV_REGEX_NS}/Kinova_YCB_B", pos=(0.5, -0.15, 0.15))

    # -- Sawyer reach (Packing + 2 YCB) --------------------------
    sawyer_robot = SAWYER_CFG.replace(prim_path="{ENV_REGEX_NS}/Sawyer_Robot")
    sawyer_table = _table("{ENV_REGEX_NS}/Sawyer_Table", style="packing")
    sawyer_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/Sawyer_YCB_A", pos=(0.5, 0.15, 0.15))
    sawyer_ycb_b = _ycb_on_table("{ENV_REGEX_NS}/Sawyer_YCB_B", pos=(0.5, -0.15, 0.15))

    # -- Flexiv Rizon4s reach (Packing table + 2 YCB) ------------
    flexiv_robot = FLEXIV_RIZON4S_CFG.replace(prim_path="{ENV_REGEX_NS}/Flexiv_Robot")
    flexiv_table = _table("{ENV_REGEX_NS}/Flexiv_Table", style="packing")
    flexiv_ycb_a = _ycb_on_table("{ENV_REGEX_NS}/Flexiv_YCB_A", pos=(0.5, 0.15, 0.15))
    flexiv_ycb_b = _ycb_on_table("{ENV_REGEX_NS}/Flexiv_YCB_B", pos=(0.5, -0.15, 0.15))

    # ============================================================
    # Quadruped locomotion (anchored on floor at z = _FLOOR_Z + native)
    # ============================================================

    anymal_c_robot = _shift_to_floor(ANYMAL_C_CFG, native_z=0.6).replace(prim_path="{ENV_REGEX_NS}/AnymalC_Robot")
    anymal_d_robot = _shift_to_floor(ANYMAL_D_CFG, native_z=0.6).replace(prim_path="{ENV_REGEX_NS}/AnymalD_Robot")
    unitree_a1_robot = _shift_to_floor(UNITREE_A1_CFG, native_z=0.42).replace(
        prim_path="{ENV_REGEX_NS}/UnitreeA1_Robot"
    )
    unitree_go1_robot = _shift_to_floor(UNITREE_GO1_CFG, native_z=0.4).replace(
        prim_path="{ENV_REGEX_NS}/UnitreeGo1_Robot"
    )
    unitree_go2_robot = _shift_to_floor(UNITREE_GO2_CFG, native_z=0.4).replace(
        prim_path="{ENV_REGEX_NS}/UnitreeGo2_Robot"
    )
    spot_robot = _shift_to_floor(SPOT_CFG, native_z=0.5).replace(prim_path="{ENV_REGEX_NS}/Spot_Robot")

    # ============================================================
    # Humanoid locomotion
    # ============================================================

    h1_robot = _shift_to_floor(H1_MINIMAL_CFG, native_z=1.05).replace(prim_path="{ENV_REGEX_NS}/H1_Robot")
    g1_robot = _shift_to_floor(G1_MINIMAL_CFG, native_z=0.74).replace(prim_path="{ENV_REGEX_NS}/G1_Robot")
    cassie_robot = _shift_to_floor(CASSIE_CFG, native_z=0.9).replace(prim_path="{ENV_REGEX_NS}/Cassie_Robot")
    digit_robot = _shift_to_floor(DIGIT_V4_CFG, native_z=1.05).replace(prim_path="{ENV_REGEX_NS}/Digit_Robot")
    gr1t2_robot = _shift_to_floor(GR1T2_CFG, native_z=0.95).replace(prim_path="{ENV_REGEX_NS}/GR1T2_Robot")
    humanoid28_robot = _shift_to_floor(HUMANOID_28_CFG, native_z=0.8).replace(
        prim_path="{ENV_REGEX_NS}/Humanoid28_Robot"
    )

    # ============================================================
    # Mobile manipulation
    # ============================================================

    # Ridgeback wheels sit at z=0 in its USD -> place on floor.
    ridgeback_franka_robot = _shift_to_floor(RIDGEBACK_FRANKA_PANDA_CFG, native_z=0.0).replace(
        prim_path="{ENV_REGEX_NS}/RidgebackFranka_Robot"
    )
    # Agibot ships with init z = -1.05 (already on the floor).
    # The default cfg points at an internal NVIDIA dev URL; override to the public Nucleus path.
    agibot_robot = AGIBOT_A2D_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Agibot_Robot",
        spawn=AGIBOT_A2D_CFG.spawn.replace(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Agibot/A2D/A2D_physics.usd",
        ),
    )
    # Galbot ships with init z = -0.8 relative to a different world origin.
    galbot_robot = _shift_to_floor(GALBOT_ONE_CHARLIE_CFG, native_z=0.25).replace(
        prim_path="{ENV_REGEX_NS}/Galbot_Robot"
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _selector_for_asset(selector, asset_name: str) -> str | None:
    """Return the task-selector name that owns an asset (for per-group lookups)."""
    for selector_name, asset_names in selector.selector_assets.items():
        if asset_name in asset_names:
            return selector_name
    return None


def _materialize_env_ids(view_env_ids, device, num_envs: int) -> torch.Tensor:
    """Convert an EnvToViewMap.env_ids (slice or tensor) into a dense tensor."""
    if isinstance(view_env_ids, slice):
        start = view_env_ids.start or 0
        stop = view_env_ids.stop if view_env_ids.stop is not None else num_envs
        return torch.arange(start, stop, device=device)
    return view_env_ids


def print_selector_info(scene: InteractiveScene) -> None:
    """Print a summary of the centralized Selector."""
    selector = scene.selector
    print("\n" + "=" * 70)
    print(f"  Selector  --  {selector}")
    print("=" * 70)
    print(f"  Total envs        : {selector.num_envs}")
    print(f"  Is heterogeneous  : {len(selector.selector_names) > 1}")
    print(f"  Registered groups : {len(selector.selector_names)}")

    for name in selector.selector_names:
        view = selector[name]
        ids = _materialize_env_ids(view.env_ids, scene.device, selector.num_envs)
        print(f"  - {name:30s} count={view.count:3d}  ids={ids.tolist()}")

    print("\n  Asset -> group registry:")
    for cat_name, cat in [
        ("articulations", scene.articulations),
        ("rigid_objects", scene.rigid_objects),
    ]:
        for aname in cat:
            group = _selector_for_asset(selector, aname)
            tag = f"[{cat_name}]"
            print(f"    {aname:28s} {tag:16s} -> group={group!r}")

    print("=" * 70 + "\n")


def reset_articulation(
    scene: InteractiveScene,
    name: str,
    env_ids: torch.Tensor,
) -> None:
    """Reset one articulation using selector-aware local ids."""
    selector = scene.selector
    art = scene[name]
    glob, local = selector.filter_reset_ids(name, env_ids)
    if local.numel() == 0:
        return

    pose = wp.to_torch(art.data.default_root_pose)[local].clone()
    vel = wp.to_torch(art.data.default_root_vel)[local].clone()
    pose[:, :3] += scene.env_origins[glob]
    art.write_root_pose_to_sim_index(root_pose=pose, env_ids=local)
    art.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=local)

    jpos = wp.to_torch(art.data.default_joint_pos)[local].clone()
    jvel = wp.to_torch(art.data.default_joint_vel)[local].clone()
    art.write_joint_position_to_sim_index(position=jpos, env_ids=local)
    art.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=local)


def randomize_dome_light(
    scene: InteractiveScene,
    base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
) -> None:
    """Pick a random HDR / intensity / tinted color for the dome light.

    This mirrors :func:`randomize_scene_lighting_domelight` from the
    franka-stack reset event so the scene visually changes every time
    the envs are re-initialized.
    """
    light_prim = scene["light"].prims[0]

    texture = random.choice(_HDR_TEXTURES)
    intensity = random.uniform(*_LIGHT_INTENSITY_RANGE)
    color = tuple(
        max(0.0, min(1.0, c + random.uniform(-_LIGHT_COLOR_VARIATION, _LIGHT_COLOR_VARIATION))) for c in base_color
    )

    light_prim.GetAttribute("inputs:texture:file").Set(texture)
    light_prim.GetAttribute("inputs:intensity").Set(intensity)
    light_prim.GetAttribute("inputs:color").Set(color)


def reset_scene(
    scene: InteractiveScene,
    env_ids: torch.Tensor | None = None,
    randomize_light: bool = True,
) -> None:
    """Reset all assets using selector-aware dispatching."""
    selector = scene.selector

    if env_ids is None:
        env_ids = torch.arange(scene.num_envs, device=scene.device)

    if randomize_light:
        randomize_dome_light(scene)

    for name in scene.articulations:
        reset_articulation(scene, name, env_ids)

    for obj_name, rigid_obj in scene.rigid_objects.items():
        glob, local = selector.filter_reset_ids(obj_name, env_ids)
        if local.numel() == 0:
            continue
        obj_pose = wp.to_torch(rigid_obj.data.default_root_pose)[local].clone()
        obj_vel = wp.to_torch(rigid_obj.data.default_root_vel)[local].clone()
        obj_pose[:, :3] += scene.env_origins[glob]
        rigid_obj.write_root_pose_to_sim_index(root_pose=obj_pose, env_ids=local)
        rigid_obj.write_root_velocity_to_sim_index(root_velocity=obj_vel, env_ids=local)

    scene.reset(env_ids)


# ------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------


def apply_random_actions(
    scene: InteractiveScene,
    active_global_ids: torch.Tensor,
) -> None:
    """Apply per-group random joint offsets only to *active* envs.

    Each articulation looks up its group's noise scale so legged
    platforms do not get the same perturbation magnitude as arms.
    """
    selector = scene.selector
    for name, art in scene.articulations.items():
        default = wp.to_torch(art.data.default_joint_pos)
        art.set_joint_position_target_index(target=default)

        _, local = selector.filter_reset_ids(name, active_global_ids)
        if local.numel() == 0:
            continue

        key = _selector_for_asset(selector, name)
        scale = _NOISE_SCALE.get(key, _DEFAULT_NOISE) if key is not None else _DEFAULT_NOISE
        n_joints = default.shape[1]
        noise = scale * torch.randn(local.shape[0], n_joints, device=scene.device)
        perturbed = default[local] + noise
        art.set_joint_position_target_index(target=perturbed, joint_ids=None, env_ids=local)


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
) -> None:
    """Loop that wiggles a random subset of envs each interval."""
    selector = scene.selector
    sim_dt = sim.get_physics_dt()
    step = 0
    reset_interval = 150
    resample_interval = 50
    n_active = min(scene.num_envs, max(scene.num_envs // 2, len(selector.selector_names)))
    active: torch.Tensor | None = None

    while simulation_app.is_running():
        if step % reset_interval == 0:
            reset_scene(scene)

        if step % resample_interval == 0:
            perm = torch.randperm(scene.num_envs, device=scene.device)
            active = perm[:n_active].sort().values
            print(f"[step {step:>5d}] active global ids ({active.numel()}): {active.tolist()}")
            for gn in selector.selector_names:
                loc, _ = selector[gn].filter(active)
                print(f"  {gn:30s}: local ids = {loc.tolist()}")

        assert active is not None
        apply_random_actions(scene, active)
        scene.write_data_to_sim()
        sim.step()
        step += 1
        scene.update(sim_dt)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])

    scene_cfg = MultiRobotSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=3.0,
        replicate_physics=False,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print_selector_info(scene)

    print(
        "[INFO] Setup complete -- starting simulation.\n"
        "  A random subset of global env-ids will wiggle each interval.\n"
        "  Watch the console to see how global ids map to per-robot locals.\n"
    )
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
