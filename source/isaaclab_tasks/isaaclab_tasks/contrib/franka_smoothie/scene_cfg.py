# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene layout for the smoothie task: workbench, robot, containers, lid, blender and packed fruit."""

import math
import os
from pathlib import Path

from isaaclab_newton.sim.schemas import NewtonMeshCollisionPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.utils.configclass import configclass

from ..franka_pour.pour_env_cfg import PourSceneCfg
from .basket_geometry import BASKET_POSITION
from .smoothie_asset import tap_articulation_cfg

ASSETS = Path(__file__).parent / "assets"
CUP_POSITION = (0.57, -0.07, 0.013)
MOTOR_POSITION = (0.64, 0.25, 0.0)
CAP_POSITION = (0.29, -0.075, 0.078)
FRUITS = ("strawberry", "blueberry", "blackberry", "mango")
# Basket-local fruit centers [m], packed with clearance at the reset yaw angles.
FRUIT_LAYOUT = {
    "strawberry": (-0.016504, 0.013801, 0.029275),
    "strawberry_2": (0.011796, -0.025169, 0.107484),
    "strawberry_3": (0.007000, -0.017021, 0.044672),
    "strawberry_4": (0.002145, 0.011383, 0.087000),
    "blueberry": (-0.026109, -0.022603, 0.051921),
    "blueberry_2": (-0.029348, 0.023428, 0.074840),
    "blueberry_3": (0.014665, 0.037911, 0.116507),
    "blueberry_4": (-0.017944, -0.038494, 0.123598),
    "blackberry": (-0.014852, 0.036472, 0.112907),
    "blackberry_2": (-0.017227, -0.029619, 0.077301),
    "blackberry_3": (0.032341, 0.020091, 0.112470),
    "blackberry_4": (0.028775, 0.012079, 0.063778),
    "mango": (-0.010620, 0.016723, 0.060005),
    "mango_2": (-0.005598, -0.003316, 0.012698),
    "mango_3": (-0.024086, 0.010443, 0.131901),
    "mango_4": (0.011269, 0.022617, 0.129368),
}
FRUIT_LAYOUT = {
    name: (BASKET_POSITION[0] + x, BASKET_POSITION[1] + y, BASKET_POSITION[2] + z)
    for name, (x, y, z) in FRUIT_LAYOUT.items()
}
FRUIT_GROUPS = {kind: tuple(name for name in FRUIT_LAYOUT if name.split("_")[0] == kind) for kind in FRUITS}
# Keep the packing rotations of the retained fruits when changing the population.
_FRUIT_RESET_YAW_INDICES = {
    "strawberry": 0,
    "strawberry_2": 1,
    "strawberry_3": 2,
    "strawberry_4": 3,
    "blueberry": 5,
    "blueberry_2": 6,
    "blueberry_3": 7,
    "blueberry_4": 8,
    "blackberry": 11,
    "blackberry_2": 12,
    "blackberry_3": 13,
    "blackberry_4": 14,
    "mango": 16,
    "mango_2": 17,
    "mango_3": 18,
    "mango_4": 19,
}


def rigid(name, file, position):
    """Configure a scene-owned dynamic body at a position [m]."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / file)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


@configclass
class TapSceneCfg(InteractiveSceneCfg):
    """Keyed workbench, robot, cup, threaded lid, blender, spring-button tap and packed fruit."""

    robot = PourSceneCfg().robot.copy()
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.81,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    workstation = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workstation",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "overrides" / "workstation.usda")),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    cup = rigid("Cup", "cup.usda", CUP_POSITION)
    basket = rigid("FruitBasket", "fruit_basket.usda", BASKET_POSITION)
    tap = tap_articulation_cfg()
    blade_cap = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/BladeCap",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "blade_cap.usda")),
        init_state=ArticulationCfg.InitialStateCfg(pos=CAP_POSITION, joint_pos={"Bearing": 0.0}),
        actuators={
            "motor": ImplicitActuatorCfg(
                joint_names_expr=["Bearing"],
                stiffness=0.0,
                damping=0.02,
                joint_effort_limit=0.5,
                joint_velocity_limit=150.0,
            )
        },
    )
    motor = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Motor",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "motor.usda")),
        init_state=ArticulationCfg.InitialStateCfg(pos=MOTOR_POSITION, joint_pos={"PowerButton": 0.0}),
        actuators={
            "button": ImplicitActuatorCfg(
                joint_names_expr=["PowerButton"], stiffness=300.0, damping=3.0, joint_effort_limit=10.0
            )
        },
    )

    def __post_init__(self):
        self.robot.spawn.usd_path = os.environ.get("ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH", str(ASSETS / "franka.usdc"))
        self.robot.actuators["panda_hand"].stiffness = 5000.0
        self.robot.actuators["panda_hand"].damping = 100.0
        self.robot.actuators["panda_hand"].joint_effort_limit = 200.0
        for name, position in FRUIT_LAYOUT.items():
            kind = name.split("_")[0]
            fruit = rigid(name.capitalize(), f"{kind}.{'usda' if kind == 'mango' else 'usdc'}", position)
            yaw = _FRUIT_RESET_YAW_INDICES[name] * 2.399963229728653
            fruit.init_state.rot = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            # Convex hulls keep the packed fruit from interpenetrating in the basket.
            fruit.spawn.collision_props = CollisionPropertiesCfg(
                mesh_collision_property=NewtonMeshCollisionPropertiesCfg(
                    mesh_approximation_name="convexHull", max_hull_vertices=-1
                )
            )
            if kind == "strawberry":
                fruit.spawn.usd_path = str(ASSETS / "overrides" / "strawberry.usda")
            setattr(self, name, fruit)
