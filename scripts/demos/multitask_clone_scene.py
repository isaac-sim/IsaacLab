# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a flat multi-robot multi-task scene and step physics only.

This demo exercises heterogeneous clone combinations and selector indexing for
OpenArm-lift, Franka-cabinet, and UR10-reach without constructing a
``ManagerBasedRLEnv``. No action, command, observation, reward, termination,
event, or curriculum managers are created.

Usage:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py --visualizer kit --num_envs 24

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo: clone-only flat multi-robot multi-task scene.")
parser.add_argument("--num_envs", type=int, default=24, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between environment origins [m].")
parser.add_argument("--sim_dt", type=float, default=1.0 / 60.0, help="Physics timestep [s].")
parser.add_argument(
    "--disable_replicate_physics",
    action="store_true",
    help="Disable replicated physics while constructing the heterogeneous scene.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Maximum physics steps before exit. Negative values run until the app closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet, sequential
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg, SelectorCfg, SelectorTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG


_TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
_DEX_CUBE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
_CABINET_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"

_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


def _asset_names(asset_cfgs: dict[str, object], names: list[str]) -> tuple[str, ...]:
    """Return configured asset names from an explicit list."""
    return tuple(name for name in names if name in asset_cfgs)


@configclass
class FlatMultiTaskSelectorCfg(SelectorCfg):
    """Selector terms for the flat clone-only multitask scene."""

    openarm_lift = SelectorTermCfg(
        func=_asset_names,
        params={"names": ["openarm_robot", "openarm_lift_table", "openarm_lift_object"]},
    )
    franka_cabinet = SelectorTermCfg(
        func=_asset_names,
        params={"names": ["franka_robot", "franka_cabinet"]},
    )
    ur10_reach = SelectorTermCfg(
        func=_asset_names,
        params={"names": ["ur10_robot", "ur10_reach_table"]},
    )


@configclass
class FlatMultiRobotMultiTaskSceneCfg(InteractiveSceneCfg):
    """Flat multi-robot multi-task scene with clone combinations only."""

    clone_cfg = CloneCfg(
        clone_strategy=sequential,
        clone_combinations=[
            InclusionSet(assets=["openarm_robot", "openarm_lift_table", "openarm_lift_object"], weight=1),
            InclusionSet(assets=["franka_robot", "franka_cabinet"], weight=1),
            InclusionSet(assets=["ur10_robot", "ur10_reach_table"], weight=1),
        ]
    )
    selector_cfg = FlatMultiTaskSelectorCfg()

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # -- OpenArm lift -----------------------------------------------------
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/OpenArm_Robot")
    openarm_lift_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_LiftTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
    )
    openarm_lift_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_LiftObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(usd_path=_DEX_CUBE_USD, scale=(0.8, 0.8, 0.8), rigid_props=_CUBE_RIGID_PROPS),
    )

    # -- Franka cabinet --------------------------------------------------
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka_Robot")
    franka_cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cabinet",
        spawn=UsdFileCfg(usd_path=_CABINET_USD, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.4),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # -- UR10 reach ------------------------------------------------------
    ur10_robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10_Robot")
    ur10_reach_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ur10_ReachTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
    )


def _format_ids(ids) -> str:
    """Return a compact string for tensor or slice IDs."""
    if isinstance(ids, slice):
        return f"slice({ids.start}, {ids.stop}, {ids.step})"
    return str(ids.detach().cpu().tolist())


def print_scene_summary(scene: InteractiveScene) -> None:
    """Print clone-plan and selector details for the constructed scene."""
    print("\n" + "=" * 72)
    print("Flat multitask clone-only scene")
    print("=" * 72)
    print(f"num_envs             : {scene.num_envs}")
    print(f"articulations        : {sorted(scene.articulations)}")
    print(f"rigid_objects        : {sorted(scene.rigid_objects)}")
    print(f"extras               : {sorted(scene.extras)}")
    print(f"selectors            : {scene.selector.selector_names}")

    for selector_name in scene.selector.selector_names:
        env_to_view = scene.selector[selector_name]
        print(f"\nselector {selector_name!r}")
        print(f"  env ids            : {_format_ids(env_to_view.env_ids)}")
        print(f"  view ids           : {_format_ids(env_to_view.view_ids)}")

    plan = scene.clone_plan
    if plan is None:
        return

    print("\nclone plan")
    for row, (source, destination) in enumerate(zip(plan.sources, plan.destinations)):
        env_ids = plan.clone_mask[row].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        print(f"  row {row:02d}: {source} -> {destination}; envs={env_ids}")


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Step physics for the scene without applying MDP actions."""
    sim_dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running():
        if args_cli.max_steps >= 0 and step >= args_cli.max_steps:
            break
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1


def main() -> None:
    """Run the flat scene-only clone demo."""
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])

    scene_cfg = FlatMultiRobotMultiTaskSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=not args_cli.disable_replicate_physics,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.write_data_to_sim()
    print_scene_summary(scene)
    print("\n[INFO] Setup complete. Stepping physics without MDP managers.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
