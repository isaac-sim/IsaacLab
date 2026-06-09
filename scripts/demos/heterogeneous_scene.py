# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Demonstrate a multi-robot heterogeneous scene with Selector.

This script shows the core :class:`Selector` workflow without any
RL machinery (no observations, rewards, or training loop).  Three
different robot types are placed in separate environment selectors,
each with its own task-specific objects.

Three task groups split the environments evenly:

* **openarm_lift**   -- OpenArm  + one DexCube
* **franka_stack**   -- Franka   + three coloured cubes
* **ur10_reach**     -- UR10     (no objects, pure reaching)

The ``clone_cfg`` on the scene config declares legal asset combinations.
The simulation loop uses :class:`Selector` to dispatch per-selector resets
and joint targets.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py --visualizer kit --num_envs 24

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Demo: multi-robot heterogeneous scene.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=24,
    help="Number of environments to spawn.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg, SelectorCfg, SelectorTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

TASK_OPENARM_LIFT = "openarm_lift"
TASK_FRANKA_STACK = "franka_stack"
TASK_UR10_REACH = "ur10_reach"

_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

_BLOCKS_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks"

# ------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------


def _asset_names(asset_cfgs: dict[str, object], names: list[str]) -> tuple[str, ...]:
    """Return configured asset names from an explicit list."""
    return tuple(name for name in names if name in asset_cfgs)


@configclass
class DemoSelectorCfg(SelectorCfg):
    """Selector terms used by the heterogeneous scene demo."""

    openarm_lift = SelectorTermCfg(
        func=_asset_names,
        params={"names": ["openarm_robot", "openarm_table", "openarm_cube"]},
    )
    franka_stack = SelectorTermCfg(
        func=_asset_names,
        params={"names": ["franka_robot", "franka_table", "franka_cube_blue", "franka_cube_red", "franka_cube_green"]},
    )
    ur10_reach = SelectorTermCfg(
        func=_asset_names,
        params={"names": ["ur10_robot", "ur10_table"]},
    )


@configclass
class MultiRobotSceneCfg(InteractiveSceneCfg):
    """Scene with three robot types, each in its own env selector.

    ``clone_cfg`` declares legal combinations.  Assets listed in each
    :class:`InclusionSet` are only cloned into that group's
    environments.
    """

    clone_cfg = CloneCfg(
        clone_combinations=[
            InclusionSet(assets=["openarm_robot", "openarm_table", "openarm_cube"], weight=1),
            InclusionSet(
                assets=["franka_robot", "franka_table", "franka_cube_blue", "franka_cube_red", "franka_cube_green"],
                weight=1,
            ),
            InclusionSet(assets=["ur10_robot", "ur10_table"], weight=1),
        ]
    )
    selector_cfg = DemoSelectorCfg()

    # -- shared across ALL envs --------------------------------
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.05),
        ),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    # -- Group 0: OpenArm + lift cube -------------------------
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
    )
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
    )
    openarm_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.055),
        ),
        spawn=UsdFileCfg(
            usd_path=(f"{_BLOCKS_DIR}/DexCube/dex_cube_instanceable.usd"),
            scale=(0.8, 0.8, 0.8),
            rigid_props=_CUBE_RIGID_PROPS,
        ),
    )

    # -- Group 1: Franka + three stacking cubes ----------------
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
    )
    franka_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
    )
    franka_cube_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_CubeBlue",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.02),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{_BLOCKS_DIR}/blue_block.usd",
            rigid_props=_CUBE_RIGID_PROPS,
        ),
    )
    franka_cube_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_CubeRed",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.05, 0.02),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{_BLOCKS_DIR}/red_block.usd",
            rigid_props=_CUBE_RIGID_PROPS,
        ),
    )
    franka_cube_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_CubeGreen",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.1, 0.02),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{_BLOCKS_DIR}/green_block.usd",
            rigid_props=_CUBE_RIGID_PROPS,
        ),
    )

    # -- Group 2: UR10 (no objects) ----------------------------
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
    )
    ur10_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def print_selector_info(scene: InteractiveScene) -> None:
    """Print a summary of the centralized Selector."""
    selector = scene.selector
    print("\n" + "=" * 60)
    print(f"  Selector  --  {selector}")
    print("=" * 60)
    print(f"  Total envs          : {selector.num_envs}")
    print(f"  Registered selectors: {selector.selector_names}")

    for name in selector.selector_names:
        env_to_view = selector[name]
        print(f"\n  Selector '{name}':")
        print(f"    env ids: {env_to_view.env_ids}")

    print("\n  Asset -> selector registry:")
    for cat_name, cat in [
        ("articulations", scene.articulations),
        ("rigid_objects", scene.rigid_objects),
    ]:
        for aname in cat:
            tag = f"[{cat_name}]"
            print(f"    {aname:22s} {tag:18s} -> selectors={selector.assets.get(aname)!r}")

    print("=" * 60 + "\n")


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


def reset_scene(
    scene: InteractiveScene,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Reset all assets using selector-aware dispatching."""
    selector = scene.selector

    if env_ids is None:
        env_ids = torch.arange(scene.num_envs, device=scene.device)

    # --- Per-group articulations ---
    for name in scene.articulations:
        reset_articulation(scene, name, env_ids)

    # --- Per-group rigid objects ---
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
    """Apply random joint offsets only to *active* environments.

    For each articulation the selector resolves which of the active
    global env-ids actually belong to that robot, yielding
    local indices.  Random offsets are written only at those local
    rows; every other environment holds its default joint target.

    This makes the env/view split directly visible: only the selected
    environments wiggle.
    """
    selector = scene.selector
    for name, art in scene.articulations.items():
        default = wp.to_torch(art.data.default_joint_pos)
        art.set_joint_position_target_index(target=default)

        _, local = selector.filter_reset_ids(name, active_global_ids)
        if local.numel() == 0:
            continue

        n_joints = default.shape[1]
        noise = 0.4 * torch.randn(local.shape[0], n_joints, device=scene.device)
        perturbed = default[local] + noise
        art.set_joint_position_target_index(target=perturbed, joint_ids=None, env_ids=local)


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
) -> None:
    """Run a loop that randomly perturbs a subset of envs.

    Every ``RESAMPLE_INTERVAL`` steps a new set of global env-ids
    is chosen.  ``apply_random_actions`` uses the selector to map
    those global ids to per-robot local indices, so only the
    matching environments wiggle while the rest hold default pose.
    """
    selector = scene.selector
    sim_dt = sim.get_physics_dt()
    step = 0
    resample_interval = 200
    n_active = min(scene.num_envs // 2, 12)
    active: torch.Tensor | None = None

    while simulation_app.is_running():
        if step % 500 == 0:
            reset_scene(scene)

        if step % resample_interval == 0:
            perm = torch.randperm(scene.num_envs, device=scene.device)
            active = perm[:n_active].sort().values
            print(f"[step {step:>5d}] active global ids = {active.tolist()}")
            for name in selector.selector_names:
                loc, _ = selector[name].filter(active)
                print(f"  {name:16s}: local ids = {loc.tolist()}")

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
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.5])

    scene_cfg = MultiRobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # --- Show what the selector index looks like ---
    print_selector_info(scene)

    print(
        "[INFO] Setup complete -- starting simulation.\n"
        "  A random subset of global env-ids will wiggle;\n"
        "  the rest hold default pose.  Watch the console\n"
        "  to see how global ids map to per-robot locals.\n"
    )
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
