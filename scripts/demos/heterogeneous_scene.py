# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Demonstrate a multi-robot heterogeneous scene with EnvLayout.

This script shows the core :class:`EnvLayout` workflow without any
RL machinery (no observations, rewards, or training loop).  Three
different robot types are placed in separate environment groups,
each with its own task-specific objects.

Three task groups split the environments evenly:

* **openarm_lift**   -- OpenArm  + one DexCube
* **franka_stack**   -- Franka   + three coloured cubes
* **ur10_reach**     -- UR10     (no objects, pure reaching)

Every robot, table, and object declares ``task_group`` so it only
appears in its group's environments.  The simulation loop uses
:class:`EnvLayout` to dispatch per-group resets and joint targets.

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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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


@configclass
class MultiRobotSceneCfg(InteractiveSceneCfg):
    """Scene with three robot types, each in its own env group.

    ``task_groups`` declares the partition.  Every per-group asset
    sets ``task_group`` so it is only cloned into the matching
    environments.
    """

    task_groups = {
        TASK_OPENARM_LIFT: 1,
        TASK_FRANKA_STACK: 1,
        TASK_UR10_REACH: 1,
    }

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
        task_group=TASK_OPENARM_LIFT,
    )
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
        task_group=TASK_OPENARM_LIFT,
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
        task_group=TASK_OPENARM_LIFT,
    )

    # -- Group 1: Franka + three stacking cubes ----------------
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
        task_group=TASK_FRANKA_STACK,
    )
    franka_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
        task_group=TASK_FRANKA_STACK,
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
        task_group=TASK_FRANKA_STACK,
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
        task_group=TASK_FRANKA_STACK,
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
        task_group=TASK_FRANKA_STACK,
    )

    # -- Group 2: UR10 (no objects) ----------------------------
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
        task_group=TASK_UR10_REACH,
    )
    ur10_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=UsdFileCfg(usd_path=_TABLE_USD),
        task_group=TASK_UR10_REACH,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def print_layout_info(scene: InteractiveScene) -> None:
    """Print a summary of the centralized EnvLayout."""
    layout = scene.layout
    print("\n" + "=" * 60)
    print(f"  EnvLayout  --  {layout}")
    print("=" * 60)
    print(f"  Total envs        : {layout.num_envs}")
    print(f"  Is heterogeneous  : {layout.is_heterogeneous}")
    print(f"  Registered groups : {layout.group_names}")

    for name in layout.group_names:
        ids = layout.env_ids(name)
        print(f"\n  Group '{name}':")
        print(f"    env count : {layout.num_envs_for(name)}")
        print(f"    env ids   : {ids}")
        print(f"    env slice : {layout.env_slice(name)}")

    print("\n  Asset -> group registry:")
    for cat_name, cat in [
        ("articulations", scene.articulations),
        ("rigid_objects", scene.rigid_objects),
    ]:
        for aname in cat:
            group = layout.group_for_asset(aname)
            tag = f"[{cat_name}]"
            print(f"    {aname:22s} {tag:18s} -> group={group!r}")

    print("=" * 60 + "\n")


def reset_articulation(
    scene: InteractiveScene,
    name: str,
    env_ids: torch.Tensor,
) -> None:
    """Reset one articulation using layout-aware local ids."""
    layout = scene.layout
    art = scene[name]
    key = layout.group_for_asset(name)

    if key is not None:
        local, glob = layout.filter_and_split(key, env_ids)
    else:
        local, glob = env_ids, env_ids
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
    """Reset all assets using layout-aware dispatching."""
    layout = scene.layout

    if env_ids is None:
        env_ids = torch.arange(scene.num_envs, device=scene.device)

    # --- Per-group articulations ---
    for name in scene.articulations:
        reset_articulation(scene, name, env_ids)

    # --- Per-group rigid objects ---
    for obj_name, rigid_obj in scene.rigid_objects.items():
        key = layout.group_for_asset(obj_name)
        if key is not None:
            local, glob = layout.filter_and_split(key, env_ids)
        else:
            local, glob = env_ids, env_ids
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

    For each articulation the layout resolves which of the active
    global env-ids actually belong to that robot's group, yielding
    local indices.  Random offsets are written only at those local
    rows; every other environment holds its default joint target.

    This makes the effect of ``global_to_local`` / ``filter_and_split``
    directly *visible*: only the selected environments wiggle.
    """
    layout = scene.layout
    for name, art in scene.articulations.items():
        default = wp.to_torch(art.data.default_joint_pos)
        art.set_joint_position_target_index(target=default)

        key = layout.group_for_asset(name)
        if key is not None:
            local, _ = layout.filter_and_split(key, active_global_ids)
        else:
            local = active_global_ids
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
    is chosen.  ``apply_random_actions`` uses the layout to map
    those global ids to per-robot local indices, so only the
    matching environments wiggle while the rest hold default pose.
    """
    layout = scene.layout
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
            for gn in layout.group_names:
                loc, _ = layout.filter_and_split(gn, active)
                print(f"  {gn:16s}: local ids = {loc.tolist()}")

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

    # --- Show what the layout looks like ---
    print_layout_info(scene)

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
