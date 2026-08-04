# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstration of heterogeneous bin-packing layouts with the cloner API.

The scene declares a bin layout as a clone combination on its
:class:`~isaaclab.cloner.CloneCfg`, and the replication pipeline then spawns
only the assets named by each combination, cycling environments through the
declared layouts. Instead of listing those assets by hand, this script defines
its own combination set, ``RandomSubsetSet``, which draws a random subset of
the grocery slots: the cloner only reads ``assets`` and ``weight`` off a
combination, so any subclass of :class:`~isaaclab.cloner.InclusionSet` that
fills them in is a valid combination. The slots cycle through eight YCB grocery
models, spawned by a custom spawner that authors the rigid bodies and colliders
the visual models ship without. The simulation periodically re-drops every
spawned grocery into its bin with pose, velocity, and mass noise, teleporting
out-of-bounds objects back in.

.. note::
    Heterogeneous per-environment object counts require the PhysX backend, so
    this demo runs on PhysX only.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/bin_packing.py

"""

from __future__ import annotations

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Demo of heterogeneous bin-packing layouts through the cloner API",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import math
from dataclasses import MISSING
from random import Random

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

##
# Pre-defined configs
##
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet, sequential
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import schemas
from isaaclab.utils import Timer
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.scene import InteractiveScene

##
# Scene Configuration
##

# Layout and spawn counts.
MAX_OBJECTS_PER_BIN = 24  # Maximum active objects we plan to fit inside the bin.
MIN_OBJECTS_PER_BIN = 1  # Lower bound for randomized active object count.
NUM_LAYOUTS = 16  # Number of distinct bin layouts declared; environments cycle through them.
NUM_OBJECTS_PER_LAYER = 4  # Number of groceries spawned on each layer of the active stack.
ACTIVE_LAYER_SPACING = 0.1  # Vertical spacing (m) between layers inside the bin.
BIN_DIMENSIONS = (0.2, 0.3, 0.15)  # Physical size (m) of the storage bin.
BIN_XY_BOUND = ((-0.2, -0.3), (0.2, 0.3))  # Valid XY region (min/max) for active groceries.

# Randomization ranges (radians for rotations, m/s and rad/s for velocities).
POSE_RANGE = {"roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)}
VELOCITY_RANGE = {"roll": (-0.2, 1.0), "pitch": (-0.2, 1.0), "yaw": (-0.2, 1.0)}

GROCERY_USDS = [
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
]

# The ``Axis_Aligned`` YCB models are visual meshes, and only four of them ship a physics-ready
# twin under ``Axis_Aligned_Physics``. Those twins reference the visual mesh and add a rigid
# body on the root and a convex-hull collider on the mesh; :func:`spawn_grocery` authors the
# same thing so all eight models can be dropped into a bin.
GROCERY_MASS = 0.3  # Spawn mass [kg]; the simulation loop re-randomizes it on every reset.

GROCERY_NAMES = [f"grocery_{slot:02d}" for slot in range(MAX_OBJECTS_PER_BIN)]


def grocery_spawn_poses(device: str = "cpu") -> torch.Tensor:
    """Create the stacked spawn grid above the bin, one pose per grocery slot.

    The slots fill layer by layer: slot ``i`` sits on layer ``i // NUM_OBJECTS_PER_LAYER``
    at position ``i % NUM_OBJECTS_PER_LAYER`` of a small XY grid over the bin opening.

    Args:
        device: Torch device for allocation (e.g., ``"cuda:0"`` or ``"cpu"``).

    Returns:
        Spawn poses in the local environment frame [m], shape
        ``(MAX_OBJECTS_PER_BIN, 7)`` with ``[x, y, z, qx, qy, qz, qw]``.
    """
    bin_x_dim, bin_y_dim, bin_z_dim = BIN_DIMENSIONS
    num_layers = math.ceil(MAX_OBJECTS_PER_BIN / NUM_OBJECTS_PER_LAYER)
    num_x_objects = math.ceil(math.sqrt(NUM_OBJECTS_PER_LAYER))
    num_y_objects = math.ceil(NUM_OBJECTS_PER_LAYER / num_x_objects)
    # Create a 3D grid that stacks NUM_OBJECTS_PER_LAYER objects per layer above the bin.
    x = torch.linspace(-bin_x_dim * (2 / 6), bin_x_dim * (2 / 6), num_x_objects, device=device)
    y = torch.linspace(-bin_y_dim * (2 / 6), bin_y_dim * (2 / 6), num_y_objects, device=device)
    z = torch.linspace(0, ACTIVE_LAYER_SPACING * (num_layers - 1), num_layers, device=device) + bin_z_dim * 2
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")  # Note Z first, this stacks the layers.
    positions = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)[:MAX_OBJECTS_PER_BIN]
    ref_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(len(positions), 1)
    return torch.cat((positions, ref_quat), dim=-1)


def spawn_grocery(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a visual YCB model and author the physics schemas it does not ship with.

    These models carry no physics schemas, so the ``rigid_props`` and ``collision_props`` of a
    spawner configuration would find nothing to modify. This spawner defines each schema
    instead: the rigid body on the asset root and a convex-hull collider on the mesh, matching
    how the physics-ready YCB models are authored.

    Args:
        prim_path: Prim path to spawn the asset at.
        cfg: Spawner configuration naming the USD file.
        translation: Translation w.r.t. the parent prim [m].
        orientation: Orientation w.r.t. the parent prim as ``(w, x, y, z)``.
        **kwargs: Forwarded to :func:`~isaaclab.sim.spawn_from_usd`.

    Returns:
        The prim of the spawned asset.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    root_path = prim.GetPath().pathString
    for mesh in sim_utils.get_all_matching_child_prims(root_path, lambda child: child.GetTypeName() == "Mesh"):
        mesh_path = mesh.GetPath().pathString
        schemas.define_collision_properties(mesh_path, schemas.CollisionBaseCfg(collision_enabled=True))
        schemas.define_mesh_collision_properties(
            mesh_path, schemas.MeshCollisionBaseCfg(mesh_approximation_name="convexHull")
        )
    schemas.define_rigid_body_properties(root_path, sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4))
    schemas.define_mass_properties(root_path, schemas.MassPropertiesCfg(mass=GROCERY_MASS))
    return prim


def grocery_cfg(slot: int) -> RigidObjectCfg:
    """Declare the grocery that occupies one slot of the spawn grid.

    Args:
        slot: Index into the spawn grid built by :func:`grocery_spawn_poses`.

    Returns:
        The rigid-object configuration for that slot.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Groceries/Grocery_{slot:02d}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(grocery_spawn_poses()[slot, :3].tolist())),
        spawn=sim_utils.UsdFileCfg(func=spawn_grocery, usd_path=GROCERY_USDS[slot % len(GROCERY_USDS)]),
    )


@configclass
class RandomSubsetSet(InclusionSet):
    """Clone combination that activates a random subset of a pool of assets.

    The cloner reads only :attr:`~isaaclab.cloner.InclusionSet.assets` and
    :attr:`~isaaclab.cloner.InclusionSet.weight` off a combination, so extending
    the combination vocabulary is a matter of subclassing
    :class:`~isaaclab.cloner.InclusionSet` and filling in the active assets.
    """

    pool: list[str] = MISSING
    """Scene asset names eligible for inclusion."""

    num_active: tuple[int, int] = MISSING
    """Inclusive lower and upper bound on the number of active assets."""

    seed: int | None = None
    """Seed for the draw. ``None`` draws a different subset for every instance."""

    def __post_init__(self):
        """Draw the active subset that this combination declares."""
        rng = Random(self.seed)
        self.assets = rng.sample(self.pool, rng.randint(*self.num_active))


@configclass
class BinPackingSceneCfg(InteractiveSceneCfg):
    """Bin-packing scene with per-environment heterogeneous grocery layouts.

    Besides the world-shared ground plane and light and the storage bin, the
    scene declares one grocery slot per object that may end up in the bin, and
    one :class:`RandomSubsetSet` clone combination per layout to pick which of
    those slots are active.
    """

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(2.0, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )

    # grocery slots, one per object that may end up in the bin
    grocery_00: RigidObjectCfg = grocery_cfg(0)
    grocery_01: RigidObjectCfg = grocery_cfg(1)
    grocery_02: RigidObjectCfg = grocery_cfg(2)
    grocery_03: RigidObjectCfg = grocery_cfg(3)
    grocery_04: RigidObjectCfg = grocery_cfg(4)
    grocery_05: RigidObjectCfg = grocery_cfg(5)
    grocery_06: RigidObjectCfg = grocery_cfg(6)
    grocery_07: RigidObjectCfg = grocery_cfg(7)
    grocery_08: RigidObjectCfg = grocery_cfg(8)
    grocery_09: RigidObjectCfg = grocery_cfg(9)
    grocery_10: RigidObjectCfg = grocery_cfg(10)
    grocery_11: RigidObjectCfg = grocery_cfg(11)
    grocery_12: RigidObjectCfg = grocery_cfg(12)
    grocery_13: RigidObjectCfg = grocery_cfg(13)
    grocery_14: RigidObjectCfg = grocery_cfg(14)
    grocery_15: RigidObjectCfg = grocery_cfg(15)
    grocery_16: RigidObjectCfg = grocery_cfg(16)
    grocery_17: RigidObjectCfg = grocery_cfg(17)
    grocery_18: RigidObjectCfg = grocery_cfg(18)
    grocery_19: RigidObjectCfg = grocery_cfg(19)
    grocery_20: RigidObjectCfg = grocery_cfg(20)
    grocery_21: RigidObjectCfg = grocery_cfg(21)
    grocery_22: RigidObjectCfg = grocery_cfg(22)
    grocery_23: RigidObjectCfg = grocery_cfg(23)

    clone_cfg: CloneCfg = CloneCfg(
        clone_combinations=[
            RandomSubsetSet(pool=GROCERY_NAMES, num_active=(MIN_OBJECTS_PER_BIN, MAX_OBJECTS_PER_BIN))
            for _ in range(NUM_LAYOUTS)
        ],
        clone_strategy=sequential,
    )


##
# Simulation Loop
##


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Runs the simulation loop that coordinates spawn randomization and stepping.

    A heterogeneous scene gives each grocery slot its own physics view whose
    instance count is the number of environments containing that slot, not
    ``num_envs``.

    Returns:
        None: The simulator side-effects are applied through ``scene`` and ``sim``.
    """
    device = scene.device
    # Resolve each spawned slot once: (asset, env ids owning an instance, world spawn poses).
    # The spawn pose is the asset's default (init-state) pose, which carries the jittered
    # grid position sampled by :func:`grocery_spawn_poses` at scene-config build time.
    groceries = []
    for name in GROCERY_NAMES:
        asset = scene[name]
        prim_paths = getattr(asset.root_view, "prim_paths", None)
        if prim_paths is not None and len(prim_paths) == asset.num_instances:
            env_ids = [int(path.partition("/env_")[2].partition("/")[0]) for path in prim_paths]
        else:
            # Homogeneous fallback: one instance per environment.
            env_ids = range(asset.num_instances)
        env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=device)
        spawn_pose_w = asset.data.default_root_pose.torch.clone()
        spawn_pose_w[:, :3] += scene.env_origins[env_ids]
        groceries.append((asset, env_ids, spawn_pose_w))

    pose_ranges = torch.tensor([POSE_RANGE[axis] for axis in ("roll", "pitch", "yaw")], device=device)
    velocity_ranges = torch.tensor([VELOCITY_RANGE[axis] for axis in ("roll", "pitch", "yaw")], device=device)
    bounds_xy = torch.as_tensor(BIN_XY_BOUND, device=device)
    sim_dt = sim.get_physics_dt()
    count = 0

    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        if count % 250 == 0:
            # reset counter
            count = 0
            # Retrieve positions
            with Timer("[INFO] Time to reset scene: "):
                for asset, env_ids, spawn_pose_w in groceries:
                    num_instances = len(env_ids)
                    # Only randomize the [roll, pitch, yaw] rotation; keep the spawn position fixed.
                    noise = math_utils.sample_uniform(
                        pose_ranges[:, 0], pose_ranges[:, 1], (num_instances, 3), device=device
                    )
                    pose = spawn_pose_w.clone()
                    noise_quat = math_utils.quat_from_euler_xyz(noise[:, 0], noise[:, 1], noise[:, 2])
                    pose[:, 3:7] = math_utils.quat_mul(pose[:, 3:7], noise_quat)
                    asset.write_root_pose_to_sim_index(root_pose=pose)
                    # Only randomize the angular velocity; leave the linear velocity at zero.
                    root_velocity = torch.zeros((num_instances, 6), device=device)
                    root_velocity[:, 3:6] = math_utils.sample_uniform(
                        velocity_ranges[:, 0], velocity_ranges[:, 1], (num_instances, 3), device=device
                    )
                    asset.write_root_velocity_to_sim_index(root_velocity=root_velocity)
                    # Vary the mass so the drop behavior differs between resets.
                    asset.set_masses_index(masses=torch.rand((num_instances, 1), device=device) * 0.2 + 0.2)
                scene.reset()

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Bring out-of-bounds objects back to the bin.
        for asset, env_ids, spawn_pose_w in groceries:
            xy = asset.data.root_link_pose_w.torch[:, :2] - scene.env_origins[env_ids, :2]
            stray = torch.nonzero(~((xy >= bounds_xy[0]) & (xy <= bounds_xy[1])).all(dim=-1)).flatten()
            if stray.numel():
                asset.write_root_pose_to_sim_index(root_pose=spawn_pose_w[stray], env_ids=stray)
                asset.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((stray.numel(), 6), device=device), env_ids=stray
                )
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function.

    Returns:
        None: The function drives the simulation for its side-effects.
    """
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view((4.0, 0.0, 4.0), (0.0, 0.0, 0.0))

        # Design scene
        scene_cfg = BinPackingSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0, replicate_physics=True)
        layouts = [combination.assets for combination in scene_cfg.clone_cfg.clone_combinations]
        print(f"[INFO] Drawn bin layouts (objects per layout): {[len(layout) for layout in layouts]}")
        with Timer("[INFO] Time to create scene: "):
            scene = scene_cfg.class_type(scene_cfg)

        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    main()
