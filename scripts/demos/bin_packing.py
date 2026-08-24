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
fills them in is a valid combination. The first layout fills the bin, which
keeps every declared slot spawned no matter how many environments run. The
slots cycle through eight YCB grocery models, spawned by a custom spawner that
authors the rigid bodies and colliders the visual models ship without. The
simulation periodically re-drops every spawned grocery into its bin with pose,
velocity, and mass noise, teleporting out-of-bounds objects back in.

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
from dataclasses import MISSING, dataclass
from random import Random

import torch
import warp as wp

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
    import omni.physics.tensors as physx
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


def grocery_cfg(slot: int) -> AssetBaseCfg:
    """Declare the grocery that occupies one slot of the spawn grid.

    Args:
        slot: Index into the spawn grid built by :func:`grocery_spawn_poses`.

    Returns:
        The spawn-only asset configuration for that slot.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Groceries/Grocery_{slot:02d}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(grocery_spawn_poses()[slot, :3].tolist())),
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
    one clone combination per layout to pick which of those slots are active:
    a full bin first, then a :class:`RandomSubsetSet` draw per remaining layout.
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
    grocery_00: AssetBaseCfg = grocery_cfg(0)
    grocery_01: AssetBaseCfg = grocery_cfg(1)
    grocery_02: AssetBaseCfg = grocery_cfg(2)
    grocery_03: AssetBaseCfg = grocery_cfg(3)
    grocery_04: AssetBaseCfg = grocery_cfg(4)
    grocery_05: AssetBaseCfg = grocery_cfg(5)
    grocery_06: AssetBaseCfg = grocery_cfg(6)
    grocery_07: AssetBaseCfg = grocery_cfg(7)
    grocery_08: AssetBaseCfg = grocery_cfg(8)
    grocery_09: AssetBaseCfg = grocery_cfg(9)
    grocery_10: AssetBaseCfg = grocery_cfg(10)
    grocery_11: AssetBaseCfg = grocery_cfg(11)
    grocery_12: AssetBaseCfg = grocery_cfg(12)
    grocery_13: AssetBaseCfg = grocery_cfg(13)
    grocery_14: AssetBaseCfg = grocery_cfg(14)
    grocery_15: AssetBaseCfg = grocery_cfg(15)
    grocery_16: AssetBaseCfg = grocery_cfg(16)
    grocery_17: AssetBaseCfg = grocery_cfg(17)
    grocery_18: AssetBaseCfg = grocery_cfg(18)
    grocery_19: AssetBaseCfg = grocery_cfg(19)
    grocery_20: AssetBaseCfg = grocery_cfg(20)
    grocery_21: AssetBaseCfg = grocery_cfg(21)
    grocery_22: AssetBaseCfg = grocery_cfg(22)
    grocery_23: AssetBaseCfg = grocery_cfg(23)

    # A slot claimed by some layout but active in none of the layouts the environments
    # actually receive is never spawned, which the scene rejects. Environments cycle
    # through the layouts in order, so a full first layout keeps every slot active for
    # any ``--num_envs``, and the random draws below stay unconstrained.
    clone_cfg: CloneCfg = CloneCfg(
        clone_combinations=[
            InclusionSet(assets=GROCERY_NAMES),
            *(
                RandomSubsetSet(pool=GROCERY_NAMES, num_active=(MIN_OBJECTS_PER_BIN, MAX_OBJECTS_PER_BIN))
                for _ in range(NUM_LAYOUTS - 1)
            ),
        ],
        clone_strategy=sequential,
    )


##
# Simulation Loop
##


@dataclass
class FlatGroceryView:
    """Demo-local state for one flat PhysX view over every spawned grocery."""

    root_view: physx.RigidBodyView
    """Rigid-body view with one row per grocery, without an environment axis."""

    env_origins: torch.Tensor
    """World origin of every view row's environment [m], shape ``(num_objects, 3)``."""

    spawn_pose_w: torch.Tensor
    """Default world pose of every grocery [m], shape ``(num_objects, 7)``."""

    all_indices: wp.array
    """Device indices covering every view row."""

    cpu_indices: wp.array
    """CPU indices covering every view row for model-property writes."""


def create_flat_grocery_view(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> FlatGroceryView:
    """Create one flat PhysX rigid-body view and derive row metadata from its paths."""
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    root_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/Groceries/Grocery_*")
    if root_view.count == 0:
        raise RuntimeError("The flat grocery view matched no rigid bodies.")

    env_ids = []
    slot_ids = []
    for path in root_view.prim_paths:
        env_ids.append(int(path.partition("/env_")[2].partition("/")[0]))
        slot_ids.append(int(path.rpartition("/Grocery_")[2]))
    env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=scene.device)
    slot_ids_t = torch.tensor(slot_ids, dtype=torch.long, device=scene.device)
    env_origins = scene.env_origins[env_ids_t]
    spawn_pose_w = grocery_spawn_poses(scene.device)[slot_ids_t]
    spawn_pose_w[:, :3] += env_origins

    all_indices_t = torch.arange(root_view.count, dtype=torch.int32, device=scene.device)
    cpu_indices_t = torch.arange(root_view.count, dtype=torch.int32, device="cpu")
    return FlatGroceryView(
        root_view=root_view,
        env_origins=env_origins,
        spawn_pose_w=spawn_pose_w,
        all_indices=wp.from_torch(all_indices_t),
        cpu_indices=wp.from_torch(cpu_indices_t),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Runs the simulation loop that coordinates spawn randomization and stepping.

    A flat physics view manages every grocery present in the heterogeneous scene.

    Returns:
        None: The simulator side-effects are applied through ``scene`` and ``sim``.
    """
    device = scene.device
    groceries = create_flat_grocery_view(sim, scene)
    print(f"[INFO] Grocery RigidBodyView: 1 flat view managing {groceries.root_view.count} rigid bodies")

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
                shape = (groceries.root_view.count,)
                # Only randomize the [roll, pitch, yaw] rotation; keep the spawn position fixed.
                noise = math_utils.sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (*shape, 3), device=device)
                pose = groceries.spawn_pose_w.clone()
                noise_quat = math_utils.quat_from_euler_xyz(noise[..., 0], noise[..., 1], noise[..., 2])
                pose[..., 3:7] = math_utils.quat_mul(pose[..., 3:7], noise_quat)
                groceries.root_view.set_transforms(wp.from_torch(pose.contiguous()), groceries.all_indices)
                # Only randomize the angular velocity; leave the linear velocity at zero.
                body_velocity = torch.zeros((*shape, 6), device=device)
                body_velocity[..., 3:6] = math_utils.sample_uniform(
                    velocity_ranges[:, 0], velocity_ranges[:, 1], (*shape, 3), device=device
                )
                groceries.root_view.set_velocities(wp.from_torch(body_velocity.contiguous()), groceries.all_indices)
                # Vary the mass so the drop behavior differs between resets.
                masses = torch.rand((groceries.root_view.count, 1), device="cpu") * 0.2 + 0.2
                groceries.root_view.set_masses(wp.from_torch(masses), groceries.cpu_indices)
                scene.reset()

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Bring out-of-bounds objects back to the bin.
        transforms = groceries.root_view.get_transforms()
        transforms_t = wp.to_torch(transforms) if isinstance(transforms, wp.array) else transforms
        xy = transforms_t[:, :2] - groceries.env_origins[:, :2]
        stray = torch.nonzero(~((xy >= bounds_xy[0]) & (xy <= bounds_xy[1])).all(dim=-1)).flatten()
        if stray.numel():
            stray_indices = wp.from_torch(stray.to(dtype=torch.int32))
            corrected_transforms = transforms_t.clone()
            corrected_transforms[stray] = groceries.spawn_pose_w[stray]
            velocities = groceries.root_view.get_velocities()
            corrected_velocities = (
                wp.to_torch(velocities).clone() if isinstance(velocities, wp.array) else velocities.clone()
            )
            corrected_velocities[stray] = 0.0
            groceries.root_view.set_transforms(wp.from_torch(corrected_transforms.contiguous()), stray_indices)
            groceries.root_view.set_velocities(wp.from_torch(corrected_velocities.contiguous()), stray_indices)
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
