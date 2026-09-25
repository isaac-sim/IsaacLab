<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Conveyor Franka

Choose between two tasks using the same pretrained Franka policy:

| Task | Layout and behavior | Newton task ID |
| --- | --- | --- |
| Racetrack transfer | Original two closed racetracks and four numbered cubes; continuous alternating transfers | `IsaacContrib-Conveyor-Franka-Newton-v0` |
| Warehouse sorting | Current extended conveyors and 24 colored parcels; two colors per circulating conveyor | `IsaacContrib-Conveyor-Franka-Newton-Play-v0` |

The sorter inherits the racetrack environment and configuration. Both reuse the same action,
observation, reward, and placement logic and the same RSL-RL agent configuration. Sorting adds
class dispatch and a four-slot parcel adapter; it needs no separately trained policy. The original
task keeps its compact geometry and four fixed cube identities. Both use 123 observations,
eight actions, 120 Hz physics, and a 60 Hz policy rate.

## Backend support

Both tasks run on Newton GPU. `IsaacContrib-Conveyor-Franka-PhysX-CPU-v0` provides a CPU-only
native PhysX reference for the original four-cube racetrack task.

The PhysX task rejects CUDA during configuration validation. In the supported Isaac Sim runtime,
enabling the native surface-velocity contact-modification path under GPU dynamics can drop the belt
contacts and let packages pass through the conveyor. CPU PhysX preserves those contacts. Use the
Newton task whenever GPU simulation or vectorized throughput is required.

The two backends deliberately share the policy tensor contract, so an RSL-RL checkpoint can be
loaded by either task without reshaping or reordering tensors. Their contact and actuator dynamics
are not numerically identical; validate task behavior when transferring a policy between them.

Surface-velocity intent and the tensorized control contract live in
`isaaclab.physics.surface_velocity`. Backend mechanics are separate: Newton's solved-contact force
pipeline lives in `isaaclab_newton.physics.surface_velocity`, while PhysX schema authoring and live
attribute control live in `isaaclab_physx.physics.surface_velocity`. The task package owns only the
racetrack geometry, backend lifecycle selection, and task-level commands.

## Original racetrack task

Newton is kitless and supports the lightweight GL viewer:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-Newton-v0 \
  --checkpoint pretrained \
  --num_envs 8 --device cuda:0 --viz newton_gl --real-time
```

The `pretrained` selector downloads the RSL-RL policy published specifically for the Newton MJWarp
backend. To evaluate another policy, replace `pretrained` with an explicit checkpoint path. The
PhysX task resolves a different backend-specific artifact name, so transferring this Newton policy
to PhysX currently requires the explicit local checkpoint path shown below.

Training uses the same task ID and defaults to 256 environments:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-Newton-v0 \
  --num_envs 256 --device cuda:0
```

## Warehouse sorting task

Select the `Newton-Play` task for warehouse sorting with the same checkpoint. The two parallel manipulation
straights and their adjoining 90-degree bends retain their original positions, widths, radii,
and 0.35 m/s surface speed. Beyond these fixed sections, two short rising feeds climb 0.10 m at less than 10 degrees and
join a shared elevated deck. Guides keep the two return lanes assigned through the upper split,
and separate descending conveyors deliver the parcels back to the original workcell approaches.
The new incline panels use slope-aligned traction and a 0.95 friction coefficient; the original
manipulation sections retain their trained 0.5 setting.

Blue conveyor frames reach the floor, and the scanner faces along the background main belt after
a 90-degree counterclockwise rotation. The cubes wear SimReady cardboard meshes normalized to
**40 × 40 × 40 mm**, centered on their original colliders; mass remains 50 g. Actions and observation
ordering stay checkpoint-compatible. The Play variant widens its travel bounds and resets all parcels as a randomized mixed batch on the
raised supply belts.
The base training and PhysX tasks retain the original compact layout.

The workcell contains **24 physical parcels**, exposed to the unchanged checkpoint through
four policy slots. The sorting command fills remote slots with misplaced arrivals; local assignments and an
active grasp stay pinned. Reassignment runs in the command manager, before the next policy observation,
and changes only identity mapping, never a physical pose. Commands, rewards, and placement checks
use the same mapping. All 24 parcels receive belt forces and participate in safety checks.
The arm parks when there is no active sorting transfer. The policy sees remote inventory in canonical waiting slots;
all local parcel poses and velocities, physical transport, rewards, and transfer checks remain
actual simulated states. This adapter is necessary because the checkpoint was trained on compact,
flat returns. Tensor ordering stays 123 observations to eight actions, but remote observation values
are intentionally adapted. Invalid actions still reach the original sanitization and termination path.
The checkpoint can still miss grasps and reset before a batch finishes. The larger randomized
inventory preserves the policy interface, but reliable complete-batch sorting is not established.

The near loop has a rounded return; the far loop has a shorter squared return with rounded corners
and a shorter supply belt. Both retain the exact original manipulation geometry. Oversized yellow
drive blocks are omitted from the workcell.

The batch contains six cartons in each of four clearly marked colors: **blue, orange, green,
and purple**. Colored paper bands wrap the textured cardboard without changing its 40 mm bounds.
Reset shuffles physical identities across 24 supply positions, independently
for each environment, using the simulation's seeded random generator. Counts remain balanced while
arrival order and initial conveyor assignments vary. Colors stay fixed throughout each batch.
The 0.043–0.052 m/s feeds release cartons through **12 cm gravity drops** onto the 0.35 m/s loops.
Blue and green belong on the positive-Y loop; orange and purple belong on the negative-Y loop.
The dispatcher requests only wrong-lane transfers and retains ownership through grasp and stable
release. Missed arrivals recirculate for another opportunity. Once the batch is sorted, the arm
parks and both loops continue running. No parcels are recolored, teleported, or replaced during sorting.

Class assignment is explicit supervisory metadata, not a claim that the unchanged state-based
checkpoint recognizes color. `commands.transfer.parcel_colors` assigns appearances and
`commands.transfer.parcel_destinations` assigns loop IDs; all parcels of a color must share a
single destination. `commands.transfer.randomize_arrivals=False` uses the authored ordering.
`sorted_parcels` counts settled, correctly routed inventory; `batch_complete` indicates that all
24 parcels are settled on their assigned loops.

The surrounding warehouse includes loaded rack aisles, packing shelves, pallet staging, scan and
outbound signs, safety markings, overhead beams, and warm/cool industrial lighting. Its additional
23.11 m parcel loop includes A29 elevated runs and A38 ramps, with twelve animated cartons traveling
along the main transport line. These background cartons remain render-only inventory and do not
add contacts or policy observations. The 24 small workcell parcels have real collision and
dynamics. A Y-divert dresses the outbound bay.

Use **Kit/RTX** to see the authored MDL textures, USD lights, and background animation:

```bash
DISPLAY=:1 uv run --extra isaacsim isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-Newton-Play-v0 \
  --checkpoint https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/IsaacContrib-Conveyor-Franka-Newton-v0_newtonmjwarp_none_rsl_rl.pt --num_envs 1 --device cuda:0 --viz kit --real-time \
  --kit_args=--/UJITSO/geometry=false
```

Kit renders the USD directly, so the Play configuration excludes visual-only meshes from the
Newton model (`sim.physics.load_visual_shapes=False`). This avoids importing warehouse dressing
into the physics model. For a static approximation in `--viz newton_gl`, explicitly pass
`env.sim.physics.load_visual_shapes=True`; materials and lighting are simplified in that viewer.
The launch override disables experimental geometry streaming to prevent disappearing meshes with
Fabric transforms. Presentation defaults to one environment. Asset references and textures are downloaded and cached
on first use, so the first launch takes longer.

### Editing the presentation

* `assets/warehouse.usda` owns the warehouse layout, referenced props, materials, lights, cameras,
  and looping parcel transform samples. It can be opened in a USD authoring application with an
  Omniverse-compatible asset resolver. `Cameras/Workcell` and `Cameras/Overview` provide two views.
* `assets/parcel_{blue,orange,green,purple}.usda` reference the carton, tint its cardboard, and
  add a colored paper band within the original bounds.
* `assets/parcel.usda` normalizes the measured SimReady `cardbox_a1` visual bounds to a centered
  40 mm cube. Keep the shell bounds aligned with the task's original collider when editing it.
* `assets/conveyor_*_supported.usd` override the lower frame vertices of the referenced conveyor
  modules. Upper-frame and belt geometry, source topology, and materials remain unchanged.
* `assets/conveyor_routes.usda` owns the elevated network's centerlines, frame meshes, material
  references, ramp traction, feed speeds, and mixed-batch spawn positions. Physics reads these same paths after application startup;
  USD is not imported during task discovery. Incline normals follow each panel's slope. All 24
  physical parcel spawn positions are authored in `conveyor:parcelSpawnPositions`.
* The Python adapter resolves remote references into the normal local asset cache, removes all
  imported physics/action-graph ownership, and samples background animation using simulation time.
  `conveyor_warehouse_geometry.py` sweeps lightweight collision proxies along the authored paths.
  `mdp/sorting.py` extends the shared transfer command with class dispatch and batch metrics;
  the action, observation, reward, and stable-placement contracts remain shared with the trained task.
* `env.conveyor_cube_pool.slot_ids` maps each environment's four policy slots to physical parcel IDs.
  `transfer_counts` records placements per physical parcel; assignments never duplicate a parcel
  within an environment, and resetting one environment does not change another's slots.

### Asset and visual references

The asset survey covered SimReady Central (`simready-central.nvidia.com`),
`omniverse://ov-isaac-dev.nvidia.com/Isaac/SimReady/Industrial/Warehouse`,
`Isaac/Environments/{Digital_Twin_Warehouse,Modular_Warehouse}`, `Isaac/Props/Conveyors`, and
`NVIDIA/Assets/DigitalTwin/Assets/Warehouse`. The composition references publicly accessible
Omniverse counterparts so playback does not require internal Nucleus credentials. Public catalog
usage is described in the [SimReady Explorer documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_browser-extensions/simready-explorer.html).

Selected assets are the Omniverse A03/A09/A12/A24/A29/A38 conveyors, `RackLarge_A1`, SimReady
`bulkstoragerack_a01` and `cardbox_a1`, the Isaac packing table, and loaded pallets. NVIDIA assets
remain under their original licenses; this package contains the scene layout and lower-frame mesh overrides.
The layout draws on [Dematic's modular conveyor examples](https://www.dematic.com/content/dam/dematic/downloads/brochures/NA_BR_1039_MCS.pdf)
and [KION's warehouse installation photographs](https://www.kiongroup.com/en/News-Stories/Stories/Innovation/Warehousing-is-being-transformed.html):
parallel transport, elevation changes, rack storage, packing zones, and clear marked aisles.
The induction and recirculation layout also follows the concepts in
[Dematic’s sortation overview](https://www.dematic.com/content/dam/dematic/downloads/whitepapers/NA_WP-1015_Sorting-Out-Sortation.pdf).

## PhysX CPU playback

The native PhysX variant requires an Isaac Sim-enabled launch and an explicit CPU device. One
environment is the default and recommended interactive configuration:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-PhysX-CPU-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1 --device cpu --viz kit --real-time \
  agent.device=cpu
```

Overriding the task to CUDA is an error by design; keep the explicit `--device cpu` in launch
commands for clarity. The native surface-velocity backend stages commands through USD, while the
Newton backend keeps its batched state, contact processing, and force application on the GPU.
