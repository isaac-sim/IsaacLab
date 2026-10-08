<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Conveyor tasks

Choose between two tasks using the same pretrained Franka policy:

| Task | Layout and behavior | Newton task ID |
| --- | --- | --- |
| Racetrack transfer | Original two closed racetracks and four numbered cubes; continuous alternating transfers | `IsaacContrib-Conveyor-Racetrack-Transfer-v0` |
| Warehouse sorting | Current extended conveyors and 24 colored parcels; two colors per circulating conveyor | `IsaacContrib-Conveyor-Warehouse-Sorting-v0` |

The sorter inherits the racetrack environment and configuration. Both reuse the same action,
observation, reward, and placement logic and the same RSL-RL agent configuration. Sorting adds
class dispatch and a four-slot parcel adapter for playback with the shared checkpoint. The original
task keeps its compact geometry and four fixed cube identities. Both use 123 observations,
eight actions, 120 Hz physics, and a 60 Hz policy rate.

## Backend support

Both tasks run on Newton GPU. `IsaacContrib-Conveyor-Racetrack-Transfer-PhysX-CPU-v0` provides a CPU-only
native PhysX reference for the original four-cube racetrack task.

The PhysX task rejects CUDA during configuration validation. In the supported Isaac Sim runtime,
enabling the native surface-velocity contact-modification path under GPU dynamics can drop the belt
contacts and let packages pass through the conveyor. CPU PhysX preserves those contacts. Use the
Newton task whenever GPU simulation or vectorized throughput is required.

The two backends deliberately share the policy tensor contract, so an RSL-RL checkpoint can be
loaded by either task without reshaping or reordering tensors. Their contact and actuator dynamics
are not numerically identical; validate task behavior when transferring a policy between them.

Surface-velocity intent and the tensorized control contract live in
`isaaclab_contrib.conveyors.surface_velocity`. Newton traction uses the complete upstream
`ConveyorForceModel` from the pinned Newton release, including its MuJoCo contact reporting.
`isaaclab_contrib.conveyors.newton` adapts replicated surfaces, parcel selection, controls, and
reset lifecycle; it contains no separate traction solver.
PhysX schema authoring and live attribute control live in `isaaclab_contrib.conveyors.physx`.
The task package owns only the racetrack geometry, backend lifecycle selection, and task-level commands.

## Original racetrack task

Newton is kitless and supports the lightweight GL viewer:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Racetrack-Transfer-v0 \
  --checkpoint pretrained \
  --num_envs 8 --device cuda:0 --viz newton_gl --real-time
```

The `pretrained` selector downloads the RSL-RL policy published specifically for the Newton MJWarp
backend. Both task keys are published on Isaac Dev. While the public mirror is syncing,
use a local checkpoint path if the download is unavailable.
The PhysX task resolves a different backend-specific artifact name, so transferring this Newton policy
to PhysX currently requires the explicit local checkpoint path shown below.

Training uses the same task ID and defaults to 256 environments:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Racetrack-Transfer-v0 \
  --num_envs 256 --device cuda:0
```

### Curriculum and evaluation

Racetrack training samples physically calibrated resets from release through moving-belt pickup.
The shared `SuccessMonitorCfg` tracks **phase progress** per reset row and weights intermediate
variants around a 50% progress target; this is not the full-transfer success rate. Sampling remains
balanced across phases, cube identities, and directions. Moving-belt starts receive at least 35%
of resets, increasing toward 90% only when completed transfers from those starts and reset-row
coverage improve. `deployment_transfer_success_rate` reports that separate rolling signal.

New RSL-RL checkpoints save the curriculum evidence alongside standard PPO state and restore it
with `--checkpoint /path/to/model.pt`. Older checkpoints load normally with fresh evidence;
loading actor weights alone leaves the current curriculum intact. Playback disables the curriculum.

Evaluate from home on moving belts, with fixed reset settings and a chosen seed:

```bash
uv run python -m isaaclab_tasks.contrib.conveyor_franka.evaluate \
  --checkpoint /path/to/model.pt --num_envs 8 --device cuda:0 \
  --steps 3600 --seed 0 --output /tmp/conveyor-evaluation.json
```

The report counts completed transfers in both directions, safety failures, non-finite observations,
and transfers per simulated environment minute. Repeat with different seeds when comparing policies;
intermediate-reset training reward alone does not establish deployment reliability.

## Warehouse sorting task

The workcell contains 24 physical 40 mm, 50 g parcels: six each in blue, orange, green,
and purple. Blue and green belong on the positive-Y conveyor; orange and purple belong
on the negative-Y conveyor. Resets shuffle a mixed batch across the supply feeds using
seeded randomness. Slow feeds deliver parcels through 12 cm gravity drops; misplaced
arrivals recirculate until transferred. Colors and physical identities remain fixed.

The two manipulation straights and adjoining 90-degree bends preserve the trained
positions, dimensions, and 0.35 m/s speed. Compact elevated returns connect them to
asymmetric routes. Incline normals follow their panels, with 0.95 traction friction;
the original manipulation sections retain their 0.5 setting.

Four policy slots map to the larger physical pool. Local parcels and the active grasp
keep their assignments; the command manager assigns misplaced arrivals to available
slots before observations are computed. Reassignment changes identity mapping, never
physical poses. Commands, rewards, placement checks, and safety checks use the same
mapping. All 24 parcels receive physical belt forces.

Remote inventory is represented through canonical waiting-slot observations because
the checkpoint learned compact, flat returns. Local states and physical interactions
remain exact. Class dispatch uses supervisory metadata; the state-based policy does
not recognize color. Playback parks the arm between transfers and after sorting.
Invalid actions still reach sanitization and termination. The unchanged checkpoint
can miss grasps and reset before finishing; reliable complete-batch sorting is not established.

`commands.transfer.parcel_colors` sets appearances and `parcel_destinations` sets
conveyor IDs; each color must have one destination. Set `randomize_arrivals=False`
for authored arrival ordering. `sorted_parcels` counts settled, correctly routed parcels;
`batch_complete` indicates that all parcels have reached their assigned loops.

Use Kit/RTX for the USD materials, warehouse lighting, and twelve animated background
cartons. Background inventory is render-only and adds no contacts or policy observations.

```bash
DISPLAY=:1 uv run --extra isaacsim isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Warehouse-Sorting-v0 \
  --checkpoint pretrained --num_envs 1 --device cuda:0 --viz kit --real-time \
  --kit_args=--/UJITSO/geometry=false
```

Kit renders USD directly; `sim.physics.load_visual_shapes=False` excludes dressing
from the Newton model. For a static approximation with `--viz newton_gl`, pass
`env.sim.physics.load_visual_shapes=True`. The Kit override disables experimental
geometry streaming to prevent disappearing meshes with Fabric transforms. Assets
and textures are downloaded and cached on first use. Presentation defaults to one environment.

### Fine-tuning

Warehouse training starts at home with parcels on the actual feeds, disables the
racetrack reset curriculum, and passes every sampled policy action through.
Idle parking applies only during playback. Warm-start with the shared checkpoint:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Warehouse-Sorting-v0 \
  --checkpoint /path/to/model.pt --num_envs 256 --device cuda:0
```

### Editing the presentation

- `assets/warehouse.usda` owns layout, props, materials, lights, cameras, and looping
  background transforms. `Cameras/Workcell` and `Cameras/Overview` provide two views.
- `assets/conveyor_routes.usda` owns route centerlines, frame meshes, traction,
  feed speeds, and `conveyor:parcelSpawnPositions`. Python builds collision proxies
  from these paths after application startup; task discovery does not import USD.
- `assets/parcel.usda` normalizes the referenced SimReady carton to the original
  centered 40 mm collider; `parcel_{blue,orange,green,purple}.usda` add color bands.
- `assets/conveyor_*_supported.usda` reference source conveyors and describe support
  extension with two scalar attributes. The spawner derives lower-frame points and normals.
- `env.conveyor_cube_pool.slot_ids` maps policy slots to physical IDs;
  `transfer_counts` records placements. Selected resets clear that batch's counters.

The spawner caches external assets, removes imported physics and action graphs,
and samples background animation using simulation time. NVIDIA geometry and textures
remain external references under their original licenses. See the
[SimReady Explorer documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_browser-extensions/simready-explorer.html)
for catalog access. Layout references include
[Dematic modular conveyors](https://www.dematic.com/content/dam/dematic/downloads/brochures/NA_BR_1039_MCS.pdf),
[Dematic sortation](https://www.dematic.com/content/dam/dematic/downloads/whitepapers/NA_WP-1015_Sorting-Out-Sortation.pdf),
and [KION warehouse installations](https://www.kiongroup.com/en/News-Stories/Stories/Innovation/Warehousing-is-being-transformed.html).

## PhysX CPU playback

The native PhysX variant requires an Isaac Sim-enabled launch and an explicit CPU device. One
environment is the default and recommended interactive configuration:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Racetrack-Transfer-PhysX-CPU-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1 --device cpu --viz kit --real-time \
  agent.device=cpu
```

Overriding the task to CUDA is an error by design; keep the explicit `--device cpu` in launch
commands for clarity. The native surface-velocity backend stages commands through USD, while the
Newton backend keeps its batched state, contact processing, and force application on the GPU.
