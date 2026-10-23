# Conveyor tasks

A Franka continuously transfers four numbered 40 mm cubes between two counter-rotating
racetrack conveyors. The trained policy uses 123 observations and eight actions at 60 Hz,
with 120 Hz physics and a 0.35 m/s belt speed.

![Racetrack transfer](../../../../../docs/source/_static/tasks/conveyor/racetrack_transfer.jpg)

Newton traction uses the upstream `ConveyorForceModel`, including its MuJoCo contact
reporting. `isaaclab_contrib.conveyors.newton` adapts replicated surfaces, parcel selection,
controls, and reset lifecycle; the task owns its geometry, commands, and reset recipes.

## Run

Newton is kitless and supports the lightweight GL viewer:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Racetrack-Transfer-v0 \
  --checkpoint pretrained \
  --num_envs 8 --device cuda:0 --viz newton_gl --real-time
```

The `pretrained` selector downloads the RSL-RL policy published specifically for the Newton MJWarp
backend. The Newton checkpoint is published on Isaac Dev. While the public mirror is syncing,
use a local checkpoint path if the download is unavailable.

Training uses the same task ID and defaults to 256 environments:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Racetrack-Transfer-v0 \
  --num_envs 256 --device cuda:0
```

### Curriculum and evaluation

Racetrack training samples physically calibrated resets from release through moving-belt pickup.
The shared `SuccessMonitor` records whether each episode completed a stable transfer from its
sampled reset row. NIST's reusable `Sampler` mixes uniform coverage with a Beta-shaped score
centered at 50% transfer success, giving partly learned rows more practice while keeping every
row eligible. Reset recipes generate the robot and cube states directly; the task needs no
cached physical-state bank or assembly-variant machinery.

Training uses standard PPO with the existing arm-and-gripper action distribution. Checkpoints
save policy and optimizer state; sampling history starts fresh when the environment is recreated,
including when resuming training. Playback disables adaptive sampling. Evaluate from moving-belt
starts to measure pickup and transfer ability independently of the easier training resets.

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

The authored USDA layers are versioned with the task because their route metadata,
collision paths, and animation must match its implementation. NVIDIA meshes and
textures stay at their original remote locations under their original licenses.

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
and samples background animation using simulation time. See the
[SimReady Explorer documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_browser-extensions/simready-explorer.html)
for catalog access. Layout references include
[Dematic modular conveyors](https://www.dematic.com/content/dam/dematic/downloads/brochures/NA_BR_1039_MCS.pdf),
[Dematic sortation](https://www.dematic.com/content/dam/dematic/downloads/whitepapers/NA_WP-1015_Sorting-Out-Sortation.pdf),
and [KION warehouse installations](https://www.kiongroup.com/en/News-Stories/Stories/Innovation/Warehousing-is-being-transformed.html).
