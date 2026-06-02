# Rough Locomotion Migration Validation

## Contents

- Source task
- How to use this validation
- Why this is the representative validation
- Direct workflow mapping
- Terrain mapping
- Sensors and observations
- Randomization and backend checks
- Result

## Source Task

This validation uses IsaacGymEnvs AnymalTerrain:

- `isaacgymenvs/tasks/anymal_terrain.py`
- `isaacgymenvs/cfg/task/AnymalTerrain.yaml`

## How To Use This Validation

Use this file as an evaluation checklist for the migration skill, not as a ready-made migrated task. A useful migration attempt should:

- Read the IsaacGymEnvs source task and config.
- Start with a direct Isaac Lab environment target.
- Produce an explicit mapping for terrain, sensors, observations, rewards, resets, randomization, and backend settings.
- Identify behavior differences between the legacy task and maintained Isaac Lab examples.
- Define small import, reset, step, and training smoke tests before scaling.

## Why This Is the Representative Validation

AnymalTerrain is representative because it combines several migration concerns that usually fail when a skill only handles simple control tasks:

- Rough triangle-mesh terrain generation with curriculum levels and terrain proportions.
- A quadruped URDF asset with PD control parameters and default joint poses.
- Velocity command sampling.
- Height measurements for terrain perception.
- Net contact force usage for feet, knees, and termination.
- Observation noise, action noise, friction randomization, random pushes, and setup-only mass randomization.
- PhysX simulation parameters such as contact collection, GPU contact pair count, and solver iterations.

## Direct Workflow Mapping

| IsaacGymEnvs AnymalTerrain | Isaac Lab direct rough Anymal-C |
| --- | --- |
| `VecTask` subclass | `AnymalCEnv`, a `DirectRLEnv` subclass |
| `AnymalTerrain.yaml` | `AnymalCRoughEnvCfg` config class |
| `numActions = 12` | `action_space = 12` |
| `numObservations = 188` | `observation_space = 235` in the maintained Isaac Lab rough task |
| `dt = 0.005`, `control.decimation = 4` | `sim.dt = 1 / 200`, `decimation = 4` |
| `urdf/anymal_c/urdf/anymal_minimal.urdf` | `isaaclab_assets.robots.anymal.ANYMAL_C_CFG` |
| `create_sim()`, `_create_trimesh()`, `_create_envs()` | `TerrainImporterCfg`, rough terrain generator, and `_setup_scene()` |
| `compute_observations()` | `_get_observations()` with robot state, commands, contact sensor data, and height scanner data |
| `compute_reward()` and reward helpers | `_get_rewards()` with explicit reward terms in the direct environment |
| `reset_idx()` | `_reset_idx()` with command and robot state reset |

## Terrain Mapping

IsaacGymEnvs creates a custom triangle mesh through `Terrain(...)` and `gym.add_triangle_mesh(...)`. In Isaac Lab, route this through `TerrainImporterCfg` with `terrain_type="generator"` and a maintained terrain generator such as `ROUGH_TERRAINS_CFG`.

The IsaacGymEnvs terrain settings map conceptually as:

| IsaacGymEnvs terrain setting | Isaac Lab target |
| --- | --- |
| `terrainType: trimesh` | `TerrainImporterCfg(terrain_type="generator")` |
| `curriculum` | terrain generator curriculum setting |
| `numLevels`, `numTerrains` | terrain generator rows / columns or difficulty grid |
| `terrainProportions` | terrain generator sub-terrain proportions |
| friction / restitution | terrain physics material |
| custom terrain origins | terrain importer environment origins |

## Sensors and Observations

IsaacGymEnvs AnymalTerrain uses net contact forces and sampled terrain heights. The maintained Isaac Lab direct rough Anymal-C task uses:

- `ContactSensorCfg` for contact history, feet air time, and undesired contacts.
- `RayCasterCfg` mounted on the robot base with rays cast against `/World/ground`.
- Direct observation assembly that includes height scanner data for rough terrain.

Do not assume observation dimensions will match exactly. A migration that needs behavior parity should compare every observation block and decide whether to preserve IsaacGymEnvs' 188-dimensional observation or adopt the maintained Isaac Lab rough locomotion observation layout.

## Randomization and Backend Checks

IsaacGymEnvs AnymalTerrain includes:

- Observation and action noise.
- Friction randomization.
- Random pushes.
- Setup-only rigid-body mass randomization.
- DOF damping, stiffness, and limit randomization.

In Isaac Lab, first migrate the direct task until it runs with the maintained baseline randomization. Then port the legacy task's additional randomization through event terms or noise models, checking each item against current source:

- Use the `isaaclab-randomizing-with-events` skill for event timing and backend compatibility.
- Treat setup-only randomization as `prestartup` or `startup` unless the backend supports runtime changes.
- Check PhysX versus Newton behavior before adding `PresetCfg` support.
- Validate CPU/GPU assumptions for each event implementation.

## Result

The migration skill is actionable for a non-toy rough-terrain locomotion task. The direct-first path remains valid because Isaac Lab has a direct rough Anymal-C task at `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/`.

This validation also shows the limits of direct copying: terrain generation, sensor models, observation layout, randomization timing, and PhysX/Newton support must be mapped deliberately through Isaac Lab docs and maintained source.
