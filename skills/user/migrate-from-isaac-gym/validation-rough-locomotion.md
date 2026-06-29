# Rough Locomotion Migration Validation

## Contents

- Source task
- How to use this validation
- Why this is the representative validation
- Direct workflow mapping
- Terrain mapping
- Sensors and observations
- Randomization and backend checks
- Training validation
- Result

## Source Task

This validation uses IsaacGymEnvs AnymalTerrain:

- `isaacgymenvs/tasks/anymal_terrain.py`
- `isaacgymenvs/cfg/task/AnymalTerrain.yaml`

## How To Use This Validation

Use this file as an evaluation checklist for the migration skill, not as a ready-made migrated task. A useful migration attempt should:

- Read the IsaacGymEnvs source task and config.
- Start with a direct Isaac Lab environment target.
- Recommend a manager-based follow-up after the direct task validates, especially for reusable velocity commands, observations, rewards, randomization events, and terminations.
- Produce an explicit mapping for terrain, sensors, observations, rewards, resets, randomization, and backend settings.
- Identify behavior differences between the legacy task and maintained Isaac Lab examples.
- Define small import, reset, step, and training smoke tests before scaling.
- Validate flat walking before rough-terrain curriculum training.

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

## Training Validation

Use IsaacGymEnvs `AnymalTerrain` as the source mapping, but validate training in stages:

1. Start from a direct flat migration target and confirm it can learn a walking policy.
2. Load the saved checkpoint in a bounded play rollout.
3. Move to the direct rough Anymal-C target and treat immediate base-contact termination as a behavioral failure even if the runner completes.
4. Convert the reusable command, observation, reward, terrain curriculum, and termination logic to the manager workflow after the direct parity pass is healthy.

A fresh-agent validation used this skill to clone IsaacGymEnvs, migrate AnymalTerrain into an external package named `isaacgym_anymal_migration`, and register `IsaacGym-AnymalTerrain-Flat-Migrated-Direct-v0`. The package implemented its own `DirectRLEnv`, preserved a 188-dimensional AnymalTerrain-style observation, and validated import, registration, reset, random steps, RSL-RL training, and checkpoint loading.

The migrated task trained for 500 RSL-RL iterations with 4096 environments. Mean reward improved from `0.003` to `3.30`, and XY velocity error improved from `0.735` to `0.222`, but the result was not a successful walking policy: final mean episode length was about `293` steps against a 1000-step horizon, yaw error stayed high at about `1.43`, success rate stayed near `0.12`, and base-contact terminations remained common. A 256-step checkpoint rollout with 64 environments had mean completed episode length `135.3` and success rate `0.165`.

This failure mode points to command staging rather than simple execution failure. IsaacGymEnvs `AnymalTerrain.yaml` uses a yaw command range of `[-3.14, 3.14]`, which made the flat walking validation unstable. Future validation agents should narrow or curriculum-stage commands, or first migrate the simpler `Anymal.yaml` flat behavior, before returning to full AnymalTerrain parity and rough-terrain curriculum.

After adding the policy-success validation loop, a fresh agent reran the migration against IsaacGymEnvs commit `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` and followed that staged path. It migrated `isaacgymenvs/tasks/anymal.py`, `cfg/task/Anymal.yaml`, and `cfg/train/AnymalPPO.yaml` into an external direct Isaac Lab package registering `IsaacGym-Anymal-Flat-Migrated-Direct-v0`. The package implemented its own `DirectRLEnv` with a 48-dimensional Anymal-style observation, IsaacGymEnvs reward/reset/command mapping, contact-sensor termination checks, and RSL-RL config.

That fresh validation defined policy-success criteria before training, then ran registration, reset/random-step smoke, short training, 500-iteration RSL-RL training with 4096 environments, TensorBoard scalar parsing, and a bounded checkpoint rollout. The final 50-iteration training window reached mean reward `72.23`, mean episode length `2493.94` of a `2499`-step horizon, XY velocity error `0.00865`, yaw velocity error `0.01475`, and success rate `1.0`. A 64-environment, 512-step rollout from `model_499.pt` had zero done events, zero base or knee contact terminations, mean XY error `0.00983`, mean yaw error `0.01006`, and tracking success rate `0.99997`.

This confirms the skill can guide a fresh agent to a successful migrated Anymal walking policy when it first validates flat locomotion and uses parsed metrics plus checkpoint rollout evidence. Rough terrain remains a follow-up stage: add terrain generation, height observations, and command/curriculum staging only after the flat migration is healthy.

## Result

The migration skill is actionable for a non-toy quadruped locomotion task. The direct-first path remains valid because Isaac Lab has direct flat and rough Anymal-C tasks at `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/`.

This validation also shows the limits of direct copying: terrain generation, sensor models, observation layout, randomization timing, PhysX/Newton support, and staged training must be mapped deliberately through Isaac Lab docs and maintained source. After the parity pass works, the agent should steer users toward manager-based task structure for reusable Isaac Lab development.
