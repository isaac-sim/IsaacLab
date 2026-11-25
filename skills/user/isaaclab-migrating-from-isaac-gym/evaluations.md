# Isaac Gym Migration Evaluations

## Contents

- Scenario 1: rough locomotion migration
- Scenario 2: sensor and terrain migration
- Scenario 3: randomization migration
- Scenario 4: Newton backend follow-up

## Scenario 1: Rough Locomotion Migration

Query: "Port this IsaacGymEnvs rough-terrain quadruped task to Isaac Lab."

Expected behavior:

- Identifies terrain generation, robot asset, commands, observations, rewards, resets, sensors, randomization, and training configuration.
- Recommends a direct environment as the first migration target.
- Maps reward, termination, command, and reset logic into direct environment methods.
- Recommends a manager-based follow-up after direct reset, step, and short training validation.
- Gives import, reset/random-step, short-training, full-training, metric-parsing, and checkpoint-rollout gates before claiming policy success.
- Stages flat locomotion before rough-terrain curriculum when the rough source command ranges or terrain curriculum prevent a healthy first policy.

Known failure modes:

- Starts by decomposing into manager terms before preserving behavior in direct form.
- Treats the direct migration as the final structure even after behavior is stable and reusable manager terms would help.
- Omits terrain or sensor behavior from the migration plan.
- Treats a completed training command or checkpoint file as policy success without parsed metrics and rollout evidence.

## Scenario 2: Sensor and Terrain Migration

Query: "Migrate the height measurements and contact force observations from my Isaac Gym quadruped terrain task."

Expected behavior:

- Maps terrain generation to `TerrainImporterCfg` and a maintained terrain generator.
- Maps height measurements to a ray-caster height scanner when appropriate.
- Maps contact-force usage to contact sensors.
- Calls out observation-block parity and shape validation.

Known failure modes:

- Drops the height measurement block because it is not part of the flat locomotion task.
- Assumes contact-force tensors map directly without sensor configuration.

## Scenario 3: Randomization Migration

Query: "Move my Isaac Gym reset randomization to Isaac Lab."

Expected behavior:

- Preserves reset randomization behavior in the direct migration first.
- Distinguishes startup, reset, and interval randomization when the user asks for event terms.
- Recommends validation with repeated resets and small environment counts.

Known failure modes:

- Places all randomization in environment constructor code.
- Does not verify reset determinism or tensor shapes.

## Scenario 4: Newton Backend Follow-Up

Query: "The migrated Isaac Gym task works on PhysX. Add Newton support."

Expected behavior:

- Keeps the validated direct workflow intact.
- Uses `PresetCfg` variants for backend-specific physics settings.
- Maps PhysX schema cfgs through the multi-backend and schema cfg docs before proposing Newton equivalents.
- Calls out parameters with no direct Newton equivalent instead of silently copying them.

Known failure modes:

- Assumes every PhysX parameter maps one-to-one to Newton.
- Replaces the PhysX config instead of adding a backend preset.
