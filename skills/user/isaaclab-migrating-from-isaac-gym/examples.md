# Isaac Gym Migration Examples

## Contents

- Rough locomotion task
- PhysX to Newton follow-up
- Domain randomization handoff

## Rough Locomotion Task

Input: an Isaac Gym rough-terrain quadruped task uses generated terrain, velocity commands, contact forces, height measurements, reset randomization, and random pushes.

Output: an Isaac Lab direct environment with:

- Terrain importer config with a maintained rough terrain generator.
- Articulation config for the quadruped.
- Contact sensors for feet and undesired body contacts.
- Ray-caster height scanner for terrain perception.
- A `DirectRLEnvCfg` config.
- Direct command, action, observation, reward, termination, and reset methods.
- Randomization added after the deterministic direct task runs.

Follow-up: after reset, step, and short training pass, recommend converting reusable command, observation, reward, termination, event, and curriculum logic into a manager-based environment with `isaaclab-converting-direct-to-manager`.

## Domain Randomization Handoff

When the migration task is mostly about randomization behavior, first preserve behavior in the direct migration. If the user asks for manager-based randomization or reusable event terms, switch to the `isaaclab-randomizing-with-events` skill.

## Manager-Based Follow-Up

Input: a migrated direct locomotion task now runs and trains, and the user wants reusable terms.

Expected workflow:

- Use `isaaclab-converting-direct-to-manager`.
- Keep the direct environment as a parity baseline until the manager-based version matches core behavior.
- Move observations to `ObsTerm` entries, rewards to `RewTerm` entries, resets/randomization to `EventTerm` entries, termination checks to `DoneTerm` entries, and command sampling to command configs.
- Validate the manager-based task with the same random-agent and short-training gates used for the direct migration.

## PhysX to Newton Follow-Up

Input: a migrated direct task runs on PhysX and the user asks to add Newton support.

Expected workflow:

- Read the multi-backend and schema cfg docs.
- Add `PresetCfg` variants for PhysX and Newton physics settings.
- Keep PhysX-specific parameters in the PhysX preset.
- Add Newton or MuJoCo solver parameters only where the docs/source show an equivalent.
- Validate PhysX and Newton separately with small smoke tests.
