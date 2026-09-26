# Isaac Gym Migration Reference

## Contents

- Direct workflow mapping
- External template projects
- Locomotion training gates
- Policy success validation loop
- Legacy force and torque sensors
- Backend mapping
- Manager-based follow-up mapping
- Current workflow
- Old patterns
- Validation checklist

## Direct Workflow Mapping

Use this mapping as the default starting point:

| Isaac Gym concept | Isaac Lab target |
| --- | --- |
| Task class with tensor buffers | `DirectRLEnv` or `DirectMARLEnv` subclass |
| YAML task config | `DirectRLEnvCfg` or `DirectMARLEnvCfg` config class |
| Asset loading in task setup | Asset config and scene config |
| Action application | Direct environment action methods |
| Observation buffer assembly | Direct observation method |
| Reward tensor functions | Direct reward method or helper function |
| Reset logic | Direct reset method |
| Domain randomization hooks | Direct reset/startup logic first, event terms if the user wants manager-style randomization |
| RL runner script | Isaac Lab training script for the chosen framework |

## External Template Projects

For a new external migration, follow the [template scaffolding workflow](../isaaclab-building-environments/SKILL.md#scaffold-a-new-task) and the maintained [generator guide](../../../docs/source/developer-tools/template_generator.rst). Select **Direct | single-agent** for a typical Isaac Gym parity pass and the target RL library. Use a fresh project directory outside Isaac Lab; preserve an established project when one already exists.

Replace the generated Cartpole implementation with the migrated environment and agent config. Preserve the current `src/<project>/tasks/<family>/config/<robot>/` layout and `isaaclab.tasks` entry point. UI extension metadata is optional, not required for task registration.

Run `uv sync` from the generated project root. Source-generated projects record editable paths to the originating Isaac Lab checkout in `[tool.uv.sources]`; verify these resolve to the intended checkout. Use `uv run python scripts/list_envs.py --show_presets` to check task discovery. Normal CLI use requires neither `PYTHONPATH` overrides nor external callbacks. A standalone Gym validation script must import `<project>.tasks` before Gym lookup; importing the passive top-level package alone does not register tasks.

For PhysX parity, use `uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx --num_envs 16`. Keep the extra and physics selection on subsequent simulation commands. The default generated environment uses Newton, so an unqualified smoke test does not establish PhysX parity.

Do not treat successful config loading as training success. Import/register, config resolution, static compilation, reset/step, random-agent, and short training are separate gates.

## Locomotion Training Gates

For quadruped and rough-terrain migrations, start with the flat walking variant before training terrain curriculum. A rough-terrain run can execute while still failing behaviorally because the robot falls or terminates on base contact immediately.

Full legacy command ranges may be too hard for a first policy validation. For example, IsaacGymEnvs `AnymalTerrain.yaml` samples yaw commands across `[-3.14, 3.14]`. When the goal is to prove a walking policy, either stage the command ranges as curriculum or validate first against the simpler flat `Anymal.yaml` behavior, and document the temporary deviation from exact terrain-task parity.

Treat these as separate gates:

- Flat task reset/step succeeds with expected observation and action shapes.
- Flat training improves reward and mean episode length toward the timeout horizon.
- The saved checkpoint loads in `play` or an equivalent bounded rollout.
- Rough-terrain training starts only after the flat policy is healthy, then tracks terrain-specific curriculum, height-scanner, and contact metrics.

## Policy Success Validation Loop

When a user asks for a migrated RL environment that trains successfully, automate the full train/evaluate/iterate loop. A training command with exit code 0 only proves that the runner executed. A checkpoint file only proves that the runner saved state. Policy success requires task-level evidence from training metrics and a loaded checkpoint rollout.

Before training, define the success criteria in the validation notes. Prefer an explicit task success metric from the source task or benchmark. If the task has no explicit success scalar, define proxy gates such as:

- Mean episode reward improves and remains stable over the last training window.
- Mean episode length approaches the task timeout horizon instead of ending mostly from falls, collisions, or invalid resets.
- Checkpoint rollout loads the saved policy and completes bounded episodes with behavior consistent with the task.
- Locomotion tasks keep base-contact or fall terminations low and report reasonable command-tracking errors.
- Any task-specific success rate, distance, velocity error, pose error, or object-state metric crosses the declared threshold.

Use this loop for each implementation iteration:

1. Register the migrated task from the external package or local module before any Gym lookup.
2. Run import and registration checks against the active Isaac Lab checkout.
3. Run a small reset and random-step smoke test; treat sensor-path warnings, invalid observation shapes, NaNs, and immediate terminations as failures.
4. Run a short training smoke only to verify runner integration.
5. Run the policy training budget needed for the task's declared success criteria.
6. Parse TensorBoard, JSON, or stdout scalars from that run; do not inspect only the final terminal lines.
7. Load the saved checkpoint in `play` or an equivalent bounded rollout and collect rollout metrics.
8. If the policy fails, modify the migration and rerun the shortest affected gate. Continue until success or until a concrete blocker is documented.

For an external template project, run from its root. Replace the task ID and log paths with the migrated task's values; the validation helper scripts below are project-owned scripts to create as needed, not generator outputs:

```bash
uv sync
uv run pytest tests/test_registration.py
uv run --extra isaacsim isaaclab random_agent \
  --task <TASK_NAME> physics=isaacsim_physx --num_envs 16
uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task <TASK_NAME> physics=isaacsim_physx \
  --device cuda:0 --num_envs 4096 --max_iterations 500
uv run python validation/parse_tensorboard.py logs/path/to/run --output validation/train_metrics.json
uv run --extra isaacsim python validation/evaluate_checkpoint.py \
  --checkpoint logs/path/to/run/model_499.pt \
  --num_envs 64 --steps 256 --device cuda:0
```

Ensure standalone simulation helpers explicitly select the same physics preset as training. Omit `--viz` for headless validation. Use `--viz none` only when a config or command would otherwise enable visualizers. These project-root `uv` commands also work from PowerShell without platform-specific import-path setup.

If no validation helper scripts exist, create the smallest scratch-only smoke, scalar parsing, and checkpoint evaluation scripts needed for the migration task. Do not add those helpers to Isaac Lab unless the user asks for committed validation files.

When a locomotion policy fails despite import/reset/step/training success, check these migration points before increasing training time:

- Command ranges and curriculum staging; broad yaw or velocity ranges may need staged validation.
- Reward signs, scales, clipping, alive terms, and episode-length scaling.
- Fall, base-contact, and timeout termination thresholds.
- Contact sensor, force sensor, ray caster, and body-name paths.
- Default pose, joint order, action scaling, PD gains, and drive modes.
- Observation order, units, clipping, normalization, noise, and missing history terms.
- Reset height, terrain origin, terrain curriculum, friction, mass, and push randomization.

For IsaacGymEnvs Anymal migration, validate flat walking before rough terrain. If `AnymalTerrain.yaml`'s full yaw command range prevents a healthy policy, narrow or curriculum-stage the commands, or first migrate the flat `Anymal.yaml` behavior, then reintroduce AnymalTerrain yaw ranges and rough-terrain curriculum.

## Legacy Force And Torque Sensors

Isaac Gym locomotion tasks may use force sensor tensors or net contact force tensors as policy inputs. In Isaac Lab, map these deliberately:

- Use `ContactSensorCfg` when the legacy observation only needs body net contact forces.
- Use `JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")` plus `SceneEntityCfg(..., body_names=[...])` when the legacy task depends on foot force and torque components. The maintained manager Ant task uses `mdp.body_incoming_wrench` this way for foot observations.
- Preserve the original observation shape only when parity requires it, and document any zero-padded or intentionally dropped torque slots.
- Validate sensor paths against the runtime asset's body names before training. Treat warnings such as "Failed to find rigid body" or "Failed to find contact report API" as validation failures, not harmless noise.

## Backend Mapping

Use PhysX as the first target when preserving Isaac Gym behavior, because Isaac Gym tasks were PhysX-based. Do not assume every PhysX parameter has a Newton equivalent.

Map backend parameters through the official schema docs. Current spawner slots may accept either property cfg wrappers or schema-fragment lists. Use property cfg wrappers for the common single-cfg path, and use fragments when a slot must combine multiple USD namespaces such as universal USD physics plus PhysX, Newton, or MuJoCo attributes.

| Isaac Gym / PhysX concern | Isaac Lab PhysX target | Newton target |
| --- | --- | --- |
| Simulation-level PhysX settings | `PhysxCfg` on `SimulationCfg.physics` | `NewtonCfg` with a solver cfg such as `MJWarpSolverCfg` |
| Rigid-body settings | `PhysxRigidBodyPropertiesCfg`, backend-portable `RigidBodyBaseCfg`, or fragments such as `UsdPhysicsRigidBodyCfg` plus `PhysxRigidBodyCfg` | `NewtonRigidBodyPropertiesCfg`, `MujocoRigidBodyPropertiesCfg`, or fragments such as `MujocoRigidBodyCfg` |
| Collision settings | `PhysxCollisionPropertiesCfg`, `CollisionBaseCfg`, or fragments such as `UsdPhysicsCollisionCfg` plus `PhysxCollisionCfg` | `NewtonCollisionPropertiesCfg`, `NewtonMeshCollisionPropertiesCfg`, `NewtonSDFCollisionPropertiesCfg`, or fragments such as `NewtonCollisionCfg` |
| Mesh cooking settings | `PhysxConvexHullPropertiesCfg`, `PhysxConvexDecompositionPropertiesCfg`, `PhysxTriangleMeshPropertiesCfg`, `PhysxTriangleMeshSimplificationPropertiesCfg`, or `PhysxSDFMeshPropertiesCfg` | `NewtonMeshCollisionPropertiesCfg` or `NewtonSDFCollisionPropertiesCfg` |
| Joint-drive settings | `JointDriveBaseCfg` or fragments such as `UsdPhysicsDriveCfg` plus `PhysxJointCfg` | `NewtonJointDrivePropertiesCfg`, `MujocoJointDrivePropertiesCfg`, or fragments such as `MujocoJointCfg` |
| Material settings | `PhysxRigidBodyMaterialCfg` or `RigidBodyMaterialBaseCfg` | `NewtonMaterialPropertiesCfg` |

For multi-backend tasks, use `PresetCfg` variants so the PhysX and Newton configs can differ cleanly. Keep backend-specific ranges, solver values, and unsupported options in separate presets.

Import backend schema classes from their backend packages, not through deprecated core shims:

- Core universal fragments/base cfgs: `from isaaclab.sim import schemas`
- PhysX cfgs/fragments: `from isaaclab_physx.sim import schemas as physx_schemas`
- Newton and MuJoCo cfgs/fragments: `from isaaclab_newton.sim import schemas as newton_schemas`

## Manager-Based Follow-Up Mapping

Use this mapping after the direct migration has reset, stepped, and trained. Recommend the manager-based follow-up when the task's observation, reward, command, event, curriculum, or termination logic should be reusable across robots, terrains, backends, or experiments. Use the `isaaclab-converting-direct-to-manager` skill for the conversion workflow.

| Direct migration concern | Manager-based target |
| --- | --- |
| Observation method | Observation manager terms |
| Reward method | Reward manager terms |
| Termination checks | Termination manager terms |
| Reset randomization | Event manager reset terms |
| Command sampling | Command manager terms |

## Current Workflow

Prefer a direct environment for the first migration pass. This is closer to Isaac Gym task structure and makes it easier to compare observations, rewards, resets, and actions against the original implementation.

After the direct migration is validated, recommend trying a manager-based version when the task should benefit from Isaac Lab's reusable managers. Keep the first pass direct, but do not leave users with the impression that direct is the desired long-term structure for reusable Isaac Lab tasks.

## Old Patterns

Legacy Isaac Gym tasks often combine asset loading, reward computation, reset logic, and randomization in one Python class. During the direct migration, keep the logic easy to compare with the original task, but structure methods according to the Isaac Lab direct workflow.

## Validation Checklist

- The migrated environment can construct with a small number of environments.
- `reset()` succeeds repeatedly.
- `step()` returns observations with expected shapes.
- Sensor paths resolve without missing rigid-body or contact-report warnings.
- Rewards and terminations match the intended task behavior.
- Training starts with the chosen RL framework.
- TensorBoard, JSON, or equivalent scalar parsing shows the declared policy-success metrics.
- A saved checkpoint loads in a bounded rollout and meets the declared rollout thresholds.
- A short training run only proves the runner can execute. Claim a successful policy only after a run of sufficient length shows stable reward improvement, episode lengths approaching the task horizon, or the task's explicit success metric.
