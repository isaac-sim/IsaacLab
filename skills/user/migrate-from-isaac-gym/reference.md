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

When validating a migration outside the Isaac Lab tree, start with the template generator instead of hand-rolling the external package structure. From the Isaac Lab checkout, run `uv run isaaclab -n`, then choose:

- `External` project.
- The scratch directory as the project path.
- A valid Python identifier for the project name, such as `isaacgym_anymal_migration`.
- `Direct | single-agent` for the first Isaac Gym parity pass.
- The target RL library, usually `rsl_rl` for locomotion validation.

Replace the generated task implementation with the migrated environment, config, registration, and agent config, preserving the generated project layout. The generated project gives agents a known `source/<project>/<project>/tasks/...` structure, `pyproject.toml`, extension metadata, scripts, and task registration pattern.

For validation, install the generated extension in editable mode with `uv pip` for the target Isaac Lab environment, or keep it importable through `PYTHONPATH`. If using `PYTHONPATH`, put the generated project's extension package first, then every package directory under the target Isaac Lab checkout's `source/` directory. This prevents Python from mixing the active checkout with another installed Isaac Lab checkout and causing duplicate Gym registration or stale task imports. If editable install makes task registration available automatically, omit external callbacks. For uninstalled scratch validation, use a small wrapper for scripts without `--external_callback`, and use the callback option when a training script exposes one.

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

For an external template project, use commands like these as templates. Replace the task id, callback, project path, extension path, and log paths with the migrated package names:

```bash
export LAB=/path/to/IsaacLab
export PROJECT=/path/to/migration-scratch/isaacgym_anymal_migration
export EXTENSION="$PROJECT/source/isaacgym_anymal_migration"
export PYTHONPATH="$EXTENSION:$(find "$LAB/source" -mindepth 1 -maxdepth 1 -type d | paste -sd: -)"
cd "$LAB"

# Optional when validating as an installed template project:
uv pip install -e "$EXTENSION"

uv run python -c "import gymnasium as gym; import my_migration; tid='My-Migrated-Task-v0'; spec=gym.spec(tid); print(spec.entry_point); print(spec.kwargs)"

uv run python "$PROJECT/validation/smoke_my_task.py" --device cuda:0 --num_envs 16 --steps 8

uv run isaaclab train --rl_library rsl_rl \
  --task My-Migrated-Task-v0 \
  --external_callback my_migration.register.register \
  --device cuda:0 --num_envs 4096 --max_iterations 500

uv run python "$PROJECT/validation/parse_tensorboard.py" logs/path/to/run --output "$PROJECT/validation/train_metrics.json"

uv run python "$PROJECT/validation/evaluate_checkpoint.py" \
  --checkpoint logs/path/to/run/model_499.pt \
  --num_envs 64 --steps 256 --device cuda:0
```

Omit `--viz` for headless validation. Use `--viz none` only when a config or command would otherwise enable visualizers.

On Windows, use the same `uv` commands from PowerShell and set the same path order:

```powershell
$lab = "C:/path/to/IsaacLab"
$project = "C:/path/to/migration-scratch/isaacgym_anymal_migration"
$extension = "$project/source/isaacgym_anymal_migration"
$srcs = Get-ChildItem "$lab/source" -Directory | ForEach-Object { $_.FullName }
$env:PYTHONPATH = $extension + ";" + ($srcs -join ";")
Set-Location $lab
```

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
