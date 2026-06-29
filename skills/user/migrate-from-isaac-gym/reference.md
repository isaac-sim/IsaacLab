# Isaac Gym Migration Reference

## Contents

- Direct workflow mapping
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

## External Scratch Packages

When validating a migration outside the Isaac Lab tree, keep the package importable through `PYTHONPATH` and register the Gym task by importing the package before `gym.make()` or registry lookups. Put the scratch package first, then every package directory under the target Isaac Lab checkout's `source/` directory. This prevents Python from mixing the active checkout with another installed Isaac Lab checkout and causing duplicate Gym registration or stale task imports. For scripts that do not expose `--external_callback`, create a small wrapper that imports the migrated package before running the Isaac Lab entry point. For training scripts that support external callbacks, point the callback to a registration function such as `my_migrated_task.register.register`.

Do not treat successful config loading as training success. Import/register, config resolution, static compilation, reset/step, random-agent, and short training are separate gates.

## Locomotion Training Gates

For quadruped and rough-terrain migrations, start with the flat walking variant before training terrain curriculum. A rough-terrain run can execute while still failing behaviorally because the robot falls or terminates on base contact immediately.

Treat these as separate gates:

- Flat task reset/step succeeds with expected observation and action shapes.
- Flat training improves reward and mean episode length toward the timeout horizon.
- The saved checkpoint loads in `play` or an equivalent bounded rollout.
- Rough-terrain training starts only after the flat policy is healthy, then tracks terrain-specific curriculum, height-scanner, and contact metrics.

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
- A short training run only proves the runner can execute. Claim a successful policy only after a run of sufficient length shows stable reward improvement, episode lengths approaching the task horizon, or the task's explicit success metric.
