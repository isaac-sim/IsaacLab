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

Only use this mapping when the user asks for manager-based organization or when the task needs reusable manager terms:

| Direct migration concern | Manager-based target |
| --- | --- |
| Observation method | Observation manager terms |
| Reward method | Reward manager terms |
| Termination checks | Termination manager terms |
| Reset randomization | Event manager reset terms |
| Command sampling | Command manager terms |

## Current Workflow

Prefer a direct environment for the first migration pass. This is closer to Isaac Gym task structure and makes it easier to compare observations, rewards, resets, and actions against the original implementation.

Move to manager-based environments only when the user asks for a modular task or after the direct migration is validated.

## Old Patterns

Legacy Isaac Gym tasks often combine asset loading, reward computation, reset logic, and randomization in one Python class. During the direct migration, keep the logic easy to compare with the original task, but structure methods according to the Isaac Lab direct workflow.

## Validation Checklist

- The migrated environment can construct with a small number of environments.
- `reset()` succeeds repeatedly.
- `step()` returns observations with expected shapes.
- Rewards and terminations match the intended task behavior.
- Training starts with the chosen RL framework.
