# Newton Asset Preparation Reference

## Contents

- Asset classifications
- Audit checklist
- Task-level control checks
- Common failures

## Asset Classifications

Use these labels when reporting status:

- PhysX-compatible: the asset works in the current PhysX task or standalone smoke.
- Newton-runnable: Newton can parse and simulate the asset enough for a limited smoke.
- Newton-clean: authored metadata, task spawn path, and control path pass the validation checklist.

## Audit Checklist

Inspect:

- Rigid body APIs and authored mass properties.
- Diagonal inertia and center-of-mass values.
- Collider types and collision approximation.
- Joint topology and fixed-joint structure.
- Friction and material overrides.
- Nested references and package dependencies.
- Whether task-level overrides apply to the same prims under Newton.

## Task-Level Control Checks

Passing asset import is not enough. Also verify:

- Actuator joint name patterns resolve to the converted USD joint names.
- Controller body names and frame names resolve.
- Action dimensions match the environment action term.
- Stiffness, damping, armature, effort limits, and friction are intentional for Newton.
- Zero-action and small nonzero-action rollouts are finite and move the expected joints or bodies.

## Common Failures

- Missing authored mass, inertia, or center of mass.
- Placeholder inertia warnings.
- Fixed-joint topology rejected by Newton.
- Nested references resolve locally but fail in CI or containers.
- Visual-only support geometry causes objects to fall or contact counts to spike.
- Asset import passes, but stale actuator or controller names break the task.
