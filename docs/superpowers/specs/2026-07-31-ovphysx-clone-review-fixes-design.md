# OvPhysX Clone Review Fixes Design

## Goal

Complete the justified review feedback on PR #6766 while preserving its nested-clone pose fix and integrating it with the clone-recipe lifecycle now present on `develop`.

## Integration strategy

Merge current `develop` into the PR branch so the update remains additive and does not require rewriting the contributor's history. Resolve the conflicts in the OvPhysX cloner and manager by retaining `develop`'s active/pending recipe lifecycle while upgrading the recipe payload from translation-only positions to final world transforms in `(x, y, z, qx, qy, qz, qw)` order.

Full-stage materialization continues to use source and target paths because authored USD preserves nested local transforms. Env-0-only runtime replay passes the final transforms to `physx.clone()`.

## Clone transform boundary

Add one private package module that owns the fixed-length clone-transform type and the conversion from `(x, y, z)` positions to identity-rotation transforms. Both the cloner and manager import this definition so their cross-module payload cannot drift.

`OvPhysxManager.register_clone()` remains the supported translation-only entry point. A private full-transform registration method records identical recipes in both the active and pending queues, allowing forced warmups to reproduce the same runtime clones.

## Validation

`OvPhysxReplicateContext.queue_mapping()` validates inputs at its exported boundary:

- Every active source prim must exist before its world transform is queried.
- A derived source-environment anchor must exist before its world transform is queried.
- A provided positions tensor must have shape `[num_envs, 3]` and contain every selected non-negative environment ID.
- A provided quaternions tensor must have shape `[num_envs, 4]` and contain every selected non-negative environment ID.
- `None` retains its intentional identity-position or identity-orientation meaning.

Malformed inputs raise `ValueError` with the input name or prim path in the message instead of silently substituting identity transforms.

## Documentation and comments

Public docstrings describe current behavior only: translation-only calls queue world positions with identity rotations; mapping calls compose source-relative rows with target environment poses. The module docstring remains the single owner of deferred clone lifecycle details. Internal comments state durable lifecycle facts without repeating tuple layouts already expressed by the shared type.

## Tests

Keep the nested-pose regression focused on the manager handoff: invoke `replicate()`, assert queue clearing, and inspect the recorded final transform once. Preserve the translation-only compatibility test against both active and pending recipes.

Add parameterized validation coverage for short or malformed position/quaternion tensors, plus invalid source and anchor prim coverage. Run each new regression against code without the corresponding guard to confirm the expected failure before implementing the production change. After implementation, run the focused OvPhysX tests and the repository-wide pre-commit hooks.

## Scope

No public API symbols are removed or renamed, no dependencies are added, and the existing OvPhysX changelog fragment remains the single fragment for the package.
