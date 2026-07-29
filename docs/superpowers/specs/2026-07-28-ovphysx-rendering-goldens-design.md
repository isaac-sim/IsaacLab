# OvPhysX Franka Rendering Goldens Design

## Goal

Replace the temporary blanket skip for Franka deformable OvPhysX rendering
tests with reviewed golden images for every currently supported OvPhysX and
OVRTX combination.

## Scope

- Generate goldens for both `franka_soft` and `franka_cloth`.
- Cover every OvPhysX + OVRTX AOV that the rendering helpers do not already
  skip for a documented renderer limitation.
- Keep the existing Newton-renderer and crashing OVRTX AOV skips unchanged.
- Use one baseline per task and AOV for both the legacy and OvStage OVRTX
  implementations.

The expected output is 23 PNG files. A requested ``rgb`` output also exposes
and validates ``rgba``, matching the existing rendering-baseline convention:

- 12 for `franka_soft`.
- 11 for `franka_cloth`, where `instance_segmentation` and `motion_vectors`
  remain skipped for their existing NVBUGs.

## Generation and Review

1. Remove the temporary blanket OvPhysX skips and their skip-only unit test.
2. Run only the legacy OvPhysX + OVRTX cases to bootstrap missing images.
3. Confirm the generated file count and inspect image montages for invalid,
   blank, clipped, duplicated, or visibly corrupted outputs.
4. Rerun the legacy cases against the new files to verify deterministic
   comparison rather than bootstrap behavior.
5. Run the OvStage variants against the same files to detect divergence
   between implementations.

No comparison thresholds will be widened as part of baseline generation.

## Validation

- Both task-specific kitless rendering suites pass for the supported OvPhysX
  + OVRTX legacy cases.
- The same cases pass with `ISAAC_LAB_OVRTX_USE_OVSTAGE=1`.
- Unsupported combinations remain explicitly skipped for their existing
  documented reasons.
- The existing deformable configuration, live task, and articulation dynamics
  regressions continue to pass.
- Full repository pre-commit passes before committing and before pushing.
