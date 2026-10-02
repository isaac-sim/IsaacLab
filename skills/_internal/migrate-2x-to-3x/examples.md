# Isaac Lab 3.0 Migration Examples

## Contents

- Quaternion migration
- ProxyArray migration
- RSL-RL compatibility migration

## Quaternion Migration

Input: downstream code contains hardcoded quaternions and direct quaternion index assignments.

Expected workflow:

1. Read the quaternion section of the official migration guide.
2. Run the documented quaternion finder tool on the downstream project.
3. Fix identity quaternions mechanically when safe.
4. Review non-identity quaternions and runtime index assignments manually.
5. Run a small task smoke test.

## ProxyArray Migration

Input: downstream code calls tensor-only methods on asset or sensor data.

Expected workflow:

1. Read the `ProxyArray` sections of the official migration guide.
2. Search for tensor-only method calls on `.data.*` properties.
3. Use the current `ProxyArray` access pattern from the docs and source.
4. Run focused tests around the migrated logic.

## RSL-RL Compatibility Migration

Input: downstream training scripts fail after the RSL-RL upgrade.

Expected workflow:

1. Inspect the current Isaac Lab RSL-RL entry points, agent configs, and `source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py`.
2. Use the compatibility helper when it applies.
3. Avoid preserving copied training scripts if a maintained Isaac Lab script can be reused or imported.
4. Run a short training or config-construction smoke test.
