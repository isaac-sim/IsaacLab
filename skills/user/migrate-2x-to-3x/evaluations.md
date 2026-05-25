# Isaac Lab 3.0 Migration Evaluations

## Contents

- Scenario 1: quaternion migration
- Scenario 2: ProxyArray migration
- Scenario 3: RSL-RL migration

## Scenario 1: Quaternion Migration

Query: "My Isaac Lab 2.x project has hardcoded quaternions and now behaves incorrectly in 3.0."

Expected behavior:

- Reads the official migration guide before proposing edits.
- Uses the quaternion finder workflow documented in the repo.
- Distinguishes quaternions from colors or other four-element values.
- Recommends a small task smoke test.

Known failure modes:

- Reorders every four-element tuple blindly.
- Ignores runtime quaternion index assignments.

## Scenario 2: ProxyArray Migration

Query: "After upgrading, `asset.data.joint_pos.detach()` fails."

Expected behavior:

- Routes to the current `ProxyArray` migration guidance.
- Verifies the current access pattern against docs/source before editing.
- Avoids stale raw `wp.to_torch()`-only guidance when the current docs recommend explicit `ProxyArray` accessors.

Known failure modes:

- Copies old raw Warp-array instructions from an external skill.
- Changes unrelated asset APIs without checking current source.

## Scenario 3: RSL-RL Migration

Query: "My old RSL-RL training script fails after migrating to Isaac Lab 3.0."

Expected behavior:

- Checks maintained Isaac Lab RSL-RL scripts and compatibility helpers.
- Identifies whether the downstream script can be replaced with or aligned to the maintained script.
- Runs a small config or training smoke test.

Known failure modes:

- Patches a copied training script without comparing to the current maintained implementation.
- Hardcodes old version-specific assumptions into the skill.
