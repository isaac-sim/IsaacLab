# Isaac Lab 2.x To 3.x Migration Reference

## Contents

- Source of truth
- Current workflow
- Old patterns
- Maintenance rule

## Source of Truth

The official migration guide is the authoritative source:

- `docs/source/migration/migrating_to_isaaclab_3-0.rst`

Use this skill to decide where to look and how to validate. Do not treat this file as a replacement for the migration guide.

## Current Workflow

Start with the official migration guide, then verify behavior against current source files and examples. Prioritize these areas:

- Visualizer CLI and headless behavior.
- Multi-backend architecture and backend-specific packages.
- Schema configuration class refactors.
- Quaternion convention changes.
- `ProxyArray` access for asset and sensor data.
- Asset and physics view API renames.
- RL workflow compatibility helpers.

## Old Patterns

Avoid carrying forward old project-specific migration notes as general guidance. The external prototype skill included useful discoveries, but it also contained downstream-project paths, stale raw Warp-array guidance, and one-off commands. Convert those into short search checks and official doc improvements.

## Maintenance Rule

If a repeated migration fix is not covered by the official guide, update `docs/source/migration/migrating_to_isaaclab_3-0.rst`. Keep this skill small so it does not become a second migration document that drifts from the codebase.
