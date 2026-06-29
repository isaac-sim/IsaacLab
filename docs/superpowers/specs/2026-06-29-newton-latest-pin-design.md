<!-- Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md). -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: BSD-3-Clause -->

# Newton Latest-Pin Compatibility Probe

## Goal

Open a focused pull request against IsaacLab's `develop` branch that installs the latest commit on Newton's default
`main` branch and uses IsaacLab CI to reveal compatibility blockers. The upstream head resolved with `git ls-remote`
on 2026-06-29 is `49ca73b9dd088e48a853aa7033f928f3bd84aa78`; `develop` currently pins
`79e95bf5571d70a0a46c8eaedc80644531d27368`.

## Scope

Use the smallest dependency update that lets CI reach runtime tests:

- Replace all seven active `newton[sim]` dependency declarations with the selected Newton commit.
- Replace IsaacLab's three active `warp-lang==1.14.0` declarations with
  `warp-lang==1.15.0.dev20260626`. Newton requires at least that development build and locks the same build itself.
- Add patch changelog fragments for `isaaclab`, `isaaclab_newton`, `isaaclab_physx`, and `isaaclab_visualizers`.
- Preserve generated `docs/CHANGELOG.rst` files and every `config/extension.toml` file.

This first PR will not adapt IsaacLab to upstream API or behavior changes. Keeping those fixes out of the initial
commit preserves CI as a compatibility probe and makes each failure attributable to the dependency update.

## Approaches Considered

1. **Resolvable pin probe (selected):** update Newton and the minimum required Warp build, then let CI expose runtime
   incompatibilities. This avoids a known resolver failure without masking API or physics changes.
2. **Newton-only pin:** change only Newton declarations. This is smaller but cannot resolve because IsaacLab pins
   Warp 1.14.0 while the selected Newton commit requires `warp-lang>=1.15.0.dev20260626`.
3. **Proactive compatibility migration:** update dependencies and immediately adapt known API changes. This would
   produce a larger PR and reduce the value of the requested CI reconnaissance; compatibility fixes can follow once
   CI supplies a complete failure inventory.

## Files and Consistency Boundaries

The Newton commit is duplicated intentionally across source-package extras and wheel metadata:

- `source/isaaclab_newton/pyproject.toml`
- `source/isaaclab_physx/pyproject.toml`
- `source/isaaclab_visualizers/pyproject.toml` (four extras)
- `tools/wheel_builder/res/python_packages.toml`

The exact Warp build is duplicated in:

- `source/isaaclab/pyproject.toml`
- `tools/wheel_builder/res/python_packages.toml` (core and Newton-extra declarations)

`source/isaaclab/test/cli/test_wheel_builder_metadata.py` is the consistency boundary for both sets of duplicates.
No new test is needed because it already compares every active declaration.

## Validation

1. Re-run `git ls-remote --symref https://github.com/newton-physics/newton.git HEAD refs/heads/main` immediately
   before editing. If `main` advanced, use the new head and reassess its Warp requirement.
2. Run the existing metadata test after changing only the canonical source declarations and confirm it fails on the
   stale replicas.
3. Update every replica and confirm the metadata test passes.
4. Run `./isaaclab.sh -f`, review any automatic edits, and run it again if it changed files.
5. Push only to the `antoine` fork, open a normal PR against `isaac-sim/IsaacLab:develop`, and monitor CI.

## Expected CI Risk Areas

The upstream range contains 31 commits across 156 files. The most likely runtime blocker is the Kamino reset API:
Newton replaced the `joint_q` and `joint_u` reset keywords with `SolverKamino.ResetConfig`, while IsaacLab still uses
the removed keywords. Other risk areas include keyword-only deprecation warnings, Featherstone body-velocity and
wrench semantics, mesh/SDF contact reduction, and VBD particle-rigid contact margins. The deprecated tiled-camera,
body-armature, and VBD Dahl-friction APIs removed in this range have no active IsaacLab call sites.

CI failures will be reported with the failing jobs and first actionable error. They will not be fixed in this PR
without a separate scope decision.
