<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Solver Tuning How-to Subsection Design

## Purpose

Group solver-specific tuning guidance under one How-to subsection and make the
Kamino guide parallel to the existing MJWarp tuning guide.

This design supersedes only the MJWarp and Kamino How-to paths and the Kamino
page contract in the physical-backend documentation reorganization design from
2026-08-13. All other content ownership, labels, and file locations from that
design remain unchanged.

## Information architecture

Create this subsection:

```text
docs/source/how-to/solver_tuning/
├── index.rst
├── tune_mjwarp.rst
└── tune_kamino.rst
```

The main How-to index exposes only `solver_tuning/index`. The subsection index
briefly states that solver tuning begins after backend, task, and asset
validation, then lists the MJWarp and Kamino guides.

Move the existing MJWarp guide without changing its tuning scope. Rename and
rewrite `enable_kamino.rst` as `solver_tuning/tune_kamino.rst`.

## Kamino guide contract

Title the page `Tune Kamino`. Preserve the existing `newton-kamino-solver`
label for compatibility and add `kamino-solver-tuning` as the parallel tuning
label.

Use this diagnose-first sequence:

1. Establish a fixed baseline from a Newton-compatible task and asset.
2. Choose PADMM or DVI based on the workload.
3. Validate reset and state consistency.
4. Choose and size collision handling.
5. Tune timestep and substeps.
6. Tune convergence and stabilization.
7. Optimize only after the physical and task metrics are stable.

Reduce Kamino enablement and preset wiring to a short prerequisite. Link to
`backends-and-presets` and the Hydra preset guide instead of maintaining a
second preset-integration procedure or code example.

Generated API documentation remains authoritative for every solver field,
accepted value, and default. The guide may name controls needed by a diagnostic
decision, but it must not restore parameter tables or universal defaults.

## Links and compatibility

Update every repository link from:

- `/source/how-to/tune_mjwarp` to
  `/source/how-to/solver_tuning/tune_mjwarp`;
- `/source/how-to/enable_kamino` to
  `/source/how-to/solver_tuning/tune_kamino`.

Preserve `mjwarp-solver-tuning` and `newton-kamino-solver`. Add
`kamino-solver-tuning`. Update affected skill references so they resolve to the
new MJWarp path.

## Validation

- Confirm the old flat files no longer exist.
- Confirm no active RST or skill Markdown references the old docnames.
- Confirm all three solver-tuning labels occur exactly once.
- Run skill validation because a skill links to the MJWarp guide.
- Run the warnings-as-errors Sphinx dummy build.
- Run repository-wide pre-commit checks before committing and after staging.
- Confirm cached and branch diffs contain no whitespace errors.

## Non-goals

- Do not move native physics API guidance.
- Do not change backend, solver, preset, or runtime behavior.
- Do not add a separate `Enable Kamino` page.
- Do not create task-support inventories or copy generated configuration
  reference material.
