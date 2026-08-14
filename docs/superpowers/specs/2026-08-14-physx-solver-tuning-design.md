<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# PhysX Solver Tuning Guide Design

## Purpose

Preserve the practical PhysX configuration guidance removed from
`docs/source/overview/core-concepts/physical-backends/physx/configuration.rst`
while placing it in the new solver-tuning how-to subsection. The existing
published path may remain deleted; no compatibility stub is required.

## Documentation structure

Create `docs/source/how-to/solver_tuning/tune_physx.rst` and list it first in
`docs/source/how-to/solver_tuning/index.rst`, before the MJWarp and Kamino
guides. Keep `docs/source/concepts/physics_backends.rst` concise and add a link
from its PhysX section to the new guide.

The page will retain the useful material from the removed configuration page:

- a practical `PhysxCfg` baseline;
- TGS and PGS solver selection;
- solver iteration controls;
- contact and stability controls;
- GPU buffer sizing and exhaustion guidance; and
- links to the authoritative `PhysxCfg` API and USD schema configuration.

The guide will frame values as tuning starting points rather than universal
recommendations. Generated API documentation remains authoritative for field
names, accepted values, and defaults.

## Scope boundaries

Do not restore the old page or duplicate the full tuning content in Concepts.
Do not change PhysX behavior, public APIs, solver defaults, or unrelated
backend documentation.

## Validation

Verify the new page is reachable from both the solver-tuning index and the
PhysX Concepts section. Run a Sphinx build with warnings treated as errors,
check for stale references to the deleted configuration path, and run the
repository pre-commit suite before committing and pushing.
