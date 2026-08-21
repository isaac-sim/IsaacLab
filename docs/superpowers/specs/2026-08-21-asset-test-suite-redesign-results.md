<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Asset Test Suite Redesign Results

## Outcome

The comparable asset and WrenchComposer scopes now finish in **134.18 s** of
subprocess wall time, down from **1,197.60 s**: an **8.93x wall-time speed-up**.
The test-runner model is unchanged: every selected file runs in a fresh
subprocess. The final scopes collect 1,460 focused cases: 1,355 pass, 69 skip
with an explicit capability reason, and 36 are expected failures for explicit
Newton contract limitations.

The reduction comes from replacing backend-independent solver matrices with a
shared contract, moving backend-only branches into tiny unit/kernel tests, and
retaining one local real-solver seam per supported asset family. Newton's real
asset tests are kitless and use no Nucleus assets. Cable and MPM remain
deliberately excluded.

## Measurement environment

- Final branch base: `0c676a15c8` from `origin/develop`, rebased before the
  final measurements.
- Original baseline base: `d7033a5a1a207f1d4284edb60d72d7838984413b`.
- Worktree-local environment: `env_isaaclab`, installed with
  `UV_PROJECT_ENVIRONMENT=env_isaaclab uv sync --frozen --inexact --extra test
  --extra isaacsim --extra ovphysx`.
- IsaacSim 6.0.1.0, Kit 110.1.2, Warp 1.16.0, OVPhysX 0.5.10.
- NVIDIA GeForce RTX 5090, driver 590.48.01, 32,607 MiB.
- Commands ran with `OMNI_KIT_ACCEPT_EULA=YES` and a warmed writable Warp cache
  at `/tmp/isaaclab-task8-warp`.
- Times in the comparison are the repository orchestrator's aggregate pytest
  and subprocess wall times, not the outer shell duration.

## Copy-ready PR performance section

The asset-test redesign reduced the five comparable CI-style scopes from
19m57.60s to 2m14.18s wall time (**8.93x faster**). This comparison uses the
same repository test orchestrator before and after, with one process per test
file. Controller-owner and OV manager lifecycle checks are reported separately
and are not included in the denominator.

| Scope | Before files / cases | After files / outcomes | Before pytest / wall | After pytest / wall | Wall speed-up |
|---|---:|---:|---:|---:|---:|
| Shared assets | 8 / 4,331 | 6 / 1,185 pass, 69 skip, 36 xfail | 64.70 / 84.02 s | 3.80 / 18.26 s | **4.60x** |
| Newton assets, no cable/MPM | 6 / 644 | 15 / 58 pass | 590.94 / 608.44 s | 16.56 / 50.53 s | **12.04x** |
| PhysX assets | 7 / 486 | 12 / 41 pass | 236.80 / 255.30 s | 6.29 / 34.07 s | **7.49x** |
| OV assets | 9 / 492 | 12 / 56 pass | 196.94 / 220.05 s | 3.49 / 24.99 s | **8.81x** |
| WrenchComposer | 3 / 412 | 2 / 15 pass | 23.34 / 29.79 s | 2.50 / 6.33 s | **4.71x** |
| **Aggregate** | **33 / 6,365** | **47 / 1,355 pass, 69 skip, 36 xfail** | **1,112.72 / 1,197.60 s** | **32.64 / 134.18 s** | **8.93x** |

Focused gate and ownership timings:

| Gate or owner | Result | Pytest / wall |
|---|---:|---:|
| Shared contract, one process | 1,081 pass, 69 skip, 36 xfail | 5.36 / 6.81 s |
| Shared contract + adjacent units, file-isolated gate | 1,185 pass, 69 skip, 36 xfail | 3.80 / 18.26 s |
| Newton backend units/kernels, including executable kitless guard | 49 pass | 11.72 / 12.78 s |
| PhysX backend units | 33 pass | 1.72 / 2.84 s |
| OV backend units | 50 pass | 1.67 / 2.77 s |
| Newton minimal real integration | 4 files, 9 pass | 5.03 / 16.87 s |
| PhysX minimal real integration | 6 files, 8 pass | 4.43 / 20.62 s |
| OV minimal real integration | 4 files, 6 pass | 2.73 / 10.14 s |
| WrenchComposer real delivery | 1 file, 1 pass | 2.48 / 4.82 s |
| Newton task-space controller owner | 3 pass | 3.03 / 4.57 s |
| PhysX actuator-runtime and termination owners | 6 pass | 1.13 / 2.00 s |
| OV mixed CPU/CUDA lifecycle owner | 1 pass | 2.01 / 2.43 s |

All warmed contract/backend-unit gates are below the 30-second target.

## Exact comparable-scope commands

`TEST_INCLUDE_FILES` matches **basenames recursively** below
`TEST_FILTER_PATTERN`. This is intentional for the comparable scopes: for
example, `test_articulation.py` selects both the integration file and
`assets/unit/test_articulation.py`. Basenames are not unique identifiers.

```bash
WARP_CACHE_PATH=/tmp/isaaclab-task8-warp \
OMNI_KIT_ACCEPT_EULA=YES \
TEST_FILTER_PATTERN=/source/isaaclab/test/assets/ \
TEST_INCLUDE_FILES=test_asset_contract_api.py,test_asset_contract_data.py,test_asset_contract_writes.py,test_articulation_ordering.py,test_articulation_ordering_kernels.py,test_asset_selector_cache.py \
TEST_RESULT_FILE=task8-final-assets-shared.xml \
./isaaclab.sh -p -m pytest tools -q

WARP_CACHE_PATH=/tmp/isaaclab-task8-warp \
OMNI_KIT_ACCEPT_EULA=YES \
TEST_FILTER_PATTERN=/source/isaaclab_newton/test/assets/ \
TEST_INCLUDE_FILES=test_articulation.py,test_articulation_fk_cache.py,test_articulation_joint_staging.py,test_articulation_ordering.py,test_articulation_ordering_kernels.py,test_newton_actuator_adaptation.py,test_newton_actuators_newton.py,test_rigid_assets_import.py,test_rigid_object.py,test_rigid_object_collection.py,test_rigid_object_collection_model_indices.py,test_rigid_object_fk_cache.py,test_rigid_object_inertial_staging.py,test_rigid_object_setter_notifications.py,test_wrench_kernels.py \
TEST_RESULT_FILE=task8-final-assets-newton.xml \
./isaaclab.sh -p -m pytest tools -q

WARP_CACHE_PATH=/tmp/isaaclab-task8-warp \
OMNI_KIT_ACCEPT_EULA=YES \
TEST_FILTER_PATTERN=/source/isaaclab_physx/test/assets/ \
TEST_INCLUDE_FILES=test_actuator_control.py,test_articulation.py,test_deformable_object.py,test_newton_actuators_physx.py,test_rigid_object.py,test_rigid_object_collection.py,test_surface_gripper.py \
TEST_RESULT_FILE=task8-final-assets-physx.xml \
./isaaclab.sh -p -m pytest tools -q

WARP_CACHE_PATH=/tmp/isaaclab-task8-warp \
OMNI_KIT_ACCEPT_EULA=YES \
TEST_FILTER_PATTERN=/source/isaaclab_ov/test/assets/ \
TEST_INCLUDE_FILES=test_actuator_control.py,test_articulation.py,test_articulation_helpers.py,test_articulation_kernels.py,test_deformable_object.py,test_deformable_views.py,test_rigid_object.py,test_rigid_object_collection.py \
TEST_RESULT_FILE=task8-final-assets-ov.xml \
./isaaclab.sh -p -m pytest tools -q

WARP_CACHE_PATH=/tmp/isaaclab-task8-warp \
OMNI_KIT_ACCEPT_EULA=YES \
TEST_FILTER_PATTERN=/source/isaaclab/test/utils/ \
TEST_INCLUDE_FILES=test_wrench_composer.py,test_wrench_composer_integration.py \
TEST_RESULT_FILE=task8-final-wrench-composer.xml \
./isaaclab.sh -p -m pytest tools -q
```

## Fast and real gate commands

Run contract definitions together, but keep the adjacent legacy-ordering unit
module as its own file. That unit intentionally controls import stubs at module
collection time; the repository orchestrator above gives it the required clean
process.

```bash
# Shared contract and backend-specific fast gates.
export WARP_CACHE_PATH=/tmp/isaaclab-task8-warp
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p -m pytest source/isaaclab/test/assets/contract -q
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/unit \
  source/isaaclab_newton/test/assets/test_articulation_ordering_kernels.py -q
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/unit -q
./isaaclab.sh -p -m pytest source/isaaclab_ov/test/assets/unit -q
```

Real-only subprocess gates must exclude `/assets/unit/`; otherwise colliding
basenames pull unit files into the result.

```bash
export WARP_CACHE_PATH=/tmp/isaaclab-task8-warp
export OMNI_KIT_ACCEPT_EULA=YES

TEST_FILTER_PATTERN=/source/isaaclab_newton/test/assets/ \
TEST_EXCLUDE_PATTERN=/assets/unit/ \
TEST_INCLUDE_FILES=test_articulation.py,test_newton_actuators_newton.py,test_rigid_object.py,test_rigid_object_collection.py \
TEST_RESULT_FILE=task8-final-integration-newton.xml \
./isaaclab.sh -p -m pytest tools -q

TEST_FILTER_PATTERN=/source/isaaclab_physx/test/assets/ \
TEST_EXCLUDE_PATTERN=/assets/unit/ \
TEST_INCLUDE_FILES=test_articulation.py,test_deformable_object.py,test_newton_actuators_physx.py,test_rigid_object.py,test_rigid_object_collection.py,test_surface_gripper.py \
TEST_RESULT_FILE=task8-final-integration-physx.xml \
./isaaclab.sh -p -m pytest tools -q

TEST_FILTER_PATTERN=/source/isaaclab_ov/test/assets/ \
TEST_EXCLUDE_PATTERN=/assets/unit/ \
TEST_INCLUDE_FILES=test_articulation.py,test_deformable_object.py,test_rigid_object.py,test_rigid_object_collection.py \
TEST_RESULT_FILE=task8-final-integration-ov.xml \
./isaaclab.sh -p -m pytest tools -q
```

## Old-module disposition

### Shared Isaac Lab and WrenchComposer

| Old module | Disposition | Coverage owner |
|---|---|---|
| `test_articulation_iface.py` | Replaced/renamed | Contract API/data/write entries; assertion cases remain in `_articulation_contract_cases.py`. |
| `test_articulation_ordering_iface.py` | Moved | `_articulation_ordering_contract_cases.py`, exposed by the three contract entries. |
| `test_rigid_object_iface.py` | Replaced/renamed | Contract API/data/write entries and `_rigid_object_contract_cases.py`. |
| `test_rigid_object_collection_iface.py` | Replaced/renamed | Contract API/data/write entries and `_rigid_object_collection_contract_cases.py`. |
| `test_iface_test_utils.py` | Replaced | Explicit backend capabilities and public-base-surface classification tests. |
| `test_articulation_ordering.py` | Retained | Solver-independent name/order/map units in its own clean process. |
| `test_articulation_ordering_kernels.py` | Retained | Tiny gather/scatter/write kernels and selector-width coverage. |
| `test_asset_selector_cache.py` | Retained | Selector identity/domain/LRU semantics. |
| `test_wrench_composer.py` | Replaced/focused | Fourteen literal `2 x 2` arithmetic, selection, reset, lazy, merge, validation, and compatibility cases. |
| `test_wrench_composer_integration.py` | Consolidated | One rotated global force-at-position delivery parity test. |
| `test_wrench_composer_vs_physx.py` | Removed as redundant | The unique rotated force/induced-torque seam is in the retained integration test; the matrix moved to literal units. |

### Newton, excluding cable and MPM

| Old module | Disposition | Coverage owner |
|---|---|---|
| `test_articulation.py` | Retained and reduced | Local floating/fixed articulation seams, partial state/property/wrench, drive, Jacobian, and mass matrix. FK, staging, and ordering moved to units; IK/OSC/gravity moved to the controller owner. |
| `test_newton_actuators_newton.py` | Retained and reduced | One real Lab/native execution-path equivalence; adaptation and target-mode branches moved to units. |
| `test_rigid_object.py` | Retained and reduced | Local CPU property/state/wrench seam plus CUDA smoke; selection, inverse inertia, FK, and notification branches moved to units. |
| `test_rigid_object_collection.py` | Retained and reduced | Local `N=2, B=2` selection/property seam; model-index mapping moved to units. |
| `test_articulation_ordering_kernels.py` | Retained | Tiny Newton ordering-kernel coverage. |
| `test_wrench_kernels.py` | Moved | `assets/unit/test_wrench_kernels.py`. |

`test_cable_object.py` and `test_mpm_object.py` are excluded, unchanged, and
absent from every benchmark command.

### PhysX

| Old module | Disposition | Coverage owner |
|---|---|---|
| `test_articulation.py` | Retained and reduced | One local ordered CPU articulation seam and one CUDA dynamics smoke; property/order conversions moved to units. |
| `test_articulation_kernels.py` | Moved/expanded | `assets/unit/test_articulation.py`. |
| `test_deformable_object.py` | Replaced | Two working local surface/volume probes plus focused classification/material/target/kernel units; the former all-skipped startup is gone. |
| `test_newton_actuators_physx.py` | Retained and reduced | One real ordered Lab/native dispatch seam; dispatch and graph branches moved to backend and shared actuator units. |
| `test_rigid_object.py` | Retained and reduced | Local state, raw mass/COM/inertia/material, and wrench delivery; staging/cache/import isolation moved to units. |
| `test_rigid_object_collection.py` | Retained and reduced | Local nontrivial body-major selection and raw property/material readback; ID/layout/staging moved to units. |
| `test_surface_gripper.py` | Replaced | Local two-cube open/close seam; filtering, partial properties, and CUDA rejection moved to units. |

### OVPhysX

| Old module | Disposition | Coverage owner |
|---|---|---|
| `test_articulation.py` | Retained and reduced | Local CPU and CUDA articulation seams with ordered state/properties, native actuation, Jacobian, and mass access. |
| `test_articulation_helpers.py` | Moved | `assets/unit/test_articulation_helpers.py`. |
| `test_articulation_kernels.py` | Moved | `assets/unit/test_articulation_kernels.py`. |
| `test_deformable_object.py` | Retained and reduced | One volume and one surface CUDA seam, including forced rewarm isolation. |
| `test_deformable_object_helpers.py` | Moved/expanded | `assets/unit/test_deformable_object.py`. |
| `test_deformable_views.py` | Moved | `assets/unit/test_deformable_views.py`. |
| `test_rigid_object.py` | Retained and reduced | Local partial state, raw inertial property, and real wrench delivery. |
| `test_rigid_object_collection.py` | Retained and reduced | Local nonidentity selection plus raw fused inertial/material mapping. |
| `test_rigid_object_helpers.py` | Moved/expanded | Rigid and fused-collection unit modules. |

## Case-family disposition and coverage holes closed

| Old case family | Disposition | Current evidence |
|---|---|---|
| Generic initialization, names, shapes, aliases, defaults, finders | Replaced | Shared API/data contracts with canonical CPU `N=2, B=3, J=4`. |
| Root/body/joint writers, partial selection, invalid shapes | Replaced | Shared write contracts with literal index/mask cases. |
| Cache timestamps and invalidation | Moved | Shared contracts plus backend-specific cache/FK units. |
| Broad environment/body/joint/device Cartesian products | Removed as redundant | One canonical case, targeted singleton/order cases, and one genuine CUDA smoke per device-specific path. |
| Repeated wrench arithmetic, frames, masks, offsets, and long rollouts | Replaced | Literal WrenchComposer units plus one real delivery per backend. |
| Remote ANYmal/Panda/ShadowHand/humanoid fixtures | Replaced | Small local authored primitives and branching articulations. |
| Backend model/view ordering and selector translation | Moved/expanded | Newton model-index/order units, PhysX body-major units, and OV fused-layout units, each backed by one real nonidentity seam. |
| Mass, COM, inertia, friction, and restitution setters | Retained and strengthened | Raw backend view/binding readback per rigid, collection, and articulation family. |
| Newton FK freshness and model notifications | Newly focused | Dedicated units with exact literal state/model flags. |
| PhysX collection COM/inertia layout | Newly covered/fixed | Real body-major write/read exposed and guards flattened COM and 3-D inertia TensorAPI layouts. |
| PhysX deformable surface/volume distinction and material fallback | Newly covered | Focused units plus two non-skipping real probes. |
| OV fused layout, CPU staging, view rewarm, and manager device reuse | Newly covered | Focused units, forced-rewarm deformable seam, and CPU-CUDA-CPU lifecycle owner. |
| IK, OSC, gravity compensation, graph capture, and termination | Moved | Newton task-space controller owner; shared PhysX actuator-runtime and termination owners. |
| Newton kinematic rigid-object parameters | Removed unsupported skips | Newton does not support these modes; the old cases executed no behavior. |
| Cable and MPM | Excluded | Assigned outside this project and unchanged. |

The public-base-surface audit classifies every member declared by `AssetBase`,
`BaseRigidObject`, `BaseRigidObjectData`, `BaseRigidObjectCollection`,
`BaseRigidObjectCollectionData`, `BaseArticulation`, and
`BaseArticulationData` as covered, explicitly unsupported, or reasoned out of
scope. Factory manager replacements are scoped to one contract test and exact
PhysX/Newton production bindings are restored; forward and reverse
cross-backend factory order is covered.

## Unsupported capabilities and expected outcomes

- Newton spatial tendons are explicitly unsupported. Contract parameters stay
  visible as reasoned skips rather than disappearing from collection.
- Contract fixtures without spatial tendons skip spatial-tendon data probes
  with `No spatial tendons configured`.
- The 36 expected failures document Newton fixed-tendon writer/property gaps
  and its position-only COM representation; they are not silent omissions.
- OV surface deformable kinematic targets remain unsupported and raise the
  asserted `ValueError` in unit and real coverage.
- The installed PhysX wheel reports an invalid surface-view `check()` flag
  while still accepting/returning nodal data. The real test asserts the stable
  supported operations rather than the inconsistent flag.

No retained backend asset file is entirely skipped. All Newton, PhysX, OV, and
Wrench real integration files executed on the reference GPU.

## Isolation and residual warnings

Final isolation checks included:

- contract factory manager access/restoration in PhysX-Newton and
  Newton-PhysX order;
- Newton rigid/collection/articulation forward and reverse repetition:
  12 passed, with four CUDA cases skipped only in the sandboxed repeat probe;
- PhysX deformables repeated in one process: 4 passed;
- OV deformable/rigid/articulation forward and reverse repetition: 8 passed;
- OV CPU-CUDA-CPU manager lifecycle: 1 passed.

Residual output is from existing runtime behavior: Torch JIT deprecations,
Isaac Lab schema deprecations, Newton shape-color and coordinate-layout future
warnings, headless Kit display/IOMMU messages, OVPhysX automatic warmup and
USD synchronization messages, and the PhysX surface-view warning described
above. None caused a skip or failure in the retained real integration gates.
