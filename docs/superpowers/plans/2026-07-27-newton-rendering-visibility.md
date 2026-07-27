<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Newton Rendering Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore procedural collider visibility in Newton-rendered scenes and replace the stale Franka deformable golden images with the already validated outputs.

**Architecture:** A private cloner helper will inspect Newton's USD import result and re-enable `ShapeFlags.VISIBLE` only for non-mesh colliders whose USD geometry is viewport-visible and whose body or static parent has no separate visible geometry. Every Isaac Lab Newton USD-import path will invoke that helper immediately after `ModelBuilder.add_usd`; the rendering tests will retain their current comparison thresholds.

**Tech Stack:** Python 3.12, USD (`pxr.Usd`, `UsdGeom`, `UsdPhysics`), Newton `ModelBuilder`, pytest, Git LFS.

## Global Constraints

- Do not change rendering-job `continue-on-error`.
- Do not change pixel-difference or SSIM thresholds.
- Do not change renderer dependency pins.
- Add no required or optional dependency.
- Verify the regression test fails without the implementation and passes with it.
- Add one `isaaclab_newton` changelog fragment and one shared `isaaclab_tasks` test-only skip fragment.
- Run commands through `./isaaclab.sh -p` and run `./isaaclab.sh -f` before committing.

---

### Task 1: Restore procedural collider visibility after Newton USD imports

**Files:**

- Modify: `source/isaaclab_newton/test/cloner/test_collision_approximation.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/cloner/replicate.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/visualization_builder.py`
- Modify: `source/isaaclab_newton/test/cloner/test_rename_builder_labels.py`

**Interfaces:**

- Consumes: `ModelBuilder.add_usd(...) -> {"path_shape_map": dict[str, int], ...}`.
- Produces: `_restore_visible_colliders_without_visual_shapes(builder: ModelBuilder, stage: Usd.Stage, path_shape_map: dict[str, int] | None) -> None`.

- [ ] **Step 1: Write the failing mixed-visual-model regression**

Add `_make_mixed_visual_stage()` with four cases:

```python
def _make_mixed_visual_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, _SOURCE)

    primitive_body = UsdGeom.Xform.Define(stage, f"{_SOURCE}/Primitive")
    UsdPhysics.RigidBodyAPI.Apply(primitive_body.GetPrim())
    primitive = UsdGeom.Cube.Define(stage, f"{_SOURCE}/Primitive/geometry")
    UsdPhysics.CollisionAPI.Apply(primitive.GetPrim())

    authored_body = UsdGeom.Xform.Define(stage, f"{_SOURCE}/Authored")
    UsdPhysics.RigidBodyAPI.Apply(authored_body.GetPrim())
    UsdGeom.Sphere.Define(stage, f"{_SOURCE}/Authored/visual")
    collider = UsdGeom.Cube.Define(stage, f"{_SOURCE}/Authored/collider")
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    UsdGeom.Xform.Define(stage, f"{_SOURCE}/StaticPrimitive")
    static_primitive = UsdGeom.Cube.Define(stage, f"{_SOURCE}/StaticPrimitive/geometry")
    UsdPhysics.CollisionAPI.Apply(static_primitive.GetPrim())

    UsdGeom.Xform.Define(stage, f"{_SOURCE}/StaticAuthored")
    UsdGeom.Sphere.Define(stage, f"{_SOURCE}/StaticAuthored/visual")
    static_collider = UsdGeom.Cube.Define(stage, f"{_SOURCE}/StaticAuthored/collider")
    UsdPhysics.CollisionAPI.Apply(static_collider.GetPrim())
    return stage
```

Add the assertion:

```python
def test_primitive_collider_remains_visible_in_mixed_visual_model(self):
    builder = _build(_make_mixed_visual_stage())
    flags_by_label = dict(zip(builder.shape_label, builder.shape_flags, strict=True))

    assert flags_by_label[f"{_SOURCE}/Primitive/geometry"] & ShapeFlags.VISIBLE
    assert not flags_by_label[f"{_SOURCE}/Authored/collider"] & ShapeFlags.VISIBLE
    assert flags_by_label[f"{_SOURCE}/StaticPrimitive/geometry"] & ShapeFlags.VISIBLE
    assert not flags_by_label[f"{_SOURCE}/StaticAuthored/collider"] & ShapeFlags.VISIBLE
```

- [ ] **Step 2: Run the regression without the fix**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/cloner/test_collision_approximation.py::TestClonerCollisionApproximation::test_primitive_collider_remains_visible_in_mixed_visual_model -v
```

Expected: FAIL because the procedural dynamic and static primitive flags do not contain `ShapeFlags.VISIBLE`.

- [ ] **Step 3: Implement the minimal shape-selection helper**

Import `UsdGeom` and add:

```python
def _has_visible_non_collision_geometry(stage: Usd.Stage, prim_path: str) -> bool:
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        return False
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible and imageable.ComputePurpose() in (
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
        ):
            return True
    return False


def _restore_visible_colliders_without_visual_shapes(
    builder: ModelBuilder, stage: Usd.Stage, path_shape_map: dict[str, int] | None
) -> None:
    if not path_shape_map:
        return
    bodies_with_visual_shapes = {
        builder.shape_body[index]
        for index, flags in enumerate(builder.shape_flags)
        if builder.shape_body[index] >= 0 and flags & ShapeFlags.VISIBLE and not flags & ShapeFlags.COLLIDE_SHAPES
    }
    static_parent_paths = {
        path.rpartition("/")[0] for path, index in path_shape_map.items() if builder.shape_body[index] < 0
    }
    static_parents_with_visual_shapes = {
        path for path in static_parent_paths if _has_visible_non_collision_geometry(stage, path)
    }
    for path, index in path_shape_map.items():
        flags = builder.shape_flags[index]
        body_index = builder.shape_body[index]
        if (
            not flags & ShapeFlags.COLLIDE_SHAPES
            or builder.shape_type[index] == GeoType.MESH
            or body_index in bodies_with_visual_shapes
            or (body_index < 0 and path.rpartition("/")[0] in static_parents_with_visual_shapes)
        ):
            continue
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        if (
            imageable
            and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy)
        ):
            builder.shape_flags[index] = flags | ShapeFlags.VISIBLE
```

Include concise Google-style docstrings explaining the selection rules.

- [ ] **Step 4: Invoke the helper at every Isaac Lab Newton import boundary**

Capture each `add_usd` result and pass its `path_shape_map` to the helper:

```python
import_result = builder.add_usd(...)
_restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
```

Apply this in:

1. `_build_source_builder`;
2. `_build_newton_builder_from_mapping` for the global builder;
3. `NewtonManager.instantiate_builder_from_stage` for flat, global cloned, and prototype builders;
4. `build_visualization_builder_from_stage_envs` for its global builder.

Update `_FakeVisualizationModelBuilder.add_usd` to return `{"path_shape_map": {}}`.

- [ ] **Step 5: Run focused Newton tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/cloner/test_collision_approximation.py \
  source/isaaclab_newton/test/cloner/test_rename_builder_labels.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the visibility implementation**

```bash
git add \
  source/isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py \
  source/isaaclab_newton/isaaclab_newton/cloner/replicate.py \
  source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py \
  source/isaaclab_newton/isaaclab_newton/physics/visualization_builder.py \
  source/isaaclab_newton/test/cloner/test_collision_approximation.py \
  source/isaaclab_newton/test/cloner/test_rename_builder_labels.py
git commit -m "Restore Newton procedural collider visibility"
```

### Task 2: Refresh validated deformable goldens

**Files:**

- Modify: the eight `newton-newton_renderer-*.png` files under `source/isaaclab_tasks/test/golden_images/franka_cloth/`
- Modify: the eight `newton-newton_renderer-*.png` files under `source/isaaclab_tasks/test/golden_images/franka_soft/`
- Create: `source/isaaclab_newton/changelog.d/antoine-fix-rendering-ci-flakes.rst`
- Create: `source/isaaclab_tasks/changelog.d/antoine-fix-rendering-ci-flakes.skip`

The exact AOV suffixes are `depth`, `distance_to_camera`, `distance_to_image_plane`, `instance_segmentation`, `normals`, `rgb`, `rgba`, and `semantic_segmentation`.

**Interfaces:**

- Consumes: validated Git LFS pointers from commit `60fab180454927e7f45d37ec8a25122427b2adae`.
- Produces: sixteen repository goldens matching the corrected Newton visibility output.

- [ ] **Step 1: Restore only the validated LFS pointers**

Fetch the sixteen objects from PR 6704 if they are not already present, then restore the exact paths from `60fab180454927e7f45d37ec8a25122427b2adae`; do not restore that commit's lockfile or any unrelated file.

- [ ] **Step 2: Verify the expected artifacts changed**

Run:

```bash
git diff --name-only -- source/isaaclab_tasks/test/golden_images/franka_cloth source/isaaclab_tasks/test/golden_images/franka_soft
git lfs pointer --check --file=source/isaaclab_tasks/test/golden_images/franka_cloth/newton-newton_renderer-rgb.png
```

Expected: exactly sixteen changed PNGs; the cloth RGB pointer has SHA-256 `9226378dbb78c6a6dfb7d871365289ea170e186b23bc1f337c6464f761c53458`.

- [ ] **Step 3: Add package changelog fragments**

Add:

```rst
Fixed
^^^^^

* Fixed Newton rendering of procedural colliders when unrelated visual-only
  geometry is present.
```

Create the `isaaclab_tasks` `.skip` fragment as an empty file because only test artifacts changed in that package.

- [ ] **Step 4: Run focused image validation where available**

Run the two kitless test modules if the local runtime supports the pinned Newton/OV packages:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_rendering_franka_cloth_kitless.py \
  source/isaaclab_tasks/test/core/test_rendering_franka_soft_kitless.py \
  -k newton-newton_warp -v
```

If simulator execution is unavailable, verify all sixteen LFS pointer object IDs match commit `60fab180454927e7f45d37ec8a25122427b2adae` and record the runtime limitation in the PR.

- [ ] **Step 5: Commit the goldens and fragments**

```bash
git add \
  source/isaaclab_newton/changelog.d/antoine-fix-rendering-ci-flakes.rst \
  source/isaaclab_tasks/changelog.d/antoine-fix-rendering-ci-flakes.skip \
  source/isaaclab_tasks/test/golden_images/franka_cloth \
  source/isaaclab_tasks/test/golden_images/franka_soft
git commit -m "Refresh Newton deformable render goldens"
```

### Task 3: Verify the Newton deliverable

**Files:**

- Verify only; no planned modifications.

**Interfaces:**

- Consumes: Tasks 1-2.
- Produces: evidence for the PR description.

- [ ] **Step 1: Run the focused unit suite again**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/cloner/test_collision_approximation.py \
  source/isaaclab_newton/test/cloner/test_rename_builder_labels.py -v
```

- [ ] **Step 2: Audit all Newton USD imports**

```bash
rg -n "add_usd\\(" \
  source/isaaclab_newton/isaaclab_newton/cloner \
  source/isaaclab_newton/isaaclab_newton/physics
```

Confirm every Isaac Lab-owned import returning a `path_shape_map` invokes `_restore_visible_colliders_without_visual_shapes` before builder replication/finalization.
