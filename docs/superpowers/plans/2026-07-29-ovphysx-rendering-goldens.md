# OvPhysX Franka Rendering Goldens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reviewed golden images for every supported Franka deformable OvPhysX and OVRTX rendering case, with both legacy and OvStage paths sharing the same baselines.

**Architecture:** The existing rendering helpers remain the source of truth for supported AOVs and image comparison. Remove the temporary blanket skip, let the legacy OVRTX path bootstrap only missing OvPhysX images, visually inspect those images, then run both legacy and OvStage paths against the committed files without changing comparison tolerances.

**Tech Stack:** pytest, OvPhysX, OVRTX, Pillow-backed Isaac Lab rendering utilities, PNG golden images, RST documentation.

## Global Constraints

- Generate goldens only for supported OvPhysX + OVRTX combinations.
- Keep existing Newton-renderer and documented crashing-AOV skips unchanged.
- Use one shared baseline set for legacy and OvStage OVRTX.
- Generate exactly 11 `franka_soft` images and 10 `franka_cloth` images.
- Do not widen pixel-difference or SSIM thresholds.
- Run `./isaaclab.sh -f` before committing and again before pushing.

---

### Task 1: Activate supported OvPhysX rendering cases

**Files:**
- Modify: `source/isaaclab_tasks/test/rendering_test_utils.py:1705-1715`
- Modify: `source/isaaclab_tasks/test/rendering_test_utils.py:1837-1847`
- Modify: `source/isaaclab_tasks/test/core/test_franka_deformable_ovphysx_cfg.py:5-85`

**Interfaces:**
- Consumes: `rendering_test_franka_soft()` and `rendering_test_franka_cloth()`.
- Produces: Supported OvPhysX cases reach environment construction and `validate_camera_outputs()`; existing per-renderer/AOV skips remain active.

- [ ] **Step 1: Remove the temporary blanket skips**

Delete these blocks from both rendering functions:

```python
if physics_backend == "ovphysx":
    pytest.skip("Franka deformable rendering tests require reviewed OvPhysX golden images.")
```

- [ ] **Step 2: Remove the skip-only regression**

Delete `test_ovphysx_deformable_rendering_waits_for_reviewed_goldens`, its parametrization, and the now-unused top-level `pytest` import from `test_franka_deformable_ovphysx_cfg.py`.

- [ ] **Step 3: Run one supported case to verify the missing-baseline failure**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_rendering_franka_soft_kitless.py \
  -k "legacy and ovphysx and rgb" -q
```

Expected: the rendering case reaches `validate_camera_outputs()` and reports that
`golden_images/franka_soft/ovphysx-ovrtx_renderer-rgb.png` was missing. The helper
may retry and pass after bootstrapping the file, so also verify the new PNG exists.

- [ ] **Step 4: Remove the single probe image before the full matrix generation**

Delete only:

```text
source/isaaclab_tasks/test/golden_images/franka_soft/ovphysx-ovrtx_renderer-rgb.png
```

This restores a uniform missing-baseline state for Task 2.

---

### Task 2: Generate and review the complete legacy baseline set

**Files:**
- Create: `source/isaaclab_tasks/test/golden_images/franka_soft/ovphysx-ovrtx_renderer-*.png`
- Create: `source/isaaclab_tasks/test/golden_images/franka_cloth/ovphysx-ovrtx_renderer-*.png`

**Interfaces:**
- Consumes: the activated rendering functions from Task 1.
- Produces: 21 PNG baselines consumed by both legacy and OvStage rendering tests.

- [ ] **Step 1: Bootstrap all legacy OvPhysX + OVRTX images**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_rendering_franka_soft_kitless.py \
  source/isaaclab_tasks/test/core/test_rendering_franka_cloth_kitless.py \
  -k "legacy and ovphysx and ovrtx" -q
```

Expected: supported cases create missing PNGs. Existing skips remain for cloth
`instance_segmentation`, cloth `motion_vectors`, and unsupported renderer cases.

- [ ] **Step 2: Verify the exact generated matrix**

Run:

```bash
find source/isaaclab_tasks/test/golden_images/franka_soft \
  -maxdepth 1 -type f -name 'ovphysx-ovrtx_renderer-*.png' | sort
find source/isaaclab_tasks/test/golden_images/franka_cloth \
  -maxdepth 1 -type f -name 'ovphysx-ovrtx_renderer-*.png' | sort
```

Expected: 11 soft paths and 10 cloth paths, with no
`ovphysx-newton_renderer-*.png` files.

- [ ] **Step 3: Build task-specific review montages**

Use Pillow through the repository Python wrapper to place each generated image,
labeled with its AOV filename, into two temporary montage PNGs under `/tmp`:

```bash
./isaaclab.sh -p -c "
from pathlib import Path
from PIL import Image, ImageDraw

for task in ('franka_soft', 'franka_cloth'):
    paths = sorted(Path('source/isaaclab_tasks/test/golden_images', task).glob('ovphysx-ovrtx_renderer-*.png'))
    opened = [Image.open(path).convert('RGB') for path in paths]
    width = max(image.width for image in opened)
    height = max(image.height for image in opened)
    cell_height = height + 32
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    montage = Image.new('RGB', (columns * width, rows * cell_height), 'white')
    draw = ImageDraw.Draw(montage)
    for index, (path, image) in enumerate(zip(paths, opened)):
        x = (index % columns) * width
        y = (index // columns) * cell_height
        montage.paste(image, (x, y))
        draw.text((x + 4, y + height + 4), path.stem.removeprefix('ovphysx-ovrtx_renderer-'), fill='black')
    montage.save(Path('/tmp') / f'{task}-ovphysx-goldens.png')
"
```

- [ ] **Step 4: Inspect both montages**

Open `/tmp/franka_soft-ovphysx-goldens.png` and
`/tmp/franka_cloth-ovphysx-goldens.png` with the local image viewer. Confirm:

- no supported output is blank or uniformly colored;
- the robot, deformable, table, and camera framing are present where applicable;
- depth, normals, segmentation, and motion outputs are structurally plausible;
- no image is clipped, corrupted, or an accidental duplicate of a different AOV.

- [ ] **Step 5: Rerun legacy comparisons**

Run the same legacy command from Step 1.

Expected: all supported cases pass by comparing against existing PNGs. No new
golden files appear and `git diff --exit-code` reports no PNG modifications.

---

### Task 3: Validate the shared goldens through OvStage

**Files:**
- Test: `source/isaaclab_tasks/test/core/test_rendering_franka_soft_kitless.py`
- Test: `source/isaaclab_tasks/test/core/test_rendering_franka_cloth_kitless.py`

**Interfaces:**
- Consumes: the 21 legacy-generated PNGs from Task 2.
- Produces: evidence that OvStage renders within the existing comparison thresholds against the same baselines.

- [ ] **Step 1: Run the OvStage matrix**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_rendering_franka_soft_kitless.py \
  source/isaaclab_tasks/test/core/test_rendering_franka_cloth_kitless.py \
  -k "ovstage and ovphysx and ovrtx" -q
```

Expected: all supported cases pass against the Task 2 PNGs; documented
unsupported combinations skip.

- [ ] **Step 2: Confirm validation did not rewrite baselines**

Run:

```bash
git diff --exit-code -- \
  source/isaaclab_tasks/test/golden_images/franka_soft \
  source/isaaclab_tasks/test/golden_images/franka_cloth
```

Expected: exit code 0 before staging, because comparisons never overwrite
existing goldens.

---

### Task 4: Apply Kelly's environment-catalog suggestions

**Files:**
- Modify: `docs/source/overview/environments.rst:1004`
- Modify: `docs/source/overview/environments.rst:1015`

**Interfaces:**
- Consumes: Kelly's inline suggestions on PR #6674.
- Produces: User-facing rows that retain the explicit `isaacsim_physx` name for the forced Kit backend.

- [ ] **Step 1: Update the KukaAllegro row**

Set the physics list to:

```rst
      - | **physics=** ``newton_mjwarp``, ``ovphysx``, ``isaacsim_physx``
```

- [ ] **Step 2: Update the soft-lift row**

Set the physics list to:

```rst
      - **physics=** ``newton_mjwarp_vbd``, ``newton_mjwarp_vbd_proxy``, ``ovphysx``, ``isaacsim_physx``
```

- [ ] **Step 3: Verify both exact rows**

Run:

```bash
sed -n '1001,1017p' docs/source/overview/environments.rst
```

Expected: neither edited row advertises the automatic `physx` selector.

---

### Task 5: Final regression, review, and PR update

**Files:**
- Modify: `source/isaaclab_tasks/changelog.d/antoiner-ovphysx-deformable-tasks.minor.rst`
- Test: all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: activated rendering coverage, 21 inspected PNGs, and corrected catalog rows.
- Produces: a verified additive commit on PR #6674.

- [ ] **Step 1: Update the tasks changelog fragment**

Under `Added`, add:

```rst
* Added rendering-correctness coverage for the OvPhysX Franka deformable
  environments.
```

- [ ] **Step 2: Run non-rendering focused regressions**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_franka_deformable_ovphysx_cfg.py \
  source/isaaclab_ovphysx/test/tasks/test_lift_franka_soft_deformable.py \
  source/isaaclab_ovphysx/test/assets/test_articulation.py::test_articulation_dynamics_fixed_base_match_raw_ovphysx_bindings \
  source/isaaclab_ovphysx/test/assets/test_articulation.py::test_articulation_dynamics_reorder_body_rows_and_joint_axes \
  source/isaaclab_ovphysx/test/assets/test_articulation.py::test_articulation_dynamics_preserve_floating_base_columns_during_joint_reordering \
  -q
```

Expected: deformable config and both live task tests pass; supported CUDA
articulation cases pass and unsupported CPU OvPhysX cases skip.

- [ ] **Step 3: Run full pre-commit**

Run:

```bash
./isaaclab.sh -f
```

Expected: every hook passes. If a hook modifies files, review and stage those
changes, then rerun the command until it passes without modifications.

- [ ] **Step 4: Review the final diff**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: only the intended helper/test/docs/changelog files and 21 PNGs are
present.

- [ ] **Step 5: Commit the implementation**

Run:

```bash
git add \
  docs/source/overview/environments.rst \
  source/isaaclab_tasks/changelog.d/antoiner-ovphysx-deformable-tasks.minor.rst \
  source/isaaclab_tasks/test/core/test_franka_deformable_ovphysx_cfg.py \
  source/isaaclab_tasks/test/rendering_test_utils.py \
  source/isaaclab_tasks/test/golden_images/franka_soft/ovphysx-ovrtx_renderer-*.png \
  source/isaaclab_tasks/test/golden_images/franka_cloth/ovphysx-ovrtx_renderer-*.png
git commit -m "Add OvPhysX deformable rendering goldens"
```

- [ ] **Step 6: Run pre-commit again and push**

Run:

```bash
./isaaclab.sh -f
git push antoine HEAD:antoiner/ovphysx-deformable-demos
```

Expected: pre-commit passes and PR #6674 points at the new commit.

- [ ] **Step 7: Reply to Kelly's inline threads**

Reply in each GitHub review thread with the exact catalog correction made, then
verify the PR remains mergeable and CI starts on the new head.
