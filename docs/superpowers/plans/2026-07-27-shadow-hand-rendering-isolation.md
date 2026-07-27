<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Shadow Hand Rendering Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the Shadow Hand rendering suite from hanging while changing backends in one Kit process, and make any future stall diagnostically useful.

**Architecture:** Split the aggregate standard Shadow Hand module into three fresh-process modules whose parameter lists are disjoint and exhaustive. A private standard-library watchdog inside each child process emits Python thread stacks before the hard timeout, while the outer test runner grants one timeout-only retry exclusively to the three new files.

**Tech Stack:** pytest parametrization, Isaac Lab `AppLauncher`, Python `faulthandler`, the custom `tools/conftest.py` subprocess runner, GitHub Actions YAML.

## Global Constraints

- Do not change rendering-job `continue-on-error`.
- Preserve every existing Shadow Hand backend/renderer/AOV case.
- Do not retry assertion failures or image mismatches.
- Do not restore global hard-timeout retries; only the three isolated Shadow Hand files receive one retry.
- Add no required or optional dependency.
- Verify each regression test fails before its implementation and passes afterward.
- Use the shared `source/isaaclab_tasks/changelog.d/antoine-fix-rendering-ci-flakes.skip` fragment.
- Run commands through `./isaaclab.sh -p` and run `./isaaclab.sh -f` before committing.

---

### Task 1: Expose disjoint rendering slices and a timeout watchdog

**Files:**

- Modify: `source/isaaclab_tasks/test/rendering_test_utils.py`
- Create: `.github/actions/run-tests/test_shadow_hand_isolation.py`

**Interfaces:**

- Produces: `PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS`, `NEWTON_ISAACSIM_RTX_AOV_COMBINATIONS`, and `PHYSX_NEWTON_WARP_AOV_COMBINATIONS`.
- Preserves: `PHYSICS_RENDERER_AOV_COMBINATIONS` as the concatenation of the three lists.
- Produces: `arm_timeout_traceback_watchdog(delay_seconds: int = 1200) -> None`.

- [ ] **Step 1: Write failing utility tests**

Load `source/isaaclab_tasks/test/rendering_test_utils.py` and assert:

```python
def test_standard_rendering_slices_are_disjoint_and_exhaustive():
    slices = (
        module.PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS,
        module.NEWTON_ISAACSIM_RTX_AOV_COMBINATIONS,
        module.PHYSX_NEWTON_WARP_AOV_COMBINATIONS,
    )
    ids = [{param.id for param in params} for params in slices]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert set.union(*ids) == {param.id for param in module.PHYSICS_RENDERER_AOV_COMBINATIONS}
```

Patch `faulthandler.dump_traceback_later` and assert:

```python
def test_timeout_watchdog_schedules_one_all_thread_dump(monkeypatch):
    calls = []
    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", lambda delay, repeat: calls.append((delay, repeat)))
    module.arm_timeout_traceback_watchdog()
    assert calls == [(1200, False)]
```

- [ ] **Step 2: Run the utility tests without the implementation**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/run-tests/test_shadow_hand_isolation.py \
  -k "standard_rendering_slices or timeout_watchdog" -v
```

Expected: FAIL because the three constants and watchdog helper do not exist.

- [ ] **Step 3: Split the aggregate matrix without changing its contents**

Replace the inline aggregate construction with:

```python
PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS = _make_sensor_data_type_params("physx", "isaacsim_rtx")
NEWTON_ISAACSIM_RTX_AOV_COMBINATIONS = _make_sensor_data_type_params("newton", "isaacsim_rtx")
PHYSX_NEWTON_WARP_AOV_COMBINATIONS = _make_sensor_data_type_params(
    "physx", "newton", _NEWTON_WARP_DATA_TYPES, flaky=False, renderer_label="newton_warp"
)

PHYSICS_RENDERER_AOV_COMBINATIONS = [
    *PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS,
    *NEWTON_ISAACSIM_RTX_AOV_COMBINATIONS,
    *PHYSX_NEWTON_WARP_AOV_COMBINATIONS,
]
```

Import `faulthandler` and add:

```python
def arm_timeout_traceback_watchdog(delay_seconds: int = 1200) -> None:
    """Schedule a Python thread dump before the rendering test hard timeout."""
    faulthandler.dump_traceback_later(delay_seconds, repeat=False)
```

- [ ] **Step 4: Run the utility tests**

Run the Step 2 command.

Expected: PASS.

### Task 2: Split Shadow Hand into three fresh simulator processes

**Files:**

- Delete: `source/isaaclab_tasks/test/core/test_rendering_shadow_hand.py`
- Create: `source/isaaclab_tasks/test/core/test_rendering_shadow_hand_physx_isaacsim_rtx.py`
- Create: `source/isaaclab_tasks/test/core/test_rendering_shadow_hand_newton_isaacsim_rtx.py`
- Create: `source/isaaclab_tasks/test/core/test_rendering_shadow_hand_physx_newton_warp.py`
- Modify: `.github/workflows/build.yaml`
- Modify: `tools/test_settings.py`
- Modify: `.github/actions/run-tests/test_shadow_hand_isolation.py`

**Interfaces:**

- Consumes: the three parameter constants and `arm_timeout_traceback_watchdog()` from Task 1.
- Produces: three test files, each with its own `AppLauncher` and simulator process.

- [ ] **Step 1: Write the failing layout regression**

Add a test that reads the workflow and source directory:

```python
def test_shadow_hand_workflow_runs_three_isolated_modules():
    expected = {
        "test_rendering_shadow_hand_physx_isaacsim_rtx.py",
        "test_rendering_shadow_hand_newton_isaacsim_rtx.py",
        "test_rendering_shadow_hand_physx_newton_warp.py",
    }
    workflow = (_REPO_ROOT / ".github/workflows/build.yaml").read_text(encoding="utf-8")
    assert "test_rendering_shadow_hand.py" not in workflow
    assert all(name in workflow for name in expected)
    assert not (_REPO_ROOT / "source/isaaclab_tasks/test/core/test_rendering_shadow_hand.py").exists()
    assert all((_REPO_ROOT / "source/isaaclab_tasks/test/core" / name).is_file() for name in expected)
```

- [ ] **Step 2: Run the layout regression without the split**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/run-tests/test_shadow_hand_isolation.py::test_shadow_hand_workflow_runs_three_isolated_modules -v
```

Expected: FAIL because the workflow still selects the aggregate module.

- [ ] **Step 3: Create each isolated module**

Each file retains the original fixture setup and test body, but imports exactly one slice constant. Immediately after importing the shared helpers, arm the watchdog:

```python
arm_timeout_traceback_watchdog()
```

For example, the PhysX/RTX module parametrizes:

```python
@pytest.mark.parametrize(
    "physics_backend,renderer,data_type",
    PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS,
)
def test_rendering_shadow_hand_physx_isaacsim_rtx(physics_backend, renderer, data_type):
    rendering_test_shadow_hand(physics_backend, renderer, data_type, _COMPARISON_SCORES)
```

Use the corresponding constant and function name in the Newton/RTX and PhysX/Newton-Warp modules.

- [ ] **Step 4: Update workflow selection and hard-timeout settings**

Replace `test_rendering_shadow_hand.py` in `.github/workflows/build.yaml` with the three filenames. In `tools/test_settings.py`, replace the old 1500-second entry with three entries, each set to `1500`.

- [ ] **Step 5: Run the layout and utility tests**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/run-tests/test_shadow_hand_isolation.py \
  -k "standard_rendering_slices or timeout_watchdog or workflow_runs" -v
```

Expected: PASS.

### Task 3: Retry only reportless hard timeouts from isolated Shadow files

**Files:**

- Modify: `tools/conftest.py`
- Modify: `.github/actions/run-tests/test_shadow_hand_isolation.py`

**Interfaces:**

- Produces: `TIMEOUT_RETRIES_BY_FILE: dict[str, int]`.
- Produces: `_timeout_retries_for_file(file_name: str) -> int`.
- Preserves: `TIMEOUT_RETRIES = 0` as the default for every unlisted file.

- [ ] **Step 1: Write the failing retry-scope regression**

```python
@pytest.mark.parametrize(
    "file_name",
    [
        "test_rendering_shadow_hand_physx_isaacsim_rtx.py",
        "test_rendering_shadow_hand_newton_isaacsim_rtx.py",
        "test_rendering_shadow_hand_physx_newton_warp.py",
    ],
)
def test_shadow_hand_files_receive_one_timeout_retry(file_name):
    assert runner._timeout_retries_for_file(file_name) == 1


def test_unrelated_file_receives_no_timeout_retry():
    assert runner._timeout_retries_for_file("test_rendering_franka_cloth.py") == 0
```

- [ ] **Step 2: Run the retry test without the implementation**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/run-tests/test_shadow_hand_isolation.py \
  -k "timeout_retry" -v
```

Expected: FAIL because `_timeout_retries_for_file` does not exist.

- [ ] **Step 3: Implement per-file timeout retry lookup**

Add:

```python
TIMEOUT_RETRIES_BY_FILE = {
    "test_rendering_shadow_hand_physx_isaacsim_rtx.py": 1,
    "test_rendering_shadow_hand_newton_isaacsim_rtx.py": 1,
    "test_rendering_shadow_hand_physx_newton_warp.py": 1,
}


def _timeout_retries_for_file(file_name: str) -> int:
    return TIMEOUT_RETRIES_BY_FILE.get(file_name, TIMEOUT_RETRIES)
```

In `_run_one_pass`, compute `timeout_retries = _timeout_retries_for_file(ctx.file_name)` once and use it in both the retry condition and attempt-count log. Keep the `kill_reason == "timeout" and not has_report` gate unchanged so completed assertion failures are never retried.

- [ ] **Step 4: Run the complete CI helper regression**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/run-tests/test_shadow_hand_isolation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Shadow Hand isolation**

```bash
git add \
  .github/actions/run-tests/test_shadow_hand_isolation.py \
  .github/workflows/build.yaml \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand_physx_isaacsim_rtx.py \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand_newton_isaacsim_rtx.py \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand_physx_newton_warp.py \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand.py \
  source/isaaclab_tasks/test/rendering_test_utils.py \
  tools/conftest.py \
  tools/test_settings.py
git commit -m "Isolate Shadow Hand rendering backends"
```

### Task 4: Verify the Shadow Hand deliverable

**Files:**

- Verify only; no planned modifications.

**Interfaces:**

- Consumes: Tasks 1-3.
- Produces: collection and unit-test evidence for the PR.

- [ ] **Step 1: Verify the aggregate matrix is still exhaustive**

Run the complete helper regression from Task 3, Step 4.

- [ ] **Step 2: Collect each isolated module when the simulator runtime is available**

```bash
./isaaclab.sh -p -m pytest --collect-only -q \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand_physx_isaacsim_rtx.py \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand_newton_isaacsim_rtx.py \
  source/isaaclab_tasks/test/core/test_rendering_shadow_hand_physx_newton_warp.py
```

Expected: the total collected cases equal the original aggregate matrix, with no duplicate node IDs.
