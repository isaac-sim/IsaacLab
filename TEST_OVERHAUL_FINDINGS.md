# Test Overhaul — Findings

Working notes from the repo-wide test overhaul. Each entry records what was found, why it
matters, and what was done. Kept untracked: `AGENTS.md` forbids committing agent working
documents.

Legend: **FIXED** — landed on this branch. **OPEN** — real problem, not addressed here.

---

## 1. Defects that silently removed coverage

These are the highest-severity findings: in each case the suite reported success while running
less than it appeared to.

### 1.1 The isaaclab fast lane ran zero tests — **FIXED**

`pytest source/isaaclab/test` aborted on four collection errors. Pytest stops the whole run on a
collection error, so the largest package in the repo had no working local profile at all.
Anyone running it saw an error, not 11k passing tests.

Two entangled causes:

- `source/isaaclab/test/assets/_iface_test_boot.py` stubbed `isaacsim.core` and
  `isaacsim.core.simulation_manager` into `sys.modules` at import time and never restored them.
  Four unrelated modules then imported a half-initialised `isaaclab_physx` and died on `carb`.
- Removing those stubs exposed a second defect. `test_articulation_ordering.py` builds
  partially-initialised articulations with `object.__new__`, which refuses an abstract class. It
  only ever passed because the leaked stub happened to make `AssetBase` concrete. It now clears
  `__abstractmethods__` on its two stand-in backends explicitly.

Result: 0 → **11,132 tests in 102 s**.

### 1.2 Phase 0 silently disabled Kit tests in CI — **FIXED**

Marking modules `requires_kit` excludes them unless `ISAACLAB_RUN_CI_TESTS=1`. Thirteen CI jobs
never set it. So `isaaclab_rl`, `isaaclab_mimic`, `isaaclab_contrib`, `isaaclab_teleop`,
`isaaclab_visualizers` and `isaaclab_assets` would have stopped running their Kit tests
altogether while still reporting green.

This was self-inflicted, introduced by the marking pass and caught by auditing every job's env
block. It is the strongest argument for the boundary guard: a marker that changes what runs must
be paired with a check that the runner opts in.

### 1.3 `pytestmark` assigned twice, discarding the first — **FIXED**

`source/isaaclab/test/controllers/test_operational_space.py` and
`source/isaaclab/test/devices/test_device_constructors.py` each assigned `pytestmark` twice at
module scope. Python keeps the last binding, so the first was dead. `test_operational_space`'s
`arm_ci` marker had **never** been applied since it was added.

Now enforced by `test_pytestmark_is_assigned_once`.

### 1.4 `isaacsim_ci` markers on files the CI marker scan cannot see — **PARTLY OPEN**

`tools/conftest.py` selects marked files with a plain substring scan for
`pytest.mark.<marker>`. That works, but the root conftest's gating is AST-based. The two
mechanisms agree today; they are independent implementations and can drift.

### 1.5 `filter-pattern` is a plain substring — **FIXED (documented)**

`filter-pattern: "isaaclab_ov"` also matches `source/isaaclab_ovphysx/`. The `isaaclab_ov` job
has therefore always been running the `isaaclab_ovphysx` suite too. This is benign — that job
installs the OV pins — but it is invisible from the job name. Renamed to
`isaaclab_ov + isaaclab_ovphysx` with a comment, rather than adding a second job that would run
the same files twice.

---

## 2. Packaging and dependency defects

### 2.1 `opencv-python` was never declared — **FIXED**

`scripts/tools/hdf5_to_mp4.py` and `scripts/tools/mp4_to_hdf5.py` import `cv2`, but no extra in
`pyproject.toml` provided it. The shipped scripts could not run, and their 19 tests had been
uncollectable indefinitely. Added an `opencv` extra; all 19 now pass.

### 2.2 Three packages had no CI job — **FIXED**

`isaaclab_experimental` (9 files) and `isaaclab_ppisp` (4 files) had no job at all. Added.
`isaaclab_ovphysx` appeared to have none but was in fact covered by the `isaaclab_ov` job via the
substring match above.

### 2.3 Extras cannot all co-resolve — **BY DESIGN, now explicit**

`pyproject.toml`'s `conflicts` table means `isaacsim` cannot co-install with `teleop`, `ov`,
`ovphysx`, `viser`, `mimic`, or even `test`. No single environment can run the whole suite. This
is why suites now declare `requires_extra(...)` and a missing extra is a loud failure naming the
install command rather than a silent skip.

---

## 3. Order dependence and shared global state

The recurring defect class. Every instance passes in isolation and fails in company, so it hides
until something changes collection order.

### 3.1 `test_cloudxr_lifecycle` and `test_control_events` share MagicMock state — **OPEN**

Both stub the same `isaacteleop.*` names into `sys.modules` using `if name not in sys.modules`.
Whichever module runs second silently inherits the first's `MagicMock` instances, including
`side_effect` iterators that have already been exhausted, producing `StopIteration`.

Run together with the `teleop` extra installed: **40 failed, 73 passed**. Run one file per
process: all pass. CI runs one file per process, so CI is green — but the tests violate the
no-order-dependence rule and the failure is invisible until someone runs the directory.

The fix is to install stubs unconditionally and restore the previous entry on teardown, instead
of "install only if absent". Not attempted here: it touches the mocking strategy of six files and
deserves an owner who knows the teleop stack.

### 3.2 Modules that stub `sys.modules` at import and never restore — **OPEN (allowlisted)**

- `source/isaaclab/test/assets/_iface_test_boot.py` — still leaks `omni.*` stubs (the `isaacsim`
  leak is fixed). Needed by the kitless OvPhysX path.
- `source/isaaclab_contrib/test/rl/test_rlinf_extension.py`
- `source/isaaclab_physx/test/renderers/test_isaac_rtx_renderer_utils.py`
- `source/isaaclab_teleop/test/test_retargeters.py` — replaces `isaaclab.sim` with a `MagicMock`
  for the rest of the session.

`test_articulation_ordering.py` also mutates `sys.modules`, but correctly pops exactly what it
inserted; it is allowlisted only because the static rule cannot see the restore.

### 3.3 Duplicate module names collided across packages — **FIXED**

Test directories are not packages, so under pytest's `prepend` import mode two files with the
same name in different packages collide and one silently shadows the other. Seven collisions
existed (`test_articulation`, `test_rigid_object`, `test_mock_articulation_view`, …).

Renamed with package suffixes. Removing empty `__init__.py` markers from `sim/` and
`test_mock_interfaces/` then surfaced three more (`test_cloner`, `test_spawn_materials`,
`test_spawn_meshes`), also renamed. Now enforced by `test_module_names_are_globally_unique`.

`tools/test_settings.py` keys per-file timeouts on the basename, so the renames required updating
`PER_TEST_TIMEOUTS` — otherwise `test_articulation.py` would have silently dropped from a 3000 s
budget to the 1000 s default.

---

## 4. Tests that cost far more than they measure

### 4.1 Simulation contexts built for tests that never use them — **FIXED**

`source/isaaclab/test/actuators/test_implicit_actuator.py` built a **simulation context for each
of its 160 cases**, and no test body referenced the fixture. The tests only construct actuator
configs and assert how effort and velocity limits resolve.

Same shape in three more files. All now Kit-less, coverage unchanged:

| file | before | after |
|---|---|---|
| `isaaclab/.../test_implicit_actuator.py` | 160 cases in the Kit lane | 160 pass in 1.4 s |
| `isaaclab/.../test_ray_caster_patterns.py` | Kit lane | 90 pass in 1.8 s |
| `isaaclab_contrib/.../test_thruster.py` | Kit lane | 102 pass in 1.3 s |
| `isaaclab_contrib/.../test_drone_geometric_controllers.py` | Kit lane | 96 pass in 1.8 s |
| `isaaclab_contrib/.../test_visuotactile_render.py` | Kit lane | 4 pass, needs `opencv` |

`isaaclab_contrib`'s fast lane went from **88 to 286 passing tests** as a result.

Checked the rest of the Kit lane for the same shape; the remaining candidates genuinely need Kit
(`omni.kit.app` import, Isaac Sim version read, launcher behaviour itself, or a
`SimulationContext` that resolves a PhysX manager).

### 4.2 Device × instance-count cross products — **FIXED for newton/physx**

Behaviour tests multiplied device and instance-count axes that were not the subject under test.
Measured on `test_articulation_physx.py`: **241 cases in 231 s → 95 cases in 122 s**, all
passing. Initialization tests keep CPU and CUDA through paired cases so shape coverage survives.

### 4.3 `isaaclab_tasks` produced 285 errors and took 257 s — **FIXED**

All 285 came from seven `test_rendering_*_kitless.py` suites that need the `ov` extra and failed
late inside a fixture. Declaring `requires_extra("ov")` gates them once, loudly.
**257 s → 9.5 s, 285 errors → 0.**

---

## 5. Traps that cost real debugging time

Recorded in `AGENTS.md` so they are not rediscovered.

### 5.1 The Kit lane must run one file per process

Starting Kit in several modules within one pytest process **segfaults with no output at all**
(exit 139, empty stdout and stderr). It reads like a hang, not a failure. A whole-package Kit
invocation is not a valid check; use one file per process or the fresh-process orchestrator.
`OMNI_KIT_ACCEPT_EULA=YES` runs it non-interactively.

### 5.2 Never run two `uv run` commands with different extras concurrently

They share one `.venv`. A concurrent invocation re-syncs packages underneath the first and
silently corrupts its results. This cost one wasted verification run.

Worse: `uv sync` does not remove the `isaacsim` extension-cache directory left in
`site-packages`, so a later `--extra test` run still finds `isaacsim` importable and tries to
bootstrap Kit. Recovering required `rm -rf .venv/lib/python3.12/site-packages/isaacsim` followed
by `uv sync --extra test --locked --reinstall`, because the stale tree had also replaced `pxr`.

### 5.3 `pytest_ignore_collect` does not apply to explicitly named files

The hook runs during directory recursion. Passing a file path directly bypasses all gating, so a
`requires_kit` module will run if you name it. Fine as behaviour, but it means per-file
verification loops do not exercise the gating.

### 5.4 Trimming a device axis can delete coverage

`isaaclab_ovphysx` marks modules `device_split` and pins the session to the first device it sees.
There the device parameter is how CI shards a file across processes, not a redundant axis.
Narrowing it dropped `test_rigid_object.py` from **52 passing tests to 5**. Reverted. Always
compare passed counts either side of a parametrize change.

### 5.5 `install_ci` is a separate rootdir

It ships its own `pytest.ini`, so it becomes the rootdir whenever addressed directly and the
repo-root `conftest.py` is not loaded. Gating it from repo-wide sweeps therefore has to be
unconditional; keying it on `--run-ci-tests` pulled it into the new kitless lane, where it fails
for want of a built wheel.

---

## 6. Product-level failures surfaced, not fixed — **OPEN**

Unblocking the lanes revealed real failures that were previously hidden behind collection errors.
None are test-infrastructure problems.

- `isaaclab_experimental/test/envs/test_frontend_cfg_conversion.py` — 2 failures. The warp
  frontend has no MDP twins for `is_terminated_term` and `pose_command_success`, so
  `Isaac-Reach-Franka` and `Isaac-Reach-UR10` cannot adapt to warp. Pre-existing.
- `isaaclab_rl` — 9 failures before the extras were declared, all from missing RL frameworks.
  Now gated; they need `--extra sb3 --extra skrl --extra rl-games --extra rsl-rl` to run.
- `isaaclab_teleop` — 40 failures when run as a directory, see 3.1.

---

## 7. Structural observations

### 7.1 `isaaclab_physx` forces almost everything into the Kit lane

`source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py:27` imports `carb` at module
scope. Anything that touches `isaaclab_physx` therefore needs Kit, including tests of pure
config logic. Making that import lazy would unlock a large amount of Kit-less coverage. Left
alone: it is a production change.

### 7.2 The three physics backends are not duplicates

Measured body similarity across `isaaclab_newton`, `isaaclab_physx` and `isaaclab_ovphysx`:

| suite | shared names | identical bodies |
|---|---|---|
| `rigid_object` | 86% | 0 of 19 |
| `rigid_object_collection` | 76% | 1 of 13 |
| `articulation` | 35% | 3 of 34, with 17 under 85% similar |

The duplication is structural, not literal. A shared backend-parametrized suite would mean
reconciling hundreds of genuine per-backend differences. Recommendation: dedupe within each
package, keep the backends separate. Accepted.

### 7.3 The docstring rule does not scale backwards

`isaaclab` has 634 tests with no docstring, but their names already state the contract
(`test_resolve_env_ids_handles_tensor_view_shape`). Adding a docstring to each would produce 634
restatements of the name, which the rule itself forbids as filler.

The allowlist is therefore a **grandfather list, not a debt to pay down**: it blocks new
undocumented tests while leaving well-named existing ones alone. Clear an entry only when a
test's contract is genuinely unclear from its name.

---

## 8. Where the rules live

- `AGENTS.md` → `## Testing Guidelines` — the full ruleset, always in agent context.
- `skills/developer/writing-tests/` — the working procedure plus before/after examples.
- `docs/source/refs/contributing.rst` — contributor-facing subset.
- `source/isaaclab/test/test_repo_test_boundary.py` — mechanical enforcement, 1,960 checks in
  ~3 s, wired into `uv run isaaclab -f`.
- `source/isaaclab/test/test_repo_test_boundary_allowlist.txt` — grandfathered files, with a
  reason per entry.
