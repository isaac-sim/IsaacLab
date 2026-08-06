# Automatic Implicit Actuator Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically capture and replay the complete built-in implicit-actuator execution sequence for eligible CUDA collections while preserving exact output, caller-owned capture, and the existing eager fallback.

**Architecture:** Resolve eligibility once when an `ActuatorCollection` is constructed. Eligible collections lazily capture their ordered fused implicit Warp launches into one collection-owned graph; collections under a Warp-visible outer capture emit their underlying launches instead, and any local capture failure permanently returns that collection to cached-eager execution. Backend submission remains outside this path.

**Tech Stack:** Python 3.11+, PyTorch, NVIDIA Warp CUDA graphs, pytest, Isaac Lab's wrapped Python and pre-commit tooling.

## Global Constraints

- Follow `/home/antoiner/Documents/IsaacLab/AGENTS.md`.
- Work only in `/home/antoiner/Documents/IsaacLab/docs/superpowers/worktrees/actuator-collection-pre-global` on `antoiner/actuators-collection-split-6248`.
- Do not change public API, configuration, backend submission, or CPU behavior.
- Capture only nonempty CUDA collections whose batches are all exact built-in `ImplicitActuator` instances and whose backend-native execution is inactive.
- Keep mixed, custom-subclass, explicit, stateful, neural, empty, CPU, and backend-native collections on their existing execution paths.
- Treat only `wp.get_device(self.device).is_capturing` on the collection's dispatch stream as caller-owned capture. Never replay or create a local graph in that state.
- Use one graph for the complete ordered implicit sequence, not one graph per batch.
- Attempt local capture at most once. Warn once on failure, execute the current call eagerly, and remain cached-eager for the collection lifetime.
- Preserve the existing pointer-stability contract; do not add rebinding, allocation, stream, event, or public invalidation machinery.
- Prove focused regression tests fail before production code is changed and pass afterward.
- Run `./isaaclab.sh -f` before committing and again before pushing.
- Do not push to `origin`; push only to the fork remote `antoine` after the maintainer has reviewed the benchmark decision.

---

### Task 1: Capture eligible implicit actuator execution automatically

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection.py`

- [ ] **Step 1: Add focused tests before changing production code**

  Add CUDA-gated tests whose names contain `implicit_cuda_graph` and reuse the existing `FakeActuatorControl`, `_assign_deterministic_inputs`, and exact-output helpers:

  1. `test_implicit_cuda_graph_matches_eager_and_replays_changed_inputs` constructs one eager collection by temporarily disabling `ActuatorCollection._ENABLE_IMPLICIT_CUDA_GRAPH` and one graph-enabled collection from the same exact built-in implicit configuration. Verify graph state is empty before the first compute, the first graph compute creates a graph, subsequent computes retain the same graph object, input/output pointers remain unchanged, and graph results exactly match eager results before and after in-place command/state changes.
  2. `test_implicit_cuda_graph_yields_to_outer_capture` first creates a local graph, then enters `wp.ScopedCapture` on the collection device and calls `compute()`. Guard `wp.capture_launch` so replay during active capture fails the test. Verify the local graph object is unchanged, mutate inputs in place, replay the outer graph, and compare exactly against an eager reference. Repeat the capture assertion with a fresh eligible collection and verify outer capture leaves its local graph unset.
  3. `test_implicit_cuda_graph_capture_failure_warns_once_and_falls_back` replaces the module's `wp.ScopedCapture` with a context manager that raises. Call `compute()` twice, verify a single capture attempt and warning, permanent failure state, no graph, and exact equality with an eager reference on both calls.
  4. `test_implicit_cuda_graph_eligibility_is_narrow` verifies the optimization is disabled for CPU, mixed implicit/explicit CUDA, custom `ImplicitActuator` subclass CUDA, and backend-native CUDA collections. Calling `compute()` must not attempt local capture for any ineligible case.

- [ ] **Step 2: Run the focused tests and preserve the RED evidence**

  Run:

  ```bash
  ./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection.py -k "implicit_cuda_graph" -q
  ```

  Confirm the new tests fail because automatic collection graph state and behavior do not exist yet. Record the failing summary in the task report before editing production code.

- [ ] **Step 3: Add construction-time eligibility and private graph state**

  In `ActuatorCollection`:

  - import `ClassVar` and add private class constant `_ENABLE_IMPLICIT_CUDA_GRAPH: ClassVar[bool] = True`;
  - after `_build_execution_batches()`, initialize `_implicit_cuda_graph: wp.Graph | None = None` and `_implicit_cuda_graph_capture_failed = False`;
  - resolve `_use_implicit_cuda_graph` once from the class switch, CUDA device, nonempty execution batches, exact `ImplicitActuator` batch types, and inactive `control.native_active`;
  - keep this state private and undocumented as public API.

- [ ] **Step 4: Add the collection graph execution helper**

  Add `_run_implicit_batches()` to execute every batch through the unchanged `_compute_implicit_batch()` method in order. Add `_compute_implicit_graph()` with this precise precedence:

  1. when the Warp device reports `is_capturing`, call `_run_implicit_batches()` and return without touching local graph state;
  2. when a local graph exists, call `wp.capture_launch()` and return;
  3. when capture previously failed, call `_run_implicit_batches()` and return;
  4. otherwise use `wp.ScopedCapture(device=self.device, force_module_load=True)` around `_run_implicit_batches()`, store `capture.graph`, then replay it for the current call;
  5. catch capture exceptions, mark failure before issuing a single `logger.warning(...)`, and execute the current call eagerly.

  In `compute()`, retain `compute_native_actuators()` as the first operation. Route eligible collections through `_compute_implicit_graph()` and return; leave the current loop byte-for-byte equivalent for all ineligible collections.

- [ ] **Step 5: Run focused and neighboring tests GREEN**

  Run:

  ```bash
  ./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection.py -k "implicit_cuda_graph" -q
  ./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection.py -q
  ./isaaclab.sh -p -m pytest source/isaaclab/test/utils/warp/test_launch_cache.py -q
  ```

  Confirm all focused tests pass and the full collection plus launch-cache suites remain green.

- [ ] **Step 6: Review, format, and commit the focused implementation**

  Inspect `git diff --check` and the complete diff. Run:

  ```bash
  ./isaaclab.sh -f
  ```

  If hooks modify files, review and stage those edits, then rerun `./isaaclab.sh -f`. Stage only the two task files and commit:

  ```bash
  git commit -m "Capture implicit actuator execution"
  ```

  Do not push yet. Record the commit SHA and verification output in the task report.

---

### Task 2: Benchmark graph replay against cached eager execution

**Files:**

- Create locally under: `benchmarks/actuator-collection-refactor/2026-08-06-implicit-graph/`
- Do not commit benchmark output unless the maintainer explicitly asks for it.

- [ ] **Step 1: Establish GPU readiness and immutable comparison identity**

  Record the implementation commit SHA, clean worktree status, `nvidia-smi` state, dependency versions, and hardware. Reject attempts that violate the telemetry gates in the design specification.

- [ ] **Step 2: Build one hashed same-revision mode launcher**

  Create a launcher that accepts `--mode eager|graph`, sets `ActuatorCollection._ENABLE_IMPLICIT_CUDA_GRAPH` before constructing the environment, then invokes `isaaclab.benchmark.entrypoints.runtime.run()` with untouched remaining arguments. Save its SHA-256 alongside every result. Use this same source revision and launcher for both modes.

- [ ] **Step 3: Run an actuator-only exactness and latency microbenchmark**

  Construct the exact built-in implicit collection with 4096 environments on CUDA, warm all kernels, and compare cached eager versus automatic graph replay. Mutate preallocated commands/states in place, verify exact outputs, verify no recapture or pointer change, and report host submission plus synchronized elapsed latency with dispersion.

- [ ] **Step 4: Validate that graph mode really captures**

  Run the untimed smoke from the design specification and assert graph mode owns a non-`None` `_implicit_cuda_graph` without failure while eager mode owns none. Profile one graph member with 100 warm-up and 200 measured steps under Nsight Systems and retain graph-launch evidence from the exported CUDA API tables.

- [ ] **Step 5: Run the 4096-environment PhysX Franka A/B protocol**

  Execute twelve fresh-process counterbalanced pairs, six eager-then-graph and six graph-then-eager, using:

  ```text
  --task Isaac-Reach-Franka --num_envs 4096
  --warmup_steps 100 --num_steps 1000 --seed 42
  --visualizer none --benchmark_formatter schema
  physics=isaacsim_physx
  ```

  Preserve all accepted and rejected attempts. Report paired graph/eager latency ratio, per-mode median latency, throughput, dispersion, and the seed-42 5000-resample paired bootstrap 95% confidence interval.

- [ ] **Step 6: Present evidence for the maintainer's keep/remove decision**

  Summarize exactness, graph validation, microbenchmark results, end-to-end Franka results, noise, memory cost, and implementation complexity without applying an arbitrary minimum-win threshold. Do not push until the maintainer decides whether this optimization remains in PR 6839.
