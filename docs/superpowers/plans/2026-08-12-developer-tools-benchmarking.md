---
orphan: true
---

<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Developer Tools Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level Developer Tools section that routes readers through concise, visual benchmark workflow, micro-benchmark, and Python API guides without removing the published RL performance results.

**Architecture:** Build from `restack/benchmarking-giga-stack`, where the public benchmark CLI and Python API used by the guides exist. Keep the three detailed guides at their current `docs/source/testing/` paths so published URLs and Sphinx labels remain stable; change only their navigation parent and content. Add one `docs/source/developer-tools/index.rst` decision page as the public entry point.

**Tech Stack:** Sphinx, reStructuredText, MyST for internal plan files, `sphinx_design`, Isaac Lab benchmark CLI, and the `isaaclab.benchmark` Python API.

## Global Constraints

- Use an isolated feature worktree based on `restack/benchmarking-giga-stack`; do not implement this plan on the dirty `antoiner/resolved-preset-banner` worktree.
- Cherry-pick design commit `c4c9cf93ff` and the commit containing this plan into the implementation branch before editing.
- Keep `docs/source/testing/benchmarks.rst`, `docs/source/testing/micro_benchmarks.rst`, and `docs/source/testing/benchmark_framework.rst` at their existing paths.
- Keep the labels `testing_benchmarks`, `testing_micro_benchmarks`, and `testing_benchmark_framework` unchanged.
- Keep the published RL performance-results page under Reinforcement Learning.
- Do not change benchmark code, public command-line behavior, metrics, schemas, output formats, or dependencies.
- Use complete commands through `./isaaclab.sh`; do not document abbreviated commands with `...`.
- Preserve metric definitions, units, defaults, warm-up rules, null semantics, timing boundaries, provenance rules, warnings, and backend differences.
- Use sentence-case headings, active voice, one main idea per sentence, and concrete verbs.
- Use tables for repeated structured data, short text diagrams for flows, and dropdowns for long output or advanced examples.
- Do not add decorative screenshots or a new documentation dependency.
- Do not add a changelog fragment because this plan changes only the repository-level documentation and touches no package.
- Run `./isaaclab.sh -f` before every commit and again before handoff. If hooks modify files, review and stage those changes, then rerun the command.

---

### Task 1: Preserve the published RL results on the benchmark-stack baseline

**Files:**

- Restore: `docs/source/overview/reinforcement-learning/performance_benchmarks.rst`
- Restore: `docs/source/overview/reinforcement-learning/index.rst`
- Restore: `docs/source/_static/benchmarks/cartpole.jpg`
- Restore: `docs/source/_static/benchmarks/cartpole_camera.jpg`
- Restore: `docs/source/_static/benchmarks/g1_rough.jpg`
- Restore: `docs/source/_static/benchmarks/shadow.jpg`
- Restore: `docs/source/refs/release_notes.rst`

**Interfaces:**

- Consumes: benchmark implementation and documentation through `restack/benchmarking-giga-stack`.
- Produces: the existing `source/overview/reinforcement-learning/performance_benchmarks` document and its toctree entry for Task 6 to cross-link.

- [ ] **Step 1: Confirm the baseline removed the results page**

Run:

```bash
test ! -f docs/source/overview/reinforcement-learning/performance_benchmarks.rst
! rg -n '^\s+performance_benchmarks$' docs/source/overview/reinforcement-learning/index.rst
```

Expected: both commands return success because commit `9193fcf382` removed the page and its navigation entry.

- [ ] **Step 2: Restore the exact pre-removal files**

Run:

```bash
git restore --source=9193fcf382^ -- \
    docs/source/overview/reinforcement-learning/performance_benchmarks.rst \
    docs/source/overview/reinforcement-learning/index.rst \
    docs/source/_static/benchmarks/cartpole.jpg \
    docs/source/_static/benchmarks/cartpole_camera.jpg \
    docs/source/_static/benchmarks/g1_rough.jpg \
    docs/source/_static/benchmarks/shadow.jpg \
    docs/source/refs/release_notes.rst
```

This reverses only the removal of the results page. Do not rewrite its tables, measurements, or methodology.

- [ ] **Step 3: Verify the restored page and assets**

Run:

```bash
test -f docs/source/overview/reinforcement-learning/performance_benchmarks.rst
rg -n '^\s+performance_benchmarks$' docs/source/overview/reinforcement-learning/index.rst
test -f docs/source/_static/benchmarks/cartpole.jpg
test -f docs/source/_static/benchmarks/cartpole_camera.jpg
test -f docs/source/_static/benchmarks/g1_rough.jpg
test -f docs/source/_static/benchmarks/shadow.jpg
git diff --check
```

Expected: every file check succeeds, the RL index prints the restored toctree entry, and `git diff --check` prints nothing.

- [ ] **Step 4: Run pre-commit and commit the retained page**

Run:

```bash
./isaaclab.sh -f
git add docs/source/overview/reinforcement-learning/performance_benchmarks.rst \
    docs/source/overview/reinforcement-learning/index.rst \
    docs/source/_static/benchmarks/cartpole.jpg \
    docs/source/_static/benchmarks/cartpole_camera.jpg \
    docs/source/_static/benchmarks/g1_rough.jpg \
    docs/source/_static/benchmarks/shadow.jpg \
    docs/source/refs/release_notes.rst
./isaaclab.sh -f
git commit -m "docs: retain RL performance results"
```

Expected: both pre-commit runs pass and the commit contains only the seven restored paths.

### Task 2: Add the Developer Tools navigation and benchmark landing page

**Files:**

- Create: `docs/source/developer-tools/index.rst`
- Modify: `docs/index.rst`
- Modify: `docs/source/testing/index.rst`

**Interfaces:**

- Consumes: the stable labels `testing_benchmarks`, `testing_micro_benchmarks`, and `testing_benchmark_framework` from the detailed guides.
- Produces: the new stable label `developer_tools_benchmarking` and the only navigation parent for the three detailed benchmark guides.

- [ ] **Step 1: Run the navigation contract before editing**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; root = Path('docs/index.rst').read_text(); landing = Path('docs/source/developer-tools/index.rst'); assert ':caption: Developer Tools' in root and landing.exists()"
```

Expected: FAIL with `AssertionError` because the Developer Tools caption and landing page do not exist.

- [ ] **Step 2: Create the landing page**

Create `docs/source/developer-tools/index.rst` with this structure and content:

```rst
.. _developer_tools_benchmarking:

Benchmarking
============

Measure Isaac Lab workloads, compare changes, and catch performance
regressions with the benchmark CLI and Python API.

Run a benchmark
---------------

Start with a runtime benchmark. This command measures 1000 environment steps
after 50 warm-up steps and prints a summary:

.. code-block:: bash

   ./isaaclab.sh benchmark runtime \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --warmup_steps 50 \
       --num_steps 1000 \
       --benchmark_formatter summary \
       --output_path ./benchmark_results \
       physics=isaacsim_physx

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 38 20 42

   * - Question
     - Tool
     - Continue with
   * - How fast does an environment step?
     - ``runtime``
     - :ref:`testing_benchmarks`
   * - How fast does a trained policy run?
     - ``play``
     - :ref:`testing_benchmarks`
   * - How fast does a policy train?
     - ``training``
     - :ref:`testing_benchmarks`
   * - Where is startup time spent?
     - ``startup``
     - :ref:`testing_benchmarks`
   * - How fast is one asset or sensor operation?
     - Micro-benchmark
     - :ref:`testing_micro_benchmarks`
   * - How do I automate or extend a benchmark?
     - Python API
     - :ref:`testing_benchmark_framework`

From workload to comparison
---------------------------

.. code-block:: text

   workload -> warm-up -> measurement -> summary/schema output -> comparison

Use the same workload, hardware, software revision, and measurement mode on
both sides of a comparison. The detailed guides define each timing boundary
and the provenance required for a valid result.

Guides
------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Run benchmarks
      :link: testing_benchmarks
      :link-type: ref

      Measure runtime, policy playback, training, or startup.

   .. grid-item-card:: Write micro-benchmarks
      :link: testing_micro_benchmarks
      :link-type: ref

      Isolate one asset method, data property, or sensor update.

   .. grid-item-card:: Use the benchmark API
      :link: testing_benchmark_framework
      :link-type: ref

      Run workflows from Python or add a benchmark producer.

Published results
-----------------

See :doc:`/source/overview/reinforcement-learning/performance_benchmarks` for
published reinforcement-learning performance results.

.. toctree::
   :hidden:
   :maxdepth: 1

   Run benchmarks <../testing/benchmarks>
   Write micro-benchmarks <../testing/micro_benchmarks>
   Use the benchmark API <../testing/benchmark_framework>
```

- [ ] **Step 3: Add the top-level navigation entry**

In `docs/index.rst`, add this toctree immediately after the Features toctree and before Experimental Features:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Developer Tools

   source/developer-tools/index
```

Keep Concepts, Features, and Experimental Features in their existing order.

- [ ] **Step 4: Remove benchmark pages from the Testing toctree**

Change `docs/source/testing/index.rst` so its toctree contains only `mock_interfaces`. Replace its introduction with two short sentences: the first says that the section covers test utilities for Isaac Lab development; the second links benchmark readers to `:ref:`developer_tools_benchmarking``.

- [ ] **Step 5: Verify navigation and build references**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; root = Path('docs/index.rst').read_text(); landing = Path('docs/source/developer-tools/index.rst').read_text(); testing = Path('docs/source/testing/index.rst').read_text(); assert ':caption: Developer Tools' in root; assert '.. _developer_tools_benchmarking:' in landing; assert all(label in landing for label in ('testing_benchmarks', 'testing_micro_benchmarks', 'testing_benchmark_framework')); assert 'benchmarks\n' not in testing and 'micro_benchmarks' not in testing and 'benchmark_framework' not in testing"
./isaaclab.sh -p -m sphinx -b dummy -W --keep-going docs /tmp/isaaclab-developer-tools-dummy
```

Expected: the contract passes and Sphinx finishes with no warnings or missing references.

- [ ] **Step 6: Run pre-commit and commit the navigation**

Run:

```bash
./isaaclab.sh -f
git add docs/index.rst docs/source/developer-tools/index.rst docs/source/testing/index.rst
./isaaclab.sh -f
git commit -m "docs: add Developer Tools benchmark hub"
```

Expected: both pre-commit runs pass and the commit contains only the navigation and landing-page files.

### Task 3: Rewrite the end-to-end benchmark guide

**Files:**

- Modify: `docs/source/testing/benchmarks.rst`

**Interfaces:**

- Consumes: the `runtime`, `play`, `training`, and `startup` CLI workflows from the benchmark stack.
- Produces: the `testing_benchmarks` guide used by the landing page, micro-benchmark guide, framework guide, and RL results page.

- [ ] **Step 1: Run the page-structure contract before editing**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; text = Path('docs/source/testing/benchmarks.rst').read_text(); assert 'Run benchmarks\n==============' in text; assert 'Choose a workflow\n-----------------' in text; assert 'Compare runs\n------------' in text"
```

Expected: FAIL with `AssertionError` because the current title and section layout do not match the approved reading order.

- [ ] **Step 2: Rewrite the opening and workflow-selection content**

Keep `.. _testing_benchmarks:` unchanged. Rename the visible page to `Run benchmarks`. Open with two sentences: one states that the page covers runtime, play, training, and startup; the other directs isolated-operation readers to `:ref:`testing_micro_benchmarks`` and Python users to `:ref:`testing_benchmark_framework``.

Replace `Choose A Workflow` with sentence-case `Choose a workflow`. Use one table with these rows and boundaries:

| Workflow | Use it to measure | Primary result | Excludes from the primary result |
| --- | --- | --- | --- |
| `runtime` | Environment-step capacity under random actions | Environment-step FPS | Policy inference and learning |
| `play` | Trained-policy rollout | Collection FPS | Learning updates |
| `training` | End-to-end learning | Total FPS and learning metrics | Startup |
| `startup` | Launch, imports, configuration, scene creation, first step | Phase duration [s] | Steady-state throughput |

Follow the table with three concise comparison rules: keep workload and provenance fixed, compare identical schema fields and measurement modes, and separate cold startup from steady state.

- [ ] **Step 3: Put each workflow in the same task order**

For runtime, play, training, and startup, use the same subsections in this order:

1. `When to use it`
2. `Run it`
3. `Read the result`
4. `What not to infer`

Keep one complete command per workflow. Preserve every existing argument and exact selector in those canonical commands. Put the first relevant summary or schema excerpt directly after `Read the result`. Move long canonical-workstation output and provenance into dropdowns; do not delete their values.

State these required facts in the matching workflow sections:

- Runtime random-action generation is outside `env.step()` timing.
- Play requires a compatible checkpoint and can return null episode metrics when no episode completes.
- Training collection FPS excludes policy updates while total FPS includes them.
- Startup phase durations are cold-start diagnostics, not throughput.
- Warm-up affects only the fields documented by each workflow; do not generalize one workflow's warm-up rule to another.

- [ ] **Step 4: Consolidate shared metric and comparison material**

After the workflow sections, use these sentence-case sections in order:

1. `Measurement boundaries`
2. `Rendered workloads`
3. `Physics backends`
4. `Read the output`
5. `Compare runs`
6. `Synchronized-step diagnostics`
7. `Troubleshooting`

In `Measurement boundaries`, keep one table that maps each schema field to its measured scope. In `Read the output`, keep the summary/schema distinction and evidence levels. In `Compare runs`, keep hardware, software, task, seed, environment count, warm-up, measured window, rendering, and power-profile provenance. Preserve the rule requiring at least three independent processes for a performance claim and the warning that synchronized-step diagnostics change the schedule being measured.

Remove duplicate `Use It When`, `Warm-Up`, `Read The Result`, and `Do Not Infer` prose after the shared tables carry the same fact. Do not remove facts listed in the Global Constraints.

- [ ] **Step 5: Verify the rewritten guide**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; text = Path('docs/source/testing/benchmarks.rst').read_text(); required = ('.. _testing_benchmarks:', 'Run benchmarks\n==============', 'Choose a workflow\n-----------------', 'Measurement boundaries\n----------------------', 'Compare runs\n------------', 'Synchronized-step diagnostics\n---------------------------------'); assert all(item in text for item in required); assert all(workflow in text for workflow in ('benchmark runtime', 'benchmark play', 'benchmark training', 'benchmark startup'))"
./isaaclab.sh -p -m sphinx -b dummy -W --keep-going docs /tmp/isaaclab-developer-tools-dummy
git diff --check -- docs/source/testing/benchmarks.rst
```

Expected: the content contract passes, Sphinx reports no warnings, and `git diff --check` prints nothing.

- [ ] **Step 6: Run pre-commit and commit the workflow guide**

Run:

```bash
./isaaclab.sh -f
git add docs/source/testing/benchmarks.rst
./isaaclab.sh -f
git commit -m "docs: simplify benchmark workflow guide"
```

Expected: both pre-commit runs pass and the commit changes only the workflow guide.

### Task 4: Rewrite the micro-benchmark guide

**Files:**

- Modify: `docs/source/testing/micro_benchmarks.rst`

**Interfaces:**

- Consumes: the top-level `./isaaclab.sh microbenchmark` command and backend-specific benchmark suites.
- Produces: the `testing_micro_benchmarks` guide linked from the landing, workflow, and framework pages.

- [ ] **Step 1: Run the page-structure contract before editing**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; text = Path('docs/source/testing/micro_benchmarks.rst').read_text(); assert 'Write micro-benchmarks\n======================' in text; assert 'Choose a suite and backend\n--------------------------' in text; assert 'Add a benchmark\n---------------' in text"
```

Expected: FAIL with `AssertionError` because the current title and section sequence differ.

- [ ] **Step 2: Rewrite the decision path and benchmark inventory**

Keep `.. _testing_micro_benchmarks:` unchanged. Rename the page `Write micro-benchmarks`. Open with the distinction between mock-view asset benchmarks and live-scene sensor benchmarks. State immediately that neither predicts end-to-end throughput.

Use these sentence-case sections first:

1. `Choose a suite and backend`
2. `Run an asset benchmark`
3. `Run a sensor benchmark`

Combine the existing family and backend matrices into one selection table with columns for question, workload, simulation mode, and supported backends. Keep every supported asset component, sensor component, and exact backend selector. Keep one complete canonical command for an asset and one for a sensor in the main flow. Put additional backend command variants in tabs or a dropdown without shortening them.

- [ ] **Step 3: Consolidate arguments, timing, and output interpretation**

Follow the run sections with:

1. `Change the workload`
2. `Understand the timing boundary`
3. `Read the result`
4. `Compare runs`

Use separate asset and sensor argument tables. Preserve defaults, units, modes, diagnostic flags, ray-caster terrain behavior, and output formatter differences. Use the existing text diagram to show that `sim.step()` and validation sit outside sensor timing. Keep definitions for synchronized completion, host submission, observer floor, p50, p95, native read, estimated non-read time, and validation. Keep the example terminal summary and JSON measurement in dropdowns labeled as illustrative rather than reference performance.

In `Compare runs`, preserve the independent-process and three-repetition protocol, designated-workstation rule, raw-artifact requirement, and the warning that contact protocols are not yet identical across backends.

- [ ] **Step 4: Make extension guidance task-based**

Replace `Adding New Benchmarks` with `Add a benchmark`. Split it into `Add an asset case` and `Add a sensor workload`. Keep each existing implementation requirement, including shared generators, `derived_from`, matched backend behavior, warm-up, untimed simulation and validation, `measure_latency`, `LatencyBenchmarkRunner`, observer-floor reporting, metadata, and validation failures.

End with `Troubleshooting`. Keep backend availability, CUDA memory, first-process cost, and noisy-result guidance. Remove repeated architectural prose already expressed by the timing diagram and extension steps.

- [ ] **Step 5: Verify the rewritten guide**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; text = Path('docs/source/testing/micro_benchmarks.rst').read_text(); required = ('.. _testing_micro_benchmarks:', 'Write micro-benchmarks\n======================', 'Choose a suite and backend\n--------------------------', 'Understand the timing boundary\n------------------------------', 'Add a benchmark\n---------------'); assert all(item in text for item in required); assert all(name in text for name in ('articulation', 'rigid_object', 'rigid_object_collection', 'contact_sensor', 'frame_transformer', 'imu', 'pva', 'joint_wrench', 'ray_caster')); assert all(backend in text for backend in ('physics=physx', 'physics=newton_mjwarp', 'physics=newton_kamino', 'physics=ovphysx'))"
./isaaclab.sh -p -m sphinx -b dummy -W --keep-going docs /tmp/isaaclab-developer-tools-dummy
git diff --check -- docs/source/testing/micro_benchmarks.rst
```

Expected: the content contract passes, Sphinx reports no warnings, and `git diff --check` prints nothing.

- [ ] **Step 6: Run pre-commit and commit the micro-benchmark guide**

Run:

```bash
./isaaclab.sh -f
git add docs/source/testing/micro_benchmarks.rst
./isaaclab.sh -f
git commit -m "docs: simplify micro-benchmark guide"
```

Expected: both pre-commit runs pass and the commit changes only the micro-benchmark guide.

### Task 5: Rewrite the benchmark API guide

**Files:**

- Modify: `docs/source/testing/benchmark_framework.rst`

**Interfaces:**

- Consumes: typed requests, workflow runners, `BenchmarkResult`, formatters, measurements, recorders, and lower-level benchmark helpers from `isaaclab.benchmark`.
- Produces: the `testing_benchmark_framework` guide for automation authors and benchmark-framework contributors.

- [ ] **Step 1: Run the page-structure contract before editing**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; text = Path('docs/source/testing/benchmark_framework.rst').read_text(); assert 'Use the benchmark API\n=====================' in text; assert 'Run one workflow\n----------------' in text; assert 'Add a custom producer\n---------------------' in text"
```

Expected: FAIL with `AssertionError` because the current title and section names differ.

- [ ] **Step 2: Lead with one complete supported API example**

Keep `.. _testing_benchmark_framework:` unchanged. Rename the page `Use the benchmark API`. Open by telling day-to-day CLI users to use `:ref:`testing_benchmarks`` and isolated-operation authors to use `:ref:`testing_micro_benchmarks``.

Use `Run one workflow` as the first section. Keep one complete runtime script that imports `BenchmarkLauncherConfig`, `BenchmarkOutputConfig`, `BenchmarkRuntimeRequest`, and `run_runtime_benchmark`; measures 1000 steps after 50 warm-up steps; disables visualizers; writes schema and summary output; prints total FPS; and prints every returned output path. Keep the exact `./isaaclab.sh -p runtime_benchmark.py` command beside it.

- [ ] **Step 3: Replace repeated request prose with API tables**

After the complete example, use these sections:

1. `Choose a request`
2. `Configure a request`
3. `Choose output formats`
4. `Read the result`
5. `Handle errors and process lifetime`

Keep the request-to-runner table for runtime, startup, training, and play. Keep one shared-fields table and one workflow-specific-fields table. Preserve the meanings of `None` and an empty `visualizers` tuple, typed fields versus `backend_args`, all five formatter contracts, `BenchmarkResult.bundle`, `BenchmarkResult.output_paths`, and typed result examples. Keep the rule that each workflow runs in a separate process and the synchronized-step warning.

Move the complete startup, training, and play scripts into separate dropdowns under `Choose a request`. Do not shorten them or merge training and play into one process.

- [ ] **Step 4: Make the extension path explicit and compact**

Replace `Extend The Framework` with `Add a custom producer`. Start with a table of measurement and metadata types. Keep one complete custom CPU workload script. Preserve explicit units, warm-up, sample count, `try/finally`, `finalize()`, returned output paths, `attach_bundle()`, manual recorder updates, and Kit frametime constraints.

Use a short `Choose a lower-level runner` subsection to distinguish `MethodBenchmarkRunner`, `measure_latency`, and `LatencyBenchmarkRunner`. End with `Troubleshooting`. Remove prose that repeats the CLI guide's measurement definitions; link to `:ref:`testing_benchmarks`` instead.

- [ ] **Step 5: Verify the rewritten API guide**

Run:

```bash
./isaaclab.sh -p -c "from pathlib import Path; text = Path('docs/source/testing/benchmark_framework.rst').read_text(); required = ('.. _testing_benchmark_framework:', 'Use the benchmark API\n=====================', 'Run one workflow\n----------------', 'Choose a request\n----------------', 'Add a custom producer\n---------------------'); assert all(item in text for item in required); assert all(symbol in text for symbol in ('BenchmarkRuntimeRequest', 'BenchmarkStartupRequest', 'BenchmarkTrainingRequest', 'BenchmarkPlayRequest', 'BenchmarkResult', 'BaseIsaacLabBenchmark', 'MethodBenchmarkRunner', 'measure_latency', 'LatencyBenchmarkRunner'))"
./isaaclab.sh -p -m sphinx -b dummy -W --keep-going docs /tmp/isaaclab-developer-tools-dummy
git diff --check -- docs/source/testing/benchmark_framework.rst
```

Expected: the content contract passes, Sphinx reports no warnings, and `git diff --check` prints nothing.

- [ ] **Step 6: Run pre-commit and commit the API guide**

Run:

```bash
./isaaclab.sh -f
git add docs/source/testing/benchmark_framework.rst
./isaaclab.sh -f
git commit -m "docs: simplify benchmark API guide"
```

Expected: both pre-commit runs pass and the commit changes only the API guide.

### Task 6: Add cross-links and validate the complete documentation

**Files:**

- Modify: `docs/source/overview/reinforcement-learning/performance_benchmarks.rst`
- Verify: `docs/index.rst`
- Verify: `docs/source/developer-tools/index.rst`
- Verify: `docs/source/testing/index.rst`
- Verify: `docs/source/testing/benchmarks.rst`
- Verify: `docs/source/testing/micro_benchmarks.rst`
- Verify: `docs/source/testing/benchmark_framework.rst`

**Interfaces:**

- Consumes: all navigation labels and page responsibilities established in Tasks 1-5.
- Produces: bidirectional navigation between benchmark tooling and retained results, warning-free HTML, and a clean repository validation result.

- [ ] **Step 1: Add a short tooling backlink to the results page**

At the end of `docs/source/overview/reinforcement-learning/performance_benchmarks.rst`, keep the existing results unchanged and replace any old script instructions with this short section:

```rst
Run these workloads
-------------------

Use the supported workflows in :ref:`developer_tools_benchmarking` to reproduce
these workload types. Record the task, backend, CPU, GPU, software revision,
environment count, seed, warm-up, and measured window with every result.
```

Do not change the result tables, images, hardware descriptions, or recorded values.

- [ ] **Step 2: Run a cross-page content audit**

Run:

```bash
rg -n ':caption: Developer Tools|developer_tools_benchmarking|testing_benchmarks|testing_micro_benchmarks|testing_benchmark_framework|performance_benchmarks' \
    docs/index.rst \
    docs/source/developer-tools/index.rst \
    docs/source/testing/index.rst \
    docs/source/testing/benchmarks.rst \
    docs/source/testing/micro_benchmarks.rst \
    docs/source/testing/benchmark_framework.rst \
    docs/source/overview/reinforcement-learning/performance_benchmarks.rst
rg -n 'comprehensive|powerful|seamless|flexible|In this section|The above|It should be noted' \
    docs/source/developer-tools/index.rst \
    docs/source/testing/benchmarks.rst \
    docs/source/testing/micro_benchmarks.rst \
    docs/source/testing/benchmark_framework.rst
```

Expected: the first command shows every intended navigation path. The second command prints nothing; rewrite any matching sentence in direct English before continuing.

- [ ] **Step 3: Build warning-free HTML outside the repository**

Run:

```bash
./isaaclab.sh -p -m sphinx -b html -W --keep-going -j auto docs /tmp/isaaclab-developer-tools-html
test -f /tmp/isaaclab-developer-tools-html/source/developer-tools/index.html
test -f /tmp/isaaclab-developer-tools-html/source/testing/benchmarks.html
test -f /tmp/isaaclab-developer-tools-html/source/testing/micro_benchmarks.html
test -f /tmp/isaaclab-developer-tools-html/source/testing/benchmark_framework.html
test -f /tmp/isaaclab-developer-tools-html/source/overview/reinforcement-learning/performance_benchmarks.html
```

Expected: Sphinx exits with status 0 and every expected HTML page exists.

- [ ] **Step 4: Check the rendered landing-page elements**

Run:

```bash
rg -n 'Developer Tools|Choose a workflow|Run benchmarks|Write micro-benchmarks|Use the benchmark API|Published results' \
    /tmp/isaaclab-developer-tools-html/source/developer-tools/index.html
rg -n 'testing/benchmarks.html|testing/micro_benchmarks.html|testing/benchmark_framework.html|performance_benchmarks.html' \
    /tmp/isaaclab-developer-tools-html/source/developer-tools/index.html
```

Expected: the first command finds all visible landing-page elements and the second finds all four rendered destinations.

- [ ] **Step 5: Run final repository checks**

Run:

```bash
git diff --check
./isaaclab.sh -f
git status --short --branch
```

Expected: `git diff --check` prints nothing, pre-commit passes without changing files, and status lists only the intended results-page backlink before staging.

- [ ] **Step 6: Commit the backlink and final consistency pass**

Run:

```bash
git add docs/source/overview/reinforcement-learning/performance_benchmarks.rst
./isaaclab.sh -f
git commit -m "docs: link benchmark tools and results"
git log -6 --oneline
```

Expected: pre-commit passes and the log shows six focused documentation commits: retained results, navigation hub, workflow guide, micro-benchmark guide, API guide, and cross-links.
