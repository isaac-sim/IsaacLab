---
orphan: true
---

<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Developer Tools benchmarking documentation design

## Summary

Add a top-level documentation section named **Developer Tools**. In the first
iteration, the section contains the benchmark documentation only. It serves
both developers who build projects with Isaac Lab and contributors who work on
Isaac Lab itself.

The section introduces a short **Benchmarking** landing page. The landing page
helps readers choose the correct workflow, then routes them to one of three
detailed guides:

- Run benchmarks.
- Write micro-benchmarks.
- Use the benchmark API.

Published reinforcement-learning performance results remain under the
reinforcement-learning overview.

## Goals

- Give developer-focused tooling a clear place in the top-level navigation.
- Help readers select the correct benchmark workflow quickly.
- Preserve the current separation between workflow, micro-benchmark, and API
  documentation.
- Make the benchmark documentation concise and easy to scan.
- Use tables, runnable examples, terminal output, and small diagrams to carry
  technical information.
- Preserve measurement definitions, defaults, units, warnings, and other
  technical details.
- Preserve existing benchmark page URLs and Sphinx reference labels.

## Non-goals

- Do not reorganize the full Developer's Guide or Testing sections.
- Do not move the existing benchmark source files in this iteration.
- Do not change benchmark code, public command-line behavior, metrics, schemas,
  or output formats.
- Do not rewrite or relocate the published reinforcement-learning performance
  results.
- Do not add a Sphinx extension or another documentation dependency.
- Do not turn Developer Tools into a complete catalog of developer tooling in
  this iteration.

## Audience

The documentation serves two overlapping audiences:

1. Isaac Lab users who need to measure their environments, policies, training
   runs, or custom operations.
2. Isaac Lab contributors who need to find regressions, automate measurements,
   or extend the benchmark framework.

The landing page serves both audiences. The detailed guides separate common
workflows from framework extension so that advanced material does not block the
quick path.

## Information architecture

Add **Developer Tools** as a top-level navigation caption after **Features**.
The initial navigation is:

```text
Developer Tools
└── Benchmarking
    ├── Run benchmarks
    ├── Write micro-benchmarks
    └── Use the benchmark API
```

The Developer Tools toctree links to a new Benchmarking landing page. The
landing page's toctree contains the three benchmark guides. The guides stop
appearing in the Testing toctree. Testing keeps its remaining testing-specific
material, including mock interfaces.

Keep the existing guide files at their current paths during this iteration.
This changes their navigation parent without changing their published URLs.
Keep their existing Sphinx labels so that `:ref:` links remain valid.

The published performance-results page stays under Reinforcement Learning. The
Benchmarking landing page links to it as an example of published results. The
performance-results page gets a short link back to the benchmark guide.
Rewriting its results or methodology is outside this iteration.

## Page responsibilities

Each page answers one primary reader question.

| Page | Reader question | Content |
| --- | --- | --- |
| Benchmarking | Which tool should I use? | Decision table, minimal command, workflow diagram, and links |
| Run benchmarks | How do I measure Isaac Lab? | Runtime, play, training, startup, metrics, comparison rules, and output |
| Write micro-benchmarks | How do I measure one operation? | Available suites, inputs, timing protocol, result interpretation, and adding a case |
| Use the benchmark API | How do I automate or extend benchmarking? | Typed requests, results, formatters, framework lifecycle, and custom producers |

### Benchmarking landing page

The landing page is a decision page rather than a fourth detailed guide. Its
first screen contains:

1. A one-sentence definition of Isaac Lab benchmarking.
2. One minimal command that writes a human-readable summary.
3. A decision table that maps questions to workflows and detailed guides.
4. A small data-flow diagram.

The decision table covers these routes:

| Question | Workflow or tool | Detailed guide |
| --- | --- | --- |
| Measure environment stepping | `runtime` | Run benchmarks |
| Measure trained-policy inference and rollout | `play` | Run benchmarks |
| Measure end-to-end learning | `training` | Run benchmarks |
| Measure launch and initialization | `startup` | Run benchmarks |
| Measure one asset or sensor operation | Micro-benchmark suite | Write micro-benchmarks |
| Automate a workflow or add a producer | Python API | Use the benchmark API |

Use a small text diagram to show the common flow:

```text
workload -> warm-up -> measurement -> summary/schema output -> comparison
```

### Run benchmarks

Organize the workflow guide in the order readers perform the work:

1. Choose a workflow.
2. Run a minimal benchmark.
3. Read the result.
4. Understand the measurement boundary.
5. Change common options.
6. Compare runs correctly.
7. Use the output in automation.
8. Troubleshoot failures.

Use a workflow table to compare `runtime`, `play`, `training`, and `startup`.
Use a metric table to state what each rate includes and excludes. Keep exact
warm-up behavior, measured call counts, null result semantics, and comparison
requirements. Avoid repeating the same prose in every workflow section when a
shared table states the rule more clearly.

### Write micro-benchmarks

Organize the guide around the micro-benchmark lifecycle:

1. Choose a suite and backend.
2. Run an existing benchmark.
3. Understand warm-up and timing.
4. Read console and exported results.
5. Add a benchmark case.
6. Avoid common measurement mistakes.
7. Troubleshoot environment and device failures.

Keep the list of supported suites, modes, configuration arguments, output
fields, and backend-specific behavior. Prefer tables for repeated suite and
argument data. Use maintained source files through `literalinclude` where a
full implementation example is useful.

### Use the benchmark API

Start with the supported typed workflow API. Explain the lower-level framework
only after the typed requests and results. The page order is:

1. Decide whether the Python API is needed.
2. Run one complete typed workflow.
3. Match requests to runners and result types.
4. Configure output and consume generated paths.
5. Choose a formatter.
6. Add a custom producer only when no supported workflow answers the question.
7. Handle lifecycle constraints and errors.
8. Test the integration.

Use one complete, runnable Python example before smaller focused examples. Use
tables for request fields, result fields, and formatters. Keep process-lifecycle
constraints and stable-schema requirements explicit.

## Writing style

Apply the following rules throughout the benchmark documentation:

- Start with the action or answer.
- Put one main idea in each sentence.
- Prefer active voice and concrete verbs.
- Use sentence-case, task-based headings.
- Remove filler adjectives such as "comprehensive," "powerful," "seamless,"
  and "flexible."
- Replace repeated prose with a table when several items share the same
  structure.
- Show one canonical command first. Show alternatives only when they help the
  reader complete the task.
- Keep commands complete and copyable.
- Define a metric beside the first example that uses it.
- State what measurements include and exclude.
- Preserve units, defaults, exact field names, and warnings.
- Move advanced details later in a page instead of deleting them.
- Link to the API reference instead of duplicating full public API definitions.

## Visual language

Visual elements must explain a relationship or save the reader from repetitive
prose.

- Use tables for workflow selection, metric boundaries, arguments, output
  formats, and API mappings.
- Place short terminal output near the command that produced it.
- Use small text diagrams for timing and data flow.
- Use tabs only for supported command variants that readers are likely to need.
- Put long JSON, complete scripts, and advanced examples in dropdowns.
- Prefer runnable scripts or `literalinclude` blocks over copied code that can
  drift from the implementation.
- Do not add decorative screenshots or diagrams that duplicate nearby text.
- Do not add another diagramming or documentation dependency.

## Compatibility

This iteration changes navigation, page titles, structure, and prose. It does
not relocate existing benchmark guide files. Their URLs therefore remain
stable. Existing labels remain on their respective pages, even if visible page
titles change.

Internal links should use Sphinx references or document links rather than raw
HTML paths. The new landing page must link to each detailed guide and to the
published reinforcement-learning performance results.

## Validation

Review the content and build output against these checks:

- A reader can select the correct workflow from the landing-page table.
- Every command is complete and matches the current CLI.
- Every documented metric states its scope and unit.
- Warm-up, timing, null-value, and comparison semantics remain present.
- Every long example adds information that is not already clear from a table or
  short example.
- Existing benchmark URLs and Sphinx labels remain valid.
- The new pages introduce no broken cross-references or build warnings.
- The documentation build succeeds.
- `./isaaclab.sh -f` passes before the change is committed.

## Initial implementation scope

The implementation should make these focused changes:

1. Add the Developer Tools caption to the root toctree after Features.
2. Add the Benchmarking landing page and its nested toctree.
3. Remove benchmark pages from the Testing toctree while keeping unrelated
   testing documentation there.
4. Rewrite the workflow, micro-benchmark, and API guides according to their
   approved responsibilities.
5. Add the cross-links between benchmark tooling and published RL results.
6. Build and lint the documentation.

Further developer tools can join the section later. Their addition does not
require another top-level navigation redesign.
