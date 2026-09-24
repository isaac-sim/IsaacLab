---
name: isaaclab-auditing-tests
description: Gates new or changed Isaac Lab tests at write time and audits existing tests for low-value, implementation-coupled, or duplicated coverage, including parametrize matrices and scene builds that repeat one code path. Use when writing, changing, reviewing, consolidating, or pruning tests, or when a test job is slow because of redundant coverage.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Auditing Tests

## When To Use

Use this skill in one of three modes that share one value bar:

- **Authoring**: gate every new or changed test before it lands.
- **Audit**: sweep one area for tests that re-assert source, duplicate stronger proof, couple to implementation, or keep test-only production seams alive. Optimize for confidence, not deletion count; continue broad audits as separate coherent follow-up PRs.
- **Campaign**: prune one package's or subsystem's whole test surface. Read [campaign.md](campaign.md) before starting one.

Read `AGENTS.md` (especially *Testing and validation*) and any scoped `AGENTS.md` first. This skill decides *whether* a test earns its cost; `isaaclab-following-coding-style` covers how it is written.

## Workflow

### Authoring gate

Before adding a test, answer four questions. A missing answer means do not add it yet:

1. What observable behavior, invariant, or independent contract does it protect?
2. What credible regression makes it fail?
3. Why does existing coverage not already catch that failure? Each contract has one primary test owner at the strongest boundary; another layer needs its own distinct risk. Prefer extending a parametrized case or a shared fixture over a near-duplicate test, and consolidate duplicated setup in the same change.
4. Does it need a production seam (export, flag, wrapper, injection hook) that no production caller needs? If yes, move the test to the real boundary instead.

Then check the test against every junk pattern below; a match fails the gate unless the retention bar names the contract it independently guards. A test that would break under behavior-preserving refactoring asserts implementation, not behavior; rewrite it at the owning boundary before landing it.

Bug regression tests must fail on the pre-fix code for the intended reason and pass after the fix. One regression at the owner boundary covers the bug; do not replay the same scenario at every layer it crosses, or on every backend unless the bug is backend-specific.

### Junk patterns

The authoring gate rejects a new test that matches one; audits hunt for existing tests that do.

- assertion-free coverage probes, including sim steps whose results are never asserted;
- self-comparisons and identity copiers, such as writing a Warp binding and asserting the same binding reads back;
- copied fixtures, inventories, manifests, or export lists;
- exact source, import, or string greps;
- private predicate or call-shape tests duplicated at real boundaries;
- duplicate invocations of the same contract, including two tests that build the same scene to check overlapping properties;
- per-backend replays of a backend-independent helper;
- tests whose only purpose is preserving test-only exports, globals, or wrappers;
- dead production code whose only callers are tests;
- expected values produced by the helper under test, or by repeating the production transformation on the backend's raw output;
- mocks that implement the asserted behavior, or one mock standing in for different APIs;
- fixtures that supply the ordering or state the owner should produce;
- capability tests that restate declared flags or config instead of exercising what the flag promises;
- negative controls that pass for an unrelated reason, such as an error raised by a different guard;
- names or docstrings that promise more than the input exercises.

Isaac Lab parametrization adds its own duplication; treat these as junk unless an axis reaches a distinct code path:

- cartesian products (device × pipeline × shape × `num_envs`) where some axes select no distinct branch. Cover each value once and rotate the other axes across the rows instead;
- CPU and GPU variants of bookkeeping logic whose kernels and Python paths are identical. Scope with `test_devices(DeviceScope.CUDA)` rather than a hard-coded device list, which also ignores the multi-GPU runtime mask;
- a fresh `SimulationContext` or scene per assertion when a sibling test already builds the same fixture.

### Value bar

Tests justify their maintenance cost by protecting behavior, a credible regression, or an independently meaningful contract. In an audit, an existing test that must change for behavior-preserving reorganization is suspect, not automatically deletable.

Before judging a candidate, read the complete test and its production owner, entry point, callers, sibling backend implementations, overlapping tests (including cross-backend interface and contract tests under `source/isaaclab/test`), CI routing in `tools/test_settings.py`, and relevant history. When the test claims dependency-backed behavior (Newton, MuJoCo Warp, PhysX, Warp), inspect the dependency source directly.

### Retention bar

Keep a test when it independently enforces a public API, config, schema, migration, backend-parity, frame or unit convention, determinism, security, platform, default-value, or packaging contract. Also keep:

- call ordering when order is observable behavior;
- regressions with a credible failure mode;
- source inspection when it is the cheapest independent guard of a user-facing contract;
- a retained test that fails on the baseline: treat it as a possible product bug, reproduce it, and fix the owner rather than deleting it.

Static or slow is not a deletion reason. A test that resembles implementation may still be the independent contract; prove otherwise before removing it. Distinct physical fixtures (different assets, contact pipelines, or solver paths) are not duplicates just because the assertions look alike.

### Discovery and evidence

Keep discovery read-only and report evidence before editing. For broad scope, run parallel read-only lanes along owner boundaries: core (`source/isaaclab`), each backend package, tasks and assets, and a cross-cutting pattern sweep. Outside campaign mode, prefer a few high-confidence candidates over a large speculative inventory.

Record every field below before editing a candidate. A missing field means it is not ready:

- exact test name and location;
- what failure it can actually detect;
- non-test callers of the covered production seam;
- stronger remaining owner-boundary proof, or why no proof is needed;
- relevant history and the reason the test exists;
- production or test-support deletion unlocked;
- risk and the focused validation command.

### Edit shape

Choose one coherent owner-boundary batch. Merge assertions into the test that already builds the fixture, keeping every distinct edge case the merged test covered. Delete test-only exports and dead production paths instead of preserving aliases. Do not add replacement tests that restate the same implementation, and do not turn uncertain candidates into deletions to raise the count.

## Validation

1. Run the narrowest owner and sibling tests: `uv run python -m pytest path/to/test_file.py`. CI runs each file in its own process through `tools/conftest.py`, so validate per file.
2. For a removed or merged test, confirm the keeper still fails when the contract breaks; a one-line mutation of the production owner that turns the keeper red is the strongest proof.
3. When the motivation is CI time, time each changed file before and after on the same machine with a warm Warp kernel cache, and separate kernel compilation from test time with a cold-cache run.
4. Run `uv run isaaclab -f`, then `git diff --check`.
5. Add the changelog fragment the change needs (usually `.skip` for test-only changes) with `isaaclab-writing-changelog-fragments`.
6. Inspect `git diff --numstat`; report production changes separately from test and test-support changes.

Commit, push, or open a PR only when authorized, using `isaaclab-preparing-pr-workflow`. Report:

- removed low-value categories and the keeper for each merged contract;
- retained false positives and why they remain valuable;
- focused proof actually run, with before and after timings when relevant;
- production versus test line counts;
- named follow-ups.

## Maintenance

Review this skill when these change:

- `AGENTS.md` testing and validation rules;
- `tools/conftest.py` and `tools/test_settings.py` (per-file CI runner, timeouts, skips);
- `source/isaaclab/isaaclab/test/utils/devices.py` (`test_devices` and `DeviceScope`).

## References

- [campaign.md](campaign.md): order of work for a whole-package pruning campaign.
- Adapted from the OpenClaw `test-audit` skill at commit `80930af` (<https://github.com/openclaw/openclaw/tree/80930af448ebabc84174146b56bc106d37fab3b4/.agents/skills/test-audit>), MIT licensed; see [LICENSE-openclaw.txt](LICENSE-openclaw.txt).
