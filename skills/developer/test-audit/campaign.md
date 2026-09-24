# Test-Pruning Campaign

Campaign mode prunes one package's or subsystem's whole test surface in one PR, such as `source/isaaclab_newton/test`. The value bar, retention bar, candidate evidence, and validation in [SKILL.md](SKILL.md) apply to every lane. This file adds the order of work. Each step ends on its completion criterion; do not start the next step early.

## 1. Baseline

Record the area's test line counts and every test file's pass/fail state and wall time at a pinned `develop` SHA, one process per file as CI runs them. Run once with a warm and once with a cold Warp kernel cache (`WARP_CACHE_PATH` pointing at an empty directory) so kernel compilation is not mistaken for test cost. Keep baseline failures in their own list; they may be real product bugs, not stale tests.

Done when every in-scope test file has a recorded baseline result.

## 2. Lanes and inventory

Split the surface into **lanes** along production owner boundaries, not file names: for a backend package, assets, sensors, physics managers, cloner, renderers, and sim utilities. Include the area's cases in cross-backend interface and contract tests under `source/isaaclab/test`.

Done when every test file the area owns belongs to exactly one lane.

## 3. Read-only ledger per lane

Give each lane to its own read-only agent. The agent reads every assigned test in full, including parametrize tables, and the production owners, entry points, callers, history, and CI routing. Each test goes into a written **ledger** with one mark. A parametrized test is one entry unless its rows need different marks; then mark each row or axis.

- `R`: retain, naming the contract and the bug it catches;
- `F`: retain the contract but repair the assertion, such as a negative check that passes when only one of several items is missing;
- `C`: consolidate, naming the keeper that absorbs the assertion: a sibling parametrize row, a test that already builds the same fixture, or a shared contract test;
- `D`: delete, naming the proof that remains, or why no contract exists.

Judge a test by its assertions, not its name.

Done when every entry in the lane has a mark and an evidence line.

## 4. Layer plan per lane

Treat the ledger as input, not as the edit list. A second read-only pass looks for the redundant **layer**, such as a backend suite replaying a backend-independent helper around a stronger cross-backend contract test. Name the **keeper** for each contract, and prefer the real simulation boundary over a mocked view. Correct any ledger errors this pass finds.

Done when each lane plan names its retired tests, its keeper per contract, the assertions to carry into keepers, and the test-only production seams unlocked.

## 5. Cutover

Edit lane by lane. Serialize changes to shared fixtures and test utilities through one owner. With each lane, remove the test-only production seams it unlocks. Update `tools/test_settings.py` entries (timeouts, skips, quarantines) for renamed or removed files. Put durable test-ownership rules in the relevant `AGENTS.md`, drawn from mistakes this campaign actually found.

Done when every lane plan is applied and each lane's keepers pass.

## 6. Preservation review

Before claiming completion, have independent reviewers compare deleted coverage against the keepers, one reviewer per lane group. They look for contracts that lost their only proof, and for new assertions that cannot fail.

For each restored contract, make one deliberate **mutation** of the production owner and confirm the keeper goes red. Then restore the source exactly.

Done when every reported gap is restored or rejected with source evidence, and every restored contract has a caught mutation.

## 7. Product defects

A baseline failure that survives into a keeper is a bug report. Fix it at its owner as a separate commit, with a regression test that fails before the fix. Record unrelated discrepancies as follow-ups instead of fixing them in the campaign.

Done when each repaired defect has a failing control and a passing candidate on the same test.

## 8. Reconcile and hand off

Campaigns outlive many `develop` commits. Merge `develop` rather than rebasing a long campaign. When `develop` modified a test the campaign deleted, keep the deletion and port the new contract into the keeper. Rerun the whole area on the merged head.

Hand off with the [SKILL.md](SKILL.md) report, plus:

- baseline and final test line counts and per-file wall times, with production counted separately;
- lanes, retired layers, and keepers;
- preservation gaps found and their mutations;
- product defects with control and candidate proof.
